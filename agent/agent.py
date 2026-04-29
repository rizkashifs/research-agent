import json
from config import MAX_ITERATIONS
from agent.state import AgentState, Step
from agent.planner import build_system_prompt, build_messages
from llm.factory import get_llm_client
from tools.registry import ToolRegistry


class ResearchAgent:
    def __init__(
        self,
        llm_client=None,
        registry: ToolRegistry | None = None,
        short_term=None,
        long_term=None,
        max_iterations: int = MAX_ITERATIONS,
        system_prompt_override: str | None = None,
    ):
        self.llm = llm_client or get_llm_client()
        self.registry = registry or ToolRegistry()
        self.short_term = short_term
        self.long_term = long_term
        self.max_iterations = max_iterations
        self.system_prompt_override = system_prompt_override

    def run(self, task: str) -> AgentState:
        state = AgentState(task=task)

        if self.system_prompt_override:
            system_prompt = self.system_prompt_override
        else:
            memory_context = self._fetch_memory_context(task)
            internet_enabled = getattr(self.registry, "internet_enabled", True)
            system_prompt = build_system_prompt(memory_context, internet_enabled=internet_enabled)

        # Initialise short-term buffer with the user task if empty
        if self.short_term is not None:
            if not self.short_term.messages:
                self.short_term.add("user", task)
        else:
            # Bare list used as fallback
            self._bare_messages = [{"role": "user", "content": task}]

        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")

        while not state.is_complete and state.iteration < self.max_iterations:
            state.iteration += 1
            print(f"--- Iteration {state.iteration} ---")

            messages = self._get_messages()
            response = self.llm.chat(
                messages=messages,
                tools=self.registry.get_schemas(),
                system=system_prompt,
            )

            thought = response["content"]
            tool_calls = response["tool_calls"]
            stop_reason = response["stop_reason"]

            if thought:
                print(f"Thought: {thought}")

            if stop_reason == "end_turn" or not tool_calls:
                # Final answer
                state.is_complete = True
                state.final_answer = thought
                step = Step(
                    thought=thought,
                    action="final_answer",
                    action_input={},
                    observation="",
                    iteration=state.iteration,
                )
                state.steps.append(step)
                self._add_assistant_message(thought)
                print(f"\nFinal Answer: {thought}\n")
                break

            # Process tool calls
            # Add assistant message with tool use blocks (for Anthropic multi-turn)
            self._add_raw_assistant_response(response, messages)

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                tool_id = tc["id"]

                print(f"Action: {tool_name}({json.dumps(tool_args)})")

                observation = self.registry.execute(tool_name, tool_args)
                print(f"Observation: {observation[:300]}{'...' if len(observation) > 300 else ''}\n")

                state.tool_calls_made.append(tool_name)
                step = Step(
                    thought=thought,
                    action=tool_name,
                    action_input=tool_args,
                    observation=observation,
                    iteration=state.iteration,
                )
                state.steps.append(step)

                # Add tool result to conversation
                tool_result_msg = self.llm.format_tool_result(tool_id, observation)
                self._add_tool_result_message(tool_result_msg)

        if not state.is_complete:
            warning = "[Warning: answer may be incomplete — iteration limit reached]"
            state.final_answer = (state.final_answer + "\n" + warning).strip()
            print(warning)

        return state

    # ------------------------------------------------------------------
    # Internal message management
    # ------------------------------------------------------------------

    def _get_messages(self) -> list[dict]:
        if self.short_term is not None:
            return self.short_term.messages
        return self._bare_messages

    def _add_assistant_message(self, content: str):
        msg = {"role": "assistant", "content": content}
        if self.short_term is not None:
            self.short_term.add_raw(msg)
        else:
            self._bare_messages.append(msg)

    def _add_raw_assistant_response(self, response: dict, messages: list[dict]):
        """Add Anthropic-style assistant message with tool_use blocks."""
        # Re-construct the raw content blocks Anthropic expects
        content_blocks = []
        if response["content"]:
            content_blocks.append({"type": "text", "text": response["content"]})
        for tc in response["tool_calls"]:
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["arguments"],
                }
            )
        msg = {"role": "assistant", "content": content_blocks}
        if self.short_term is not None:
            self.short_term.add_raw(msg)
        else:
            self._bare_messages.append(msg)

    def _add_tool_result_message(self, tool_result_msg: dict):
        if self.short_term is not None:
            self.short_term.add_raw(tool_result_msg)
        else:
            self._bare_messages.append(tool_result_msg)

    def _fetch_memory_context(self, task: str) -> str:
        if self.long_term is None:
            return ""
        try:
            notes = self.long_term.get_recent_notes(limit=5)
            if not notes:
                return ""
            lines = [f"- [{n['topic']}] {n['content']}" for n in notes]
            return "\n".join(lines)
        except Exception:
            return ""
