import anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL
from llm.base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str | None = None):
        self.model = model or LLM_MODEL
        kwargs = {}
        if ANTHROPIC_API_KEY:
            # Session ingress tokens (sk-ant-si-*) use Bearer auth; regular keys use X-Api-Key
            if ANTHROPIC_API_KEY.startswith("sk-ant-si-"):
                kwargs["auth_token"] = ANTHROPIC_API_KEY
            else:
                kwargs["api_key"] = ANTHROPIC_API_KEY
        self.client = anthropic.Anthropic(**kwargs)

    def chat(self, messages, tools=None, system=None):
        kwargs = {"model": self.model, "max_tokens": 4096, "messages": messages}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self.normalize_tools(tools)

        response = self.client.messages.create(**kwargs)

        text_content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {"name": block.name, "arguments": block.input, "id": block.id}
                )

        stop_reason_map = {
            "end_turn": "end_turn",
            "tool_use": "tool_use",
            "max_tokens": "max_tokens",
        }
        return {
            "content": text_content,
            "tool_calls": tool_calls,
            "stop_reason": stop_reason_map.get(response.stop_reason, response.stop_reason),
        }

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }
            ],
        }

    def normalize_tools(self, tools: list[dict]) -> list[dict]:
        """Convert canonical schema → Anthropic tools format."""
        normalized = []
        for t in tools:
            normalized.append(
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
            )
        return normalized
