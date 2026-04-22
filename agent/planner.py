from agent.state import AgentState


_BASE_SYSTEM = """\
You are a research assistant that uses tools to answer questions thoroughly.

## Mandatory tool-call workflow — follow this order every time:

1. **recall(query)** — ALWAYS call this first to check saved notes before anything else.
2. **search(query)** — Search the web for up-to-date information (may return "No results" in offline mode).
3. **summarize(text)** — After gathering any text (from search results OR recall results that have >3 sentences), call summarize to condense the key points.
4. **save_note(content, topic)** — Before writing your final answer, ALWAYS call save_note at least once to preserve the key finding you are about to report. Topic should be a short label like "RAG", "fine-tuning", "transformers", etc.
5. **Final answer** — Only after completing steps 1-4, write your comprehensive final answer WITHOUT calling any tool.

## Important rules:
- Never skip recall() — even if you think you know the answer already.
- Never skip save_note() — always save at least one note before your final answer.
- If search returns no results, still call summarize() on your recall results or your own knowledge summary.
- The topic in save_note must be a short keyword, not a sentence.

{memory_context}\
"""


def build_system_prompt(memory_context: str = "") -> str:
    ctx = f"\n## What you already know\n{memory_context}\n" if memory_context.strip() else ""
    return _BASE_SYSTEM.format(memory_context=ctx)


def build_messages(state: AgentState, short_term_messages: list[dict]) -> list[dict]:
    """Construct the full message list for the next LLM call."""
    messages = list(short_term_messages)

    if not messages:
        messages.append({"role": "user", "content": state.task})

    return messages
