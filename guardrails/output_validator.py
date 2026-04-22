import re
from agent.state import AgentState
from config import MAX_ITERATIONS


_URL_RE = re.compile(r"https?://[^\s\"'>]+")


def validate_output(state: AgentState, search_urls: list[str] | None = None) -> str:
    """
    Post-process the agent's final answer and return the (possibly annotated) answer.
    Warnings are appended rather than blocking — callers can inspect state.
    """
    answer = state.final_answer.strip()
    warnings: list[str] = []

    if not answer:
        warnings.append("[Warning: agent produced an empty answer]")

    if answer and answer.strip().lower() == state.task.strip().lower():
        warnings.append("[Warning: answer is a verbatim repetition of the query]")

    if not state.is_complete and state.iteration >= MAX_ITERATIONS:
        limit_warning = "[Warning: answer may be incomplete — iteration limit reached]"
        if limit_warning not in answer:
            warnings.append(limit_warning)

    # Hallucination check — URLs in the answer that never appeared in search results
    if search_urls is not None:
        answer_urls = set(_URL_RE.findall(answer))
        allowed = set(search_urls)
        fabricated = answer_urls - allowed
        if fabricated:
            urls_str = ", ".join(sorted(fabricated))
            warnings.append(
                f"[Warning: answer contains URLs not returned by search: {urls_str}]"
            )

    if warnings:
        answer = answer + "\n\n" + "\n".join(warnings)

    return answer


def extract_search_urls(state: AgentState) -> list[str]:
    """Collect every URL that appeared in a search tool observation."""
    urls: list[str] = []
    for step in state.steps:
        if step.action == "search":
            urls.extend(_URL_RE.findall(step.observation))
    return urls
