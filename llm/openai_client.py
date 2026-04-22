from llm.base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """Stub — not implemented."""

    def __init__(self, model: str | None = None):
        raise NotImplementedError(
            "OpenAI provider is not yet implemented. "
            "Set LLM_PROVIDER=anthropic or implement this client."
        )

    def chat(self, messages, tools=None, system=None):
        raise NotImplementedError

    def format_tool_result(self, tool_call_id, content):
        raise NotImplementedError

    def normalize_tools(self, tools):
        raise NotImplementedError
