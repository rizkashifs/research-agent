from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """Abstract base for all LLM provider clients."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> dict:
        """Send messages and return a normalized response dict.

        Returns:
            {
              "content": str,           # text content (may be empty)
              "tool_calls": [           # list of tool invocations (may be empty)
                {"name": str, "arguments": dict, "id": str}
              ],
              "stop_reason": str,       # "end_turn" | "tool_use" | "max_tokens"
            }
        """

    @abstractmethod
    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        """Return a message dict representing a tool result for this provider."""

    @abstractmethod
    def normalize_tools(self, tools: list[dict]) -> list[dict]:
        """Convert the registry's canonical tool schema to provider format."""
