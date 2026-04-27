from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict       # JSON Schema object
    fn: Callable


class ToolRegistry:
    def __init__(self, internet_enabled: bool = False):
        self.internet_enabled = internet_enabled
        self._tools: dict[str, ToolDefinition] = {}
        self._register_defaults()

    def register(self, name: str, description: str, parameters: dict, fn: Callable):
        self._tools[name] = ToolDefinition(name, description, parameters, fn)

    def execute(self, name: str, arguments: dict) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        try:
            return self._tools[name].fn(arguments)
        except Exception as e:
            return f"Error running tool '{name}': {e}"

    def get_schemas(self) -> list[dict]:
        """Return canonical tool schemas (provider-agnostic)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def _register_defaults(self):
        from tools.search import run as search_run
        from tools.summarize import run as summarize_run
        from tools.save_note import run as save_note_run
        from tools.recall import run as recall_run

        def offline_search(args: dict) -> str:
            query = args.get("query", "").strip()
            query_text = f" for query: {query}" if query else ""
            return (
                "Internet search is disabled because offline mode is enabled"
                f"{query_text}. Use recall() and existing knowledge instead, and tell the user "
                "that fresh web verification was not performed."
            )

        self.register(
            name="search",
            description=(
                "Internet search is disabled in offline mode. Call this only if you need to "
                "record that fresh web verification was not performed."
                if not self.internet_enabled
                else (
                    "Search the web using DuckDuckGo and return current results with URLs. "
                    "Use this for latest, recent, or time-sensitive information."
                )
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string."},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return, from 1 to 10. Default is 5.",
                        "default": 5,
                    },
                    "timelimit": {
                        "type": "string",
                        "description": (
                            "Optional freshness filter: d=past day, w=past week, "
                            "m=past month, y=past year."
                        ),
                        "enum": ["d", "w", "m", "y"],
                    },
                },
                "required": ["query"],
            },
            fn=search_run if self.internet_enabled else offline_search,
        )

        self.register(
            name="summarize",
            description="Summarize a block of text into key points.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to summarize."},
                    "max_sentences": {
                        "type": "integer",
                        "description": "Maximum number of sentences in the summary (default 5).",
                        "default": 5,
                    },
                },
                "required": ["text"],
            },
            fn=summarize_run,
        )

        self.register(
            name="save_note",
            description="Save an important note or finding to memory for future reference.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The note content to save."},
                    "topic": {
                        "type": "string",
                        "description": "A short topic label for the note (e.g. 'RAG', 'fine-tuning').",
                    },
                },
                "required": ["content", "topic"],
            },
            fn=save_note_run,
        )

        self.register(
            name="recall",
            description=(
                "Search your saved notes for information relevant to a query. "
                "Always call this BEFORE searching the web."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic or question to look up in saved notes.",
                    }
                },
                "required": ["query"],
            },
            fn=recall_run,
        )
