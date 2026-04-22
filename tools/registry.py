from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict       # JSON Schema object
    fn: Callable


class ToolRegistry:
    def __init__(self):
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

        self.register(
            name="search",
            description="Search the web using DuckDuckGo and return a summary of results.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string."}
                },
                "required": ["query"],
            },
            fn=search_run,
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
