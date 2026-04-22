from config import SHORT_TERM_MEMORY_SIZE


class ShortTermMemory:
    def __init__(self, max_messages: int = SHORT_TERM_MEMORY_SIZE):
        self.max_messages = max_messages
        self._messages: list[dict] = []

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    def add(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})
        self._trim()

    def add_raw(self, message: dict):
        self._messages.append(message)
        self._trim()

    def _trim(self):
        # Always keep at least the first user message
        if len(self._messages) > self.max_messages:
            first = self._messages[:1]
            rest = self._messages[-(self.max_messages - 1):]
            self._messages = first + rest

    def load(self, messages: list[dict]):
        self._messages = list(messages)
        self._trim()

    def clear(self):
        self._messages = []

    def to_list(self) -> list[dict]:
        return list(self._messages)
