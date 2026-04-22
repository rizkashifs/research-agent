import re
from pydantic import BaseModel, field_validator, model_validator


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now",
    r"disregard\s+(all\s+)?prior",
    r"forget\s+(all\s+)?previous",
    r"new\s+persona",
    r"act\s+as\s+if\s+you",
    r"jailbreak",
    r"dan\s+mode",
]


class QueryInput(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query must not be empty.")
        return v.strip()

    @field_validator("query")
    @classmethod
    def min_words(cls, v: str) -> str:
        if len(v.split()) < 3:
            raise ValueError("Query must be at least 3 words.")
        return v

    @field_validator("query")
    @classmethod
    def max_length(cls, v: str) -> str:
        if len(v) > 500:
            raise ValueError("Query must not exceed 500 characters.")
        return v

    @field_validator("query")
    @classmethod
    def no_injection(cls, v: str) -> str:
        lower = v.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, lower):
                raise ValueError(
                    "Query contains a prompt injection pattern and was rejected."
                )
        return v


def validate_query(query: str) -> tuple[bool, str]:
    """Returns (is_valid, error_message). error_message is '' if valid."""
    try:
        QueryInput(query=query)
        return True, ""
    except Exception as e:
        return False, str(e)
