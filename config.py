import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "8"))
SHORT_TERM_MEMORY_SIZE = int(os.getenv("SHORT_TERM_MEMORY_SIZE", "20"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "research_memory.db")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = "research_notes"
LONG_TERM_CONTEXT_NOTES = 5


def _resolve_anthropic_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key:
        return key
    token_file = os.getenv("CLAUDE_SESSION_INGRESS_TOKEN_FILE", "")
    if token_file:
        p = Path(token_file)
        if p.exists():
            return p.read_text().strip()
    return ""


ANTHROPIC_API_KEY = _resolve_anthropic_key()
