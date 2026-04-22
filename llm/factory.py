from config import LLM_MODEL, LLM_PROVIDER
from llm.base import BaseLLMClient


def get_llm_client(provider: str | None = None, model: str | None = None) -> BaseLLMClient:
    p = (provider or LLM_PROVIDER).lower()
    m = model or LLM_MODEL

    if p == "anthropic":
        from llm.anthropic_client import AnthropicClient
        return AnthropicClient(model=m)
    elif p == "openai":
        from llm.openai_client import OpenAIClient
        return OpenAIClient(model=m)
    elif p == "gemini":
        from llm.gemini_client import GeminiClient
        return GeminiClient(model=m)
    else:
        raise ValueError(f"Unknown LLM provider: '{p}'. Choose from: anthropic, openai, gemini")
