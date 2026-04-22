# Research Agent

A local Research Agent built from scratch in raw Python — no LangChain, no LlamaIndex.

Covers agent fundamentals hands-on: tool calling, memory, state management, planning, retrieval, evaluation, and guardrails.

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python main.py "What is the difference between RAG and fine-tuning?"
```

## Features

- **ReAct loop** with configurable iteration cap
- **Tool registry** with DuckDuckGo search, summarize, save_note, recall
- **Short-term memory** — conversation buffer (last N messages)
- **Long-term memory** — SQLite persistence across sessions
- **Vector retrieval** — ChromaDB with sentence-transformers embeddings
- **Guardrails** — Pydantic input validation + output checks
- **Eval harness** — 8 test cases with per-case scoring

## Provider Switching

```bash
LLM_PROVIDER=anthropic python main.py "query"   # default
LLM_PROVIDER=openai    python main.py "query"   # raises NotImplementedError (stub)
LLM_PROVIDER=gemini    python main.py "query"   # raises NotImplementedError (stub)
```
