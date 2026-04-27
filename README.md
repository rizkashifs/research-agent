# Research Agent

A local Research Agent built from scratch in raw Python — no LangChain, no LlamaIndex.

Covers agent fundamentals hands-on: tool calling, memory, state management, planning, retrieval, evaluation, and guardrails.

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/rizkashifs/research-agent.git
cd research-agent
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy the example environment file and add your API keys:
```bash
cp .env.example .env
```
Edit `.env` and configure your preferences (see `CLAUDE.md` for all options). Minimum requirement is an API key for your chosen provider (e.g., `ANTHROPIC_API_KEY`).

### 5. Run the Research Agent
```bash
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
