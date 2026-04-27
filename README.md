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
By default, the agent runs in offline mode and does not make internet requests.
```bash
python main.py "What is the difference between RAG and fine-tuning?"
```

### 6. Save output to a file (Optional)
You can save the final report automatically to the `results/` directory:
```bash
python main.py "Your query" --output
```

### 7. Enable internet search (Optional)
The agent runs offline by default. Add `--online` when you want it to search
the web for latest or current information:
```bash
python main.py "What are the latest OpenAI model releases?" --online
```

## Features

- **ReAct loop** with configurable iteration cap
- **Tool registry** with offline-first search, DuckDuckGo web search via `--online`, summarize, save_note, recall
- **Short-term memory** — conversation buffer (last N messages)
- **Long-term memory** — SQLite persistence across sessions
- **Vector retrieval** — ChromaDB with sentence-transformers embeddings
- **Guardrails** — Pydantic input validation + output checks
- **Eval harness** — 8 test cases with per-case scoring

## Internet Search for Latest Information

The agent is offline by default. The `search` tool is still registered, but it
returns a disabled/offline message instead of making a network request.

When you pass `--online`, the same tool can search the internet through
`tools/search.py`, which uses DuckDuckGo via the `ddgs` package. Ask a
time-sensitive question with `--online` and the planner will call `search()`
before answering:

```bash
python main.py "What are the latest OpenAI model releases?" --online
python main.py "Find recent news about vector databases this week" --online
```

The search tool supports optional freshness controls:

- `timelimit="d"` for the past day
- `timelimit="w"` for the past week
- `timelimit="m"` for the past month
- `timelimit="y"` for the past year

Make sure dependencies are installed with `pip install -r requirements.txt`.

## Provider Switching

```bash
LLM_PROVIDER=anthropic python main.py "query"   # default
LLM_PROVIDER=openai    python main.py "query"   # raises NotImplementedError (stub)
LLM_PROVIDER=gemini    python main.py "query"   # raises NotImplementedError (stub)
```
