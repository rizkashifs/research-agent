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

## Daily Email Report

The repo includes a daily report job at `scripts/daily_ai_research_report.py`.
It runs a fixed Senior AI Research Lead query, saves a markdown file under
`results/daily/`, puts the report content in the email body, and attaches the
same `.md` file using SMTP settings from `.env`.

The daily query covers GenAI Ops, MLOps Platform/Infra,
Deployment/Release Engineering, RAG/Data Quality, Cost/Performance Engineering,
Production Reliability/Security, one deep technical concept, and one
architecture design challenge.
The job uses at most 3 research/tool iterations, then performs a final no-tools
synthesis pass if needed so the email still contains a complete report.

Add these email settings to `.env`:

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@example.com
SMTP_PASSWORD=your_app_password
SMTP_USE_TLS=true
EMAIL_FROM=your_email@example.com
EMAIL_TO=your_email@example.com
```

Keep real credentials only in `.env`. The repo ignores `.env`, `.env.*`, and
common secret-file names; commit only placeholder values in `.env.example`.

Run it once manually:

```bash
python scripts/daily_ai_research_report.py --online
```

To generate the file without emailing:

```bash
python scripts/daily_ai_research_report.py --online --no-email
```

On Windows, register a daily scheduled task:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/register_daily_report_task.ps1 -Time 11:00 -Online
```

Without `-Online`, the scheduled job will run in offline mode and will not
perform web lookup.

The scheduled task is configured with `StartWhenAvailable`, `WakeToRun`, and
three retries at 60-minute intervals. If the laptop is asleep or hibernating and
Windows can wake it, the report should run. If the laptop is fully shut down at
11 AM, the task will run when Windows is next available.

## Provider Switching

```bash
LLM_PROVIDER=anthropic python main.py "query"   # default
LLM_PROVIDER=openai    python main.py "query"   # raises NotImplementedError (stub)
LLM_PROVIDER=gemini    python main.py "query"   # raises NotImplementedError (stub)
```
