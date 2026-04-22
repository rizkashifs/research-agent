# Research Agent — Claude Code Guide

## Project Purpose
A local Research Agent built from scratch in raw Python to learn agent fundamentals:
tool calling, memory, state management, planning, retrieval, evaluation, and guardrails.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # edit API keys
python main.py "Your research question"
```

## Key Commands
```bash
python main.py "query"            # fresh session
python main.py "query" --continue # restore last session's short-term buffer
python main.py "query" --fresh    # ignore memory, start clean
python eval/run_eval.py           # run all 8 evaluation test cases
```

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` / `openai` / `gemini` |
| `LLM_MODEL` | `claude-haiku-4-5` | Model ID for the active provider |
| `MAX_ITERATIONS` | `8` | ReAct loop iteration cap |
| `SHORT_TERM_MEMORY_SIZE` | `20` | Max messages in short-term buffer |
| `ANTHROPIC_API_KEY` | — | Required when provider=anthropic |
| `OPENAI_API_KEY` | — | Required when provider=openai |
| `GEMINI_API_KEY` | — | Required when provider=gemini |

## Architecture
```
main.py → agent/agent.py (ReAct loop)
              → agent/planner.py  (builds messages each step)
              → agent/state.py    (AgentState dataclass)
              → tools/registry.py (executes tools, builds schemas)
              → llm/factory.py    (returns correct LLM client)
              → memory/short_term.py + long_term.py
              → retrieval/vector_store.py
              → guardrails/input_validator.py + output_validator.py
```

## Week-by-Week Feature Map
- **Week 1**: LLM abstraction, tool registry, ReAct loop
- **Week 2**: Short-term (buffer) + long-term (SQLite) memory
- **Week 3**: ChromaDB vector retrieval + recall tool
- **Week 4**: Pydantic guardrails + eval harness (8 test cases)

## Adding a New Tool
1. Create `tools/my_tool.py` with a `run(args) -> str` function
2. Register it in `tools/registry.py` inside `_register_defaults()`
3. No changes needed in `agent.py`

## Adding a New LLM Provider
1. Create `llm/myprovider_client.py` implementing `BaseLLMClient`
2. Add the provider key to `llm/factory.py`
3. Set `LLM_PROVIDER=myprovider` in `.env`
