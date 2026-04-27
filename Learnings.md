# Research Agent Learnings

This document explains the internal architecture of the Research Agent and provides a detailed walkthrough of its execution flow.

## Project Structure & File Explanations

### Core Entry Point
- **`main.py`**: The CLI entry point. It orchestrates the entire lifecycle: validating input, setting up memory, choosing offline/online mode, initializing the agent, running the query, validating output, and persisting session data. The agent is offline by default; `--online` enables internet search.

### Agent Logic (`agent/`)
- **`agent.py`**: Contains the `ResearchAgent` class which implements the **ReAct (Reason + Act) loop**. It manages the iteration cycles, handles tool calls, and updates the state.
- **`planner.py`**: Responsible for constructing the system and user prompts. It formats the context (searches, memory, current date, and offline/online mode) into a structure the LLM understands.
- **`state.py`**: Defines `AgentState` and `Step` dataclasses to track the agent's progress, thoughts, actions, and observations during a session.

### Tools (`tools/`)
- **`registry.py`**: The central hub for all tools. It registers tool functions and provides their JSON schemas to the LLM. It defaults to `internet_enabled=False`, so direct `ToolRegistry()` usage is offline unless explicitly enabled.
- **`search.py`**: Performs web searches using DuckDuckGo when online mode is enabled. In offline mode, the registry keeps a `search` tool registered, but swaps in a disabled stub that tells the agent fresh web verification was not performed.
- **`summarize.py`**: Uses a separate LLM call to condense long search results or documents.
- **`save_note.py`**: Allows the agent to save specific findings into the long-term SQLite database.
- **`recall.py`**: Uses the vector store to search through historical notes and memories.

### Memory & Storage (`memory/` & `retrieval/`)
- **`short_term.py`**: Manages the "conversation buffer" (limited by `SHORT_TERM_MEMORY_SIZE`). It keeps track of the current session's back-and-forth.
- **`long_term.py`**: Interfaces with a SQLite database to persist session history and snippets (notes) across different runs.
- **`vector_store.py`**: Uses ChromaDB to store and retrieve high-dimensional embeddings of notes, enabling semantic search (RAG).

### LLM Abstraction (`llm/`)
- **`base.py`**: Defines the `BaseLLMClient` interface that all providers must implement.
- **`factory.py`**: A factory pattern to instantiate the correct client (Anthropic, OpenAI, or Gemini) based on environment variables.
- **`anthropic_client.py`**: Implementation for Anthropic's Claude API.

### Guardrails (`guardrails/`)
- **`input_validator.py`**: Uses Pydantic to ensure the user query is safe and meaningful (e.g., preventing empty or overly short prompts).
- **`output_validator.py`**: Checks the final response for quality, ensures URLs are cited, and cleans up the text.

### Evaluation (`eval/`)
- **`run_eval.py`**: An automated testing script that runs the agent against predefined test cases in `test_cases.json` and scores the results.

---

## What happens when you run `python main.py "query"`?

Here is the step-by-step execution flow of the command:

### 1. Initialization & Input Guardrails
- **`main.py`** starts and parses your query.
- It checks whether `--online` was passed. Without it, offline mode remains active and internet search is disabled.
- It calls **`guardrails/input_validator.py`** to ensure the query is valid.
- It initializes **`LongTermMemory`** (SQLite) and creates a new `session_id`.

### 2. Memory Loading
- If you didn't use `--fresh`, it checks for any existing short-term memory snapshots in the database to provide context from previous interactions.

### 3. Agent & LLM Setup
- **`llm/factory.py`** creates an LLM client (defaulting to Anthropic Claude).
- **`tools/registry.py`** loads all available tools (Search, Summarize, Save Note, Recall). `main.py` passes `internet_enabled=args.online`, so web search is available only when `--online` is used.
- The **`ResearchAgent`** is initialized with these components.

### 4. The ReAct Loop (The "Brain" at Work)
The agent enters a loop (up to `MAX_ITERATIONS`):
- **Step A: Plan/Thought**: The agent sends the current conversation history to the LLM. The LLM generates a "Thought" about what to do next.
- **Step B: Action**: If the LLM decides it needs information, it returns a "Tool Call" (e.g., `search(query="RAG vs fine-tuning")`).
- **Step C: Observation**: The agent executes the requested tool and adds the result to the conversation as an "Observation". In offline mode, a `search` call returns a clear disabled message. In online mode, `search` runs through **`tools/search.py`** and returns DuckDuckGo results with URLs and search time metadata.
- **Step D: Repeat**: The loop continues. The agent might "Summarize" the search results or "Save Note" for future use.

### 5. Final Answer Generation
- Once the agent has enough information (or hits the iteration limit), it generates a final response instead of a tool call.
- The loop terminates, and `state.is_complete` becomes `True`.

### 6. Output Guardrails & Persistence
- **`guardrails/output_validator.py`** scans the final answer to ensure it's high quality and includes necessary citations.
- **`LongTermMemory`** saves a snapshot of the conversation buffer so you can `--continue` later.
- If the `--output` flag was used, the agent slugifies your query and saves the final report as a `.md` file in the `results/` directory.
- The final answer is printed to your terminal.
