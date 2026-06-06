# Ojas

**The Local-First, Autonomous AI Developer**

Ojas is an incredibly fast, highly capable AI agent that lives entirely on your local machine. Powered by LangChain, LangGraph, and Ollama, Ojas acts as an autonomous developer that can read your codebase, plan complex features, write code, manage its own dependencies, and execute scripts safely inside an isolated Docker sandbox.

Zero cloud APIs. Zero subscription costs. Absolute privacy.

## Core Features

* **Intelligent Multi-Model Routing:** Ojas automatically scans your local Ollama library. It assigns your **largest available model** (e.g., Llama 3 70B, Command-R, etc.) to handle deep reasoning, planning, and web searches. It specifically delegates code generation and tool execution to **Qwen Coder**, ensuring the perfect balance of high-IQ planning and flawless syntax generation.
* **Zero-Setup Local RAG:** Simply open Ojas in any directory. It automatically indexes your `.py`, `.md`, and `.js` files using **ChromaDB** and `nomic-embed-text`. The agent instantly "knows" your entire project without you having to manually link files.
* **Automated Sandbox Execution:** Ojas doesn't just write code; it runs it. It spins up an ephemeral `python:3.11-slim` Docker container, reads its own code to determine required `pip` dependencies, installs them on the fly using cached volumes, executes the script, and learns from the `stderr/stdout` output.
* **Sequential Subagents (`/delegate`):** Tackle massive tasks without melting your RAM. Ojas unloads the primary agent and spins up a temporary sub-agent with a specific goal. Once the sub-agent finishes, it passes the results back to the main thread.
* **Persistent Local Memory:** Uses LangGraph's SQLite `MemorySaver` to checkpoint every state, allowing instant `/rollback` and resume capabilities. High-level facts and user preferences are saved to `MEMORY.md` and injected into the prompt.
* **Beautiful Terminal UI:** Built with Python's `rich` library, providing a glass-box view into the agent's thought process, RAG retrievals, and Docker execution states.

---

## Installation and Auto-Setup

Ojas is designed for a frictionless, zero-config startup.

**Prerequisites:**

* Python 3.11+
* Docker Desktop / Engine
* Ollama

**Installation:**

```bash
pip install -e .
ojas start
```

### The Auto-Boot Sequence

When you run `ojas start` for the first time, the CLI performs a smart environment check:

1. **Ollama Check:** Pings `http://localhost:11434`.
2. **Hardware and Model Discovery:** Runs `ollama list`.
   * It identifies your largest downloaded model to act as the `PLANNER_MODEL`.
   * It checks for `qwen2.5-coder` (or similar Qwen variant) for the `CODER_MODEL`.
   * It checks for `nomic-embed-text` for the RAG engine.
3. **Missing Models:** If `qwen` or `nomic-embed-text` are missing, Ojas provides a friendly recommendation and automatically runs the `ollama pull` commands for you.
4. **Vector DB Initialization:** ChromaDB silently spins up in the background and begins indexing your current working directory.

---

## Architecture (The LangGraph State Machine)

Ojas utilizes LangGraph to create an autonomous "Vibe Coding" loop. The agent is modeled as a cyclic state machine.

### The Agent State

```python
class OjasState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    workspace_context: str       # Snippets injected by ChromaDB
    generated_code: str          # The script Qwen wants to run
    dependencies: List[str]      # e.g., ["requests", "fastapi", "torch"]
    docker_output: str           # The stdout/stderr from the sandbox
    retry_count: int             # Self-correction attempts
    cycle_complete: bool         # Whether the current cycle is done
```

### The Node Workflow

```
User Input --> [retrieval] --> [planner] --> [coder] --> [dependency] --> [executor] --> [evaluator]
                                               ^                                           |
                                               |___________(error? retry)__________________|
                                                            (success? END)
```

1. **Retrieval Node (ChromaDB):** Intercepts the user's prompt, embeds it via `nomic-embed-text`, queries the local ChromaDB vector store, and injects highly relevant project files into the context window.
2. **Planner Node (Biggest Model):** The heavy-weight model analyzes the user request and the RAG context. It outputs a step-by-step architectural plan.
3. **Coder Node (Qwen):** Takes the plan and writes the specific Python script to accomplish the current step.
4. **Dependency Analyzer Node:** A fast, local AST parser reads Qwen's code, extracts third-party imports (like `bs4` or `pandas`), and updates the `dependencies` state.
5. **Execution Node (Docker):** Mounts a persistent named volume (`ojas-pip-cache`), runs `pip install` for any missing dependencies (cached for instant re-use), executes the generated script inside the `python:3.11-slim` container, captures output and destroys the container.
6. **Evaluation Node:** If the `docker_output` contains a Python stack trace or error, the graph automatically loops back to the **Coder Node** to self-correct (max 3 retries). If successful, it renders the output to the user.

---

## Command Line Interface and Tools

Ojas interacts directly with your terminal using standard text, but supports powerful `/` commands and `@` references.

### Context References

Type `@` to instantly inject specific local context before the LangGraph loop starts.

* `@file:src/main.py` -- Injects the full file.
* `@file:src/main.py:10-50` -- Injects specific lines.
* `@folder:tests/` -- Injects the directory tree.
* `@diff` -- Injects your current unstaged Git changes.

### CLI Commands

* `/delegate [task]` -- Pauses the main thread, unloads the main model, and spawns a sequential sub-agent to complete a specific sub-task, merging the result back into memory.
* `/rollback` -- Reverts the LangGraph state to the previous checkpoint if the agent went down the wrong path.
* `/memory [add/remove]` -- Manually adjusts the persistent `MEMORY.md` file.
* `/clear` -- Wipes the current session history and starts a fresh thread.

### Agent Tools

Ojas natively has access to the following tool functions to navigate your system safely:

* `execute_code(script: str)` -- Triggers the Docker sandbox.
* `read_file(path: str)` -- Reads local files.
* `list_directory(path: str)` -- Explores the file system.
* `write_file(path: str, content: str)` -- Modifies local files (prompts user for confirmation via the UI before saving).
* `search_web(query: str)` -- Spawns a headless Playwright instance inside the Docker container to scrape documentation or search results privately.

---

## Project Structure

```
src/ojas/
  __init__.py          # Package metadata
  __main__.py          # python -m ojas entry
  cli.py               # Rich terminal UI + command parsing
  config.py            # Ollama auto-discovery, model assignment
  state.py             # OjasState TypedDict
  graph.py             # LangGraph state machine (nodes + edges)
  context.py           # @file, @folder, @diff reference parser
  delegate.py          # /delegate sub-agent spawner
  utils.py             # Shared helpers
  nodes/
    retrieval.py       # ChromaDB RAG node
    planner.py         # Planner node (biggest model)
    coder.py           # Coder node (Qwen)
    dependency.py      # AST-based dependency analyzer
    executor.py        # Docker sandbox execution
    evaluator.py       # Output evaluation + self-correction
  tools/
    file_ops.py        # read_file, write_file, list_directory
    docker_exec.py     # Docker container management
    web_search.py      # Playwright-in-Docker web scraper
  rag/
    embeddings.py      # nomic-embed-text via Ollama
    indexer.py          # File walker + ChromaDB ingestion
    retriever.py        # Similarity search interface
  memory/
    checkpoint.py      # LangGraph MemorySaver wrapper
    persistent.py      # MEMORY.md read/write
```

---

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
