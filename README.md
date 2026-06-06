# 31agents

## Introduction

`31agents` is a small experimental project for building AI agents. It currently ships two demo agents:

- **`knowledge-retrival-agent.py`** — a **Retrieval-Augmented Generation (RAG)** agent built on [LangChain](https://www.langchain.com/):
  1. Loads `.txt` documents from `./docs/`.
  2. Splits them into overlapping chunks with `RecursiveCharacterTextSplitter`.
  3. Embeds the chunks and stores them in an in-memory vector store.
  4. Retrieves the top-`k` chunks for a question and feeds them, with a system prompt, to a local LLM served by [Ollama](https://ollama.com/) to produce a grounded answer.
- **`react.py`** — a **ReAct** (Reasoning + Acting) agent backed by the `ReactAgent` class in `agents/`. Tools are passed in as a list of functions; the agent auto-builds the OpenAI-compatible tool-calling schema from each function's signature, type hints, and Google-style `Args:` docstring, then loops over LLM ↔ tool calls until the model returns a final answer. The demo wires it to two sample tools (`add`, `get_weather`) and runs a REPL.

`main.py` is a placeholder entrypoint that simply prints a greeting.

## Prerequisites

- **Python 3.12** (pinned in `pyproject.toml`).
- **[uv](https://docs.astral.sh/uv/)** — dependency management and venv (`uv.lock`, `.venv/`).
- **Ollama** — local chat + embedding models. Install from <https://ollama.com/download> and make sure `ollama serve` is running.
- Ollama models (defaults; override via `.env`):
  - Chat: `qwen3.6:35b` — `ollama pull qwen3.6:35b`
  - Embedding: `mxbai-embed-large` — `ollama pull mxbai-embed-large`

## Setup

1. **Clone**
   ```sh
   git clone git@github.com:netium/31agents.git
   cd 31agents
   ```

2. **Create `.env`** in the project root:
   ```env
   # Ollama endpoint and auth (defaults work for a local install)
   OLLAMA_BASE_URL=http://localhost:11434/v1
   OLLAMA_API_KEY=ollama

   # Models
   OLLAMA_LLM_MODEL=qwen3.6:35b
   OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
   ```
   Add new keys to `.env` only — never commit secrets.

3. **Install dependencies**
   ```sh
   uv sync              # runtime deps
   uv sync --group dev  # adds pytest
   ```

4. **Add knowledge sources** (RAG agent only): drop `.txt` files into `./docs/`.

## Run

```sh
uv run python main.py                       # placeholder
uv run python knowledge-retrival-agent.py   # RAG demo (hard-coded query against ./docs/*.txt)
uv run python react.py                      # ReAct REPL — type a question, /exit to quit
```

The RAG agent prints the answer and the number of retrieved chunks, e.g.:
```
Answer: ...
Sources Len: 4
```

## Testing

The test suite lives in `tests/` and has two tiers:

- **Unit tests** (`test_react_agent.py`) — deterministic, mocked LLM, fast.
- **LLM-as-judge tests** (`test_react_agent_llm_judge.py`, marked `@pytest.mark.llm`) — require a running Ollama; a separate Ollama call evaluates the agent's behavior against a rubric. Slow (~30s+ per test on CPU with `qwen3.6:35b`). Tests auto-skip if Ollama isn't reachable.

```sh
uv run --group dev pytest                  # all tests
uv run --group dev pytest -m "not llm"     # fast unit tests only
uv run --group dev pytest -m llm           # LLM-as-judge only
```

## Using `ReactAgent`

```python
from openai import OpenAI
from agents import ReactAgent

def get_weather(location: str) -> str:
    """Get the current weather for a given location.

    Args:
        location: The location to get the weather for.
    """
    return f"Sunny in {location}."

llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
agent = ReactAgent(llm, tools=[get_weather])
print(agent.call("What's the weather in Paris?"))
```

The function's `__name__` becomes the tool name, type hints become JSON-schema types, and the `Args:` block of the docstring populates per-parameter descriptions.

## Project layout

```
.
├── agents/                          # ReactAgent subpackage
│   ├── __init__.py                  # re-exports ReactAgent
│   ├── react_agent.py               # ReactAgent class
│   └── _tools.py                    # private schema helpers
├── tests/                           # pytest
│   ├── test_react_agent.py          # unit tests
│   └── test_react_agent_llm_judge.py# LLM-as-judge tests (@pytest.mark.llm)
├── main.py                          # placeholder entrypoint
├── knowledge-retrival-agent.py      # RAG agent (LangChain + Ollama)
├── react.py                         # ReAct demo (REPL wired to add + get_weather)
├── docs/                            # drop .txt knowledge files here
├── pyproject.toml                   # project metadata & dependencies
├── uv.lock                          # uv lockfile
├── AGENTS.md                        # agent-friendly project notes
└── .env                             # local secrets (not committed)
```
