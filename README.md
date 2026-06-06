# 31agents

## Introduction

`31agents` is a small experimental project for building AI agents on top of [LangChain](https://www.langchain.com/). The current entrypoint, `knowledge-retrival-agent.py`, is a **Retrieval-Augmented Generation (RAG) knowledge agent**:

1. Loads `.txt` documents from the `./docs/` directory.
2. Splits them into overlapping chunks with `RecursiveCharacterTextSplitter`.
3. Embeds the chunks and stores them in an in-memory vector store.
4. Retrieves the top‑`k` relevant chunks for a question and feeds them, together with a system prompt, to a local LLM served by [Ollama](https://ollama.com/) to produce a grounded answer.

`main.py` is a placeholder entrypoint that simply prints a greeting.

## Prerequisites

Before setting up the project, make sure the following are installed and available on your machine:

- **Python 3.12** (the project pins `requires-python = ">=3.12"`, see `pyproject.toml`).
- **[uv](https://docs.astral.sh/uv/)** — used for dependency management and the local virtual environment (lockfile: `uv.lock`, venv: `.venv/`).
- **Ollama** — the agent talks to a local Ollama server for both the chat model and the embedding model. Install it from <https://ollama.com/download> and make sure the `ollama` daemon is running (`ollama serve` if you are not using the desktop app).
- The two Ollama models used by the agent (defaults shown; override via `.env`):
  - Chat model: `qwen3.6:35b` → `ollama pull qwen3.6:35b`
  - Embedding model: `mxbai-embed-large` → `ollama pull mxbai-embed-large`

## Setup

1. **Clone the repository**
   ```sh
   git clone git@github.com:netium/31agents.git
   cd 31agents
   ```

2. **Create a `.env` file** in the project root and add any secrets/keys you need, for example:
   ```env
   OLLAMA_LLM_MODEL=qwen3.6:35b
   OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
   ```
   Add new keys to `.env` only — never commit secrets to the repository.

3. **Install dependencies** (creates `.venv/` from `uv.lock`):
   ```sh
   uv sync
   ```

4. **Add knowledge sources**: drop one or more `.txt` files into the `./docs/` directory. The agent reads `*.txt` from this folder at runtime.

## Run

Run any script inside the managed virtual environment with `uv run`:

```sh
# Placeholder entrypoint
uv run python main.py

# Knowledge retrieval agent — answers a hard-coded query against ./docs/*.txt
uv run python knowledge-retrival-agent.py
```

The agent will print the model's answer and the number of source chunks it used, e.g.:

```
Answer: ...
Sources Len: 4
```

## Project layout

```
.
├── main.py                      # Placeholder entrypoint
├── knowledge-retrival-agent.py  # RAG agent (LangChain + Ollama)
├── docs/                        # Drop .txt knowledge files here
├── pyproject.toml               # Project metadata & dependencies
├── uv.lock                      # uv lockfile
└── .env                         # Local secrets (not committed)
```
