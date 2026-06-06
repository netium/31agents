# 31agents

## Stack
- Python 3.12 (see `.python-version` and `pyproject.toml` `requires-python`)
- Dependency / venv management: `uv` (lockfile `uv.lock`, venv at `.venv/`)
- Runtime deps: `langchain`, `dotenv`

## Layout
- `main.py` — placeholder entrypoint (prints hello)
- `knowledge-retrival-agent.py` — agent script; loads `.env` via `dotenv`. Filename is misspelled ("retrival"); keep the existing name unless intentionally renaming.
- `docs/` — empty
- `README.md` — empty

There is no `src/`, no package layout, no tests directory.

## Environment
- Secrets are read from `.env` (loaded with `dotenv.load_dotenv()`).
- Current key in `.env`: `MINIMAX_API_KEY`. Add new keys to `.env`, not into tracked files.

## Common commands
Run a script in the managed venv:
```
uv run python main.py
uv run python knowledge-retrival-agent.py
uv run python react.py
```
Install / sync deps after editing `pyproject.toml` or `uv.lock`:
```
uv sync
uv sync --group dev        # include pytest
```
Run the test suite:
```
uv run --group dev pytest
```
Run only the fast unit tests (skip the LLM-as-judge set, which hits Ollama and is slow):
```
uv run --group dev pytest -m "not llm"
```
Run only the LLM-as-judge tests (requires a running Ollama; ~30s+ per test with `qwen3.6:35b`):
```
uv run --group dev pytest -m llm
```

## Layout
- `agents/` — subpackage: `__init__.py` re-exports `ReactAgent`; `react_agent.py` holds the class; `_tools.py` holds private schema helpers.
- `react.py` — demo entrypoint that wires the agent to two sample tools (`add`, `get_weather`) and runs a REPL.
- `tests/` — pytest tests (config in `pyproject.toml` under `[tool.pytest.ini_options]`).

## What is NOT configured
Do not look for any of the following — they have not been set up:
- Linter / formatter (no `ruff`, `black`, `mypy` config)
- CI workflows (no `.github/`)
- Pre-commit hooks (`.git/hooks/` has no active hooks)
- `.gitignore` at repo root (only the empty `.venv/.gitignore` exists)

If a task requires any of these, set them up before relying on them.

## Git
- Default branch: `main` (only branch, single initial commit)
- Remote: `git@github.com:netium/31agents.git`
