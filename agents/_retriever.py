"""Semantic pre-filtering for tool selection.

Builds a small vector index over each tool's name + description, then exposes
a `Retriever` abstraction with two implementations:

* `OllamaEmbeddingRetriever` — embeds with an Ollama embedding model and uses
  an in-memory vector store (langchain-core / langchain-ollama).
* `IdentityRetriever` — returns every tool in declaration order. Used in unit
  tests and as a safety fallback when embeddings are unavailable.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Protocol

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings


class Retriever(Protocol):
    def select(self, query: str, top_k: int) -> list[str]:
        """Return tool names ordered by relevance to `query`, up to `top_k`."""


class IdentityRetriever:
    """Returns every registered tool in declaration order. No embedding call."""

    def __init__(self, tool_names: list[str]):
        self._tool_names = list(tool_names)

    def select(self, query: str, top_k: int) -> list[str]:
        return self._tool_names[: max(0, int(top_k))]


class OllamaEmbeddingRetriever:
    """Embeds each tool's `name + description` once and retrieves by similarity.

    The vector store is built lazily on first use so that constructing the
    agent does not require Ollama to be reachable.
    """

    def __init__(
        self,
        tool_docs: dict[str, str],
        embedding_model: str | None = None,
        base_url: str | None = None,
    ):
        self._tool_docs = dict(tool_docs)
        self._embedding_model = embedding_model or os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
        )
        self._base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1"
        )
        self._store: InMemoryVectorStore | None = None

    def _ensure_index(self) -> InMemoryVectorStore:
        if self._store is None:
            embeddings = OllamaEmbeddings(
                model=self._embedding_model,
                base_url=self._base_url,
            )
            self._store = InMemoryVectorStore(embedding=embeddings)
            self._store.add_texts(
                list(self._tool_docs.values()),
                metadatas=[{"name": name} for name in self._tool_docs],
                ids=list(self._tool_docs.keys()),
            )
        return self._store

    def select(self, query: str, top_k: int) -> list[str]:
        store = self._ensure_index()
        k = max(0, int(top_k))
        if k == 0 or not self._tool_docs:
            return []
        docs = store.similarity_search(query, k=min(k, len(self._tool_docs)))
        return [d.id for d in docs if d.id in self._tool_docs]


def build_retriever(
    tools: list[Callable],
    tool_index_text: Callable[[Callable], str],
) -> Retriever:
    """Pick the right retriever implementation based on env / availability.

    Returns an `OllamaEmbeddingRetriever` by default. Callers that need to
    avoid the embedding call (e.g. unit tests) can construct an
    `IdentityRetriever` directly.
    """
    if os.getenv("REACT_TOOL_RETRIEVER", "ollama").lower() == "identity":
        return IdentityRetriever([f.__name__ for f in tools])

    tool_docs: dict[str, str] = {f.__name__: tool_index_text(f) for f in tools}
    return OllamaEmbeddingRetriever(tool_docs=tool_docs)
