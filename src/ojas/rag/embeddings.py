"""Ollama embedding wrapper using nomic-embed-text."""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings


def get_embeddings(
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> OllamaEmbeddings:
    """Return an embedding function backed by Ollama.

    Parameters
    ----------
    model:
        The Ollama model to use for embedding.  Defaults to
        ``nomic-embed-text``.
    base_url:
        The Ollama server URL.
    """
    return OllamaEmbeddings(model=model, base_url=base_url)
