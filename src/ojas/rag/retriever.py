"""Similarity search interface over the indexed ChromaDB collection."""

from __future__ import annotations

import chromadb

from ojas.rag.embeddings import get_embeddings


def retrieve(
    query: str,
    collection: chromadb.Collection,
    embed_model: str = "nomic-embed-text",
    ollama_url: str = "http://localhost:11434",
    n_results: int = 5,
) -> str:
    """Embed *query* and return the top-*n_results* matching chunks.

    Returns a formatted string ready for injection into the LLM context.
    """
    embedder = get_embeddings(model=embed_model, base_url=ollama_url)
    query_embedding = embedder.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return ""

    parts: list[str] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        source = meta.get("source", "unknown")
        chunk_idx = meta.get("chunk_index", 0)
        total = meta.get("total_chunks", 1)
        header = (
            f"--- {source} (chunk {chunk_idx + 1}/{total}, "
            f"similarity {1 - dist:.2f}) ---"
        )
        parts.append(f"{header}\n{doc}")

    return "\n\n".join(parts)
