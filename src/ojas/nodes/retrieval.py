"""Retrieval node -- injects relevant workspace context via ChromaDB.

This node intercepts the latest user message, runs a similarity search
against the indexed codebase, and writes the results into
``state["workspace_context"]``.
"""

from __future__ import annotations

from typing import Any

import chromadb
from langchain_core.messages import HumanMessage

from ojas.rag.retriever import retrieve


def make_retrieval_node(
    collection: chromadb.Collection,
    embed_model: str,
    ollama_url: str,
):
    """Return a retrieval node function bound to the given ChromaDB collection."""

    def retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
        """Extract the latest user query and retrieve matching workspace context."""
        # Find the most recent human message.
        query = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        if not query:
            return {"workspace_context": ""}

        context = retrieve(
            query=query,
            collection=collection,
            embed_model=embed_model,
            ollama_url=ollama_url,
            n_results=5,
        )
        return {"workspace_context": context}

    return retrieval_node
