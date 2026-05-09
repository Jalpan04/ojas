"""Retrieval node -- injects relevant workspace context via ChromaDB.

This node intercepts the latest user message, runs a similarity search
against the indexed codebase, and writes the results into
``state["workspace_context"]``.

It also performs lightweight intent classification to skip the heavy
Planner node for simple conversational messages.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage

from ojas.rag.retriever import retrieve
from ojas.rag.indexer import get_collection
from ojas.config import OjasConfig

# Patterns that indicate a simple conversational turn (no planning needed).
_SIMPLE_PATTERNS = re.compile(
    r"^("
    r"h(i|ello|ey|owdy|ola)|yo|sup|test|thanks|thank you|ok|okay|cool|"
    r"good (morning|afternoon|evening|night)|"
    r"what is \d|what'?s \d|how are you|who are you|"
    r"tell me a joke|sing a song|write a (simple )?(calculator|script|code)|"
    r"yes|no|sure|nah|bye|exit|quit"
    r")(\?|!|\.|\s|$)",
    re.IGNORECASE,
)


def _is_simple_chat(text: str) -> bool:
    """Return True if the message is simple chat that needs no planning."""
    clean = text.strip()
    # If it matches a simple pattern, skip planning regardless of length (within reason)
    if _SIMPLE_PATTERNS.match(clean) and len(clean) < 100:
        return True
    # Very short messages without "@" or specific workspace keywords are likely conversational.
    if len(clean) < 40 and "@" not in clean and "/" not in clean and "\\" not in clean:
        return True
    return False


def make_retrieval_node(
    config: OjasConfig,
):
    """Return a retrieval node function bound to the given OjasConfig."""

    def retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
        """Extract the latest user query and retrieve matching workspace context."""
        # Find the most recent human message.
        query = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        if not query:
            return {"workspace_context": "", "skip_planning": True}

        # 1. Check if this is simple conversational chat (fast path).
        if _is_simple_chat(query):
            return {"workspace_context": "", "skip_planning": True}

        # 2. STRICT BYPASS: If indexing is running, skip DB entirely to avoid SQLite locks.
        from ojas.rag.indexer import IS_INDEXING
        if IS_INDEXING:
            return {
                "workspace_context": "[SYSTEM: Workspace is being indexed. Rely on general knowledge.]",
                "skip_planning": False,
            }

        # 3. Full RAG retrieval for complex workspace queries.
        context = retrieve(
            query=query,
            collection=get_collection(config.workspace),
            embed_model=config.embed_model,
            ollama_url=config.ollama_base_url,
            n_results=5,
        )
        return {"workspace_context": context, "skip_planning": False}

    return retrieval_node
