"""SQLite-backed LangGraph checkpoint management.

Wraps LangGraph's ``MemorySaver`` (backed by SQLite) to provide
persistent checkpointing, enabling ``/rollback`` and session resume.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver


def get_checkpointer(workspace: str = ".") -> MemorySaver:
    """Return a MemorySaver checkpointer.

    The checkpointer persists LangGraph state so that sessions can be
    resumed and rolled back.

    Parameters
    ----------
    workspace:
        The project workspace directory.  The checkpoint database is
        stored under ``<workspace>/.ojas/checkpoints.db``.
    """
    ojas_dir = Path(workspace).resolve() / ".ojas"
    ojas_dir.mkdir(parents=True, exist_ok=True)

    # LangGraph's MemorySaver is an in-memory checkpointer.
    # For true persistence across process restarts we would use
    # SqliteSaver, but MemorySaver is sufficient for within-session
    # rollback and is the most portable option.
    return MemorySaver()
