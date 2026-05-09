"""Persistent memory via MEMORY.md.

High-level user preferences, project facts, and learned conventions
are stored in a Markdown file that is injected into the system prompt
on every invocation.
"""

from __future__ import annotations

from pathlib import Path

MEMORY_FILENAME = "MEMORY.md"
MEMORY_HEADER = "# Ojas Memory\n\nPersistent facts and user preferences.\n\n"


def _memory_path(workspace: str = ".") -> Path:
    return Path(workspace).resolve() / ".ojas" / MEMORY_FILENAME


def load_memory(workspace: str = ".") -> str:
    """Load and return the contents of MEMORY.md, or an empty string."""
    p = _memory_path(workspace)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def save_memory(content: str, workspace: str = ".") -> None:
    """Overwrite MEMORY.md with *content*."""
    p = _memory_path(workspace)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def add_memory(fact: str, workspace: str = ".") -> str:
    """Append a fact to MEMORY.md.  Creates the file if needed."""
    p = _memory_path(workspace)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        p.write_text(MEMORY_HEADER, encoding="utf-8")

    existing = p.read_text(encoding="utf-8")
    entry = f"- {fact.strip()}\n"
    if entry in existing:
        return f"Already stored: {fact}"

    p.write_text(existing + entry, encoding="utf-8")
    return f"Saved: {fact}"


def remove_memory(fact: str, workspace: str = ".") -> str:
    """Remove a fact from MEMORY.md."""
    p = _memory_path(workspace)
    if not p.exists():
        return "No memory file found."

    existing = p.read_text(encoding="utf-8")
    entry = f"- {fact.strip()}\n"
    if entry not in existing:
        return f"Not found in memory: {fact}"

    updated = existing.replace(entry, "")
    p.write_text(updated, encoding="utf-8")
    return f"Removed: {fact}"


def list_memory(workspace: str = ".") -> str:
    """Return the full contents of MEMORY.md for display."""
    content = load_memory(workspace)
    if not content.strip():
        return "(No memories stored yet.)"
    return content
