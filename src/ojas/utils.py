"""Shared utility functions for Ojas."""

from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Directory / file exclusion patterns
# ---------------------------------------------------------------------------

EXCLUDED_DIRS: set[str] = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".eggs",
    "egg-info",
    ".ojas",
}

INDEXABLE_EXTENSIONS: set[str] = {
    ".py",
    ".md",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
    ".bash",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".php",
    ".sql",
    ".dockerfile",
    ".txt",
    ".env",
    ".gitignore",
}


def is_indexable(path: Path) -> bool:
    """Return True if *path* should be indexed by the RAG engine."""
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return False
    if path.name.startswith(".") and path.suffix not in INDEXABLE_EXTENSIONS:
        return False
    return path.suffix.lower() in INDEXABLE_EXTENSIONS or path.name.lower() in {
        "dockerfile",
        "makefile",
        "procfile",
    }


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Split *text* into overlapping character-level chunks.

    Uses a simple line-boundary heuristic so that chunks avoid splitting
    mid-line whenever possible.
    """
    if len(text) <= chunk_size:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) > chunk_size and current:
            chunks.append("".join(current))
            # Keep *overlap* characters worth of trailing lines.
            kept: list[str] = []
            kept_len = 0
            for prev in reversed(current):
                if kept_len + len(prev) > overlap:
                    break
                kept.insert(0, prev)
                kept_len += len(prev)
            current = kept
            current_len = kept_len
        current.append(line)
        current_len += len(line)

    if current:
        chunks.append("".join(current))

    return chunks


def extract_traceback(output: str) -> str | None:
    """Return the Python traceback from *output*, or ``None``."""
    match = re.search(
        r"(Traceback \(most recent call last\):.*?)(?:\n(?!\s)|$)",
        output,
        re.DOTALL,
    )
    return match.group(1).strip() if match else None


def has_error(output: str) -> bool:
    """Heuristically detect whether *output* contains a Python error."""
    error_indicators = [
        "Traceback (most recent call last):",
        "SyntaxError:",
        "IndentationError:",
        "NameError:",
        "TypeError:",
        "ValueError:",
        "ImportError:",
        "ModuleNotFoundError:",
        "AttributeError:",
        "KeyError:",
        "IndexError:",
        "FileNotFoundError:",
        "OSError:",
        "RuntimeError:",
        "ZeroDivisionError:",
    ]
    return any(indicator in output for indicator in error_indicators)
