"""Context reference parser for @ mentions.

Supports:
- @file:path/to/file.py        -- inject the full file
- @file:path/to/file.py:10-50  -- inject specific lines
- @folder:path/to/dir/          -- inject directory tree
- @diff                         -- inject unstaged git changes
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def parse_references(text: str, workspace: str = ".") -> tuple[str, str]:
    """Parse @ references from *text* and return (cleaned_text, context).

    Parameters
    ----------
    text:
        The raw user input that may contain @ references.
    workspace:
        The workspace root directory.

    Returns
    -------
    tuple[str, str]
        A 2-tuple of (text with references removed, assembled context block).
    """
    ws = Path(workspace).resolve()
    context_parts: list[str] = []
    cleaned = text

    # @file:path[:start-end]
    for match in re.finditer(r"@file:(\S+?)(?::(\d+)-(\d+))?(?:\s|$)", text):
        filepath = match.group(1)
        start = int(match.group(2)) if match.group(2) else None
        end = int(match.group(3)) if match.group(3) else None
        cleaned = cleaned.replace(match.group(0), "").strip()

        fpath = ws / filepath
        if not fpath.exists():
            context_parts.append(f"[ERROR: File not found: {filepath}]")
            continue

        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            context_parts.append(f"[ERROR reading {filepath}: {exc}]")
            continue

        if start is not None and end is not None:
            lines = content.splitlines()
            selected = lines[max(0, start - 1) : end]
            content = "\n".join(selected)
            header = f"--- {filepath} (lines {start}-{end}) ---"
        else:
            header = f"--- {filepath} ---"

        context_parts.append(f"{header}\n{content}")

    # @folder:path/
    for match in re.finditer(r"@folder:(\S+?)(?:\s|$)", text):
        folder = match.group(1)
        cleaned = cleaned.replace(match.group(0), "").strip()

        fpath = ws / folder
        if not fpath.is_dir():
            context_parts.append(f"[ERROR: Directory not found: {folder}]")
            continue

        tree_lines = [f"--- Directory tree: {folder} ---"]
        for child in sorted(fpath.rglob("*")):
            if any(part.startswith(".") for part in child.relative_to(fpath).parts):
                continue
            rel = child.relative_to(fpath)
            prefix = "  " * (len(rel.parts) - 1)
            indicator = "/" if child.is_dir() else ""
            tree_lines.append(f"{prefix}{rel.name}{indicator}")

        context_parts.append("\n".join(tree_lines))

    # @diff
    if "@diff" in text:
        cleaned = cleaned.replace("@diff", "").strip()
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=10,
            )
            diff = result.stdout.strip()
            if diff:
                context_parts.append(f"--- Unstaged Git diff ---\n{diff}")
            else:
                context_parts.append("[No unstaged changes detected.]")
        except Exception as exc:  # noqa: BLE001
            context_parts.append(f"[ERROR getting git diff: {exc}]")

    context = "\n\n".join(context_parts) if context_parts else ""
    return cleaned.strip(), context
