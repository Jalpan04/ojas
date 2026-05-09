"""File operation tools for the Ojas agent.

These are exposed as LangChain ``@tool`` functions so the LLM can read,
write, and explore the local filesystem.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from rich.console import Console
from rich.prompt import Confirm

console = Console()


@tool
def read_file(path: str) -> str:
    """Read and return the contents of a local file.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        The file's text content, or an error message.
    """
    try:
        p = Path(path).resolve()
        return p.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"ERROR: File not found: {path}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR reading {path}: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a local file (with user confirmation).

    Args:
        path: Absolute or relative path to the target file.
        content: The full content to write.

    Returns:
        A success or error message.
    """
    p = Path(path).resolve()
    console.print(f"\n  [bold yellow]Agent wants to write to:[/] {p}")
    console.print(f"  [dim]({len(content)} characters)[/]")

    if not Confirm.ask("  Allow this write?", default=True):
        return "CANCELLED: User denied the write operation."

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"SUCCESS: Wrote {len(content)} characters to {p}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR writing {path}: {exc}"


@tool
def list_directory(path: str = ".") -> str:
    """List the contents of a directory.

    Args:
        path: Directory path.  Defaults to the current directory.

    Returns:
        A newline-separated list of entries with type indicators.
    """
    try:
        p = Path(path).resolve()
        if not p.is_dir():
            return f"ERROR: Not a directory: {path}"

        entries: list[str] = []
        for child in sorted(p.iterdir()):
            prefix = "[DIR] " if child.is_dir() else "[FILE]"
            entries.append(f"  {prefix} {child.name}")

        header = f"Contents of {p}:\n"
        return header + "\n".join(entries) if entries else header + "  (empty)"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR listing {path}: {exc}"


# Convenience list for graph / tool binding.
FILE_TOOLS = [read_file, write_file, list_directory]
