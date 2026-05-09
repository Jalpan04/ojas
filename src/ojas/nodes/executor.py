"""Execution node -- runs generated code inside a Docker sandbox.

Lifecycle:
1. Creates an ephemeral ``python:3.11-slim`` container.
2. Mounts a persistent named volume (``ojas-pip-cache``) at
   ``/root/.cache/pip`` for dependency caching.
3. Installs any required pip packages.
4. Writes the generated script into the container.
5. Executes the script and captures stdout + stderr.
6. Destroys the container.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

from ojas.tools.docker_exec import run_in_sandbox

console = Console()


def executor_node(state: dict[str, Any]) -> dict[str, Any]:
    """Execute the generated code in a Docker sandbox."""
    code = state.get("generated_code", "")
    if not code:
        return {"docker_output": "", "cycle_complete": True}

    deps = state.get("dependencies", [])

    console.print("  [dim]Spinning up Docker sandbox ...[/]")
    output = run_in_sandbox(code=code, dependencies=deps)
    console.print("  [dim]Sandbox execution complete.[/]")

    return {"docker_output": output}
