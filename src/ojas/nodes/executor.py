"""Execution node -- runs generated code inside a Docker sandbox.

Lifecycle:
1. Creates an ephemeral ``python:3.11-slim`` container.
2. Mounts a persistent named volume (``ojas-pip-cache``) at
   ``/root/.cache/ojas_pkgs`` for dependency caching.
3. Mounts the workspace at ``/workspace`` for file access.
4. Installs any required pip packages (cached across runs).
5. Writes the generated script into the workspace, executes it,
   and captures stdout + stderr.
6. Cleans up the container and temp file.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

from ojas.tools.docker_exec import run_in_sandbox

console = Console()


def make_executor_node(workspace_dir: str):
    """Return an executor node function bound to the given workspace."""

    def executor_node(state: dict[str, Any]) -> dict[str, Any]:
        """Execute the generated code in a Docker sandbox."""
        code = state.get("generated_code", "")
        if not code:
            return {"docker_output": "", "cycle_complete": True}

        deps = state.get("dependencies", [])

        console.print("  [dim]Spinning up Docker sandbox ...[/]")
        output = run_in_sandbox(
            code=code,
            dependencies=deps,
            workspace_dir=workspace_dir,
        )
        console.print("  [dim]Sandbox execution complete.[/]")

        return {"docker_output": output}

    return executor_node
