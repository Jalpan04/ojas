"""Docker sandbox execution for Ojas.

Creates ephemeral ``python:3.11-slim`` containers with:
- A persistent pip cache volume (``ojas-pip-cache``) mounted at
  ``/root/.cache/ojas_pkgs`` so dependencies survive across runs.
- The user's workspace mounted at ``/workspace`` so scripts can
  read/write local files.
- CPU and memory limits to prevent runaway scripts from freezing
  the host.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import docker
import docker.errors
from rich.console import Console

console = Console()

DOCKER_IMAGE = "python:3.11-slim"
PIP_CACHE_VOLUME = "ojas-pip-cache"
EXECUTION_TIMEOUT = 60  # seconds

# Global client -- lazily initialized.
_client: docker.DockerClient | None = None


def _get_client() -> docker.DockerClient | None:
    """Return a Docker client, or None if Docker is unavailable."""
    global _client
    if _client is None:
        try:
            _client = docker.from_env()
        except docker.errors.DockerException:
            _client = None
    return _client


def _ensure_image(client: docker.DockerClient) -> None:
    """Pull the Python base image if not already present."""
    try:
        client.images.get(DOCKER_IMAGE)
    except docker.errors.ImageNotFound:
        console.print(f"  [yellow]Pulling {DOCKER_IMAGE} ...[/]")
        client.images.pull(DOCKER_IMAGE)


def _ensure_pip_cache_volume(client: docker.DockerClient) -> str:
    """Ensure the persistent pip cache volume exists.  Returns the volume name."""
    try:
        client.volumes.get(PIP_CACHE_VOLUME)
    except docker.errors.NotFound:
        client.volumes.create(name=PIP_CACHE_VOLUME)
    return PIP_CACHE_VOLUME


def run_in_sandbox(
    code: str,
    dependencies: list[str] | None = None,
    workspace_dir: str | None = None,
    timeout: int = EXECUTION_TIMEOUT,
) -> str:
    """Execute *code* inside a Docker container and return combined output.

    Parameters
    ----------
    code:
        The Python script to execute.
    dependencies:
        List of pip package names to install before execution.
    workspace_dir:
        Host directory to mount into the container at ``/workspace``.
        Defaults to the current working directory.
    timeout:
        Maximum execution time in seconds.

    Returns
    -------
    str
        Combined stdout and stderr from the container.
    """
    client = _get_client()
    if client is None:
        return (
            "DOCKER_ERROR: Cannot connect to Docker daemon.\n"
            "Make sure Docker Desktop is running."
        )

    if workspace_dir is None:
        workspace_dir = os.getcwd()

    _ensure_image(client)
    volume_name = _ensure_pip_cache_volume(client)

    # Write the script to a temporary file in the workspace so it is
    # available inside the container at /workspace/<filename>.
    script_filename = "_ojas_temp_exec.py"
    script_path = os.path.join(workspace_dir, script_filename)

    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
    except OSError as exc:
        return f"FILE_ERROR: Could not write temp script: {exc}"

    # -- Build the command chain ---------------------------------------------
    command_parts: list[str] = []

    if dependencies:
        deps_str = " ".join(dependencies)
        # Install into the cached target directory so packages persist
        # across container runs via the mounted volume.
        command_parts.append(
            f"pip install --target=/root/.cache/ojas_pkgs {deps_str} -q"
        )

    # Set PYTHONPATH so Python can import from the cached packages.
    command_parts.append(
        f"PYTHONPATH=/root/.cache/ojas_pkgs python /workspace/{script_filename}"
    )

    final_command = " && ".join(command_parts)

    try:
        container = client.containers.run(
            image=DOCKER_IMAGE,
            command=["sh", "-c", final_command],
            volumes={
                # Mount the user's workspace for file access.
                workspace_dir: {"bind": "/workspace", "mode": "rw"},
                # Mount the persistent pip cache volume.
                volume_name: {"bind": "/root/.cache/ojas_pkgs", "mode": "rw"},
            },
            working_dir="/workspace",
            detach=True,
            remove=False,  # keep briefly to grab logs, then manually remove
            mem_limit="1g",
            cpu_period=100000,
            cpu_quota=50000,  # limit to ~50% of one CPU core
        )

        # Wait for execution to finish (with timeout).
        result = container.wait(timeout=timeout)
        logs = container.logs(stdout=True, stderr=True).decode(
            "utf-8", errors="replace"
        )

        exit_code = result.get("StatusCode", -1)
        container.remove(force=True)

        # Clean up the temp script on the host.
        _cleanup_temp(script_path)

        if exit_code == 0:
            return logs
        else:
            return f"EXIT_CODE {exit_code}:\n{logs}"

    except docker.errors.ContainerError as exc:
        _cleanup_temp(script_path)
        stderr_text = ""
        if exc.stderr:
            stderr_text = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr)
            )
        return f"CONTAINER_ERROR (exit {exc.exit_status}):\n{stderr_text}"

    except docker.errors.APIError as exc:
        _cleanup_temp(script_path)
        return f"DOCKER_API_ERROR: {exc}"

    except Exception as exc:  # noqa: BLE001
        _cleanup_temp(script_path)
        return f"EXECUTION_ERROR: {exc}"


def _cleanup_temp(path: str) -> None:
    """Remove the temporary script file if it exists."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
