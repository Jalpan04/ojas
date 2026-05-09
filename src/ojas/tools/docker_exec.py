"""Docker sandbox execution for Ojas.

Creates ephemeral ``python:3.11-slim`` containers, installs dependencies
using a persistent pip cache volume, executes the generated script, and
captures all output.
"""

from __future__ import annotations

import docker
from docker.errors import DockerException, ImageNotFound
from rich.console import Console

console = Console()

DOCKER_IMAGE = "python:3.11-slim"
PIP_CACHE_VOLUME = "ojas-pip-cache"
EXECUTION_TIMEOUT = 120  # seconds


def _ensure_image(client: docker.DockerClient) -> None:
    """Pull the Python image if not already present."""
    try:
        client.images.get(DOCKER_IMAGE)
    except ImageNotFound:
        console.print(f"  [yellow]Pulling {DOCKER_IMAGE} ...[/]")
        client.images.pull(DOCKER_IMAGE)


def _ensure_volume(client: docker.DockerClient) -> None:
    """Create the pip cache volume if it does not exist."""
    volumes = {v.name for v in client.volumes.list()}
    if PIP_CACHE_VOLUME not in volumes:
        client.volumes.create(PIP_CACHE_VOLUME)


def run_in_sandbox(
    code: str,
    dependencies: list[str] | None = None,
    timeout: int = EXECUTION_TIMEOUT,
) -> str:
    """Execute *code* inside a Docker container and return combined output.

    Parameters
    ----------
    code:
        The Python script to execute.
    dependencies:
        List of pip package names to install before execution.
    timeout:
        Maximum execution time in seconds.

    Returns
    -------
    str
        Combined stdout and stderr from the container.
    """
    try:
        client = docker.from_env()
    except DockerException as exc:
        return (
            f"DOCKER_ERROR: Cannot connect to Docker daemon.\n"
            f"Make sure Docker Desktop is running.\n{exc}"
        )

    _ensure_image(client)
    _ensure_volume(client)

    # Build the shell command sequence.
    commands: list[str] = []

    if dependencies:
        dep_str = " ".join(dependencies)
        commands.append(f"pip install --cache-dir /pip-cache {dep_str} 2>&1")

    # Write the script to a temp file and execute it.
    # We use a heredoc-style approach via echo + python.
    commands.append("python /tmp/script.py 2>&1")

    shell_cmd = " && ".join(commands)

    try:
        container = client.containers.run(
            image=DOCKER_IMAGE,
            command=["bash", "-c", shell_cmd],
            volumes={
                PIP_CACHE_VOLUME: {"bind": "/pip-cache", "mode": "rw"},
            },
            # Write the script into the container via environment + entrypoint trick.
            environment={"OJAS_SCRIPT": code},
            entrypoint=[
                "bash",
                "-c",
                # First write the script, then run the actual command.
                f'echo "$OJAS_SCRIPT" > /tmp/script.py && {shell_cmd}',
            ],
            detach=False,
            remove=True,
            network_mode="bridge",
            mem_limit="512m",
            stderr=True,
            stdout=True,
            timeout=timeout,
        )

        # container.run with detach=False returns bytes.
        if isinstance(container, bytes):
            return container.decode("utf-8", errors="replace")
        return str(container)

    except docker.errors.ContainerError as exc:
        output = ""
        if exc.stderr:
            output = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr)
            )
        return f"CONTAINER_ERROR (exit {exc.exit_status}):\n{output}"
    except docker.errors.APIError as exc:
        return f"DOCKER_API_ERROR: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"EXECUTION_ERROR: {exc}"
