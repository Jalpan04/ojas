"""Ojas CLI -- Beautiful terminal interface built with Rich.

Provides the main interactive loop, command parsing, and rendering of
the agent's thought process, RAG retrievals, and Docker execution states.
"""

from __future__ import annotations

import os
import sys
import uuid
from functools import partial
from typing import Any

# Force UTF-8 output on Windows to prevent cp1252 encoding errors
# when Rich renders Markdown with special Unicode characters.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

import click
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from ojas.config import OjasConfig, auto_configure
from ojas.context import parse_references
from ojas.delegate import run_delegate
from ojas.graph import build_graph
from ojas.memory.persistent import add_memory, list_memory, remove_memory
from ojas.rag.indexer import index_workspace

# ---------------------------------------------------------------------------
# Rich theme
# ---------------------------------------------------------------------------

OJAS_THEME = Theme(
    {
        "ojas.header": "bold bright_cyan",
        "ojas.user": "bold green",
        "ojas.agent": "bold bright_magenta",
        "ojas.dim": "dim",
        "ojas.error": "bold red",
        "ojas.success": "bold green",
        "ojas.warn": "bold yellow",
    }
)

console = Console(theme=OJAS_THEME, force_terminal=True)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = r"""
   ____     _
  / __ \   (_)  ____ _   _____
 / / / /  / /  / __ `/  / ___/
/ /_/ /  / /  / /_/ /  (__  )
\____/__/ /   \__,_/  /____/
    /___/

 The Local-First, Autonomous AI Developer
"""


def _print_banner() -> None:
    console.print(
        Panel(
            Text(BANNER, style="bright_cyan"),
            border_style="bright_cyan",
            expand=False,
        )
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_agent_message(content: str) -> None:
    """Render an agent message with appropriate formatting."""
    # Detect message types by prefix tags.
    if content.startswith("[PLAN]"):
        label = "PLANNER"
        body = content[len("[PLAN]") :].strip()
        style = "bright_yellow"
    elif content.startswith("[CODE]"):
        label = "CODER"
        body = content[len("[CODE]") :].strip()
        style = "bright_green"
    elif content.startswith("[EVAL]"):
        label = "EVALUATOR"
        body = content[len("[EVAL]") :].strip()
        style = "bright_magenta"
    else:
        label = "OJAS"
        body = content
        style = "bright_cyan"

    console.print()
    try:
        console.print(
            Panel(
                Markdown(body),
                title=f"[bold]{label}[/]",
                border_style=style,
                expand=True,
                padding=(1, 2),
            )
        )
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Fallback: render as plain text if Markdown causes encoding issues.
        console.print(
            Panel(
                body,
                title=f"[bold]{label}[/]",
                border_style=style,
                expand=True,
                padding=(1, 2),
            )
        )


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _handle_command(
    raw: str,
    config: OjasConfig,
    graph: Any,
    session_id: str,
    build_graph_fn: Any,
) -> bool:
    """Process a /command.  Returns True if the input loop should continue."""
    parts = raw.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/clear":
        console.print("  [ojas.warn]Session cleared.[/]")
        return True  # caller will reset state

    if cmd == "/rollback":
        console.print("  [ojas.warn]Rolling back to previous checkpoint ...[/]")
        # With MemorySaver, rollback means re-invoking from the
        # previous thread snapshot.  For MVP we reset the session.
        console.print("  [ojas.dim]Rollback complete (session reset).[/]")
        return True

    if cmd == "/memory":
        sub_parts = arg.split(maxsplit=1)
        sub_cmd = sub_parts[0].lower() if sub_parts else "list"
        sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

        if sub_cmd == "add" and sub_arg:
            result = add_memory(sub_arg, config.workspace)
            console.print(f"  [ojas.success]{result}[/]")
        elif sub_cmd == "remove" and sub_arg:
            result = remove_memory(sub_arg, config.workspace)
            console.print(f"  [ojas.warn]{result}[/]")
        else:
            console.print(
                Panel(
                    Markdown(list_memory(config.workspace)),
                    title="[bold]Memory[/]",
                    border_style="bright_cyan",
                )
            )
        return False

    if cmd == "/delegate":
        if not arg:
            console.print("  [ojas.error]Usage: /delegate <task description>[/]")
            return False
        result = run_delegate(
            task=arg,
            build_graph_fn=build_graph_fn,
            config=config,
        )
        _render_agent_message(result)
        return False

    console.print(f"  [ojas.error]Unknown command: {cmd}[/]")
    console.print(
        "  [ojas.dim]Available: /clear, /rollback, "
        "/memory [add|remove|list], /delegate <task>[/]"
    )
    return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _run_loop(config: OjasConfig, collection: Any) -> None:
    """The main interactive REPL loop."""
    graph = build_graph(config, collection)
    build_fn = partial(build_graph, config, collection)
    session_id = str(uuid.uuid4())[:8]
    thread_config = {"configurable": {"thread_id": f"ojas-{session_id}"}}

    console.print(
        "[ojas.dim]Ready. Type your request, or use / commands. "
        "Press Ctrl+C to exit.[/]\n"
    )

    while True:
        try:
            user_input = console.input("[ojas.user]you > [/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[ojas.dim]Goodbye.[/]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # -- Handle / commands -----------------------------------------------
        if user_input.startswith("/"):
            should_reset = _handle_command(
                user_input, config, graph, session_id, build_fn
            )
            if should_reset:
                session_id = str(uuid.uuid4())[:8]
                thread_config = {
                    "configurable": {"thread_id": f"ojas-{session_id}"}
                }
            continue

        # -- Parse @ references ----------------------------------------------
        cleaned, ref_context = parse_references(user_input, config.workspace)
        if ref_context:
            cleaned = f"{cleaned}\n\n<injected_context>\n{ref_context}\n</injected_context>"

        # -- Build initial state ---------------------------------------------
        state: dict[str, Any] = {
            "messages": [HumanMessage(content=cleaned)],
            "workspace_context": "",
            "generated_code": "",
            "dependencies": [],
            "docker_output": "",
            "retry_count": 0,
            "cycle_complete": False,
        }

        # -- Invoke the graph ------------------------------------------------
        try:
            console.print("[ojas.dim]  Thinking ...[/]")
            final_state = graph.invoke(state, config=thread_config)
        except KeyboardInterrupt:
            console.print("\n  [ojas.warn]Interrupted.[/]")
            continue
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [ojas.error]Error: {exc}[/]")
            continue

        # -- Render all new AI messages --------------------------------------
        for msg in final_state.get("messages", []):
            if isinstance(msg, AIMessage):
                _render_agent_message(msg.content)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Ojas -- The Local-First, Autonomous AI Developer."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(start)


@main.command()
@click.option(
    "--workspace",
    "-w",
    default=".",
    help="Workspace directory to index and operate in.",
)
def start(workspace: str) -> None:
    """Start the Ojas agent in the current (or specified) workspace."""
    _print_banner()

    workspace = os.path.abspath(workspace)

    # -- Auto-boot -----------------------------------------------------------
    config = auto_configure(workspace=workspace)

    # -- Index workspace via ChromaDB ----------------------------------------
    console.print("[ojas.dim]  Initializing RAG engine ...[/]")
    collection = index_workspace(
        workspace=workspace,
        embed_model=config.embed_model,
        ollama_url=config.ollama_base_url,
    )

    # -- Enter interactive loop ----------------------------------------------
    _run_loop(config, collection)


if __name__ == "__main__":
    main()
