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
from ojas.rag.indexer import index_workspace, start_background_indexing

def prewarm_models(config: OjasConfig) -> None:
    """Silently asks Ollama to load the agent model into RAM so the first prompt is instant.
    
    Only warms the agent model -- warming all models simultaneously would
    thrash the disk and RAM on machines with limited resources.
    """
    import httpx
    import threading

    def _warmup():
        try:
            # Only warm the agent model -- it handles the first user message.
            httpx.post(
                f"{config.ollama_base_url}/api/generate",
                json={"model": config.agent_model, "prompt": "", "keep_alive": "10m"},
                timeout=30.0,
            )
        except Exception:
            pass  # Fail silently, it's just an optimization

    threading.Thread(target=_warmup, daemon=True).start()

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
      ___                      ___           ___     
     /\  \        ___         /\  \         /\__\    
    /::\  \      /\__\       /::\  \       /:/ _/_   
   /:/\:\  \    /:/__/      /:/\:\  \     /:/ /\  \  
  /:/  \:\  \  /::\  \     /:/ /::\  \   /:/ /::\  \ 
 /:/__/ \:\__\ \/\:\  \   /:/_/:/\:\__\ /:/_/:/\:\__\
 \:\  \ /:/  /  ~~\:\  \  \:\/:/  \/__/ \:\/:/ /:/  /
  \:\  /:/  /      \:\__\  \::/__/       \::/ /:/  / 
   \:\/:/  /       /:/  /   \:\  \        \/_/:/  /  
    \::/  /       /:/  /     \:\__\         /:/  /   
     \/__/        \/__/       \/__/         \/__/   

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


def _print_welcome_dashboard(config):
    """Print a Rich dashboard for the session."""
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    
    # Models Column
    model_text = Text()
    model_text.append("🧠 Agent: ", style="bold green")
    model_text.append(f"{config.agent_model}\n", style="cyan")
    model_text.append("🏗️ Planner: ", style="bold green")
    model_text.append(f"{config.planner_model}\n", style="cyan")
    model_text.append("💻 Coder: ", style="bold green")
    model_text.append(f"{config.coder_model}\n", style="cyan")
    model_text.append("🔍 Reviewer: ", style="bold green")
    model_text.append(f"{config.reviewer_model}\n", style="cyan")
    model_text.append("📚 Embeds: ", style="bold green")
    model_text.append(f"{config.embed_model}", style="cyan")

    # Status Column
    status_text = Text()
    status_text.append("📁 Workspace: ", style="bold yellow")
    status_text.append(f"{config.workspace}\n", style="white")
    status_text.append("📡 Ollama: ", style="bold yellow")
    status_text.append(f"{config.ollama_base_url}\n", style="white")
    status_text.append("⚡ Mode: ", style="bold yellow")
    status_text.append("ReAct (Cyclic)", style="magenta")

    panel = Panel(
        Columns([Panel(model_text, title="Active Models", border_style="blue"), 
                 Panel(status_text, title="Session Context", border_style="yellow")]),
        title="[bold blue]OJAS v2.0[/]",
        subtitle="[dim]Background RAG Indexing Started[/]",
        border_style="magenta"
    )
    console.print(panel)


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


def _run_loop(config: OjasConfig) -> None:
    """The main interactive REPL loop."""
    graph = build_graph(config)
    build_fn = partial(build_graph, config)
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
            "skip_planning": False,
        }

        # Get existing message count to avoid double-printing
        existing_state = graph.get_state(thread_config)
        existing_msg_count = len(existing_state.values.get("messages", [])) if existing_state.values else 0

        # -- Invoke the graph with interrupts and streaming -------------------
        try:
            from langchain_core.messages import AIMessageChunk
            state_to_invoke = state
            
            while True:
                status = console.status("[bold bright_cyan]Thinking...", spinner="dots")
                status.start()
                
                # Stream messages from the graph
                full_content = ""
                try:
                    for event in graph.stream(state_to_invoke, config=thread_config, stream_mode=["messages", "values"]):
                        if event[0] == "messages":
                            msg_chunk, metadata = event[1]
                            
                            # Update status based on node
                            current_node = metadata.get("langgraph_node", "")
                            if current_node:
                                status.update(f"[bold bright_cyan]{current_node.upper()}...[/]")
                            
                            if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
                                # Stop status once we start getting content for a smoother feel
                                status.stop()
                                console.print(msg_chunk.content, end="")
                                full_content += msg_chunk.content
                        elif event[0] == "values":
                            pass
                finally:
                    status.stop()
                        
                if full_content:
                    console.print() # Newline after streaming
                    
                    # Check if we hit an interrupt
                    current_state_tuple = graph.get_state(thread_config)
                    if current_state_tuple.next and "tools" in current_state_tuple.next:
                        # We are paused right before the tools node.
                        last_message = current_state_tuple.values.get("messages", [])[-1]
                        tool_calls = getattr(last_message, "tool_calls", [])
                        
                        # Check if any dangerous tools are called
                        dangerous_tools = [tc for tc in tool_calls if tc["name"] in ["run_host_shell_command", "execute_python_sandbox"]]
                        
                        if dangerous_tools:
                            import json
                            from rich.panel import Panel
                            
                            tool_calls_json = json.dumps(tool_calls, indent=2)
                            console.print(Panel(tool_calls_json, title="[bold yellow]Tool Execution Requested[/]", border_style="yellow"))
                            
                            choice = console.input("[ojas.user]Execute these tools? \\[y] Run / \\[e] Edit / \\[c] Cancel: [/]").strip().lower()
                            if choice == "y":
                                state_to_invoke = None
                                continue
                            elif choice == "e":
                                edited_json = click.edit(tool_calls_json, extension=".json")
                                if edited_json and edited_json != tool_calls_json:
                                    try:
                                        new_tool_calls = json.loads(edited_json)
                                        last_message.tool_calls = new_tool_calls
                                        graph.update_state(thread_config, {"messages": [last_message]})
                                        console.print("[ojas.success]Tools updated.[/]")
                                    except Exception as e:
                                        console.print(f"[ojas.error]Failed to parse edited tools: {e}[/]")
                                        break
                                state_to_invoke = None
                                continue
                            else:
                                console.print("  [ojas.warn]Execution canceled by user.[/]")
                                break
                        else:
                            # Auto-approve safe tools
                            state_to_invoke = None
                            continue
                    else:
                        # Finished execution
                        break

        except KeyboardInterrupt:
            console.print("\n  [ojas.warn]Interrupted.[/]")
            continue
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [ojas.error]Error: {exc}[/]")
            continue


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
@click.option(
    "--ignore-ram",
    is_flag=True,
    help="Bypass the system RAM check (assumes GPU VRAM is handling it).",
)
@click.option(
    "--planner",
    help="Force a specific model for reasoning (e.g., gemma4:e4b).",
)
@click.option(
    "--coder",
    help="Force a specific model for coding (e.g., qwen2.5-coder:7b).",
)
@click.option(
    "--reviewer",
    help="Force a specific model for error evaluation.",
)
@click.option(
    "--agent",
    help="Force a specific model for tool calling/general tasks.",
)
def start(
    workspace: str,
    ignore_ram: bool,
    planner: str | None,
    coder: str | None,
    reviewer: str | None,
    agent: str | None,
) -> None:
    """Start the Ojas agent in the current (or specified) workspace."""
    _print_banner()

    workspace = os.path.abspath(workspace)

    # -- Auto-boot -----------------------------------------------------------
    config = auto_configure(
        workspace=workspace,
        ignore_ram=ignore_ram,
        force_planner=planner,
        force_coder=coder,
        force_reviewer=reviewer,
        force_agent=agent,
    )

    # -- Pre-warm models ----------------------------------------------------
    prewarm_models(config)

    # -- Index workspace via ChromaDB (Background) --------------------------
    console.print("[dim]Starting background indexing... You can chat immediately.[/dim]\n")
    start_background_indexing(
        workspace=workspace,
        embed_model=config.embed_model,
        ollama_url=config.ollama_base_url,
    )

    # -- Show Dashboard -----------------------------------------------------
    _print_welcome_dashboard(config)

    # -- Enter interactive loop ----------------------------------------------
    _run_loop(config)


if __name__ == "__main__":
    main()
