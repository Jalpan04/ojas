"""Sequential sub-agent delegation for /delegate.

Unloads the primary agent, spawns a temporary sub-agent graph with a
scoped task, runs it to completion, and returns the result for merging
back into the main conversation.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console

console = Console()


def run_delegate(
    task: str,
    build_graph_fn: Any,
    config: Any,
) -> str:
    """Spawn a delegate sub-agent to complete *task*.

    Parameters
    ----------
    task:
        A natural-language description of the sub-task.
    build_graph_fn:
        A callable that returns a compiled LangGraph (same as the main
        graph but with a fresh state).
    config:
        The ``OjasConfig`` instance.

    Returns
    -------
    str
        The delegate's final output text.
    """
    console.print(
        f"\n  [bold magenta]Delegating:[/] {task}\n"
        "  [dim]Spawning sub-agent ...[/]"
    )

    graph = build_graph_fn()

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "workspace_context": "",
        "generated_code": "",
        "dependencies": [],
        "docker_output": "",
        "retry_count": 0,
        "cycle_complete": False,
    }

    thread_config = {"configurable": {"thread_id": f"delegate-{hash(task)}"}}

    try:
        final_state = graph.invoke(initial_state, config=thread_config)
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]Delegate failed: {exc}[/]")
        return f"DELEGATE_ERROR: {exc}"

    # Extract the last AI message as the result.
    messages = final_state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            result = msg.content
            console.print("  [green]Delegate completed.[/]\n")
            return result

    return "(Delegate produced no output.)"
