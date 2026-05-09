"""Evaluator node -- decides whether to self-correct or finish.

Checks the Docker output for Python errors.  If errors are found and
the retry budget has not been exhausted, routes back to the Coder node.
Otherwise marks the cycle as complete.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from ojas.utils import has_error

MAX_RETRIES = 3


def evaluator_node(state: dict[str, Any]) -> dict[str, Any]:
    """Evaluate Docker output and decide the next step."""
    output = state.get("docker_output", "")
    retries = state.get("retry_count", 0)

    if not output:
        return {"cycle_complete": True, "retry_count": retries}

    if has_error(output) and retries < MAX_RETRIES:
        return {
            "cycle_complete": False,
            "retry_count": retries + 1,
            "messages": [
                AIMessage(
                    content=(
                        f"[EVAL] Execution error detected (attempt "
                        f"{retries + 1}/{MAX_RETRIES}). Routing back to Coder "
                        f"for self-correction.\n\nError output:\n```\n{output}\n```"
                    )
                )
            ],
        }

    # Success (or max retries exceeded).
    status = "success" if not has_error(output) else "max retries reached"
    return {
        "cycle_complete": True,
        "retry_count": retries,
        "messages": [
            AIMessage(
                content=(
                    f"[EVAL] Execution {status}.\n\nOutput:\n```\n{output}\n```"
                )
            )
        ],
    }


def should_retry(state: dict[str, Any]) -> str:
    """Conditional edge function: 'coder' if retrying, 'end' otherwise."""
    if state.get("cycle_complete", True):
        return "end"
    return "coder"
