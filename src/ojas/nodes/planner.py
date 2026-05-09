"""Planner node -- uses the largest available model for deep reasoning.

The planner receives the user request plus RAG context and produces a
step-by-step architectural plan that the Coder node will implement.

If the planner model fails (e.g., OOM), it falls back to the coder model.
"""

from __future__ import annotations
import re
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from rich.console import Console

console = Console()

PLANNER_SYSTEM_PROMPT = """You are the architectural planner for Ojas, an autonomous AI developer.
Analyze the user's request and the provided workspace context. 

DECISION LOGIC:
1. If the user is just saying hello, asking a general question, or if NO code execution is required, output a standard conversational response AND end your response with exactly: `REQUIRES_CODE: NO`.
2. Otherwise, provide a step-by-step coding plan and end your response with exactly: `REQUIRES_CODE: YES`."""


def make_planner_node(
    model_name: str,
    ollama_url: str = "http://localhost:11434",
    fallback_model: str | None = None,
    workspace_dir: str = ".",
):
    """Return a planner node function bound to the given model.

    Parameters
    ----------
    model_name:
        Primary planner model name.
    ollama_url:
        Ollama API base URL.
    fallback_model:
        If the primary model fails (e.g., OOM), fall back to this model.
    workspace_dir:
        The root directory of the workspace to look for project instructions.
    """

    def _make_llm(name: str) -> ChatOllama:
        return ChatOllama(
            model=name,
            base_url=ollama_url,
            temperature=0.3,
        )

    # Defer LLM creation to first use to avoid blocking startup.
    _llm_cache: dict[str, ChatOllama] = {}

    def _get_llm(name: str) -> ChatOllama:
        if name not in _llm_cache:
            _llm_cache[name] = _make_llm(name)
        return _llm_cache[name]

    def planner_node(state: dict[str, Any]) -> dict[str, Any]:
        """Invoke the planner model and append its plan to messages."""
        import os
        
        # Check for project-level instructions
        project_rules = ""
        for rule_file in [".ojas.md", ".cursorrules"]:
            rule_path = os.path.join(workspace_dir, rule_file)
            if os.path.exists(rule_path):
                try:
                    with open(rule_path, "r", encoding="utf-8") as f:
                        project_rules = f"\n\n<project_rules>\n{f.read()}\n</project_rules>"
                except Exception:
                    pass
                break

        workspace_ctx = state.get("workspace_context", "")
        context_block = ""
        if workspace_ctx:
            context_block = (
                "\n\n<workspace_context>\n"
                f"{workspace_ctx}\n"
                "</workspace_context>"
            )

        system = SystemMessage(
            content=PLANNER_SYSTEM_PROMPT + project_rules + context_block
        )

        # Build the message list: system + conversation history.
        messages = [system] + list(state.get("messages", []))

        # Try the primary model first; fall back on failure.
        llm = _get_llm(model_name)
        used_model = model_name
        try:
            response = llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            if fallback_model and fallback_model != model_name:
                console.print(
                    f"  [yellow]Planner model failed ({exc}). "
                    f"Falling back to {fallback_model}.[/]"
                )
                llm = _get_llm(fallback_model)
                used_model = fallback_model
                response = llm.invoke(messages)
            else:
                raise

        plan_text = response.content if hasattr(response, "content") else str(response)

        requires_code = True
        if re.search(r"REQUIRES_CODE:\s*NO", plan_text, re.IGNORECASE):
            requires_code = False
        
        # Strip the trigger tag from the visible plan text.
        clean_plan = re.sub(r"REQUIRES_CODE:\s*(YES|NO)", "", plan_text, flags=re.IGNORECASE).strip()

        return {
            "messages": [AIMessage(content=f"[PLAN]\n{clean_plan}")],
            "requires_code": requires_code,
        }

    return planner_node
