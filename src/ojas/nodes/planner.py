"""Planner node -- uses the largest available model for deep reasoning.

The planner receives the user request plus RAG context and produces a
step-by-step architectural plan that the Coder node will implement.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama

PLANNER_SYSTEM_PROMPT = """\
You are the Planner inside Ojas, an autonomous AI developer agent.

Your role:
1. Analyze the user's request together with the workspace context (relevant code
   snippets from the project) that has been retrieved for you.
2. Produce a clear, step-by-step plan that a code-generation model will follow.
3. Identify which files need to be created or modified.
4. Note any third-party dependencies that will be required.
5. If the task requires running code, describe what the script should do.

Guidelines:
- Be precise and technical.
- Reference specific files from the workspace context when applicable.
- Keep the plan concise but complete.
- Do NOT write the code yourself -- that is the Coder's job.
- If the user's request is a simple question (not a coding task), answer it
  directly and mark the plan as "ANSWER_ONLY".
"""


def make_planner_node(
    model_name: str,
    ollama_url: str = "http://localhost:11434",
):
    """Return a planner node function bound to the given model."""

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=0.3,
    )

    def planner_node(state: dict[str, Any]) -> dict[str, Any]:
        """Invoke the planner model and append its plan to messages."""
        workspace_ctx = state.get("workspace_context", "")
        context_block = ""
        if workspace_ctx:
            context_block = (
                "\n\n<workspace_context>\n"
                f"{workspace_ctx}\n"
                "</workspace_context>"
            )

        system = SystemMessage(
            content=PLANNER_SYSTEM_PROMPT + context_block
        )

        # Build the message list: system + conversation history.
        messages = [system] + list(state.get("messages", []))

        response = llm.invoke(messages)
        plan_text = response.content if hasattr(response, "content") else str(response)

        return {
            "messages": [AIMessage(content=f"[PLAN]\n{plan_text}")],
        }

    return planner_node
