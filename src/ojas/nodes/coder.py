"""Coder node -- uses Qwen for precise code generation.

Takes the plan produced by the Planner and generates executable Python code.
On self-correction loops, it also receives the previous error output.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama

CODER_SYSTEM_PROMPT = """\
You are the Coder inside Ojas, an autonomous AI developer agent.

Your role:
1. Read the plan provided by the Planner (in the conversation history).
2. Write the exact Python code that implements the current step.
3. Output ONLY the code inside a single ```python ... ``` fenced block.
4. If there was a previous execution error, fix the code based on the error output.

Rules:
- Write clean, production-quality Python.
- Include all necessary imports.
- Do NOT include explanations outside the code block.
- If the plan says "ANSWER_ONLY", output "NO_CODE_NEEDED" instead of code.
"""


def _extract_code(text: str) -> str:
    """Pull the first Python fenced code block from *text*."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try generic fenced block.
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def make_coder_node(
    model_name: str,
    ollama_url: str = "http://localhost:11434",
):
    """Return a coder node function bound to the Qwen model."""

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=0.1,
    )

    def coder_node(state: dict[str, Any]) -> dict[str, Any]:
        """Generate (or fix) code based on the plan and any prior errors."""
        system = SystemMessage(content=CODER_SYSTEM_PROMPT)
        messages = [system] + list(state.get("messages", []))

        # If there was a previous docker error, append it for context.
        docker_output = state.get("docker_output", "")
        if docker_output and state.get("retry_count", 0) > 0:
            from langchain_core.messages import HumanMessage

            messages.append(
                HumanMessage(
                    content=(
                        "The previous execution produced this error:\n"
                        f"```\n{docker_output}\n```\n"
                        "Please fix the code."
                    )
                )
            )

        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        code = _extract_code(raw)

        if "NO_CODE_NEEDED" in code.upper():
            return {
                "generated_code": "",
                "cycle_complete": True,
                "messages": [AIMessage(content=raw)],
            }

        return {
            "generated_code": code,
            "messages": [AIMessage(content=f"[CODE]\n```python\n{code}\n```")],
        }

    return coder_node
