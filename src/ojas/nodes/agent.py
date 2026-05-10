from __future__ import annotations
import os
from typing import Any, List
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from rich.console import Console

console = Console()

JARVIS_SYSTEM_PROMPT = """You are Ojas, a highly capable, autonomous generalized AI assistant operating on the user's local machine.

You have access to a wide variety of tools, including web searching, local file manipulation, host terminal execution, and a Python Docker sandbox for complex calculations.

INSTRUCTIONS:
1. Do not assume every request requires writing code.
2. If the user asks a simple question, just answer it.
3. If the user asks you to perform a system task (like "organize my downloads folder"), use your shell or file manipulation tools.
4. If you encounter an error using a tool, analyze the error and try a different approach. You are autonomous; do not ask the user for help unless you are completely stuck."""

def make_agent_node(
    model_name: str,
    tools: List[BaseTool],
    ollama_url: str = "http://localhost:11434",
    workspace_dir: str = ".",
):
    """Return an agent node function bound to the model and tools."""

    # Detect if this is a qwen3 "thinking" model so we can disable
    # the internal reasoning chain for faster responses.
    is_thinking_model = "qwen3" in model_name.lower()

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=0.1,
        streaming=True,
        # Keep context window small for speed on low-RAM systems.
        num_ctx=2048,
    )

    # Bind tools to the model
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    def agent_node(state: dict[str, Any]) -> dict[str, Any]:
        """The core agent brain."""
        
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

        # For qwen3 models, disable the internal thinking chain.
        # Without this, qwen3 generates hundreds of hidden <think> tokens
        # before producing any visible output, making responses very slow.
        think_prefix = "/no_think\n" if is_thinking_model else ""

        system = SystemMessage(content=think_prefix + JARVIS_SYSTEM_PROMPT + project_rules)
        messages = [system] + list(state.get("messages", []))

        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as exc:
            from langchain_core.messages import AIMessage
            error_msg = f"Agent model error: {exc}"
            return {"messages": [AIMessage(content=error_msg)]}

    return agent_node
