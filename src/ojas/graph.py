"""LangGraph state machine for Ojas.

Wires the nodes into a cyclic ReAct graph.
"""

from __future__ import annotations

from typing import Any

import chromadb
from langgraph.graph import END, StateGraph
from langchain_core.messages import ToolMessage

from ojas.config import OjasConfig
from ojas.memory.checkpoint import get_checkpointer
from ojas.nodes.agent import make_agent_node
from ojas.nodes.planner import make_planner_node
from ojas.nodes.retrieval import make_retrieval_node
from ojas.state import OjasState
from ojas.tools.sandbox import execute_python_sandbox
from ojas.tools.host import run_host_shell_command
from ojas.tools.file import read_local_file
from ojas.tools.web_search import web_search_tool
from ojas.mcp_client import load_mcp_tools


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: OjasState):
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}
        
        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": []}
        
        outputs = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            tool = self.tools_by_name.get(tool_name)
            if not tool:
                result = f"Error: Tool '{tool_name}' not found."
            else:
                try:
                    result = str(tool.invoke(tool_args))
                except Exception as e:
                    result = f"Error executing tool: {e}"
            
            outputs.append(ToolMessage(
                content=result,
                name=tool_name,
                tool_call_id=tool_id
            ))
            
        return {"messages": outputs}


def make_context_manager_node():
    """Create a node to truncate context to prevent token limits."""
    def context_manager_node(state: OjasState) -> dict[str, Any]:
        from langchain_core.messages import RemoveMessage
        messages = state.get("messages", [])
        if len(messages) > 15:
            msgs_to_remove = messages[2:-5]
            removals = [RemoveMessage(id=m.id) for m in msgs_to_remove if m.id]
            if removals:
                return {"messages": removals}
        return {}
    return context_manager_node

def route_after_agent(state: OjasState):
    """If the LLM decided to use a tool, go to tools. Otherwise, end."""
    messages = state.get("messages", [])
    if not messages:
        return END
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    return END

def route_after_context(state: OjasState):
    """Router to skip the heavy planner for simple conversational turns."""
    if state.get("skip_planning", False):
        return "agent"
    return "planner"

def build_graph(
    config: OjasConfig,
) -> Any:
    """Construct and compile the full Ojas LangGraph."""
    
    # -- Initialize Tools ----------------------------------------------------
    tools = [
        web_search_tool,
        run_host_shell_command,
        read_local_file,
        execute_python_sandbox,
    ]
    
    # Dynamically load MCP tools
    mcp_tools = load_mcp_tools(config.workspace)
    tools.extend(mcp_tools)
    
    # -- Create node functions -----------------------------------------------
    context_manager = make_context_manager_node()
    retrieval = make_retrieval_node(config)
    
    planner = make_planner_node(
        model_name=config.planner_model,
        ollama_url=config.ollama_base_url,
        fallback_model=config.coder_model,
        workspace_dir=config.workspace,
    )
    
    agent = make_agent_node(
        model_name=config.agent_model, # Use the model specifically optimized for tool-calling
        tools=tools,
        ollama_url=config.ollama_base_url,
        workspace_dir=config.workspace,
    )
    
    tool_node = BasicToolNode(tools)

    # -- Build the graph -----------------------------------------------------
    graph = StateGraph(OjasState)

    graph.add_node("context_manager", context_manager)
    graph.add_node("retrieval", retrieval)
    graph.add_node("planner", planner)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)

    # -- Wire edges ----------------------------------------------------------
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "context_manager")
    
    graph.add_conditional_edges(
        "context_manager",
        route_after_context,
        {
            "planner": "planner",
            "agent": "agent"
        }
    )
    
    graph.add_edge("planner", "agent")
    
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # Loop back to agent after tools
    graph.add_edge("tools", "agent")

    # -- Compile with checkpointing ------------------------------------------
    checkpointer = get_checkpointer(config.workspace)
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"]
    )

    return compiled
