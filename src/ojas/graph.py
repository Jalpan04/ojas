"""LangGraph state machine for Ojas.

Wires the six nodes (retrieval, planner, coder, dependency, executor,
evaluator) into a cyclic graph with conditional self-correction edges.

    User Input
        |
        v
    [retrieval] -- ChromaDB context injection
        |
        v
    [planner]   -- Architectural reasoning (biggest model)
        |
        v
    [coder]     -- Code generation (Qwen) <--.
        |                                    |
        v                                    |
    [dependency] -- AST import extraction     |
        |                                    |
        v                                    |
    [executor]  -- Docker sandbox            |
        |                                    |
        v                                    |
    [evaluator] ---(error?)------------------'
        |
        v (success)
      END
"""

from __future__ import annotations

from typing import Any

import chromadb
from langgraph.graph import END, StateGraph

from ojas.config import OjasConfig
from ojas.memory.checkpoint import get_checkpointer
from ojas.nodes.coder import make_coder_node
from ojas.nodes.dependency import dependency_node
from ojas.nodes.evaluator import evaluator_node, should_retry
from ojas.nodes.executor import make_executor_node
from ojas.nodes.planner import make_planner_node
from ojas.nodes.retrieval import make_retrieval_node
from ojas.state import OjasState


def build_graph(
    config: OjasConfig,
    collection: chromadb.Collection,
) -> Any:
    """Construct and compile the full Ojas LangGraph.

    Parameters
    ----------
    config:
        Resolved runtime configuration (model names, URLs, etc.).
    collection:
        The populated ChromaDB collection for RAG retrieval.

    Returns
    -------
    CompiledGraph
        A LangGraph ready to ``.invoke()`` or ``.stream()``.
    """
    # -- Create node functions -----------------------------------------------
    retrieval = make_retrieval_node(
        collection=collection,
        embed_model=config.embed_model,
        ollama_url=config.ollama_base_url,
    )
    planner = make_planner_node(
        model_name=config.planner_model,
        ollama_url=config.ollama_base_url,
        fallback_model=config.coder_model,
    )
    executor = make_executor_node(
        workspace_dir=config.workspace,
    )
    coder = make_coder_node(
        model_name=config.coder_model,
        ollama_url=config.ollama_base_url,
    )

    # -- Build the graph -----------------------------------------------------
    graph = StateGraph(OjasState)

    graph.add_node("retrieval", retrieval)
    graph.add_node("planner", planner)
    graph.add_node("coder", coder)
    graph.add_node("dependency", dependency_node)
    graph.add_node("executor", executor)
    graph.add_node("evaluator", evaluator_node)

    # -- Wire edges ----------------------------------------------------------
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "dependency")
    graph.add_edge("dependency", "executor")
    graph.add_edge("executor", "evaluator")

    # Conditional: evaluator -> coder (retry) or END (success).
    graph.add_conditional_edges(
        "evaluator",
        should_retry,
        {
            "coder": "coder",
            "end": END,
        },
    )

    # -- Compile with checkpointing ------------------------------------------
    checkpointer = get_checkpointer(config.workspace)
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled
