import os
import sys
from langchain_core.messages import HumanMessage
from ojas.config import auto_configure
from ojas.graph import build_graph
from ojas.rag.indexer import index_workspace

def validate():
    workspace = os.path.abspath(".")
    config = auto_configure(workspace=workspace, ignore_ram=True) # Bypass RAM for validation
    
    print("\n[STEP 1] Indexing workspace...")
    collection = index_workspace(workspace, config.embed_model, config.ollama_base_url)
    
    print("\n[STEP 2] Building graph...")
    graph = build_graph(config, collection)
    
    print("\n[STEP 3] Running end-to-end task...")
    # This task requires:
    # 1. Planning (Planner)
    # 2. Writing code (Coder)
    # 3. Extracting 'requests' (Dependency)
    # 4. Running in Docker (Executor)
    # 5. Evaluating (Evaluator)
    user_input = "Fetch the current time from 'http://worldtimeapi.org/api/timezone/Etc/UTC' using the requests library and print the 'datetime' field."
    
    state = {
        "messages": [HumanMessage(content=user_input)],
        "workspace_context": "",
        "generated_code": "",
        "dependencies": [],
        "docker_output": "",
        "retry_count": 0,
        "cycle_complete": False,
    }
    
    thread_config = {"configurable": {"thread_id": "validation-test"}}
    
    print(f"Executing task: {user_input}")
    try:
        final_state = graph.invoke(state, config=thread_config)
        
        print("\n[RESULTS]")
        for msg in final_state.get("messages", []):
            print(f"--- {msg.__class__.__name__} ---")
            print(msg.content)
            print("-" * 20)
            
        print(f"\nFinal State - Cycle Complete: {final_state.get('cycle_complete')}")
        print(f"Final State - Dependencies: {final_state.get('dependencies')}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    validate()
