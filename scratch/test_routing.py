import os
from langchain_core.messages import HumanMessage
from ojas.config import auto_configure
from ojas.graph import build_graph
from ojas.rag.indexer import index_workspace

def test_routing():
    workspace = os.path.abspath(".")
    config = auto_configure(workspace=workspace, ignore_ram=True)
    collection = index_workspace(workspace, config.embed_model, config.ollama_base_url)
    graph = build_graph(config, collection)
    
    prompts = [
        "Hello Ojas! How are you?",
        "Can you help me write a python script to count files?"
    ]
    
    for prompt in prompts:
        print(f"\n>>> USER: {prompt}")
        state = {
            "messages": [HumanMessage(content=prompt)],
            "workspace_context": "",
            "generated_code": "",
            "dependencies": [],
            "docker_output": "",
            "retry_count": 0,
            "cycle_complete": False,
        }
        thread_config = {"configurable": {"thread_id": f"test-{prompt[:10]}"}}
        
        final_state = graph.invoke(state, config=thread_config)
        
        print(f"REQUIRES_CODE: {final_state.get('requires_code')}")
        for msg in final_state.get("messages", []):
            if "[PLAN]" in msg.content:
                print(f"PLAN: {msg.content}")

if __name__ == "__main__":
    test_routing()
