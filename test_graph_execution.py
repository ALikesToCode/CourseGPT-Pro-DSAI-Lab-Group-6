import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Add project root to sys.path
sys.path.append(os.getcwd())

from api.graph.graph import graph
from api.config import get_settings

load_dotenv()

def test_graph():
    settings = get_settings()
    print(f"Using Google API Key: {settings.google_api_key[:5]}..." if settings.google_api_key else "No Google API Key found")
    
    # Test Query: Math problem to trigger Math Agent
    query = "Calculate the 10th Fibonacci number."
    print(f"\n--- Sending Query: {query} ---")
    
    state = {"messages": [HumanMessage(content=query)]}
    
    try:
        # Run the graph
        print("\n--- Running Graph with Real API ---")
        for event in graph.stream(state):
            for key, value in event.items():
                print(f"\nNode: {key}")
                if "messages" in value:
                    last_msg = value["messages"][-1] if isinstance(value["messages"], list) else value["messages"]
                    print(f"Message Type: {type(last_msg).__name__}")
                    if hasattr(last_msg, "content"):
                        print(f"Content: {last_msg.content[:200]}...")
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(f"Tool Calls: {last_msg.tool_calls}")
                        
    except Exception as e:
        print(f"\nError running graph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_graph()
