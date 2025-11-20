import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from unittest.mock import patch

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
    
    # Mock the LLM to avoid 429 errors and verify routing logic
    with patch("api.services.gemini_client.Gemini3Client.invoke") as mock_invoke:
        # Create a mock response that simulates the Router Agent calling the math_agent_handoff tool
        mock_tool_call = {
            "name": "math_agent_handoff",
            "args": {
                "task_summary": "Calculate 10th Fibonacci number",
                "thinking_outline": ["1. Identify n=10.", "2. Use Binet's formula or iteration.", "3. Calculate.", "4. Verify."],
                "route_plan": ["/math"],
                "route_rationale": "Math problem.",
                "handoff_plan": "/math",
                "todo_list": ["- [ ] Calculate"],
                "acceptance_criteria": ["Result is 55"],
                "expected_artifacts": ["Answer"],
                "difficulty": "introductory",
                "tags": ["math"],
                "metrics": {},
                "compute_budget": {},
                "repro": {},
                "citation_policy": "",
                "io_schema": {},
                "requires_browse": False
            },
            "id": "call_123"
        }
        
        # The router returns an AIMessage with tool_calls
        from langchain_core.messages import AIMessage
        mock_response = AIMessage(content="", tool_calls=[mock_tool_call])
        
        # Set the side_effect to return the mock response for the first call (Router)
        # and then maybe let others fail or mock them too if needed. 
        # For now, let's just see if it routes to 'math_agent'.
        mock_invoke.return_value = mock_response

        # Run the graph
        print("\n--- Running Graph with Mocked Router Response ---")
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
                
                # If we reached math_agent, we can stop the test as success
                if key == "math_agent":
                    print("\nSUCCESS: Routed to math_agent!")
                    return

if __name__ == "__main__":
    test_graph()
