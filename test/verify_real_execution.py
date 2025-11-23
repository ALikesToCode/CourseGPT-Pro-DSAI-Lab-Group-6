import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# Ensure api is in path
sys.path.append(os.getcwd())

from api.config import get_settings
from api.services.ai_search import AISearchService
from api.graph.graph import graph
from langchain.messages import HumanMessage

async def main():
    load_dotenv()
    settings = get_settings()
    
    print("--- Configuration Check ---")
    print(f"RAG Configured: {settings.has_ai_search}")
    print(f"Router Agent URL: {settings.router_agent_url or 'Using Gemini Default'}")
    print(f"General Agent URL: {settings.general_agent_url or 'Using Gemini Default'}")
    
    if not settings.has_ai_search:
        print("\nWARNING: Cloudflare RAG credentials are missing in .env. RAG tests will fail or be skipped.")
        # We can still test the graph flow, but RAG tool will return error/message.

    # 1. Test RAG Service Directly
    if settings.has_ai_search:
        print("\n--- Testing Real RAG Service Connection ---")
        service = AISearchService(settings)
        try:
            # Search for something generic
            results = await service.search({"query": "course syllabus", "max_num_results": 1})
            print("RAG Service Call Successful!")
            print(f"Results found: {len(results.get('result', []))}")
        except Exception as e:
            print(f"RAG Service Call Failed: {e}")

    # 2. Test Full Graph Execution (Streaming)
    print("\n--- Testing Full Graph Execution (Streaming) ---")
    print("User Query: 'What does the uploaded document say about the Sprague-Grundy theorem?'")
    print("(This should trigger General Agent -> RAG Tool)")
    
    input_message = HumanMessage(content="What does the uploaded document say about the Sprague-Grundy theorem?")
    config = {"configurable": {"thread_id": "verify_test", "user_id": "test_user"}}
    
    print("\n--- Stream Output ---")
    try:
        async for event in graph.astream({"messages": [input_message]}, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                print(f"\n[Node: {node_name}]")
                
                messages = update.get("messages", [])
                if messages:
                    # Check for tool calls
                    for msg in messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"  >>> TOOL CALL: {tc['name']}")
                                print(f"  >>> ARGS: {tc['args']}")
                        
                        if hasattr(msg, "content") and msg.content:
                            print(f"  >>> CONTENT: {msg.content[:100]}...")
                            
    except Exception as e:
        print(f"\nGraph Execution Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
