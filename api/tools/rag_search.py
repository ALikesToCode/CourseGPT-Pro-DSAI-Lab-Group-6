from typing import Optional
from langchain.tools import tool
from api.services.ai_search import AISearchService
from api.config import get_settings
import asyncio

@tool
def rag_search(query: str) -> str:
    """
    Search the knowledge base (RAG) for relevant documents and information.
    Use this tool when the user asks about specific course content, documents, or uploaded files.
    """
    settings = get_settings()
    service = AISearchService(settings)
    
    if not service.is_configured:
        return "RAG search is not configured. Please check the server settings."

    try:
        # Since tools are often synchronous in this setup, we run the async method
        # This might need adjustment if the agent runner is fully async
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(service.search({"query": query}))
        loop.close()
        
        # Format the result for the LLM
        if "result" in result and result["result"]:
            hits = result["result"]
            formatted_hits = []
            for hit in hits:
                # Adjust based on actual response structure from Cloudflare
                text = hit.get("payload", {}).get("text", "") or hit.get("text", "")
                score = hit.get("score", 0)
                formatted_hits.append(f"[Score: {score:.2f}] {text}")
            
            return "\n\n".join(formatted_hits)
        else:
            return "No relevant documents found."
            
    except Exception as e:
        return f"Error performing RAG search: {str(e)}"
