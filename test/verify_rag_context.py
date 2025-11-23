import asyncio
import os
import sys
from dotenv import load_dotenv

# Ensure api is in path
sys.path.append(os.getcwd())

from api.config import get_settings
from api.services.ai_search import AISearchService
from api.routes.graph_call import _fetch_rag_context

async def main():
    load_dotenv()
    settings = get_settings()
    
    if not settings.has_ai_search:
        print("RAG not configured.")
        return

    service = AISearchService(settings)
    
    print("Fetching RAG context...")
    
    # Debug: Check raw response
    raw_resp = await service.search({"query": "course syllabus", "max_num_results": 5})
    print(f"DEBUG Raw Response keys: {raw_resp.keys()}")
    if "result" in raw_resp:
        print(f"DEBUG 'result' type: {type(raw_resp['result'])}")
        if isinstance(raw_resp['result'], dict):
             print(f"DEBUG 'result' keys: {raw_resp['result'].keys()}")
             if "data" in raw_resp['result'] and raw_resp['result']['data']:
                 print(f"DEBUG first hit: {raw_resp['result']['data'][0]}")

    try:
        contexts = await _fetch_rag_context(service, "course syllabus", "user_id")
        print(f"Success! Retrieved {len(contexts)} contexts.")
        for ctx in contexts:
            print(f"- {ctx.get('text')[:50]}...")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
