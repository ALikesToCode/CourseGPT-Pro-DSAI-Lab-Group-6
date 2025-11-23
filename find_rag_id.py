import asyncio
import os
import httpx
from dotenv import load_dotenv

async def list_rags():
    load_dotenv()
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    token = os.getenv("CLOUDFLARE_AI_SEARCH_TOKEN")
    
    print(f"Account ID: {account_id}")
    print(f"Token: {token[:5]}..." if token else "Token: None")

    if not account_id or not token:
        print("Missing credentials.")
        return

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/autorag/rags"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    try:
        async with httpx.AsyncClient() as client:
            print(f"Requesting: {url}")
            response = await client.get(url, headers=headers)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(list_rags())
