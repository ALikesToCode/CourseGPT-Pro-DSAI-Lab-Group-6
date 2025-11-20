import os
from typing import Optional, Dict, Any, Generator

import requests

class APIClient:
    def __init__(self, base_url: Optional[str] = None):
        # Prefer env override so deployed Streamlit UIs can point at a remote FastAPI instance.
        self.base_url = base_url or os.getenv("COURSEGPT_API_BASE", "http://127.0.0.1:8000")

    def chat(self, prompt: str, thread_id: str, user_id: str, file: Optional[tuple] = None) -> Generator[str, None, None]:
        """
        Send a chat message to the API and yield the response chunks.
        Note: The current API returns a single JSON response, not a stream.
        We will simulate streaming for the UI if needed, or just yield the result.
        """
        url = f"{self.base_url}/chat"
        data = {
            "prompt": prompt,
            "thread_id": thread_id,
            "user_id": user_id,
        }
        files = {}
        if file:
            # file is a tuple of (filename, file_bytes, content_type)
            files["file"] = file

        try:
            response = requests.post(url, data=data, files=files if files else None)
            response.raise_for_status()
            result = response.json()
            yield {
                "text": result.get("latest_message", ""),
                "router_debug": result.get("router_debug"),
            }
        except requests.RequestException as e:
            yield f"Error communicating with API: {str(e)}"

    def upload_file(self, file_bytes: bytes, filename: str, content_type: str = "application/pdf") -> Dict[str, Any]:
        """
        Upload a file to the API.
        """
        url = f"{self.base_url}/files/"
        files = {
            "file": (filename, file_bytes, content_type)
        }
        try:
            response = requests.post(url, files=files)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Upload failed: {str(e)}")

    def list_files(self, limit: int = 50) -> Dict[str, Any]:
        """
        List files from the API.
        """
        url = f"{self.base_url}/files/"
        params = {"limit": limit}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"List files failed: {str(e)}")

    def delete_file(self, object_key: str) -> Dict[str, Any]:
        """
        Delete a file via the API.
        """
        # The API expects the object_key in the path. 
        # Since object_key can contain slashes, we might need to handle it carefully.
        # requests handles URL encoding, but let's ensure it's correct.
        url = f"{self.base_url}/files/{object_key}"
        try:
            response = requests.delete(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Delete failed: {str(e)}")

    def get_file_url(self, object_key: str, expires_in: int = 900) -> str:
        """
        Get a temporary view URL for a file.
        """
        url = f"{self.base_url}/files/view/{object_key}"
        params = {"expires_in": expires_in}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get("url", "")
        except requests.RequestException:
            return ""

# Singleton instance
api_client = APIClient()
