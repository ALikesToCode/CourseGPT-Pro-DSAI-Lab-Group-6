import os
from typing import Optional, Dict, Any, Generator

import requests

class APIClient:
    def __init__(self, base_url: Optional[str] = None):
        # Prefer env override so deployed Streamlit UIs can point at a remote FastAPI instance.
        self.base_url = base_url or os.getenv("COURSEGPT_API_BASE", "http://127.0.0.1:8000")

    def chat(self, prompt: str, thread_id: str, user_id: str, file: Optional[tuple] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Send a chat message to the API and yield the response chunks via SSE.
        """
        url = f"{self.base_url}/chat"
        data = {
            "prompt": prompt,
            "thread_id": thread_id,
            "user_id": user_id,
        }
        files = {}
        if file:
            files["file"] = file

        try:
            import json
            with requests.post(url, data=data, files=files if files else None, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                yield data_json
                            except json.JSONDecodeError:
                                continue
        except requests.RequestException as e:
            yield {"type": "error", "content": f"Error communicating with API: {str(e)}"}

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
