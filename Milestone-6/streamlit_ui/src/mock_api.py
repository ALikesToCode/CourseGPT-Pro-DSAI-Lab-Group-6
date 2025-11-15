"""Lightweight mock API for CourseGPT frontend demo.

This module implements an in-memory mock API used by the Streamlit
frontend. It simulates document storage and streaming chat responses
with configurable latency to mimic network conditions.
"""

import time
import uuid
import random
from datetime import datetime
import base64

DEFAULT_LATENCY = 0.45


class MockAPI:
    """In-memory mock API for documents and chat streaming.

    Attributes:
        latency (float): Artificial latency factor applied to simulated
            operations (seconds multiplier).
    """

    def __init__(self):
        self.latency = DEFAULT_LATENCY

    def init_documents(self, session_state):
        """Populate `session_state['documents']` with sample documents.

        Args:
            session_state: Streamlit session state mapping to mutate.
        """
        if 'documents' not in session_state or not session_state['documents']:
            docs = []
            # Create placeholder documents without prefilled sample content.
            for i in range(4):
                did = str(uuid.uuid4())
                docs.append({
                    'id': did,
                    'title': f"Lecture Notes — Topic {i+1}",
                    'tags': ['lecture', 'ml'] if i % 2 == 0 else ['reading', 'theory'],
                    'size_kb': 0,
                    'uploaded_at': (datetime.now()).isoformat(),
                    'snippet': '',
                    'content': '',
                })
            session_state['documents'] = docs

    def list_documents(self, session_state):
        """Return the list of documents stored in session state.

        Args:
            session_state: Streamlit session state mapping.

        Returns:
            list: Documents (possibly empty).
        """
        return session_state.get('documents', [])

    def upload_document(self, session_state, title, tags, content_bytes, progress_callback=None):
        """Simulate uploading a document and insert into session state.

        Args:
            session_state: Streamlit session state mapping to modify.
            title (str): Document title.
            tags (str): Comma-separated tags string.
            content_bytes (bytes): File content.
            progress_callback (callable): Optional callback called with
                progress percentage (0-100) during simulated upload.

        Returns:
            dict: The inserted document metadata.
        """
        # Simulate progress and latency
        size_kb = max(1, len(content_bytes) // 1024)
        did = str(uuid.uuid4())
        text = None
        try:
            text = content_bytes.decode('utf-8')
        except Exception:
            text = base64.b64encode(content_bytes).decode('utf-8')

        for p in range(0, 101, 20):
            time.sleep(self.latency * 0.12)
            if progress_callback:
                progress_callback(p)

        doc = {
            'id': did,
            'title': title or f'Uploaded Document {len(session_state.get("documents",[]))+1}',
            'tags': [t.strip() for t in (tags or '').split(',') if t.strip()],
            'size_kb': size_kb,
            'uploaded_at': datetime.now().isoformat(),
            'snippet': (text or '')[:140] + '...',
            'content': text,
        }
        session_state.setdefault('documents', []).insert(0, doc)
        return doc

    def delete_document(self, session_state, doc_id):
        """Delete a document by id from session state.

        Args:
            session_state: Streamlit session state mapping to modify.
            doc_id (str): Document identifier to remove.

        Returns:
            bool: True when deletion completed (always true in mock).
        """
        docs = session_state.get('documents', [])
        new = [d for d in docs if d['id'] != doc_id]
        session_state['documents'] = new
        return True

    def get_document(self, session_state, doc_id):
        """Retrieve a document by id from session state.

        Returns None if not found.
        """
        for d in session_state.get('documents', []):
            if d['id'] == doc_id:
                return d
        return None

    def chat_response(self, messages, session_state):
        """Simulate streaming chat response: yield successive chunks.

        Args:
            messages (list): List of message dicts with 'sender' and 'text'.
            session_state: Streamlit session state mapping (used for latency).

        Yields:
            str: Successive text chunks mimicking streaming tokens.
        """
        # Short deterministic pseudo-response based on last user message
        user_msg = ""
        for m in reversed(messages):
            if m.get('sender') == 'user':
                user_msg = m.get('text', '')
                break

        # Build a mock answer in chunks
        base_answer = (
            f"CourseGPT analysis of: '{user_msg[:80]}'\n\n"
            "Key points:\n1) Understand the core concept.\n2) Provide examples.\n3) Suggest reading.\n\n"
            "References:\n- Lecture notes\n- Suggested exercises\n\n" 
            "If you want, I can generate a study plan or extract flashcards from this document."
        )

        words = base_answer.split()
        chunk_size = max(6, int(len(words) * 0.12))
        out = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            # simulate variable latency influenced by session setting
            time.sleep(self.latency * (0.05 + random.random() * 0.15))
            out.append(chunk + (' ' if i + chunk_size < len(words) else ''))
            yield chunk


mock_api = MockAPI()
