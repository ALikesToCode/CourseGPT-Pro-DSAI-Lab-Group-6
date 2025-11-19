# CourseGPT Pro - API Documentation

Complete API reference for developers integrating with CourseGPT Pro.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Authentication](#2-authentication)
3. [Base URL](#3-base-url)
4. [Endpoints](#4-endpoints)
5. [Request/Response Formats](#5-requestresponse-formats)
6. [Error Handling](#6-error-handling)
7. [Rate Limits](#7-rate-limits)
8. [Code Examples](#8-code-examples)

---

## 1. Overview

The CourseGPT Pro API provides programmatic access to:
- **File Management**: Upload, list, and delete documents in cloud storage
- **AI Search**: Query the RAG-powered document search
- **Multi-Agent Chat**: Interact with specialized educational AI agents
- **Health Monitoring**: Check service status

**API Style:** RESTful
**Data Format:** JSON (with multipart/form-data for file uploads)
**Protocol:** HTTP/HTTPS

---

## 2. Authentication

**Current Status:** No authentication required for development/local deployments

**Production Recommendations:**
- Implement API key authentication
- Use bearer tokens or JWT
- Rate limiting per API key

**Example Future Authentication Header:**
```http
Authorization: Bearer YOUR_API_KEY_HERE
```

---

## 3. Base URL

**Local Development:**
```
http://localhost:8000
```

**Production (Example):**
```
https://your-deployment-domain.com
```

**Interactive API Docs:**
- Swagger UI: `{BASE_URL}/docs`
- ReDoc: `{BASE_URL}/redoc`

---

## 4. Endpoints

### 4.1 Health Check

#### `GET /`

Check if the service is running.

**Request:**
```http
GET / HTTP/1.1
Host: localhost:8000
```

**Response:**
```json
{
  "status": "ok",
  "message": "CourseGPT graph service running"
}
```

**Status Codes:**
- `200 OK` - Service is healthy

---

### 4.2 File Management (Cloudflare R2)

#### `POST /files`

Upload a file to cloud storage.

**Request:**
```http
POST /files HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="lecture-notes.pdf"
Content-Type: application/pdf

[Binary file content]
------WebKitFormBoundary
Content-Disposition: form-data; name="prefix"

course-materials/
------WebKitFormBoundary--
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | File to upload (any format) |
| `prefix` | String | No | Folder path prefix (e.g., "user123/docs/") |

**Response:**
```json
{
  "message": "File uploaded successfully",
  "object_key": "course-materials/lecture-notes_20250119_143022.pdf",
  "url": "https://your-r2-bucket.r2.cloudflarestorage.com/course-materials/lecture-notes_20250119_143022.pdf"
}
```

**Status Codes:**
- `200 OK` - Upload successful
- `400 Bad Request` - No file provided
- `500 Internal Server Error` - Upload failed

---

#### `GET /files`

List files in cloud storage.

**Request:**
```http
GET /files?prefix=course-materials/&limit=50 HTTP/1.1
Host: localhost:8000
```

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prefix` | String | No | "" | Filter by folder prefix |
| `limit` | Integer | No | 50 | Max results (1-1000) |

**Response:**
```json
{
  "objects": [
    {
      "key": "course-materials/lecture-notes_20250119_143022.pdf",
      "size": 2048576,
      "last_modified": "2025-01-19T14:30:22.000Z",
      "etag": "\"d41d8cd98f00b204e9800998ecf8427e\""
    },
    {
      "key": "course-materials/chapter1.pdf",
      "size": 1024000,
      "last_modified": "2025-01-18T10:15:00.000Z",
      "etag": "\"098f6bcd4621d373cade4e832627b4f6\""
    }
  ],
  "count": 2,
  "is_truncated": false
}
```

**Status Codes:**
- `200 OK` - List retrieved
- `400 Bad Request` - Invalid parameters
- `500 Internal Server Error` - List failed

---

#### `GET /files/view/{object_key}`

Generate a presigned URL for viewing/downloading a file.

**Request:**
```http
GET /files/view/course-materials/lecture-notes_20250119_143022.pdf?expires_in=3600 HTTP/1.1
Host: localhost:8000
```

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `object_key` | String | Yes | Full object key/path |

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `expires_in` | Integer | No | 900 | Expiration in seconds (60-3600) |

**Response:**
```json
{
  "object_key": "course-materials/lecture-notes_20250119_143022.pdf",
  "presigned_url": "https://your-r2-bucket.r2.cloudflarestorage.com/course-materials/lecture-notes_20250119_143022.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=...",
  "expires_in": 3600
}
```

**Usage:**
- Open `presigned_url` in browser to view/download
- URL expires after specified duration

**Status Codes:**
- `200 OK` - URL generated
- `400 Bad Request` - Invalid parameters
- `500 Internal Server Error` - Generation failed

---

#### `DELETE /files/{object_key}`

Delete a file from cloud storage.

**Request:**
```http
DELETE /files/course-materials/lecture-notes_20250119_143022.pdf HTTP/1.1
Host: localhost:8000
```

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `object_key` | String | Yes | Full object key/path |

**Response:**
```json
{
  "message": "File deleted successfully",
  "object_key": "course-materials/lecture-notes_20250119_143022.pdf"
}
```

**Status Codes:**
- `200 OK` - Deletion successful
- `500 Internal Server Error` - Deletion failed

---

### 4.3 AI Search (RAG)

#### `POST /ai-search/query`

Search documents using natural language.

**Request:**
```http
POST /ai-search/query HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "query": "What are the key principles of object-oriented programming?",
  "filters": {
    "user_id": "user123"
  },
  "max_num_results": 5,
  "filter_operator": "AND"
}
```

**Body Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | String | Yes | - | Natural language search query |
| `filters` | Object | No | {} | Metadata filters (e.g., user_id) |
| `max_num_results` | Integer | No | 10 | Max results to return |
| `filter_operator` | String | No | "AND" | Filter logic: "AND" or "OR" |

**Response:**
```json
{
  "data": [
    {
      "id": "doc_123",
      "text": "Object-oriented programming (OOP) is based on four key principles: encapsulation, inheritance, polymorphism, and abstraction...",
      "score": 0.92,
      "metadata": {
        "filename": "oop-lecture.pdf",
        "page": 3,
        "user_id": "user123"
      }
    },
    {
      "id": "doc_456",
      "text": "Encapsulation is the bundling of data and methods that operate on that data...",
      "score": 0.87,
      "metadata": {
        "filename": "oop-textbook.pdf",
        "page": 12,
        "user_id": "user123"
      }
    }
  ],
  "count": 2,
  "query": "What are the key principles of object-oriented programming?"
}
```

**Status Codes:**
- `200 OK` - Search successful
- `400 Bad Request` - Invalid query
- `502 Bad Gateway` - Cloudflare API error
- `503 Service Unavailable` - AI Search not configured

---

#### `GET /ai-search/files`

List indexed documents and their ingestion status.

**Request:**
```http
GET /ai-search/files?page=1&per_page=20&status_filter=completed HTTP/1.1
Host: localhost:8000
```

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `page` | Integer | No | 1 | Page number (1-indexed) |
| `per_page` | Integer | No | 20 | Results per page |
| `status_filter` | String | No | null | Filter by status: "pending", "processing", "completed", "failed" |

**Response:**
```json
{
  "files": [
    {
      "id": "file_abc123",
      "filename": "oop-lecture.pdf",
      "status": "completed",
      "indexed_at": "2025-01-19T10:00:00Z",
      "size_bytes": 2048576,
      "chunks": 45
    },
    {
      "id": "file_def456",
      "filename": "data-structures.pdf",
      "status": "processing",
      "indexed_at": null,
      "size_bytes": 3145728,
      "chunks": null
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 2,
    "total_pages": 1
  }
}
```

**Status Codes:**
- `200 OK` - List retrieved
- `502 Bad Gateway` - Cloudflare API error
- `503 Service Unavailable` - AI Search not configured

---

#### `PATCH /ai-search/sync`

Trigger re-indexing of documents from R2 storage.

**Request:**
```http
PATCH /ai-search/sync HTTP/1.1
Host: localhost:8000
```

**Response:**
```json
{
  "message": "Sync triggered successfully",
  "status": "processing"
}
```

**Status Codes:**
- `200 OK` - Sync started
- `502 Bad Gateway` - Cloudflare API error
- `503 Service Unavailable` - AI Search not configured

---

### 4.4 Multi-Agent Chat

#### `POST /chat`

**⚠️ Note:** This endpoint exists in code but is NOT currently registered in `main.py`. See [Integration Status](#9-integration-status).

Interact with the multi-agent educational assistant.

**Request:**
```http
POST /chat HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="prompt"

Explain how binary search works and provide a Python implementation
------WebKitFormBoundary
Content-Disposition: form-data; name="thread_id"

session_user123_20250119
------WebKitFormBoundary
Content-Disposition: form-data; name="user_id"

user123
------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="algorithms.pdf"
Content-Type: application/pdf

[Binary PDF content]
------WebKitFormBoundary--
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | String | Yes | User's question or request |
| `thread_id` | String | Yes | Unique conversation thread ID |
| `user_id` | String | No | User identifier (for RAG filtering) |
| `file` | File (PDF) | No | Optional document for context |

**Response:**
```json
{
  "response": "Binary search is an efficient algorithm for finding an item in a sorted array...\n\nHere's a Python implementation:\n\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        \n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1\n```\n\n**Time Complexity:** O(log n)\n**Space Complexity:** O(1)"
}
```

**Status Codes:**
- `200 OK` - Response generated
- `400 Bad Request` - Missing required parameters
- `500 Internal Server Error` - Processing failed

**Conversation Threading:**
- Use the same `thread_id` for follow-up questions
- System maintains conversation history within a thread
- Each user should have unique `thread_id` values

---

## 5. Request/Response Formats

### Content Types

**Request:**
- JSON endpoints: `Content-Type: application/json`
- File upload endpoints: `Content-Type: multipart/form-data`

**Response:**
- Always: `Content-Type: application/json`

### Common Response Fields

**Success Response:**
```json
{
  "message": "Operation successful",
  "data": { ... },
  "timestamp": "2025-01-19T14:30:22.000Z"
}
```

**Error Response:**
```json
{
  "detail": "Error description",
  "error_type": "ValidationError",
  "status_code": 400
}
```

---

## 6. Error Handling

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| `200 OK` | Success | Request processed successfully |
| `400 Bad Request` | Client error | Invalid parameters, missing required fields |
| `404 Not Found` | Resource not found | Invalid endpoint or object_key |
| `500 Internal Server Error` | Server error | Unexpected server-side issue |
| `502 Bad Gateway` | Upstream error | External API (Cloudflare) failure |
| `503 Service Unavailable` | Service down | Feature not configured or temporarily unavailable |

### Error Response Format

```json
{
  "detail": "Human-readable error message",
  "error_type": "ErrorCategory",
  "status_code": 400,
  "request_id": "req_abc123"
}
```

### Handling Errors

**Python Example:**
```python
import httpx

try:
    response = httpx.post("http://localhost:8000/chat", data={...})
    response.raise_for_status()
    result = response.json()
except httpx.HTTPStatusError as e:
    print(f"HTTP error: {e.response.status_code}")
    print(f"Details: {e.response.json().get('detail')}")
except httpx.RequestError as e:
    print(f"Request failed: {e}")
```

---

## 7. Rate Limits

**Current Status:** No rate limiting implemented

**Recommendations for Production:**
- 10-20 requests per minute per user
- 100 requests per hour per user
- Exponential backoff on 429 (Too Many Requests) responses

**Future Response Headers:**
```http
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 15
X-RateLimit-Reset: 1705674600
```

---

## 8. Code Examples

### 8.1 Python

#### Upload File

```python
import httpx

BASE_URL = "http://localhost:8000"

async def upload_file(file_path: str, prefix: str = ""):
    async with httpx.AsyncClient() as client:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "application/pdf")}
            data = {"prefix": prefix} if prefix else {}

            response = await client.post(
                f"{BASE_URL}/files",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()

# Usage
result = await upload_file("lecture-notes.pdf", "user123/")
print(f"Uploaded: {result['object_key']}")
```

#### Chat Request

```python
async def chat(prompt: str, thread_id: str, user_id: str = None):
    async with httpx.AsyncClient(timeout=30.0) as client:
        data = {
            "prompt": prompt,
            "thread_id": thread_id,
        }
        if user_id:
            data["user_id"] = user_id

        response = await client.post(
            f"{BASE_URL}/chat",
            data=data
        )
        response.raise_for_status()
        return response.json()

# Usage
result = await chat(
    prompt="Explain quicksort algorithm",
    thread_id="session_123",
    user_id="user_456"
)
print(result["response"])
```

#### AI Search

```python
async def search_documents(query: str, user_id: str = None):
    async with httpx.AsyncClient() as client:
        payload = {
            "query": query,
            "max_num_results": 5
        }
        if user_id:
            payload["filters"] = {"user_id": user_id}

        response = await client.post(
            f"{BASE_URL}/ai-search/query",
            json=payload
        )
        response.raise_for_status()
        return response.json()

# Usage
results = await search_documents("binary search algorithm", user_id="user_123")
for result in results["data"]:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text'][:200]}...\n")
```

---

### 8.2 JavaScript (Node.js)

#### Upload File

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

async function uploadFile(filePath, prefix = '') {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));
  if (prefix) {
    form.append('prefix', prefix);
  }

  const response = await axios.post(`${BASE_URL}/files`, form, {
    headers: form.getHeaders()
  });

  return response.data;
}

// Usage
uploadFile('./lecture-notes.pdf', 'user123/')
  .then(result => console.log('Uploaded:', result.object_key))
  .catch(err => console.error('Error:', err.response.data));
```

#### Chat Request

```javascript
async function chat(prompt, threadId, userId = null) {
  const form = new FormData();
  form.append('prompt', prompt);
  form.append('thread_id', threadId);
  if (userId) {
    form.append('user_id', userId);
  }

  const response = await axios.post(`${BASE_URL}/chat`, form, {
    headers: form.getHeaders(),
    timeout: 30000
  });

  return response.data;
}

// Usage
chat('Explain merge sort', 'session_123', 'user_456')
  .then(result => console.log(result.response))
  .catch(err => console.error('Error:', err.response.data));
```

#### AI Search

```javascript
async function searchDocuments(query, userId = null) {
  const payload = {
    query: query,
    max_num_results: 5
  };

  if (userId) {
    payload.filters = { user_id: userId };
  }

  const response = await axios.post(
    `${BASE_URL}/ai-search/query`,
    payload
  );

  return response.data;
}

// Usage
searchDocuments('data structures', 'user_123')
  .then(results => {
    results.data.forEach(result => {
      console.log(`Score: ${result.score}`);
      console.log(`Text: ${result.text.substring(0, 200)}...\n`);
    });
  })
  .catch(err => console.error('Error:', err.response.data));
```

---

### 8.3 cURL

#### Upload File

```bash
curl -X POST http://localhost:8000/files \
  -F "file=@lecture-notes.pdf" \
  -F "prefix=user123/"
```

#### List Files

```bash
curl "http://localhost:8000/files?prefix=user123/&limit=10"
```

#### Get Presigned URL

```bash
curl "http://localhost:8000/files/view/user123/lecture-notes.pdf?expires_in=3600"
```

#### Delete File

```bash
curl -X DELETE "http://localhost:8000/files/user123/lecture-notes.pdf"
```

#### AI Search

```bash
curl -X POST http://localhost:8000/ai-search/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is polymorphism?",
    "filters": {"user_id": "user123"},
    "max_num_results": 3
  }'
```

#### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -F "prompt=Explain bubble sort algorithm" \
  -F "thread_id=session_123" \
  -F "user_id=user_456"
```

#### Chat with PDF

```bash
curl -X POST http://localhost:8000/chat \
  -F "prompt=Summarize the key points from this document" \
  -F "thread_id=session_123" \
  -F "file=@algorithms-chapter.pdf"
```

---

## 9. Integration Status

### Currently Available Endpoints

✅ Fully implemented and registered in `main.py`:
- `GET /` - Health check
- `POST /files` - Upload file
- `GET /files` - List files
- `GET /files/view/{object_key}` - Get presigned URL
- `DELETE /files/{object_key}` - Delete file
- `POST /ai-search/query` - Search documents
- `GET /ai-search/files` - List indexed files
- `PATCH /ai-search/sync` - Trigger sync

⚠️ Implemented but NOT registered:
- `POST /chat` - Multi-agent chat (exists in `routes/graph_call.py` but not included in `main.py`)

### To Enable Chat Endpoint

Add to [main.py](../Milestone-6/course_gpt_graph/main.py):

```python
from routes import graph_call

app.include_router(graph_call.router)
```

---

## 10. Webhooks (Future Feature)

Not currently implemented. Future consideration for:
- Document indexing completion notifications
- Long-running chat request callbacks
- Storage quota alerts

---

## Support

- **OpenAPI Schema**: `GET {BASE_URL}/openapi.json`
- **Interactive Docs**: `{BASE_URL}/docs`
- **GitHub Issues**: [Repository Issues Page]

---

*Last Updated: 2025-01-19*
*API Documentation Version: 1.0*
