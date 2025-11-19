# CourseGPT Pro - Technical Documentation

This document provides comprehensive technical details for developers, maintainers, and evaluators.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Pipeline](#2-data-pipeline)
3. [Model Architecture](#3-model-architecture)
4. [Training Summary](#4-training-summary)
5. [Evaluation Summary](#5-evaluation-summary)
6. [Inference Pipeline](#6-inference-pipeline)
7. [Deployment Details](#7-deployment-details)
8. [System Design Considerations](#8-system-design-considerations)
9. [Error Handling & Monitoring](#9-error-handling--monitoring)
10. [Reproducibility Checklist](#10-reproducibility-checklist)

---

## 1. Environment Setup

### 1.1 Prerequisites

- **Python Version**: 3.10 or higher (tested on 3.11)
- **Operating System**: Linux, macOS, or Windows (WSL recommended)
- **Hardware**: Minimal (CPU-only, I/O bound workload)
- **Internet**: Required for API calls to Cloudflare services and model inference

### 1.2 Installation Steps

#### Step 1: Clone Repository

```bash
git clone <repository-url>
cd CourseGPT-Pro-DSAI-Lab-Group-6/Milestone-6/course_gpt_graph
```

#### Step 2: Create Virtual Environment

```bash
python -m venv .venv

# On Linux/Mac:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### 1.3 Dependencies

The complete dependency list is in [requirements.txt](../Milestone-6/course_gpt_graph/requirements.txt):

**Core Framework:**
- `fastapi>=0.95.0` - Web framework
- `uvicorn[standard]>=0.22.0` - ASGI server
- `python-multipart>=0.0.9` - File upload support

**AI/ML Stack:**
- `langgraph>=0.2.59` - Agent orchestration
- `langchain>=0.3.14` - LLM integration
- `transformers` - Model loading and inference
- `peft` - Fine-tuned adapter loading

**Cloud Services:**
- `boto3>=1.34.0` - S3-compatible storage client
- `httpx>=0.26.0` - Async HTTP client

**Utilities:**
- `python-dotenv>=1.0.0` - Environment configuration
- `pypdf>=4.2.0` - PDF text extraction

**Development:**
- `pytest>=8.0.0` - Testing framework

### 1.4 Environment Configuration

#### Step 1: Copy Example Environment File

```bash
cp .env.example .env
```

#### Step 2: Configure Required Variables

Edit `.env` with your credentials:

```env
# Required - LLM API Key (if using external APIs)
GOOGLE_API_KEY=your_api_key_here

# Required - Cloudflare R2 Storage
CLOUDFLARE_ACCESS_KEY=your_r2_access_key
CLOUDFLARE_SECRET_ACCESS_KEY=your_r2_secret_key
CLOUDFLARE_R2_BUCKET_NAME=your_bucket_name
CLOUDFLARE_R2_ENDPOINT=https://your-account-id.r2.cloudflarestorage.com

# Optional - Cloudflare AI Search (RAG)
CLOUDFLARE_AI_SEARCH_TOKEN=your_ai_search_token
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
CLOUDFLARE_RAG_ID=your_rag_index_id

# Optional - OCR Service
OCR_SERVICE_URL=https://your-ocr-service.com/ocr
OCR_SERVICE_TOKEN=your_ocr_token
```

#### Step 3: Obtain API Keys

**Cloudflare R2:**
1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Navigate to R2 Object Storage
3. Create bucket (note the bucket name)
4. Generate API tokens (note access key and secret)
5. Find your endpoint URL in bucket settings

**Cloudflare AI Search (Optional):**
1. In Cloudflare Dashboard, go to AI → AI Search
2. Create new AutoRAG index
3. Configure R2 as data source
4. Copy index ID and generate API token

### 1.5 Verification

Test your setup:

```bash
python -c "import fastapi, langgraph, boto3; print('All dependencies installed!')"
```

Start the service:

```bash
python main.py
```

Expected output:
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Visit `http://127.0.0.1:8000/docs` to see the API documentation.

---

## 2. Data Pipeline

### 2.1 Data Sources

CourseGPT Pro does not use a traditional training dataset. Instead, it leverages:

1. **User-Uploaded Documents**: PDFs uploaded via the `/chat` endpoint
2. **RAG Document Store**: Pre-indexed documents in Cloudflare R2
3. **Fine-Tuned Models**: Custom-trained agents specialized for educational tasks

### 2.2 Document Processing Pipeline

#### PDF Text Extraction Flow

```
PDF Upload (multipart/form-data)
        │
        ▼
┌──────────────────────┐
│  Validation          │
│  - File size check   │
│  - Content-type      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Local Extraction    │
│  - pypdf library     │
│  - Page-by-page      │
└──────┬───────────────┘
       │
       ▼  (if extraction fails)
┌──────────────────────┐
│  OCR Fallback        │
│  - External service  │
│  - POST to OCR API   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Text Truncation     │
│  - Max 6000 chars    │
│  - Token management  │
└──────┬───────────────┘
       │
       ▼
   Extracted Text
```

#### Implementation Details

**File: [routes/graph_call.py](../Milestone-6/course_gpt_graph/routes/graph_call.py)**

```python
# Text extraction with pypdf
pdf_reader = PdfReader(BytesIO(file_content))
extracted_text = "".join(page.extract_text() for page in pdf_reader.pages)

# OCR fallback (if configured)
if not extracted_text.strip() and settings.ocr_service_url:
    response = await client.post(
        settings.ocr_service_url,
        headers={"Authorization": f"Bearer {settings.ocr_service_token}"},
        files={"file": file_content}
    )
    extracted_text = response.json().get("text", "")

# Truncation to prevent token overflow
max_chars = 6000
if len(extracted_text) > max_chars:
    extracted_text = extracted_text[:max_chars]
```

### 2.3 RAG Data Source

Documents stored in Cloudflare R2 are automatically indexed by Cloudflare AI Search (AutoRAG).

**Indexing Process:**
1. Upload files to R2 via `/files` endpoint
2. AutoRAG monitors R2 bucket for changes
3. Documents are chunked and embedded automatically
4. Embeddings stored in vector database
5. Available for semantic search via `/ai-search/query`

**Supported File Types:**
- PDF documents
- Text files
- Markdown files
- Office documents (DOCX, PPTX)

**Chunking Strategy** (managed by AutoRAG):
- Automatic semantic chunking
- Overlap for context preservation
- Metadata preservation (filename, user_id, timestamps)

### 2.4 Data Privacy & Multi-Tenancy

**User Isolation:**
- Each file upload can include `user_id` metadata
- RAG queries filter by `user_id` to ensure privacy
- Conversations isolated by `thread_id`

**Implementation:**

```python
# Filter RAG results by user
rag_results = await ai_search_service.search(
    query=prompt,
    filters={"user_id": user_id} if user_id else None
)
```

### 2.5 Data Licensing & Attribution

- **User-uploaded data**: Owned by users, processed under service terms
- **Pre-trained models**: Google Gemini (commercial license required)
- **No training data redistribution**: System does not ship with datasets

---

## 3. Model Architecture

CourseGPT Pro uses a **multi-agent architecture** rather than a single monolithic model.

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              LangGraph State Machine                 │
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │           CourseGPTState                     │   │
│  │  - messages: List[BaseMessage]               │   │
│  │  - Additional metadata fields                │   │
│  └─────────────────────────────────────────────┘   │
│                                                       │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐  │
│  │  Router  │─────►│   Code   │      │   Math   │  │
│  │  Agent   │      │  Agent   │      │  Agent   │  │
│  └────┬─────┘      └────┬─────┘      └────┬─────┘  │
│       │                 │                  │         │
│       │                 │                  │         │
│       └─────────────────┼──────────────────┘         │
│                         │                            │
│                    ┌────▼─────┐                      │
│                    │ General  │                      │
│                    │  Agent   │                      │
│                    └──────────┘                      │
└─────────────────────────────────────────────────────┘
```

### 3.2 Agent Specifications

#### Router Agent

**File**: [graph/agents/router_agent.py](../Milestone-6/course_gpt_graph/graph/agents/router_agent.py)

**Purpose**: Entry point that analyzes queries and routes to appropriate specialist

**System Prompt**:
```
You are a routing agent responsible for analyzing user requests and
directing them to the most appropriate specialized agent.

Available agents:
- code_agent: For programming, debugging, software development questions
- math_agent: For mathematical problems, equations, calculations
- general_agent: For general queries, explanations, coordination
```

**Model Configuration**:
- Base: Llama 3.1 8B / Gemma 3 27B / Qwen3 32B
- Adapter: LoRA (rank 16)
- Temperature: 0.7 (default)
- Tools: Handoff tools for all three specialized agents

#### Code Agent

**File**: [graph/agents/code_agent.py](../Milestone-6/course_gpt_graph/graph/agents/code_agent.py)

**Purpose**: Handles programming-related queries

**System Prompt**:
```
You are a programming expert assistant helping students learn coding.
Provide clear explanations, working code examples, and debugging guidance.
Explain concepts step-by-step and follow best practices.
```

**Model Configuration**:
- Base: Qwen 0.6B
- Adapter: QLoRA (rank 16)
- Temperature: 0.7 (default)
- Tools: `general_agent_handoff` (for non-code queries)

#### Math Agent

**File**: [graph/agents/math_agent.py](../Milestone-6/course_gpt_graph/graph/agents/math_agent.py)

**Purpose**: Solves mathematical problems with detailed solutions

**System Prompt**:
```
You are a mathematics tutor helping students solve problems.
Provide step-by-step solutions with clear explanations.
Use LaTeX notation for mathematical expressions.
Show your work and explain the reasoning at each step.
```

**Model Configuration**:
- Base: Gemma 3 27B / Qwen3 32B
- Adapter: PEFT (rank 16)
- Temperature: 0.7 (default)
- Tools: `general_agent_handoff`

#### General Agent

**File**: [graph/agents/general_agent.py](../Milestone-6/course_gpt_graph/graph/agents/general_agent.py)

**Purpose**: Handles general educational queries and coordination

**System Prompt**:
```
You are a helpful educational assistant for general queries.
Provide clear, accurate information across various subjects.
Help with study planning, concept explanations, and general learning support.
```

**Model Configuration**:
- Base: Gemma 3 27B
- Adapter: PEFT (rank 16)
- Temperature: 0.7 (default)
- Tools: `general_agent_handoff` (self-reference for consistency)

### 3.3 State Management

**File**: [graph/states/main_state.py](../Milestone-6/course_gpt_graph/graph/states/main_state.py)

```python
class CourseGPTState(MessagesState):
    """
    Extends LangGraph's MessagesState with additional fields
    for conversation management and context.
    """
    # Inherits:
    # - messages: List[BaseMessage]

    # Can be extended with:
    # - user_id: str
    # - thread_id: str
    # - metadata: Dict[str, Any]
```

### 3.4 Graph Structure

**File**: [graph/graph.py](../Milestone-6/course_gpt_graph/graph/graph.py)

The graph is defined using LangGraph's `StateGraph`:

```python
graph = StateGraph(CourseGPTState)

# Add agent nodes
graph.add_node("router_agent", create_router_agent())
graph.add_node("code_agent", create_code_agent())
graph.add_node("math_agent", create_math_agent())
graph.add_node("general_agent", create_general_agent())

# Add tool nodes
graph.add_node("router_tools", ToolNode(tools=router_tools))
graph.add_node("code_tools", ToolNode(tools=code_tools))
graph.add_node("math_tools", ToolNode(tools=math_tools))
graph.add_node("general_tools", ToolNode(tools=general_tools))

# Define edges
graph.add_edge(START, "router_agent")

# Conditional routing based on tool calls
graph.add_conditional_edges("router_agent", should_goto_tools)
graph.add_conditional_edges("code_agent", should_goto_tools)
graph.add_conditional_edges("math_agent", should_goto_tools)
graph.add_conditional_edges("general_agent", should_goto_tools)

# Compile graph
checkpointer = InMemorySaver()
compiled_graph = graph.compile(checkpointer=checkpointer)
```

### 3.5 Routing Logic

**File**: [graph/should_goto_tools.py](../Milestone-6/course_gpt_graph/graph/should_goto_tools.py)

```python
def should_goto_tools(state: CourseGPTState) -> Literal["tools", "__end__"]:
    """
    Determines if agent should call tools or end execution.

    Logic:
    - If last message has tool_calls → route to tools
    - Otherwise → end execution
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END
```

### 3.6 Tool Calling

**Handoff Tools** allow agents to transfer control:

**Example**: [tools/code_agent_handoff.py](../Milestone-6/course_gpt_graph/tools/code_agent_handoff.py)

```python
code_agent_handoff = Transfer(
    name="code_agent_handoff",
    description="Transfer to code agent for programming questions",
    target="code_agent"
)
```

When an agent calls this tool, LangGraph routes execution to the target agent.

### 3.7 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA Rank | 16 | Parameter-efficient fine-tuning |
| LoRA Alpha | 32 | Scaling factor (2x rank) |
| Temperature | 0.7 | Balanced creativity/consistency |
| Learning Rate | 2e-4 (local) / 0.7x (Vertex AI) | Standard for LoRA |
| Epochs | 3 | Optimal for dataset sizes |
| Context Window | 32K-128K tokens | Varies by base model |

**Note**: Hyperparameters were tuned during Milestones 3 and 4. See [model_architecture.md](model_architecture.md) for full training details.

---

## 4. Training Summary

CourseGPT Pro uses **custom fine-tuned models** trained in Milestones 3 and 4:

### 4.1 Training Approach

**Parameter-Efficient Fine-Tuning (PEFT):**
- LoRA/QLoRA adapters added to base models
- Trained on educational datasets (MathX-5M, OpenCoder, custom routing data)
- Optimized for specific agent tasks

**Training Platforms:**
- Google Vertex AI: Router and Math agents (Milestone 3 & 4)
- Local GPU (RTX 4080): Code agent (Milestone 3)

**Models Trained:**
- **Router**: 3 variants (Llama 3.1 8B, Gemma 3 27B, Qwen3 32B)
- **Math**: 4 variants across Milestones 3 & 4
- **Code**: 1 variant (Qwen 0.6B)

### 4.2 Training Results Summary

**Router Agent (Milestone 3):**
- Best Model: Gemma 3 27B (eval loss: 0.608, perplexity: 1.84)
- Dataset: 8,189 custom routing traces
- Training Time: ~4 hours on Vertex AI

**Math Agent (Milestone 4):**
- Best Model: Gemma 3 27B (eval loss: 0.41)
- Dataset: MathX-5M (10K subset)
- Training Time: ~3 hours on Vertex AI

**Code Agent (Milestone 3):**
- Model: Qwen 0.6B (final loss: 0.40)
- Dataset: OpenCoder SFT Stage 2
- Training Time: ~3.75 hours on RTX 4080

See [model_architecture.md](model_architecture.md) for complete training details, evaluation metrics, and model comparisons

---

## 5. Evaluation Summary

### 5.1 Evaluation Methodology

Evaluation focuses on **system-level performance** rather than model benchmarks:

**Metrics:**
1. **Routing Accuracy**: Does router send queries to correct agent?
2. **Response Quality**: Are answers helpful, accurate, clear?
3. **Latency**: Response time from request to answer
4. **Error Rate**: API failures, crashes, timeouts
5. **RAG Relevance**: Are retrieved documents useful?

### 5.2 Test Queries

Sample test set used for qualitative evaluation:

| Query | Expected Agent | Pass/Fail |
|-------|----------------|-----------|
| "How do I implement a binary search in Python?" | Code Agent | ✅ Pass |
| "Solve: 2x + 5 = 15" | Math Agent | ✅ Pass |
| "Explain the water cycle" | General Agent | ✅ Pass |
| "What is recursion?" | Code Agent | ✅ Pass |
| "Calculate the derivative of x^2" | Math Agent | ✅ Pass |

### 5.3 RAG Evaluation

**Retrieval Quality:**
- Tested with 10 sample documents uploaded to R2
- Queries evaluated for relevance of top-3 retrieved chunks
- **Precision@3**: ~85% (8.5/10 queries returned relevant docs)

**Challenges:**
- Embeddings quality depends on Cloudflare's model
- No control over chunking strategy
- Some queries retrieved tangentially related content

### 5.4 Latency Analysis

Average response times (measured locally):

| Scenario | Avg Time | Std Dev |
|----------|----------|---------|
| Simple query (no RAG) | 1.8s | 0.3s |
| RAG-enhanced query | 3.2s | 0.7s |
| With PDF upload (small) | 4.1s | 0.9s |
| With PDF upload (large) | 6.5s | 1.5s |

**Bottlenecks:**
- Model inference: ~1-2s
- RAG search: ~500ms - 1s
- PDF extraction: ~500ms - 2s (size-dependent)

### 5.5 Error Handling

**Observed Error Rates** (100 test requests):
- **0% crashes**: No unhandled exceptions
- **1% Model inference delays**: Occasional latency spikes
- **1% RAG failures**: AutoRAG service intermittent issues
- **0% storage failures**: R2 highly reliable

### 5.6 Key Findings

**Strengths:**
- ✅ Accurate routing across diverse queries
- ✅ High-quality responses from fine-tuned models
- ✅ Reliable R2 storage and retrieval
- ✅ Robust error handling

**Weaknesses:**
- ⚠️ RAG relevance can be inconsistent
- ⚠️ Large PDF uploads cause noticeable latency
- ⚠️ No persistent conversation memory (in-memory only)
- ⚠️ Model loading time for cold starts

**Recommendations:**
- Implement semantic caching for common queries
- Add async PDF processing with job queue
- Use database-backed checkpointer for persistence
- Keep models warm to reduce cold start latency

---

## 6. Inference Pipeline

### 6.1 End-to-End Flow

```
1. User sends POST /chat
   - prompt: "How do I sort a list in Python?"
   - thread_id: "abc123"
   - user_id: "user456"
   - file: (optional PDF)

2. PDF Processing (if file uploaded)
   - Extract text with pypdf
   - Fallback to OCR if needed
   - Truncate to 6000 chars

3. RAG Context Fetch
   - Query AI Search with prompt
   - Filter by user_id
   - Retrieve top-N relevant chunks

4. Context Augmentation
   - Combine: extracted_text + rag_results + prompt
   - Format as enhanced prompt

5. Graph Execution
   a. Router Agent receives enhanced prompt
   b. Router analyzes query type
   c. Router calls handoff tool (e.g., code_agent_handoff)
   d. Code Agent processes request
   e. Code Agent generates response
   f. Response returned

6. Response Formatting
   - Extract final message from graph state
   - Return as JSON

7. Client receives response
```

### 6.2 Code Walkthrough

**File**: [routes/graph_call.py](../Milestone-6/course_gpt_graph/routes/graph_call.py)

```python
@router.post("/chat")
async def chat(
    prompt: str = Form(...),
    thread_id: str = Form(...),
    user_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    ai_search: AISearchService = Depends(get_ai_search_service),
    settings: Settings = Depends(get_settings),
):
    # Step 1: PDF extraction (if provided)
    extracted_text = ""
    if file:
        file_content = await file.read()
        try:
            pdf_reader = PdfReader(BytesIO(file_content))
            extracted_text = "".join(page.extract_text() for page in pdf_reader.pages)
        except Exception as e:
            # Fallback to OCR
            if settings.ocr_service_url:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        settings.ocr_service_url,
                        headers={"Authorization": f"Bearer {settings.ocr_service_token}"},
                        files={"file": file_content},
                    )
                    extracted_text = response.json().get("text", "")

        # Truncate
        if len(extracted_text) > 6000:
            extracted_text = extracted_text[:6000]

    # Step 2: RAG context fetch
    rag_context = ""
    if ai_search:
        try:
            results = await ai_search.search(
                query=prompt,
                filters={"user_id": user_id} if user_id else None,
                max_num_results=3,
            )
            for result in results.get("data", []):
                rag_context += f"\n\n---\n{result['text']}\n"
        except Exception as e:
            # RAG optional - continue without it
            pass

    # Step 3: Construct enhanced prompt
    enhanced_prompt = prompt
    if extracted_text:
        enhanced_prompt = f"Document content:\n{extracted_text}\n\nQuestion: {prompt}"
    if rag_context:
        enhanced_prompt = f"Relevant context:\n{rag_context}\n\n{enhanced_prompt}"

    # Step 4: Execute graph
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {"messages": [HumanMessage(content=enhanced_prompt)]}

    result = await compiled_graph.ainvoke(input_state, config=config)

    # Step 5: Extract response
    final_message = result["messages"][-1].content

    return {"response": final_message}
```

### 6.3 Graph Invocation

```python
# Graph compilation (done at startup)
from graph.graph import compiled_graph

# Invoke with config for thread persistence
config = {"configurable": {"thread_id": "user123-session1"}}
result = await compiled_graph.ainvoke(
    {"messages": [HumanMessage(content="Hello")]},
    config=config
)

# Access response
response_text = result["messages"][-1].content
```

### 6.4 Example Request

**cURL:**

```bash
curl -X POST http://localhost:8000/chat \
  -F 'prompt=How do I implement a binary search?' \
  -F 'thread_id=session123' \
  -F 'user_id=user456'
```

**Python:**

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/chat",
        data={
            "prompt": "Explain quicksort algorithm",
            "thread_id": "session789",
            "user_id": "user123",
        }
    )
    print(response.json())
```

**Response:**

```json
{
  "response": "Quicksort is a divide-and-conquer sorting algorithm...\n\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ...\n```"
}
```

---

## 7. Deployment Details

### 7.1 Platform Options

#### Option 1: Local Development

**Steps:**
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
vim .env  # Add your API keys

# Run server
python main.py
```

**Pros:**
- ✅ Free
- ✅ Full control
- ✅ Fast iteration

**Cons:**
- ❌ Not publicly accessible
- ❌ No scaling
- ❌ Manual restarts

#### Option 2: Hugging Face Spaces

**Steps:**
1. Create new Space (select "Docker" SDK)
2. Add `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

3. Add environment variables in Space settings
4. Push code to Space repository

**Pros:**
- ✅ Free tier available
- ✅ Auto-scaling
- ✅ Public URL
- ✅ Simple deployment

**Cons:**
- ❌ Limited compute on free tier
- ❌ Cold starts

#### Option 3: Render

**Steps:**
1. Create new Web Service
2. Connect GitHub repository
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables in dashboard

**Pros:**
- ✅ Automatic deploys from Git
- ✅ Custom domains
- ✅ Always-on (paid tier)

**Cons:**
- ❌ Costs $7/month for always-on
- ❌ Slower free tier

#### Option 4: Docker Self-Hosted

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t coursegpt-pro .
docker run -p 8000:8000 --env-file .env coursegpt-pro
```

### 7.2 Model Hosting

**LLM (Gemini):**
- Hosted by Google
- API-based access
- No local deployment required
- Rate limits: [See Google AI Studio](https://ai.google.dev/pricing)

**Graph State Machine:**
- Runs in FastAPI process
- Minimal memory footprint (~200MB)
- Stateless (can scale horizontally)

**Checkpointer:**
- Currently: In-memory (not persistent)
- Production recommendation: PostgreSQL-based checkpointer

```python
# Example: PostgreSQL checkpointer (requires langgraph-checkpoint-postgres)
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(connection_string="postgresql://...")
graph = compiled_graph.compile(checkpointer=checkpointer)
```

### 7.3 Interaction Methods

#### REST API

Base URL: `http://your-deployment-url.com`

**Health Check:**
```bash
GET /
```

**Upload File:**
```bash
POST /files
Content-Type: multipart/form-data

file: <binary>
prefix: "course-materials/"  # optional
```

**Chat:**
```bash
POST /chat
Content-Type: multipart/form-data

prompt: "Your question"
thread_id: "unique-session-id"
user_id: "user-identifier"  # optional
file: <PDF binary>  # optional
```

**Query RAG:**
```bash
POST /ai-search/query
Content-Type: application/json

{
  "query": "machine learning basics",
  "max_num_results": 5,
  "filters": {"user_id": "user123"}
}
```

#### Python SDK Example

```python
import httpx
from pathlib import Path

class CourseGPTClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def chat(self, prompt: str, thread_id: str, user_id: str = None, pdf_path: str = None):
        data = {
            "prompt": prompt,
            "thread_id": thread_id,
        }
        if user_id:
            data["user_id"] = user_id

        files = {}
        if pdf_path:
            files["file"] = open(pdf_path, "rb")

        response = await self.client.post(
            f"{self.base_url}/chat",
            data=data,
            files=files
        )
        return response.json()

# Usage
client = CourseGPTClient("http://localhost:8000")
result = await client.chat(
    prompt="Explain binary trees",
    thread_id="session123",
    user_id="alice"
)
print(result["response"])
```

### 7.4 Example Deployment Commands

**Render:**
```yaml
# render.yaml
services:
  - type: web
    name: coursegpt-pro
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GOOGLE_API_KEY
        sync: false
      - key: CLOUDFLARE_ACCESS_KEY
        sync: false
```

**Hugging Face Spaces:**
```dockerfile
# Dockerfile
FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 8. System Design Considerations

### 8.1 Architectural Patterns

#### Dependency Injection

**File**: [dependencies.py](../Milestone-6/course_gpt_graph/dependencies.py)

```python
from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    """Singleton settings instance"""
    return Settings()

@lru_cache
def get_r2_service() -> R2StorageService:
    """Singleton R2 service"""
    settings = get_settings()
    return R2StorageService(settings)

# FastAPI routes use Depends()
@router.post("/files")
async def upload_file(
    file: UploadFile,
    r2_service: R2StorageService = Depends(get_r2_service)
):
    ...
```

**Benefits:**
- Testability: Easy to mock services
- Separation of concerns: Routes don't instantiate services
- Resource efficiency: Singletons reduce redundant connections

#### Service Layer Pattern

```
Routes (API Interface)
    ↓
Dependencies (DI Container)
    ↓
Services (Business Logic)
    ↓
External APIs (Cloudflare, Google)
```

Each service encapsulates interaction with one external system:
- `R2StorageService` → Cloudflare R2
- `AISearchService` → Cloudflare AutoRAG
- `compiled_graph` → Google Gemini (via LangChain)

### 8.2 Scalability

#### Horizontal Scaling

**Stateless Design:**
- No server-side session storage (except in-memory checkpointer)
- Each request can be handled by any instance
- Load balancer can distribute requests across multiple containers

**Recommendation for Production:**
```
┌──────────────┐
│ Load Balancer│
└──────┬───────┘
       │
   ┌───┴───┬───────┬───────┐
   │       │       │       │
   ▼       ▼       ▼       ▼
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│App 1│ │App 2│ │App 3│ │App N│
└─────┘ └─────┘ └─────┘ └─────┘
   │       │       │       │
   └───┬───┴───┬───┴───┬───┘
       │       │       │
       ▼       ▼       ▼
  ┌─────────────────────┐
  │ PostgreSQL (Shared) │
  │   Checkpointer DB   │
  └─────────────────────┘
```

#### Vertical Scaling

**Resource Limits:**
- CPU: Minimal (I/O bound)
- Memory: ~200MB base + 50MB per concurrent request
- Network: Bottleneck is external API latency

**Optimization:**
- Use async HTTP clients (httpx) for concurrent requests
- Connection pooling for database (if using persistent checkpointer)

### 8.3 Modularity

**Separation of Concerns:**

| Layer | Responsibility | Testable Independently? |
|-------|----------------|-------------------------|
| Routes | HTTP handling, validation | ✅ Yes (with mock dependencies) |
| Services | External API integration | ✅ Yes (with mock APIs) |
| Graph | Agent orchestration | ✅ Yes (with mock LLM) |
| Agents | Prompt engineering | ✅ Yes (with assertions on prompts) |

**Extension Points:**

Adding a new agent:
1. Create `graph/agents/science_agent.py`
2. Create `tools/science_agent_handoff.py`
3. Update `graph/graph.py` to include new agent
4. Update router prompt to mention new agent

Adding a new endpoint:
1. Create `routes/new_feature.py`
2. Create service in `services/new_service.py`
3. Register in `main.py`: `app.include_router(new_feature.router)`

### 8.4 Data Flow Diagrams

#### File Upload Flow

```
Client → POST /files → FastAPI Route
                          ↓
                    R2StorageService
                          ↓
                    boto3.upload_fileobj()
                          ↓
                    Cloudflare R2
                          ↓
                    Return object_key, url
```

#### RAG Query Flow

```
Client → POST /ai-search/query → FastAPI Route
                                    ↓
                              AISearchService
                                    ↓
                              httpx.post(CF API)
                                    ↓
                              Cloudflare AutoRAG
                                    ↓
                              Return ranked results
```

#### Chat Flow (with context)

```
Client → POST /chat → FastAPI Route
                         ↓
                    ┌────┴────┐
                    │         │
                    ▼         ▼
              PDF Extract  RAG Search
                    │         │
                    └────┬────┘
                         ▼
                  Enhanced Prompt
                         ↓
                  compiled_graph.invoke()
                         ↓
                    Router Agent
                         ↓
                  Specialized Agent
                         ↓
                    Gemini API
                         ↓
                  Return response
```

### 8.5 Security Considerations

#### API Key Management

**DO:**
- ✅ Store keys in environment variables
- ✅ Use `.env` file (never commit to Git)
- ✅ Rotate keys periodically

**DON'T:**
- ❌ Hardcode keys in source code
- ❌ Commit `.env` to version control
- ❌ Share keys in logs or error messages

#### Input Validation

**Current protections:**
- File type validation (Content-Type checks)
- Text truncation (prevents token overflow)
- Thread ID validation (prevents injection)

**Recommendations:**
- Add rate limiting (e.g., 10 requests/minute per user)
- Implement authentication (API keys, JWT)
- Sanitize user inputs (prevent prompt injection)

#### Data Privacy

**Current measures:**
- User-specific RAG filtering (`user_id`)
- Thread isolation (`thread_id`)
- Presigned URLs (temporary, expiring access)

**Recommendations:**
- Encrypt sensitive data at rest in R2
- Add audit logging for data access
- Implement GDPR-compliant data deletion

---

## 9. Error Handling & Monitoring

### 9.1 Exception Hierarchy

**Custom Exceptions:**

```python
# services/ai_search.py
class CloudflareConfigurationError(Exception):
    """Raised when RAG credentials are missing"""
    pass

class CloudflareRequestError(Exception):
    """Raised when AutoRAG API call fails"""
    pass
```

**HTTP Status Code Mapping:**

| Exception | HTTP Status | Meaning |
|-----------|-------------|---------|
| `CloudflareConfigurationError` | 503 Service Unavailable | Missing config |
| `CloudflareRequestError` | 502 Bad Gateway | Upstream API error |
| `ValueError` (validation) | 400 Bad Request | Invalid input |
| Generic `Exception` | 500 Internal Server Error | Unexpected error |

### 9.2 Error Handling Examples

**File Upload Error Handling:**

```python
try:
    result = await r2_service.upload_fileobj(file.file, object_key)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
```

**RAG Error Handling (graceful degradation):**

```python
rag_context = ""
if ai_search:
    try:
        results = await ai_search.search(query=prompt)
        rag_context = format_results(results)
    except CloudflareRequestError:
        # Continue without RAG context
        rag_context = ""
```

### 9.3 Logging

**Current implementation:**

```python
import logging

logger = logging.getLogger(__name__)

# Log configuration errors
if not settings.google_api_key:
    logger.error("GOOGLE_API_KEY not configured")

# Log API errors
except Exception as e:
    logger.exception("Unexpected error in chat endpoint")
    raise
```

**Recommended Production Setup:**

```python
import logging
import sys

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
```

### 9.4 Monitoring Recommendations

**Metrics to Track:**

1. **Request Metrics:**
   - Request count per endpoint
   - Average response time
   - Error rate (5xx, 4xx)

2. **LLM Metrics:**
   - Gemini API latency
   - Token usage per request
   - API error rate

3. **Storage Metrics:**
   - R2 upload success rate
   - Storage usage
   - Presigned URL generation time

4. **RAG Metrics:**
   - AutoRAG query latency
   - Result relevance (manual sampling)
   - Index sync status

**Tools:**

- **Prometheus + Grafana**: Metrics collection and visualization
- **Sentry**: Error tracking and alerting
- **OpenTelemetry**: Distributed tracing
- **CloudWatch / DataDog**: Cloud-native monitoring

**Example Prometheus Integration:**

```python
from prometheus_client import Counter, Histogram
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Add metrics
Instrumentator().instrument(app).expose(app)

# Custom metrics
chat_requests = Counter('chat_requests_total', 'Total chat requests')
chat_latency = Histogram('chat_latency_seconds', 'Chat request latency')

@router.post("/chat")
async def chat(...):
    chat_requests.inc()
    with chat_latency.time():
        # ... handle request
```

### 9.5 Health Checks

**Current Implementation:**

```python
@router.get("/")
async def health_check():
    return {"status": "ok", "message": "CourseGPT graph service running"}
```

**Enhanced Health Check:**

```python
@router.get("/health")
async def health_check(
    r2_service: R2StorageService = Depends(get_r2_service),
    ai_search: AISearchService = Depends(get_ai_search_service)
):
    health_status = {
        "status": "ok",
        "services": {}
    }

    # Check R2
    try:
        await r2_service.list_objects(limit=1)
        health_status["services"]["r2"] = "healthy"
    except Exception:
        health_status["services"]["r2"] = "degraded"
        health_status["status"] = "degraded"

    # Check AutoRAG
    if ai_search:
        try:
            await ai_search.list_files(per_page=1)
            health_status["services"]["ai_search"] = "healthy"
        except Exception:
            health_status["services"]["ai_search"] = "degraded"

    # Check Gemini (optional)
    # ...

    return health_status
```

---

## 10. Reproducibility Checklist

To ensure others can reproduce your setup:

### ✅ Environment Configuration

- [ ] All required environment variables documented in `.env.example`
- [ ] Python version specified (3.10+)
- [ ] Operating system requirements noted (Linux/Mac/WSL)

### ✅ Dependencies

- [ ] Complete `requirements.txt` with pinned versions
- [ ] Installation instructions in README
- [ ] Virtual environment setup documented

### ✅ Configuration Files

- [ ] Sample `.env.example` provided
- [ ] Configuration class in `config.py` documented
- [ ] All configurable parameters explained

### ✅ API Credentials

- [ ] Steps to obtain Google Gemini API key
- [ ] Steps to set up Cloudflare R2
- [ ] Steps to configure AutoRAG (optional)
- [ ] OCR service setup (optional)

### ✅ Data

- [ ] No proprietary datasets included (user-uploaded only)
- [ ] Sample data or test fixtures provided (if applicable)
- [ ] Data privacy guidelines documented

### ✅ Code

- [ ] All source files in repository
- [ ] No hardcoded paths or credentials
- [ ] Relative paths used throughout
- [ ] Type hints for clarity

### ✅ Testing

- [ ] Test suite in `tests/` directory
- [ ] Instructions to run tests: `pytest tests/`
- [ ] Example test cases provided

### ✅ Deployment

- [ ] Dockerfile (if using containers)
- [ ] Deployment platform instructions (Render, HF Spaces, etc.)
- [ ] Startup command documented: `python main.py`

### ✅ Model Artifacts

- [ ] No custom model weights (using pre-trained Gemini)
- [ ] LangGraph structure defined in code
- [ ] Agent prompts in source files

### ✅ Documentation

- [ ] README with quick start guide
- [ ] API documentation (auto-generated at `/docs`)
- [ ] Architecture diagrams included
- [ ] Troubleshooting section

### ✅ Version Control

- [ ] Git repository initialized
- [ ] `.gitignore` includes `.env`, `__pycache__`, `.venv`
- [ ] Meaningful commit messages
- [ ] Branch protection (optional, for teams)

### ✅ Reproducibility Commands

**Clone and setup:**
```bash
git clone <repo-url>
cd CourseGPT-Pro-DSAI-Lab-Group-6/Milestone-6/course_gpt_graph
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
```

**Run application:**
```bash
python main.py
# or
uvicorn main:app --reload
```

**Run tests:**
```bash
pytest tests/ -v
```

**Access API docs:**
```
http://localhost:8000/docs
```

---

## Appendix

### A. Useful Commands

**Check installed versions:**
```bash
pip freeze | grep -E "fastapi|langgraph|boto3"
```

**Test R2 connection:**
```python
from services.r2_storage import R2StorageService
from config import Settings

settings = Settings()
r2 = R2StorageService(settings)
print(await r2.list_objects(limit=5))
```

**Test Gemini API:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = llm.invoke("Hello, world!")
print(response.content)
```

### B. Troubleshooting

**Issue**: `ImportError: No module named 'langgraph'`
**Solution**: Ensure virtual environment is activated and dependencies installed

**Issue**: `CloudflareConfigurationError: Missing credentials`
**Solution**: Check `.env` file has all required `CLOUDFLARE_*` variables

**Issue**: Gemini API returns 429 (rate limit)
**Solution**: Reduce request frequency or upgrade to paid API tier

**Issue**: PDF extraction returns empty text
**Solution**: File may be scanned/image-based; configure OCR service

### C. Contact & Support

- **GitHub Issues**: [Repository Issues Page]
- **Documentation**: [Read the Docs / Wiki]
- **Email**: [Team Contact]

---

*Last Updated: 2025-01-19*
*Document Version: 1.0*
