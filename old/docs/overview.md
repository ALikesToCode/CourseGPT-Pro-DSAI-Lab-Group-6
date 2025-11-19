# CourseGPT Pro - Project Overview

## Purpose

CourseGPT Pro is an AI-powered educational assistant system designed to help students with various learning needs including programming, mathematics, and general educational queries. The system leverages multiple specialized AI agents orchestrated through a state machine to provide contextually appropriate responses based on the type of query received.

### Problem Statement

Students often need different types of assistance across various subjects and domains. A single general-purpose AI may not provide optimal responses for specialized queries like coding problems or mathematical proofs. CourseGPT Pro addresses this by:

- **Intelligent Routing**: Automatically routes queries to specialized agents based on content
- **Document Context**: Allows students to upload PDFs and ask questions about their course materials
- **RAG Integration**: Uses Retrieval-Augmented Generation to provide accurate answers based on stored documents
- **Multi-Modal Support**: Handles text, code, mathematical notation (LaTeX), and document processing

### Key Objectives

1. Provide specialized educational assistance across multiple domains
2. Enable contextual learning through document upload and RAG
3. Maintain conversation history and context across multiple turns
4. Offer scalable cloud-based storage and search capabilities
5. Deliver production-ready API for integration with various frontends

---

## Architecture Summary

CourseGPT Pro is built as a microservice architecture with the following components:

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Applications                      │
│              (Streamlit, Mobile Apps, Web UI)                 │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            │ HTTP/REST API
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                     FastAPI Service Layer                      │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐   │
│  │  Health    │  │   Files    │  │    AI Search         │   │
│  │  Endpoints │  │  (R2 API)  │  │    (RAG API)         │   │
│  └────────────┘  └────────────┘  └──────────────────────┘   │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │         Graph Chat Endpoint (Multi-Agent)            │    │
│  └──────────────────────────────────────────────────────┘    │
└───────────────────────────┬───────────────────────────────────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
                ▼           ▼           ▼
    ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐
    │  R2 Storage  │  │  AI Search  │  │     Agent llm    |
    │ (Cloudflare) │  │  (AutoRAG)  │  │                  │
    └──────────────┘  └─────────────┘  └──────────────────┘
```

### Multi-Agent Graph Architecture

The core intelligence of CourseGPT Pro is powered by a LangGraph-based multi-agent system:

```
                         ┌──────────────┐
                         │    START     │
                         └──────┬───────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Router Agent        │
                    │  (Query Analysis)     │
                    └───┬───────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌────────────┐ ┌──────────────┐
│ Code Agent   │ │ Math Agent │ │General Agent │
│ (Programming)│ │(Mathematics)│ │  (General)   │
└──────┬───────┘ └─────┬──────┘ └──────┬───────┘
       │               │               │
       │      ┌────────┴────────┐      │
       │      │   Handoff Tools │      │
       └──────►  (Agent Switch) ◄──────┘
              └────────┬────────┘
                       │
                       ▼
                  ┌────────┐
                  │  END   │
                  └────────┘
```

**Agent Responsibilities:**

1. **Router Agent**: Entry point that analyzes incoming queries and routes to appropriate specialist
2. **Code Agent**: Handles programming questions, debugging, code examples
3. **Math Agent**: Solves mathematical problems with step-by-step solutions and LaTeX formatting
4. **General Agent**: Answers general educational queries and coordinates overall tasks

### Data Flow

```
User Query + Optional PDF
        │
        ▼
┌──────────────────┐
│  PDF Processing  │ (if file uploaded)
│  - pypdf extract │
│  - OCR fallback  │
└────────┬─────────┘
         │
         ▼
┌─────────────────────┐
│  RAG Context Fetch  │
│  - Query AI Search  │
│  - Retrieve chunks  │
└────────┬────────────┘
         │
         ▼
┌──────────────────────┐
│  Router Agent        │
│  - Analyze query     │
│  - Route to agent    │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Specialized Agent   │
│  - Process request   │
│  - Generate response │
└────────┬─────────────┘
         │
         ▼
    Response to User
```

---

## Deployed Components

### Current Deployment Status

The CourseGPT Pro system consists of the following components:

#### 1. **Backend API Service**
- **Platform**: FastAPI microservice
- **Location**: Can be deployed on Render, Hugging Face Spaces, or local server
- **Components**:
  - RESTful API endpoints
  - Multi-agent graph processing
  - File upload handling
  - RAG integration

#### 2. **Cloud Storage (Cloudflare R2)**
- **Purpose**: Document and file storage
- **Features**:
  - S3-compatible object storage
  - Presigned URL generation for secure access
  - Folder organization via prefixes
  - Automatic timestamp-based naming

#### 3. **AI Search (Cloudflare AutoRAG)**
- **Purpose**: Semantic search and retrieval-augmented generation
- **Features**:
  - Automatic document indexing
  - Natural language queries
  - Metadata filtering (user_id for multi-tenancy)
  - Relevance scoring

#### 4. **Custom Fine-Tuned Models**
- **Router Agent Models**:
  - Llama 3.1 8B + LoRA adapter
  - Gemma 3 27B + LoRA adapter (best performance)
  - Qwen3 32B + LoRA adapter
- **Math Agent Models**:
  - Gemma 3 4B + QLoRA adapter (Milestone-3)
  - Gemma 3 27B + PEFT adapter (Milestone-4, best quality)
  - Qwen3 32B + PEFT adapter (Milestone-4)
- **Code Agent Model**:
  - Qwen 0.6B + QLoRA adapter
- **Deployment**: Models published to Hugging Face with ZeroGPU support
- **Production**: Fine-tuned models optimized for educational tasks

#### 5. **Optional: OCR Service**
- **Purpose**: Extract text from scanned PDFs
- **Integration**: External microservice (configurable endpoint)
- **Fallback**: Local pypdf extraction

### Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│         Production Environment                   │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │      FastAPI Application                 │  │
│  │  - Uvicorn ASGI Server                   │  │
│  │  - Health monitoring                     │  │
│  │  - Environment-based configuration       │  │
│  └──────────────┬───────────────────────────┘  │
│                 │                               │
│                 │ HTTPS                         │
│                 │                               │
└─────────────────┼───────────────────────────────┘
                  │
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌─────────┐  ┌──────────┐  ┌──────────┐
│   R2    │  │   RAG    │  │   AI     │
│ Storage │  │  Search  │  │  Models  │
└─────────┘  └──────────┘  └──────────┘
```

### Environment Requirements

The system requires the following environment variables to be configured:

**Required:**
- `GOOGLE_API_KEY`: LLM API access (if needed)
- `CLOUDFLARE_ACCESS_KEY`: R2 access credentials
- `CLOUDFLARE_SECRET_ACCESS_KEY`: R2 secret key
- `CLOUDFLARE_R2_BUCKET_NAME`: Target bucket name
- `CLOUDFLARE_R2_ENDPOINT`: R2 endpoint URL

**Optional (for RAG):**
- `CLOUDFLARE_AI_SEARCH_TOKEN`: AutoRAG API token
- `CLOUDFLARE_ACCOUNT_ID`: Cloudflare account identifier
- `CLOUDFLARE_RAG_ID`: RAG index identifier

**Optional (for OCR):**
- `OCR_SERVICE_URL`: External OCR service endpoint
- `OCR_SERVICE_TOKEN`: OCR service authentication token

---

## Key Features

### 1. Intelligent Agent Routing
- Automatically detects query type (code, math, general)
- Routes to specialized agent for optimal responses
- Supports agent handoff for complex multi-domain queries

### 2. Document-Aware Responses
- Upload PDFs and ask questions about content
- Automatic text extraction with OCR fallback
- Combines uploaded context with RAG-retrieved information

### 3. Retrieval-Augmented Generation (RAG)
- Semantic search across all indexed documents
- User-specific filtering for privacy
- Automatic document synchronization from R2

### 4. Conversation Memory
- Thread-based conversation tracking
- Multi-turn dialogue support
- User-specific conversation isolation

### 5. Scalable Storage
- Cloud-based object storage via Cloudflare R2
- Presigned URLs for secure file access
- Automatic file organization with timestamps

### 6. Production-Ready API
- Auto-generated OpenAPI documentation
- Comprehensive error handling
- Health check endpoints
- CORS support for web frontends

---

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API Framework** | FastAPI | High-performance async web framework |
| **AI Orchestration** | LangGraph | Multi-agent state machine |
| **LLM Integration** | LangChain | LLM abstraction and tooling |
| **Base Models** | Llama 3.1 8B, Gemma 3 4B/27B, Qwen3 32B, Qwen 0.6B | Foundation models for agents |
| **Fine-Tuning** | LoRA/QLoRA/PEFT | Parameter-efficient adaptation |
| **Training Platform** | Google Vertex AI + Local GPU | Model training infrastructure |
| **Model Hosting** | Hugging Face (ZeroGPU) | Fine-tuned model deployment |
| **Object Storage** | Cloudflare R2 | Document and file storage |
| **Vector Search** | Cloudflare AI Search | RAG and semantic search |
| **PDF Processing** | pypdf | Document text extraction |
| **HTTP Client** | httpx | Async API communication |
| **Testing** | pytest | Unit and integration tests |
| **Server** | Uvicorn | ASGI application server |

---

## Performance Characteristics

### Latency
- **Simple queries**: < 2 seconds (without RAG)
- **RAG-enhanced queries**: 2-5 seconds (depends on document count)
- **With PDF upload**: +1-3 seconds (depends on file size)

### Scalability
- **Concurrent requests**: Limited by Uvicorn worker count
- **Storage**: Unlimited via Cloudflare R2
- **Search**: Scales with Cloudflare infrastructure
- **Models**: Limited by GPU availability and model hosting

### Resource Requirements
- **Memory**: ~200MB base + ~50MB per concurrent request
- **CPU**: Minimal (I/O bound workload)
- **Storage**: Depends on user uploads (stored in R2, not locally)

---

## Use Cases

1. **Programming Tutoring**: Students can ask coding questions and receive explained solutions
2. **Math Homework Help**: Step-by-step mathematical solutions with proper notation
3. **Document Q&A**: Upload lecture notes or textbooks and ask specific questions
4. **General Learning**: Explanations of concepts across various subjects
5. **Course Planning**: Help organizing study schedules and project planning

---

## Future Enhancements

Potential areas for expansion:

- **Persistent Checkpointing**: Store conversation state in database for multi-session continuity
- **Model Performance Optimization**: Enhanced quantization and caching for faster inference
- **Additional Agents**: Science, language learning, test preparation specialists
- **Image Analysis**: Support for diagram understanding and visual learning materials
- **Voice Interface**: Speech-to-text and text-to-speech integration
- **Collaborative Learning**: Group study sessions with shared context
- **Analytics Dashboard**: Track learning progress and topic coverage
- **Continuous Fine-Tuning**: Implement feedback loops to improve models from user interactions

---

## Links and Resources

- **API Documentation**: Available at `/docs` when service is running
- **GitHub Repository**: [Add your repository URL]
- **Demo Video**: [Add YouTube/demo link if available]
- **Support**: [Add contact information]

---

*Last Updated: 2025-01-19*
