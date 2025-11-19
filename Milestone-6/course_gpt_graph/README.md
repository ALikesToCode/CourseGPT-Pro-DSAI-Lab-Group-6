# CourseGPT Pro - Graph Microservice

AI-powered educational assistant with specialized agents for programming, mathematics, and general learning.

This is a production-ready FastAPI microservice that uses LangGraph to orchestrate multiple fine-tuned AI agents. Each agent specializes in a different domain (code, math, general education) and is powered by custom-trained models developed in Milestones 3 and 4.

**🎓 What makes this special:**
- Custom fine-tuned models (Llama 3.1, Gemma 3, Qwen3) trained on educational datasets
- Multi-agent routing for optimal responses
- RAG integration with Cloudflare AI Search
- Cloud storage with Cloudflare R2
- Document Q&A with PDF support

**Quick overview**
- FastAPI app (`main.py`) that now exposes production-ready routes for
  Cloudflare R2 uploads and AI Search (AutoRAG) querying.
- LangGraph-driven `graph/` package orchestrating multiple CourseGPT agents.
- Services layer (`services/`) that wraps Cloudflare APIs with clean interfaces.

**✨ Key Features**

**API Endpoints:**
- 🏥 `GET /` - Health check for uptime monitoring
- 📁 `POST /files` - Upload documents to Cloudflare R2
- 📋 `GET /files` - List stored files with pagination
- 👁️ `GET /files/view/{key}` - Generate presigned URLs for secure viewing
- 🗑️ `DELETE /files/{key}` - Remove objects from storage
- 🔍 `POST /ai-search/query` - RAG queries via Cloudflare AI Search
- 📊 `GET /ai-search/files` - Check document indexing status
- 🔄 `PATCH /ai-search/sync` - Trigger AutoRAG sync pipeline
- 💬 `POST /chat` - Multi-agent chat (implemented but see [Known Issues](#known-issues))

**AI Components:**
- 🤖 Specialized agents: Router, Code, Math, General
- 🎯 Intelligent query routing
- 🧠 Fine-tuned models for educational tasks
- 📚 Document-aware responses with RAG
- 🔗 Agent handoff for complex multi-domain queries

**Fine-Tuned Models (Milestones 3 & 4):**
| Agent | Model | Parameters | Method | Status |
|-------|-------|------------|--------|--------|
| Router | Gemma 3 27B | 25.6B | LoRA | ⭐ Best |
| Router | Qwen3 32B | 31.2B | LoRA | ✅ Strong |
| Router | Llama 3.1 8B | 7.4B | LoRA | ✅ Good |
| Math | Gemma 3 27B | 25.6B | PEFT | ⭐ Best |
| Math | Qwen3 32B | 31.2B | PEFT | ✅ Strong |
| Code | Qwen 0.6B | 0.6B | QLoRA | ⭐ Production |

See [Model Architecture Documentation](../../docs/model_architecture.md) for details.

Requirements
------------
- Python 3.9+ recommended
- See `requirements.txt` for library versions (FastAPI, uvicorn, langgraph, langchain)

Environment
-----------
Create a `.env` file in `Milestone-6/course_gpt_graph/` (already excluded via
`.gitignore`). The following variables are used:

| Variable | Purpose |
| --- | --- |
| `GOOGLE_API_KEY` | Used by the LangGraph agents |
| `CLOUDFLARE_ACCESS_KEY` / `CLOUDFLARE_SECRET_ACCESS_KEY` | R2 object storage (aka S3 credentials) |
| `CLOUDFLARE_R2_BUCKET_NAME` | Target bucket for uploads |
| `CLOUDFLARE_R2_ENDPOINT` | R2 endpoint (e.g. `https://<account>.r2.cloudflarestorage.com`) |
| `CLOUDFLARE_AI_SEARCH_TOKEN` | Bearer token for AutoRAG REST API |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account id |
| `CLOUDFLARE_RAG_ID` | AutoRAG index id |
| `OCR_SERVICE_URL` | Optional endpoint for the OCR microservice used to extract text from uploads |
| `OCR_SERVICE_TOKEN` | Optional bearer token sent to the OCR service |

The loader also accepts the dash-separated versions shown in the screenshot
(`CLOUDFLARE-AI-SEARCH-TOKEN`, etc.) so you can copy/paste directly.

Do **not** commit production secrets — use a vault or environment-specific
configuration.

Install
-------
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run (development)
-----------------
Start the FastAPI app with uvicorn:

```bash
# from the project root (course_gpt_graph)
python -m uvicorn main:app --reload
# or
python main.py
```

Then visit http://127.0.0.1:8000/ — you'll get a health JSON like:

```json
{"status": "ok", "message": "CourseGPT graph service running"}
```

### Example API usage

Upload a PDF to Cloudflare R2:

```bash
curl -X POST http://127.0.0.1:8000/files \
  -F "file=@./docs/syllabus.pdf" \
  -F "prefix=course-materials"
```

Run a RAG query using Cloudflare AI Search:

```bash
curl -X POST http://127.0.0.1:8000/ai-search/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize week 1 topics", "max_num_results": 5}'
```

Project layout
--------------
Key files and folders:

- `main.py` — FastAPI application entrypoint and router wiring.
- `requirements.txt` — Python dependencies.
- `.env` — environment variables (do not commit secrets for real projects).
- `routes/` — API routers (`health`, `files`, `ai-search`).
- `graph/` — graph controller and agent code:
	- `graph/graph.py` — main graph orchestration logic.
	- `graph/agents/` — agent modules (code_agent.py, general_agent.py, math_agent.py, router_agent.py).
	- `graph/states/` — state management for graph flows.
- `tools/` — helper scripts (agent handoff examples).

Notes on the graph module
-------------------------
The `graph/` package is a compact coordinator for agent components. It is
intended to demonstrate how multiple agent types might be wired together and
driven by a central graph. The exact behaviour depends on `langgraph` and
`langchain` primitives; treat the included code as a reference implementation.

Graph diagram
-------------
The following diagram shows the high-level flow between the `router_agent`
and the various agent components and their tool sets. This image is stored
in the repository at `graph/graph.png`.

![Graph diagram](graph/graph.png)

Development tips
----------------
- Add more HTTP routes by extending the FastAPI routers (`routes/` directory).
- Keep secrets out of the repo; use environment variables or a secrets store.
- If you modify agent implementations, run targeted tests (add tests under
	`tests/` if you want CI coverage).

## Deployment

### Option 1: Local Development
```bash
python main.py
# or
uvicorn main:app --reload
```

### Option 2: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t coursegpt-graph .
docker run -p 8000:8000 --env-file .env coursegpt-graph
```

### Option 3: Hugging Face Spaces
See [deployment documentation](../../docs/technical_doc.md#7-deployment-details) for Render, HF Spaces, and other platforms.

---

## Known Issues

⚠️ **Note:** The `/chat` endpoint has been registered and is now available for use.

---

## Documentation

📚 **Complete Documentation** in `/docs`:
- [Overview](../../docs/overview.md) - Project purpose and architecture
- [Technical Documentation](../../docs/technical_doc.md) - Detailed setup and deployment
- [Model Architecture](../../docs/model_architecture.md) - Training details and results
- [User Guide](../../docs/user_guide.md) - How to use the system
- [API Documentation](../../docs/api_doc.md) - Complete API reference
- [Licenses](../../docs/licenses.md) - Legal and attribution information

---

## Testing

Run tests:
```bash
pytest tests/ -v
```

Integration tests for R2 storage:
```bash
pytest tests/test_integration_r2.py -v
```

---

## Contributing

Contributions welcome! Steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run existing tests to ensure nothing breaks
6. Commit with clear messages
7. Push and open a pull request

**Coding Standards:**
- Follow PEP 8 for Python code
- Add type hints
- Document functions with docstrings
- Keep functions focused and testable

---

## License

See [licenses.md](../../docs/licenses.md) for complete licensing information.

**Project Code:** To be determined - recommend MIT or Apache 2.0

**Third-Party Components:**
- Models: Llama 3.1 (Meta License), Gemma 3 (Google Terms), Qwen3 (Apache 2.0)
- Datasets: MathX-5M (MIT), OpenCoder (Apache 2.0)
- Libraries: See [licenses.md](../../docs/licenses.md#4-dependencies--libraries)

---

## Support

- 📖 **Documentation**: See `/docs` directory
- 🐛 **Issues**: GitHub Issues page (add your repository URL)
- 💬 **Questions**: Your contact or discussion forum

---

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [Hugging Face](https://huggingface.co/) - Model hosting and datasets
- [Cloudflare](https://cloudflare.com/) - R2 storage and AI Search

Thanks to the open-source community for making this project possible! 🙏
