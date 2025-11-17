# course_gpt_graph

FastAPI + LangGraph example microservice for the CourseGPT graph component.

This repository contains a minimal FastAPI application that integrates with
LangGraph/LangChain-style agents and a small graph module that coordinates
agents (code, math, router, general). It's intended as a lightweight
component you can run locally for experimentation or to use as a starting
point for building agent-driven services.

**Quick overview**
- Provides a small FastAPI app (`main.py`) exposing a health endpoint.
- Contains a `graph/` package with a graph controller and several agents.
- Uses `langgraph` and `langchain` for agent/graph functionality (abstracted).

**Features**
- FastAPI HTTP server with a single health endpoint (`GET /`).
- Pluggable agent modules under `graph/agents` (code, general, math, router).
- Sample `tools/` scripts to illustrate agent handoff patterns.

Requirements
------------
- Python 3.9+ recommended
- See `requirements.txt` for library versions (FastAPI, uvicorn, langgraph, langchain)

Environment
-----------
- Create a `.env` file (example included). The app will read environment
	variables from the environment (or a `.env` loader if you add one).
- Current repository includes a `.env` file with a `GOOGLE_API_KEY` variable.
	Do not commit production secrets — replace with your own keys or use a
	secret manager.

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

Then visit http://127.0.0.1:8000/ — you'll get a small health JSON like:

```json
{"status": "ok", "message": "CourseGPT running"}
```

Project layout
--------------
Key files and folders:

- `main.py` — FastAPI application entrypoint.
- `requirements.txt` — Python dependencies.
- `.env` — environment variables (do not commit secrets for real projects).
- `routes/health.py` — health route definition.
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
- If you plan to extend the HTTP API, add routes under `routes/` and include
	them in `main.py` (or an APIRouter).
- Keep secrets out of the repo; use environment variables or a secrets store.
- If you modify agent implementations, run targeted tests (add tests under
	`tests/` if you want CI coverage).

Deployment
----------
- For production, build a container image and run behind an ASGI server such
	as Uvicorn/Gunicorn. Example Dockerfile is not included but is straightforward.

Contributing
------------
Contributions are welcome. Suggested steps:

1. Fork the repo and create a feature branch.
2. Add tests for any non-trivial behavior.
3. Open a pull request describing the change.

License
-------
This project uses no explicit license file in the repo. If you need a
license, add an appropriate `LICENSE` file (MIT is a common choice for small
examples).

Contact / Next steps
--------------------
If you'd like, I can:
- Add a Dockerfile and `docker-compose` for local containerized runs.
- Add a small integration test that starts the FastAPI app and checks `/`.
- Expand the README with example agent calls and sample outputs.

Tell me which of the above you'd like next.
