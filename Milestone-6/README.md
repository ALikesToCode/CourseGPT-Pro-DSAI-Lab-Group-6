# 🚀 Milestone 6 — Project Overview

Welcome to Milestone 6. This page gives a quick, scannable guide to the main components, key scripts, and how to run the demo Streamlit UI. It's designed to be a developer-friendly landing page.

[![Docs](https://img.shields.io/badge/docs-Milestone--6-blue)](#) [![Streamlit](https://img.shields.io/badge/streamlit-demo-orange)](#)

---

## Table of Contents

- [Quick Start](#quick-start)
- [At a Glance](#at-a-glance)
- [Folder Guide (short)](#folder-guide-short)
- [Running the Streamlit UI](#running-the-streamlit-ui)
- [Notes & Next Steps](#notes--next-steps)

---

## Quick Start

Get the Streamlit UI running quickly (recommended smoke test):

```powershell
cd "d:\DSAI-Project\CourseGPT-Pro-DSAI-Lab-Group-6\Milestone-6\streamlit_ui"
pip install -r requirements.txt
streamlit run src/main.py
```

Open http://localhost:8501 in your browser.

---

## At a Glance

- Purpose: development and evaluation platform for CourseGPT agents and router logic.
- Primary focus areas in Milestone 6: orchestration (graph), agent handlers, router evaluation & a lightweight frontend for demos.

---

## Folder Guide (short)

Below are the most important folders and files with one-line descriptions.

- `.streamlit/`
  - `config.toml` — Streamlit runtime options and theme tweaks.

- `agents/`
  - `base.py` — common agent base classes and helpers used across handlers.

- `code-agent/`
  - `handler.py` — code agent entrypoint; see local README for wiring details.

- `course_gpt_graph/`
  - `main.py` — runner for graph orchestration.
  - `graph/` — routing and agent-state logic used to connect agents and tools.
  - `routes/` — HTTP routes (health, handoff endpoints).

- `general-agent/`, `math-agent/`, `ocr-service/`
  - Each exposes a `handler.py` and a short README describing environment variables and usage patterns.

- `router-agent/`
  - Evaluation tooling (`schema_score.py`, `generate_router_benchmark.py`, `router_benchmark_runner.py`).
  - `docs/` — design & technical docs.
  - `hf_space/` — packaged Hugging Face Space for demos and tests.

- `streamlit_ui/`
  - A self-contained frontend (Chat, Documents, Settings) that can run with a mock in-memory API: `src/mock_api.py`.

---

## Running the Streamlit UI (detailed)

From the project root run:

```powershell
cd "d:\DSAI-Project\CourseGPT-Pro-DSAI-Lab-Group-6\Milestone-6\streamlit_ui"
pip install -r requirements.txt
streamlit run src/main.py
```

Tips:
- If optional UI packages are missing, the app will fall back to plain Streamlit widgets (safe to run without extras).
- For production/demo deploys, consider setting Streamlit's server options in `.streamlit/config.toml`.

---