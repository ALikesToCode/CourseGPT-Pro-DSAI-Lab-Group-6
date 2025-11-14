# course_gpt_graph

Minimal FastAPI application for the CourseGPT graph component.

Run locally:

```bash
# Install deps (in a venv)
pip install -r requirements.txt

# Run using uvicorn (recommended)
python -m uvicorn main:app --reload

# Or run the module directly (uses uvicorn.run)
python main.py
```

The app exposes a single root endpoint at `/` that returns a small health JSON.
