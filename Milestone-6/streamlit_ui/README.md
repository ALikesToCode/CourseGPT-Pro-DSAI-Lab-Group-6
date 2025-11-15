# CourseGPT — Streamlit UI

Frontend-only Streamlit UI for CourseGPT (Milestone 6).

Run:

```powershell
streamlit run "Milestone 6/streamlit_ui/src/main.py"
```

This folder contains a modular, theme-driven Streamlit UI using an academic theme. Components are split into `navigation`, `components_chat`, `components_docs`, `components_settings`, `mock_api`, and `ui_styles`.

Design notes:
- All CSS is injected from `ui_styles.py`.
- Mock API simulates latency and streaming responses.
- State is managed in `st.session_state`.
