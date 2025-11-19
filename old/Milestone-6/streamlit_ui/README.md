# CourseGPT — Streamlit UI

Frontend-only Streamlit UI for CourseGPT (Milestone 6).

Run:

```powershell
streamlit run "Milestone 6/streamlit_ui/src/main.py"
```

This folder contains a modular, theme-driven Streamlit UI using an academic theme. Components are split into `navigation`, `components_chat`, `components_docs`, `components_settings`, `mock_api`, and `ui_styles`.

### Highlights (updated)
- Redesigned top navigation with quick actions (`＋ New Chat`) and live theme switching.
- Chat workspace now surfaces learning metrics, quick prompts, timeline history, and graceful autoscroll.
- Document workspace includes upload progress, preview panel, analytics tab, and deletion/preview actions.
- Settings center ships tabs for General, Integrations, and Research Lab presets with reset + toast feedback.
- `ui_styles.py` injects a Notion-inspired palette with light/dark parity and chip-like controls.

Design notes:
- All CSS is injected from `ui_styles.py`.
- Mock API simulates latency and streaming responses.
- State is managed in `st.session_state`.
