"""Settings page UI for CourseGPT Streamlit application.

Uses shadcn-ui controls when available with Streamlit fallbacks. The
controls are arranged in a two-column layout for alignment.
"""

import streamlit as st


def _get_ui():
    try:
        import streamlit_shadcn_components as shadcn

        class UI:
            def radio(self, *a, **k):
                return shadcn.ui.radio(*a, **k)

            def checkbox(self, *a, **k):
                return shadcn.ui.checkbox(*a, **k)

            def slider(self, *a, **k):
                return shadcn.ui.slider(*a, **k)

            def select(self, *a, **k):
                return shadcn.ui.select(*a, **k)

        return UI()
    except Exception:
        class UIFallback:
            def radio(self, label, options, index=0, key=None):
                return st.radio(label, options=options, index=index, key=key)

            def checkbox(self, label, value=False, key=None):
                return st.checkbox(label, value=value, key=key)

            def slider(self, label, min_value=0, max_value=100, value=None, step=None, key=None):
                return st.slider(label, min_value=min_value, max_value=max_value, value=value, step=step, key=key)

            def select(self, label, options, index=0, key=None):
                return st.selectbox(label, options=options, index=index, key=key)

        return UIFallback()


ui = _get_ui()


def _notify(message: str):
    if hasattr(st, "toast"):
        st.toast(message)
    else:
        st.success(message)


def _render_theme_mode_selector():
    """Provide an accessible light/dark selector with contextual help."""
    current_mode = st.session_state.get("theme_mode", "light")
    options = {
        "Light (default · bright, high contrast)": "light",
        "Dark (low-light · reduced glare)": "dark",
    }
    inverse = {v: k for k, v in options.items()}
    default_label = inverse.get(current_mode, "Light (default · bright, high contrast)")
    selection = st.radio(
        "Theme mode",
        list(options.keys()),
        index=list(options.keys()).index(default_label),
        help="Choose the overall appearance for CourseGPT.",
    )
    st.session_state["theme_mode"] = options[selection]
    st.caption("Switch between light and dark to match your environment.")


def render_settings(mock_api):
    """Render the Settings page controls using a 2-column grid."""

    container = st.container()
    with container:
        st.subheader("Control Center")
        st.caption("Calibrate CourseGPT's experience, data sources, and experimentation options.")

        tab_general, tab_integrations, tab_research = st.tabs(["General", "Integrations", "Research Lab"])

        with tab_general:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Appearance**")
                _render_theme_mode_selector()
                st.markdown("**Conversation UX**")
                ui.checkbox("Auto-summarize after each session", value=st.session_state.get("auto_summary", True), key="auto_summary")
                ui.checkbox("Surface citations automatically", value=st.session_state.get("citations", True), key="citations")
            with col2:
                st.markdown("**Model & Performance**")
                ui.select(
                    "Model preset",
                    options=["coursegpt-small", "coursegpt-base", "coursegpt-pro"],
                    index=1,
                    key="model_select",
                )
                ui.slider(
                    "Mock network latency (seconds)",
                    min_value=0.05,
                    max_value=1.5,
                    value=float(st.session_state.get("mock_latency", 0.45)),
                    step=0.05,
                    key="mock_latency",
                )
                ui.checkbox("Developer Mode", value=st.session_state.get("dev_mode", False), key="dev_mode")

        with tab_integrations:
            st.markdown("**Cloudflare**")
            st.text_input(
                "Account ID",
                key="cf_account",
                placeholder="e.g., 1234567890abcdef1234567890abcdef",
                help="Find this under Cloudflare → Overview → Account ID.",
            )
            st.text_input(
                "R2 Bucket",
                key="cf_bucket",
                placeholder="e.g., coursegpt-uploads",
                help="Name of the Cloudflare R2 bucket storing uploaded files.",
            )
            st.text_input(
                "AI Search Dataset ID",
                key="cf_dataset",
                placeholder="e.g., rag-xyz123",
                help="Dataset identifier from Cloudflare AI Search.",
            )
            st.markdown("**Optional Vector Stores**")
            st.text_input(
                "Pinecone Index",
                key="pinecone_idx",
                placeholder="e.g., coursegpt-prod",
                help="Optional: specify a Pinecone index if you plan to sync data outside of Cloudflare.",
            )
            st.text_input(
                "Weaviate URL",
                key="weaviate_url",
                placeholder="https://your-instance.weaviate.network",
                help="Optional: add a Weaviate endpoint for hybrid retrieval.",
            )
            st.caption("These credentials are for display only — the mock UI never transmits them.")

        with tab_research:
            st.markdown("Design experiments for upcoming cohorts.")
            focus = st.multiselect(
                "Focus areas",
                options=["Math proofs", "Code reviews", "Research papers", "Labs", "Slides"],
                default=st.session_state.get("focus_areas", ["Labs", "Slides"]),
                key="focus_areas",
            )
            st.caption("Pick the study artifacts you want CourseGPT to emphasize.")
            weekly_hours = st.slider(
                "Target study hours / week",
                min_value=2,
                max_value=20,
                value=int(st.session_state.get("weekly_hours", 8)),
                key="weekly_hours",
            )
            st.caption("CourseGPT uses this target when proposing study schedules.")
            st.text_area("Experiment brief", key="experiment_brief", height=120)
            if st.button("Save research preset", key="save_research"):
                _notify("Preset captured — this is a front-end only demo.")

        st.markdown("---")
        if st.button("Reset settings to defaults", key="reset_settings"):
            keys = [
                "theme_mode",
                "dev_mode",
                "auto_summary",
                "citations",
                "model_select",
                "mock_latency",
                "focus_areas",
                "weekly_hours",
                "experiment_brief",
                "cf_account",
                "cf_bucket",
                "cf_dataset",
                "pinecone_idx",
                "weaviate_url",
            ]
            for k in keys:
                st.session_state.pop(k, None)
            _notify("Settings reset")
