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


def render_settings(mock_api):
    """Render the Settings page controls using a 2-column grid."""

    container = st.container()
    with container:
        st.subheader("Settings")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Appearance**")
            current_mode = st.session_state.get("theme_mode", "light")
            # Use the same key as the session_state so Streamlit manages the value
            mode = ui.radio("Theme Mode", options=["light", "dark"], index=0 if current_mode == "light" else 1, key="theme_mode")

            st.markdown("**Developer**")
            # Checkbox uses key `dev_mode` and will automatically populate session_state
            dev = ui.checkbox("Developer Mode", value=st.session_state.get("dev_mode", False), key="dev_mode")

        with col2:
            st.markdown("**Model & Performance**")
            ui.select("Model", options=["coursegpt-small", "coursegpt-base", "coursegpt-pro"], index=1, key="model_select")

            # Slider uses key `mock_latency` and updates session_state automatically
            latency = ui.slider("Mock network latency (seconds)", min_value=0.05, max_value=1.5, value=float(st.session_state.get("mock_latency", 0.45)), step=0.05, key="mock_latency")

        st.markdown("---")
        st.markdown("Developer toggles and model knobs are frontend-only and used to simulate different behaviors.")
