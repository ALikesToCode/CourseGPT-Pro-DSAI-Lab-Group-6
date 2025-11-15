"""Settings page UI for CourseGPT Streamlit application."""

import streamlit as st


def render_settings(mock_api):
    """Render the Settings page controls."""

    # >>> FIX: Wrap everything in a Streamlit container (iframe-safe)
    settings_wrapper = st.container()

    with settings_wrapper:
        st.subheader("Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Appearance**")
            theme = st.radio(
                "Theme Mode",
                options=["light", "dark"],
                index=0
                if st.session_state.get("theme_mode", "light") == "light"
                else 1,
            )
            st.session_state["theme_mode"] = theme

            st.markdown("**Developer**")
            dev = st.checkbox(
                "Developer Mode",
                value=st.session_state.get("dev_mode", False),
            )
            st.session_state["dev_mode"] = dev

        with col2:
            st.markdown("**Model & Performance**")

            st.selectbox(
                "Model",
                options=[
                    "coursegpt-small",
                    "coursegpt-base",
                    "coursegpt-pro",
                ],
                index=1,
            )

            latency = st.slider(
                "Mock network latency (seconds)",
                0.05,
                1.5,
                value=float(st.session_state.get("mock_latency", 0.45)),
                step=0.05,
            )
            st.session_state["mock_latency"] = latency

        st.markdown("---")
        st.markdown(
            "Developer toggles and model knobs are frontend-only and used to simulate different behaviors."
        )
