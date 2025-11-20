"""CourseGPT Streamlit UI entrypoint.

This module wires together the UI components (navigation, chat,
documents, settings), initializes `st.session_state` with safe defaults,
and applies global styles from `ui_styles.apply_styles`.

Run with:
    streamlit run app/src/main.py
"""

import streamlit as st

from ui_styles import apply_styles
from api_client import api_client
import navigation
import components_chat
import components_docs
import components_settings


# -------------------------------------------------------------
# Initialize application state
# -------------------------------------------------------------
def init_state():
    ss = st.session_state

    ss.setdefault("selected_page", "Chat")

    # Start with an empty chat history; render a transient welcome
    # banner in the chat UI instead of seeding session state to avoid
    # duplicate welcome messages across reloads.
    ss.setdefault("chat_history", [])

    ss.setdefault("documents", [])
    ss.setdefault("theme_mode", "dark")
    ss.setdefault("dev_mode", False)
    ss.setdefault("mock_latency", 0.45)
    
    # Ensure user_id and thread_id exist for API calls
    if "user_id" not in ss:
        import uuid
        ss["user_id"] = str(uuid.uuid4())
    if "thread_id" not in ss:
        import uuid
        ss["thread_id"] = str(uuid.uuid4())


# -------------------------------------------------------------
# Main application entry point
# -------------------------------------------------------------
def main():
    st.set_page_config(page_title="CourseGPT", layout="wide")

    # Initialize state once
    init_state()
    ss = st.session_state

    # Initialize API client (no-op here as it's a singleton, but good for consistency)
    # mock_api.mock_api.latency = ss.get("mock_latency", 0.45)
    # mock_api.mock_api.init_documents(ss)

    # ---------------------------------------------------------
    # Render navigation first
    # (because it updates theme_mode and selected_page)
    # ---------------------------------------------------------
    navigation.render_topbar()

    # ---------------------------------------------------------
    # Apply styles AFTER navigation so theme changes take effect
    # ---------------------------------------------------------
    apply_styles(ss.get("theme_mode", "light"))

    # ---------------------------------------------------------
    # Page Routing
    # ---------------------------------------------------------
    page = ss.get("selected_page", "Chat")

    if page == "Chat":
        col_chat, col_docs = st.columns([1.7, 1])

        with col_chat:
            components_chat.render_chat(api_client)

        with col_docs:
            components_docs.render_documents(api_client, variant="sidebar")

    elif page == "Documents":
        components_docs.render_documents(api_client, variant="full")

    elif page == "Settings":
        components_settings.render_settings(api_client)


# -------------------------------------------------------------
# Entry hook
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
