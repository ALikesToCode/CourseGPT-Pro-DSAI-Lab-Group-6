"""Chat components for CourseGPT Streamlit UI.

This module prefers `streamlit-shadcn-ui` components when available
and falls back to basic Streamlit widgets otherwise. The goal is a
clean, minimal chat UI with a focused input area and rounded bubbles.
"""

import streamlit as st
from typing import List, Dict


def _build_ui_helper():
    """Return a small `ui` wrapper that exposes `button`, `input`, and
    `textarea` methods. Prefer `streamlit_elements` and `streamlit_extras`
    when available; otherwise fall back to plain Streamlit widgets.
    """
    try:
        import streamlit_elements as s_e
        import streamlit_extras as s_x

        class UI:
            def button(self, label, key=None, **kwargs):
                # For now, use Streamlit button; when `streamlit_elements` is
                # used we could render MUI buttons via the elements context.
                return st.button(label, key=key)

            def input(self, label, key=None, value="", **kwargs):
                return st.text_input(label, key=key, value=value)

            def textarea(self, label, key=None, value="", **kwargs):
                return st.text_area(label, key=key, value=value)

        return UI()
    except Exception:
        class UIFallback:
            def button(self, label, key=None, **_):
                return st.button(label, key=key)

            def input(self, label, key=None, value="", **_):
                return st.text_input(label, key=key, value=value)

            def textarea(self, label, key=None, value="", **_):
                return st.text_area(label, key=key, value=value)

        return UIFallback()


ui = _build_ui_helper()

# Optional modern container from streamlit-extras
try:
    from streamlit_extras import stylable_container
except Exception:
    stylable_container = None


def render_chat(mock_api):
    """Render a minimal, Notion-like chat interface.

    Uses `ui` wrapper for buttons/inputs. Streaming responses come from
    `mock_api.chat_response`.
    """

    chat_history: List[Dict] = st.session_state.get("chat_history", [])

    # Use a stylable container if available for nicer card visuals
    if stylable_container is not None:
        try:
            ctx = stylable_container()
        except Exception:
            ctx = None
    else:
        ctx = None

    if ctx is not None:
        with ctx:
            st.subheader("CourseGPT Chat")
            st.markdown('<div class="chat-window" id="chat-window">', unsafe_allow_html=True)
            for msg in chat_history:
                if msg.get("sender") == "user":
                    st.markdown(f"<div class='msg-user'>{msg.get('text')}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='msg-ai'>{msg.get('text')}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        container = st.container()
        with container:
            st.markdown('<div class="cg-card chat-column">', unsafe_allow_html=True)
            st.subheader("CourseGPT Chat")

            # Messages
            st.markdown('<div class="chat-window" id="chat-window">', unsafe_allow_html=True)
            for msg in chat_history:
                if msg.get("sender") == "user":
                    st.markdown(f"<div class='msg-user'>{msg.get('text')}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='msg-ai'>{msg.get('text')}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Messages
        st.markdown('<div class="chat-window" id="chat-window">', unsafe_allow_html=True)
        for msg in chat_history:
            if msg.get("sender") == "user":
                st.markdown(f"<div class='msg-user'>{msg.get('text')}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='msg-ai'>{msg.get('text')}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Input row
        with st.form("chat_form", clear_on_submit=True):
            # Avoid empty placeholders — textarea has a visible label
            user_input = ui.textarea("Message", key="chat_input", value="")
            # Use Streamlit's form submit button when inside a form
            submitted = st.form_submit_button("📤 Send", key="send_button")

        if submitted and user_input and user_input.strip():
            # Append user message
            chat_history.append({"sender": "user", "text": user_input})
            st.session_state["chat_history"] = chat_history

            ai_ph = st.empty()
            # Typing indicator
            typing = st.empty()
            typing.markdown("<div class='typing'>CourseGPT is thinking...</div>", unsafe_allow_html=True)

            response_text = ""
            for chunk in mock_api.chat_response(chat_history, st.session_state):
                response_text += chunk
                ai_ph.markdown(f"<div class='msg-ai'>{response_text}</div>", unsafe_allow_html=True)
                # autoscroll (best-effort)
                st.markdown("""
                    <script>
                    const el = document.getElementById('chat-window');
                    if (el) { el.scrollTop = el.scrollHeight; }
                    </script>
                """, unsafe_allow_html=True)

            typing.empty()
            chat_history.append({"sender": "ai", "text": response_text})
            st.session_state["chat_history"] = chat_history

        st.markdown('</div>', unsafe_allow_html=True)
