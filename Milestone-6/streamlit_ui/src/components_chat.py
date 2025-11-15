"""Chat components for CourseGPT Streamlit UI.

This module renders the chat pane, handles user input, and displays
AI responses using the provided `mock_api` interface. Messages are kept
in `st.session_state['chat_history']` as a list of dicts with keys
`sender` and `text`.
"""

import streamlit as st
from typing import List, Dict


def render_chat(mock_api):
    """Render the chat UI and stream responses."""

    # >>> FIX: Wrap everything in a Streamlit container so it stays inside the iframe.
    chat_wrapper = st.container()

    with chat_wrapper:
        # Replace raw div wrapper with markdown INSIDE container (iframe-safe)
        st.markdown('<div class="cg-card chat-column">', unsafe_allow_html=True)
        st.subheader('CourseGPT Chat')

        chat_history: List[Dict] = st.session_state.get('chat_history', [])

        # Messages area
        with st.container():
            st.markdown('<div class="chat-window" id="chat-window">', unsafe_allow_html=True)

            for msg in chat_history:
                if msg["sender"] == "user":
                    st.markdown(
                        f"<div class='msg-user'>{msg['text']}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='msg-ai'>{msg['text']}</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("</div>", unsafe_allow_html=True)

            # Auto-scroll
            st.markdown(
                """
                <script>
                const el = document.getElementById('chat-window');
                if (el) {
                    el.scrollTop = el.scrollHeight;
                }
                </script>
                """,
                unsafe_allow_html=True,
            )

        # Input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Message",
                placeholder="Ask CourseGPT a question and press Send",
                key="chat_input",
                height=90,
            )
            submitted = st.form_submit_button("📤 Send")

        # Handle new message
        if submitted and user_input.strip():
            chat_history.append({"sender": "user", "text": user_input})
            st.session_state["chat_history"] = chat_history

            typing = st.empty()
            ai_placeholder = st.empty()

            typing.markdown(
                """
                <div class="typing">
                    CourseGPT is thinking...
                    <span class="dots"><span></span><span></span><span></span></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Streaming response
            response_text = ""
            for chunk in mock_api.chat_response(chat_history, st.session_state):
                response_text += chunk
                ai_placeholder.markdown(
                    f"<div class='msg-ai'>{response_text}</div>",
                    unsafe_allow_html=True,
                )

                # Keep autoscroll working
                st.markdown(
                    """
                    <script>
                    const el = document.getElementById('chat-window');
                    if (el) {
                        el.scrollTop = el.scrollHeight;
                    }
                    </script>
                    """,
                    unsafe_allow_html=True,
                )

            typing.empty()
            chat_history.append({"sender": "ai", "text": response_text})
            st.session_state["chat_history"] = chat_history

        st.markdown("</div>", unsafe_allow_html=True)
