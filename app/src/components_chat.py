"""Chat components for CourseGPT Streamlit UI.

This module prefers `streamlit-shadcn-ui` components when available
and falls back to basic Streamlit widgets otherwise. The goal is a
clean, minimal chat UI with a focused input area and rounded bubbles.
"""

import json
import streamlit as st
from typing import List, Dict, Any


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
                return st.button(label, key=key, **kwargs)

            def input(self, label, key=None, value="", **kwargs):
                return st.text_input(label, key=key, value=value, **kwargs)

            def textarea(self, label, key=None, value="", **kwargs):
                return st.text_area(label, key=key, value=value, **kwargs)

        return UI()
    except Exception:
        class UIFallback:
            def button(self, label, key=None, **kwargs):
                return st.button(label, key=key, **kwargs)

            def input(self, label, key=None, value="", **kwargs):
                return st.text_input(label, key=key, value=value, **kwargs)

            def textarea(self, label, key=None, value="", **kwargs):
                return st.text_area(label, key=key, value=value, **kwargs)

        return UIFallback()


ui = _build_ui_helper()


def _notify(message: str, fallback="info"):
    if hasattr(st, "toast"):
        st.toast(message)
    else:
        getattr(st, fallback, st.info)(message)

# Optional modern container from streamlit-extras
try:
    from streamlit_extras import stylable_container
except Exception:
    stylable_container = None


def _render_metrics(chat_history: List[Dict]):
    user_turns = len([m for m in chat_history if m.get("sender") == "user"])
    ai_turns = len([m for m in chat_history if m.get("sender") == "ai"])
    streak = max(0, user_turns - ai_turns)
    docs_attached = len(st.session_state.get("documents", []))

    m1, m2, m3 = st.columns(3)
    m1.metric("Learner turns", user_turns, delta=f"+{streak}" if streak else None)
    m2.metric("AI responses", ai_turns)
    m3.metric("Linked docs", docs_attached)


def _render_quick_prompts():
    st.caption("Quick prompts · Click to auto-fill the composer.")
    prompts = [
        "Summarize today's lecture in 3 bullets",
        "Generate 5 quiz questions",
        "Build a study roadmap for Week 4",
        "Turn notes into flashcards",
    ]
    cols = st.columns(len(prompts))
    for idx, prompt in enumerate(prompts):
        if cols[idx].button(prompt, key=f"prompt_chip_{idx}"):
            st.session_state["chat_prefill"] = prompt
            _notify("Prompt added to composer")


def _render_messages(chat_history: List[Dict]):
    st.markdown('<div class="chat-window" id="chat-window">', unsafe_allow_html=True)
    if not chat_history:
        st.markdown(
            "<div class='msg-ai'>👋 Welcome! Ask me about any lecture or upload a document to ground the analysis.</div>",
            unsafe_allow_html=True,
        )
    for msg in chat_history:
        cls = "msg-user" if msg.get("sender") == "user" else "msg-ai"
        st.markdown(f"<div class='{cls}'>{msg.get('text')}</div>", unsafe_allow_html=True)
        if msg.get("router_debug") and msg.get("sender") == "ai":
            rd = msg["router_debug"]
            tool = rd.get("tool", "")
            route = ""
            handoff = ""
            if isinstance(rd.get("content"), dict):
                content = rd["content"]
                handoff_target = content.get("handoff", "unknown agent")
                task_summary = content.get("task_summary", "")
                route_rationale = content.get("route_rationale", "")
                
                st.markdown(
                    f"<div class='msg-ai' style='background:#0f1724;opacity:0.92;border-left: 3px solid #3b82f6; padding: 10px; margin-bottom: 10px;'>"
                    f"<strong>🔄 Handoff to {handoff_target.replace('_', ' ').title()}</strong><br/>"
                    f"<span style='color:#94a3b8;font-size:0.9em'>{task_summary}</span><br/>"
                    f"<div style='margin-top:5px;font-size:0.85em;color:#cbd5e1'><em>Rationale: {route_rationale}</em></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='msg-ai' style='background:#0f1724;opacity:0.92'>"
                    f"<strong>Router handoff</strong><br/>Tool: <code>{tool}</code>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with st.expander("Router payload (debug)", expanded=False):
                st.code(json.dumps(rd, indent=2), language="json")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_timeline(chat_history: List[Dict]):
    recent = chat_history[-6:]
    if not recent:
        return
    with st.expander("Conversation timeline (latest 6 turns)", expanded=False):
        for msg in recent[::-1]:
            label = "You" if msg.get("sender") == "user" else "CourseGPT"
            st.markdown(f"**{label}:** {msg.get('text')}", unsafe_allow_html=False)


def render_chat(mock_api):
    """Render the refreshed chat workspace with metrics and quick prompts."""

    chat_history: List[Dict] = st.session_state.get("chat_history", [])
    st.session_state.setdefault("chat_input", "")

    container = st.container()
    with container:
        st.markdown('<div class="cg-card chat-column" id="chat-card">', unsafe_allow_html=True)
        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.subheader("CourseGPT Workspace")
            st.caption("Context-aware mentor tuned for the CourseGPT curriculum.")
        with header_cols[1]:
            if st.button("Clear conversation", key="clear_chat", width="stretch", help="Remove every message in this chat thread"):
                st.session_state["chat_history"] = []
                chat_history = []
                _notify("Conversation cleared.")

        _render_metrics(chat_history)

        st.markdown("---")
        _render_quick_prompts()
        st.markdown("---")

        # Scrollable chat window
        # Create a placeholder for messages to allow re-rendering on submit
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            _render_messages(chat_history)

        # Input row pinned to bottom via CSS/JS scroll helper
        prefill = st.session_state.pop("chat_prefill", None)
        if prefill:
            st.session_state["chat_input"] = prefill

        # Sticky composer container
        st.markdown('<div class="composer-sticky">', unsafe_allow_html=True)
        st.markdown('<div id="composer-wrapper">', unsafe_allow_html=True)
        st.markdown("#### Message CourseGPT")
        with st.form("chat_form", clear_on_submit=True):
            user_input = ui.textarea(
                "Message CourseGPT",
                key="chat_input",
                placeholder="Ask a question, request a study plan, or paste a snippet to analyze...",
                help="Messages stay local to this Streamlit session.",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send message", key="send_button", width="stretch", help="Submit your prompt to CourseGPT")
        st.markdown("</div></div>", unsafe_allow_html=True)

        if submitted and user_input and user_input.strip():
            chat_history.append({"sender": "user", "text": user_input})
            st.session_state["chat_history"] = chat_history

            # Re-render messages to show the new user message immediately
            with chat_placeholder.container():
                _render_messages(chat_history)
                
                ai_ph = st.empty()
                typing = st.empty()
                typing.markdown("<div class='typing'>CourseGPT is synthesizing context…</div>", unsafe_allow_html=True)

                response_text = ""
                router_debug: Any = None
                for chunk in mock_api.chat(
                    prompt=user_input,
                    thread_id=st.session_state["thread_id"],
                    user_id=st.session_state["user_id"]
                ):
                    if isinstance(chunk, dict):
                        msg_type = chunk.get("type")
                        if msg_type == "token":
                            response_text += chunk.get("content", "")
                        elif msg_type == "handoff":
                            router_debug = chunk.get("content")
                        elif msg_type == "tool_use":
                            tool_name = chunk.get("tool")
                            tool_input = chunk.get("input")
                            ai_ph.markdown(
                                f"<div class='msg-ai' style='background:#1e293b; border: 1px solid #334155; padding: 8px; margin-bottom: 5px; font-size: 0.85em; color: #94a3b8;'>"
                                f"🛠️ <strong>Using Tool:</strong> <code>{tool_name}</code><br/>"
                                f"<span style='font-family:monospace'>{str(tool_input)[:100]}...</span>"
                                f"</div>", 
                                unsafe_allow_html=True
                            )
                        elif msg_type == "error":
                            st.error(chunk.get("content"))
                    else:
                        response_text += str(chunk)
                    
                    # Only update the main text bubble if we have text
                    if response_text:
                        ai_ph.markdown(f"<div class='msg-ai'>{response_text}</div>", unsafe_allow_html=True)
                    st.markdown(
                        """
                        <script>
                        const el = window.parent.document.getElementById('chat-window');
                        if (el) { el.scrollTop = el.scrollHeight; }
                        </script>
                        """,
                        unsafe_allow_html=True,
                    )

                typing.empty()
                chat_history.append({"sender": "ai", "text": response_text, "router_debug": router_debug})
                st.session_state["chat_history"] = chat_history

        _render_timeline(chat_history)
        st.markdown("</div>", unsafe_allow_html=True)
