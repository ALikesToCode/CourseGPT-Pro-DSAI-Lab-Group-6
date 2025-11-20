"""Clean, minimal Notion-like top navigation for CourseGPT.

Layout:
- Left: `CourseGPT` logo text
- Center: three tab buttons (Chat, Documents, Settings)
- Right: Theme toggle button + small profile avatar + "Professor · Demo" label

This implementation intentionally avoids any empty inputs or placeholder
widgets that can render blank full-width bars. It uses only `st.columns`,
`st.button`, and compact `st.markdown` for the logo/profile.
"""

import streamlit as st

# Optional modern components (graceful fallback if APIs differ)
try:
    from streamlit_toggle_switch import st_toggle_switch
except Exception:
    st_toggle_switch = None

try:
    from streamlit_avatar import avatar as st_avatar
except Exception:
    st_avatar = None


def _notify(message: str):
    if hasattr(st, "toast"):
        st.toast(message)
    else:
        st.success(message)


def render_topbar():
    """Render a compact, single-row top navigation bar.

    Updates `st.session_state['selected_page']` and `st.session_state['theme_mode']`.
    """

    ss = st.session_state
    ss.setdefault("selected_page", "Chat")
    ss.setdefault("theme_mode", "light")

    # Wrap the topbar in a lightweight HTML container for styling
    st.markdown('<div class="cg-topbar"><div class="cg-topbar-inner">', unsafe_allow_html=True)

    col_logo, col_tabs, col_actions = st.columns([1.3, 3.2, 1.8])

    with col_logo:
        st.markdown(
            "<div class='cg-logo-stack'><div class='cg-logo'>CourseGPT</div>"
            "<div class='cg-logo-sub'>Curriculum Intelligence Suite</div></div>",
            unsafe_allow_html=True,
        )

    pages = ["Chat", "Documents", "Settings"]
    with col_tabs:
        tab_cols = st.columns(len(pages))
        for idx, label in enumerate(pages):
            active = ss.get("selected_page") == label
            button_kwargs = {"key": f"nav_{label.lower()}", "width": "stretch"}
            if tab_cols[idx].button(label, **button_kwargs):
                ss["selected_page"] = label
            tab_cols[idx].markdown(
                f"<div class='cg-tab-indicator {'active' if active else ''}'></div>",
                unsafe_allow_html=True,
            )

    # Controls: theme toggle + quick actions + profile
    with col_actions:
        current = ss.get("theme_mode", "light")
        controls_cols = st.columns([0.9, 1.1, 1])

        with controls_cols[0]:
            if st.button("＋ New Chat", key="new_chat", width="stretch"):
                ss["chat_history"] = []
                _notify("Started a fresh chat thread.")

        with controls_cols[1]:
            st.caption("Theme")
            icon = "🌙" if current == "light" else "☀️"
            if st_toggle_switch is not None:
                try:
                    toggled = st_toggle_switch(
                        label="Toggle theme",
                        key="toggle_theme",
                        default=(current == "dark"),
                    )
                    ss["theme_mode"] = "dark" if toggled else "light"
                except Exception:
                    if st.button(icon, key="toggle_theme_icon"):
                        ss["theme_mode"] = "dark" if current == "light" else "light"
            else:
                if st.button(icon, key="toggle_theme_icon"):
                    ss["theme_mode"] = "dark" if current == "light" else "light"

        with controls_cols[2]:
            st.markdown("<div class='cg-profile'>", unsafe_allow_html=True)
            if st_avatar is not None:
                try:
                    st_avatar(name="Professor Demo", size=30)
                except Exception:
                    st.markdown(
                        "<div class='cg-avatar'>👩‍🏫</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("<div class='cg-avatar'>👩‍🏫</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='cg-profile-label'>Professor · Demo</div></div>",
                unsafe_allow_html=True,
            )
    # close wrapper
    st.markdown('</div></div>', unsafe_allow_html=True)
