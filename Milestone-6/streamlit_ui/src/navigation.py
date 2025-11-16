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


def render_topbar():
    """Render a compact, single-row top navigation bar.

    Updates `st.session_state['selected_page']` and `st.session_state['theme_mode']`.
    """

    ss = st.session_state
    ss.setdefault("selected_page", "Chat")
    ss.setdefault("theme_mode", "light")

    # Wrap the topbar in a lightweight HTML container for styling
    st.markdown('<div class="cg-topbar"><div class="cg-topbar-inner">', unsafe_allow_html=True)

    # Single-level layout: logo | tab1 | tab2 | tab3 | controls
    col_logo, col_tab1, col_tab2, col_tab3, col_controls = st.columns([1, 1, 1, 1, 1])

    # Logo
    with col_logo:
        st.markdown("<div style='font-weight:600;font-size:20px'>CourseGPT</div>", unsafe_allow_html=True)

    # Tabs (each in its own column to avoid nested column artifacts)
    with col_tab1:
        if st.button("Chat", key="nav_chat"):
            ss["selected_page"] = "Chat"
    with col_tab2:
        if st.button("Documents", key="nav_docs"):
            ss["selected_page"] = "Documents"
    with col_tab3:
        if st.button("Settings", key="nav_settings"):
            ss["selected_page"] = "Settings"

    # Controls: theme toggle + profile
    with col_controls:
        current = ss.get("theme_mode", "light")
        # Theme toggle: prefer the toggle-switch widget if available
        if st_toggle_switch is not None:
            try:
                toggled = st_toggle_switch(label="", key="toggle_theme", default=(current == "dark"))
                ss["theme_mode"] = "dark" if toggled else "light"
            except Exception:
                # fallback to simple button
                icon = "🌙" if current == "light" else "☀️"
                if st.button(icon, key="toggle_theme"):
                    ss["theme_mode"] = "dark" if current == "light" else "light"
        else:
            icon = "🌙" if current == "light" else "☀️"
            if st.button(icon, key="toggle_theme"):
                ss["theme_mode"] = "dark" if current == "light" else "light"

        # Profile avatar: prefer `streamlit-avatar` if available
        if st_avatar is not None:
            try:
                st_avatar(name="Professor Demo", size=34)
                st.markdown("<div style='font-size:13px;color:var(--muted);display:inline-block;margin-left:8px'>Professor · Demo</div>", unsafe_allow_html=True)
            except Exception:
                st.markdown("<div style='display:flex;align-items:center;gap:8px'><div style='width:28px;height:28px;border-radius:50%;background:#E6EEF8;display:inline-flex;align-items:center;justify-content:center;font-size:14px'>👩‍🏫</div><div style='font-size:13px;color:var(--muted)'>Professor · Demo</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='display:flex;align-items:center;gap:8px'><div style='width:28px;height:28px;border-radius:50%;background:#E6EEF8;display:inline-flex;align-items:center;justify-content:center;font-size:14px'>👩‍🏫</div><div style='font-size:13px;color:var(--muted)'>Professor · Demo</div></div>", unsafe_allow_html=True)
    # close wrapper
    st.markdown('</div></div>', unsafe_allow_html=True)

