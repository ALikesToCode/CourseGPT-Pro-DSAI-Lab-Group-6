"""Navigation bar for the CourseGPT Streamlit UI.

This version FIXES Streamlit's event handling by ensuring all widgets
(buttons) are inside a REAL Streamlit container rather than raw HTML.
The topbar HTML wrapper remains for styling, but Streamlit widgets are
not nested inside the raw HTML tag directly.
"""

import streamlit as st


def render_topbar():
    """Render the sticky top navigation bar with working events."""

    # Outer container ensures widgets behave correctly
    topbar = st.container()

    with topbar:
        # HTML wrapper controls styling only (safe because widgets are inside a container)
        st.markdown('<div class="cg-topbar">', unsafe_allow_html=True)

        col_left, col_center, col_right = st.columns([1, 3, 1])

        # ----------------------------------------------------
        # Logo
        # ----------------------------------------------------
        with col_left:
            st.markdown('<div class="cg-logo">CourseGPT</div>', unsafe_allow_html=True)

        # ----------------------------------------------------
        # Navigation Tabs + Theme Toggle
        # ----------------------------------------------------
        with col_center:
            t1, t2, t3, t4 = st.columns([1, 1, 1, 1])

            # Use icons for faster recognition and consistent labels
            chat_label = "💬 Chat"
            docs_label = "📄 Documents"
            settings_label = "⚙️ Settings"

            if t1.button(chat_label, key="nav_chat"):
                st.session_state["selected_page"] = "Chat"

            if t2.button(docs_label, key="nav_docs"):
                st.session_state["selected_page"] = "Documents"

            if t3.button(settings_label, key="nav_settings"):
                st.session_state["selected_page"] = "Settings"

            # THEME TOGGLE — show moon or sun depending on current mode
            current = st.session_state.get("theme_mode", "light")
            theme_label = "🌙 Toggle Theme" if current == "light" else "☀️ Toggle Theme"
            if t4.button(theme_label, key="nav_theme"):
                st.session_state["theme_mode"] = "dark" if current == "light" else "light"

        # ----------------------------------------------------
        # Avatar section
        # ----------------------------------------------------
        with col_right:
            img_col, name_col = st.columns([1, 3])

            with img_col:
                st.image("https://picsum.photos/40", width=36)

            with name_col:
                st.markdown(
                    '<div style="text-align:right;font-size:13px;color:var(--muted-text)">'
                    'Professor • Demo'
                    '</div>',
                    unsafe_allow_html=True,
                )

        # Close topbar HTML wrapper
        st.markdown("</div>", unsafe_allow_html=True)
