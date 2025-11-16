"""Document manager components for CourseGPT Streamlit UI.

Prefers `streamlit-shadcn-ui` for buttons/badges but falls back to
Streamlit widgets when unavailable. Layout is kept minimal and uses
small CSS cards for a Notion-like look.
"""

import streamlit as st


def _get_ui():
    try:
        import streamlit_elements as s_e
        import streamlit_extras as s_x

        class UI:
            def button(self, label, key=None, variant=None, **kwargs):
                # Placeholder: render as Streamlit button for now.
                return st.button(label, key=key)

            def badge(self, label, **kwargs):
                # streamlit_elements or extras can render nicer badges; keep
                # a simple markdown fallback for now.
                return st.markdown(f"<span class='tag-pill'>{label}</span>", unsafe_allow_html=True)

        return UI()
    except Exception:
        class UIFallback:
            def button(self, label, key=None, **_):
                return st.button(label, key=key)

            def badge(self, label, **_):
                return st.markdown(f"<span class='tag-pill'>{label}</span>", unsafe_allow_html=True)

        return UIFallback()


ui = _get_ui()

# Optional modern components
try:
    from streamlit_card import card as st_card
except Exception:
    st_card = None

try:
    from st_annotated_text import annotated_text
except Exception:
    annotated_text = None


def render_documents(mock_api):
    """Render the document manager UI (title, snippet, tags, actions)."""

    container = st.container()
    with container:
        st.subheader("Document Manager")

        # Upload area: keep Streamlit uploader but use minimal layout
        with st.expander("📁 Upload Document", expanded=True):
            st.markdown('<div class="doc-upload">', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Drag & drop a document, or click to browse",
                type=["txt", "md", "pdf"],
                key="file_uploader",
            )
            title = st.text_input("Title", key="doc_title")
            tags = st.text_input("Tags (comma separated)", key="doc_tags")
            if ui.button("⬆️ Upload", key="upload_btn") and uploaded is not None:
                content = uploaded.read()
                mock_api.upload_document(st.session_state, title, tags, content)
                st.success("Uploaded")
            st.markdown('</div>', unsafe_allow_html=True)

        # Search controls: moved into an expander so they don't render
        # by default in the Chat view and won't create layout artifacts.
        query = ""
        tag_filter = ""
        with st.expander("Search & Filter", expanded=False):
            qcol1, qcol2 = st.columns([4, 1])
            query = qcol1.text_input("Search documents", key="doc_search")
            tag_filter = qcol2.text_input("Filter tag", key="doc_tag_filter")

        docs = mock_api.list_documents(st.session_state)
        if query:
            docs = [d for d in docs if query.lower() in (d.get("title", "") + d.get("snippet", "")).lower()]
        if tag_filter:
            docs = [d for d in docs if tag_filter.lower() in " ".join(d.get("tags", [])).lower()]

        # Grid of cards
        for d in docs:
            title = d.get('title', 'Untitled')
            snippet = d.get('snippet', '')
            tags = d.get('tags', [])

            if st_card is not None:
                try:
                    # streamlit_card provides a nicer preview card when available
                    with st_card(title=title, text=snippet):
                        if annotated_text is not None and tags:
                            annotated_text(*[ (t, "", "#888") for t in tags ])
                        # Actions on a single horizontal row
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.button("👁️ View", key=f"view_{d['id']}")
                        with c2:
                            st.button("🗑️ Delete", key=f"del_{d['id']}")
                except Exception:
                    # fallback to simple rendering
                    st.markdown('<div class="doc-card">', unsafe_allow_html=True)
                    st.markdown(f"### {title}")
                    if snippet:
                        st.markdown(f"<div style='color:var(--text-muted-light);font-size:13px'>{snippet}</div>", unsafe_allow_html=True)
                    if tags:
                        st.markdown("<div style='margin-top:8px'>" + ", ".join([f"<span class='tag-pill'>{t}</span>" for t in tags]) + "</div>", unsafe_allow_html=True)
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.button("👁️ View", key=f"view_{d['id']}")
                    with c2:
                        st.button("🗑️ Delete", key=f"del_{d['id']}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="doc-card">', unsafe_allow_html=True)
                st.markdown(f"### {title}")
                if snippet:
                    st.markdown(f"<div style='color:var(--text-muted-light);font-size:13px'>{snippet}</div>", unsafe_allow_html=True)
                if tags:
                    st.markdown("<div style='margin-top:8px'>" + ", ".join([f"<span class='tag-pill'>{t}</span>" for t in tags]) + "</div>", unsafe_allow_html=True)
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.button("👁️ View", key=f"view_{d['id']}")
                with c2:
                    st.button("🗑️ Delete", key=f"del_{d['id']}")
                st.markdown('</div>', unsafe_allow_html=True)
