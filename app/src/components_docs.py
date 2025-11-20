"""Document manager components for CourseGPT Streamlit UI.

Prefers `streamlit-shadcn-ui` for buttons/badges but falls back to
Streamlit widgets when unavailable. Layout is kept minimal and uses
small CSS cards for a Notion-like look.
"""

from collections import Counter
from typing import List, Dict

import streamlit as st


def _get_ui():
    try:
        import streamlit_elements as s_e
        import streamlit_extras as s_x

        class UI:
            def button(self, label, key=None, variant=None, **kwargs):
                # Placeholder: render as Streamlit button for now.
                return st.button(label, key=key, **kwargs)

            def badge(self, label, **kwargs):
                # streamlit_elements or extras can render nicer badges; keep
                # a simple markdown fallback for now.
                return st.markdown(f"<span class='tag-pill'>{label}</span>", unsafe_allow_html=True)

        return UI()
    except Exception:
        class UIFallback:
            def button(self, label, key=None, **kwargs):
                return st.button(label, key=key, **kwargs)

            def badge(self, label, **_):
                return st.markdown(f"<span class='tag-pill'>{label}</span>", unsafe_allow_html=True)

        return UIFallback()


ui = _get_ui()


def _notify(message: str, level="success"):
    if hasattr(st, "toast"):
        st.toast(message)
    else:
        getattr(st, level, st.success)(message)

# Optional modern components
try:
    from streamlit_card import card as st_card
except Exception:
    st_card = None

try:
    from st_annotated_text import annotated_text
except Exception:
    annotated_text = None


def _render_empty_state(message: str):
    st.info(message)


def _render_preview(api_client, preview_id: str):
    # API doesn't have a direct "get document content" endpoint for previewing text content
    # except via RAG context or if we download it.
    # For now, we can try to get a view URL if it's a file.
    url = api_client.get_file_url(preview_id)
    if url:
        st.markdown(f"**Preview:** [Open File]({url})")
    else:
        st.info("Preview not available for this document.")


def _render_insights(docs: List[Dict]):
    if not docs:
        _render_empty_state("Upload at least one document to unlock insights.")
        return
    total_tokens = sum(len(d.get("content", "")) for d in docs)
    recent = docs[0].get("uploaded_at", "—") if docs else "—"
    top_tags = Counter(tag for d in docs for tag in d.get("tags", []))
    top_tag_label = ", ".join([f"{tag} ({count})" for tag, count in top_tags.most_common(3)]) or "—"

    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(docs))
    c2.metric("Characters indexed", f"{total_tokens:,}")
    c3.metric("Top topics", top_tag_label)

    st.dataframe(
        [
            {
                "Title": d.get("title", ""),
                "Tags": ", ".join(d.get("tags", [])),
                "Size (KB)": d.get("size_kb", 0),
                "Uploaded": d.get("uploaded_at", "")[:16],
            }
            for d in docs
        ],
        hide_index=True,
        use_container_width=True,
    )


def _render_card_actions(doc_id: str, api_client):
    cols = st.columns(2)
    with cols[0]:
        if st.button("Preview document", key=f"view_{doc_id}", width="stretch", help="Open a read-only preview in this workspace"):
            st.session_state["doc_preview"] = doc_id
            _notify("Preview loaded", level="info")
    with cols[1]:
        if st.button("Delete document", key=f"del_{doc_id}", width="stretch", help="Remove this document from the current session"):
            api_client.delete_file(doc_id)
            _notify("Document deleted", level="warning")


def _render_card(title: str, snippet: str, tags: List[str], doc_id: str, api_client):
    if st_card is not None:
        try:
            with st_card(title=title, text=snippet):
                if annotated_text is not None and tags:
                    annotated_text(*[(t, "", "#6B7280") for t in tags])
                _render_card_actions(doc_id, api_client)
            return
        except Exception:
            pass

    st.markdown('<div class="doc-card">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    if snippet:
        st.markdown(
            f"<div style='color:var(--text-muted-light);font-size:13px'>{snippet}</div>",
            unsafe_allow_html=True,
        )
    if tags:
        st.markdown(
            "<div style='margin-top:8px'>"
            + "".join([f"<span class='tag-pill'>{t}</span>" for t in tags])
            + "</div>",
            unsafe_allow_html=True,
        )
    _render_card_actions(doc_id, api_client)
    st.markdown("</div>", unsafe_allow_html=True)


def render_documents(api_client, variant: str = "full"):
    """
    Render the document UI.

    Args:
        mock_api: The mock backend service.
        variant: "full" renders the complete workspace (tabs, upload, analytics).
                 "sidebar" renders a compact summary suited for the chat column.
    """

    if variant == "sidebar":
        _render_documents_sidebar(api_client)
        return

    _render_documents_workspace(api_client)


def _render_documents_sidebar(api_client):
    try:
        docs_resp = api_client.list_files()
        docs = docs_resp.get("files", [])
    except Exception as exc:
        docs = []
        st.warning("Documents unavailable (storage not configured). Quick upload is still available.")
    container = st.container()
    with container:
        st.subheader("Documents")
        st.caption("Quick upload & recent context")

        uploaded = st.file_uploader(
            "Quick upload",
            type=["txt", "md", "pdf"],
            label_visibility="visible",
            key="sidebar_file_uploader",
            help="Drop a document here to add it.",
        )
        if uploaded is not None and st.button(
            "Upload to workspace",
            key="sidebar_upload_btn",
            use_container_width=True,
            help="Adds the selected file to the Documents workspace.",
        ):
            content = uploaded.read()
            api_client.upload_file(
                content,
                uploaded.name,
            )
            _notify("Document uploaded — open workspace to review.")

        st.divider()
        if docs:
            st.caption("Recently added")
            for d in docs[:3]:
                st.markdown(
                    f"<div class='doc-card'><strong>{d.get('title', 'Untitled')}</strong>"
                    f"<div class='doc-snippet'>{d.get('snippet', '')[:110]}...</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No recent uploads yet. Use the uploader above to add your first document.")

        if st.button("Go to document workspace", width="stretch", help="Open the full-page document manager"):
            st.session_state["selected_page"] = "Documents"


def _render_documents_workspace(api_client):
    container = st.container()
    try:
        docs_resp = api_client.list_files()
        docs = docs_resp.get("files", [])
    except Exception as exc:
        docs = []
        st.warning("Document storage not ready. Check Cloudflare R2 settings to enable uploads.")

    with container:
        st.markdown('<div class="cg-card doc-panel">', unsafe_allow_html=True)
        st.subheader("Document Workspace")
        st.caption("Ground CourseGPT with curated lecture notes, readings, and transcripts.")

        tab_library, tab_insights = st.tabs(["Library", "Insights"])

        with tab_library:
            _render_upload_section(api_client)

            query, tag_filter = _render_search_filter()

            filtered = docs
            if query:
                filtered = [d for d in filtered if query.lower() in (d.get("title", "") + d.get("snippet", "")).lower()]
            if tag_filter:
                filtered = [
                    d for d in filtered if tag_filter.lower() in " ".join(d.get("tags", [])).lower()
                ]

            if not filtered:
                empty_message = "No documents match your filters." if docs else "No documents yet. Upload lecture notes, assignments, or transcripts to get started."
                _render_empty_state(empty_message)
            else:
                for d in filtered:
                    _render_card(d.get("key", "Untitled"), "", [], d["key"], api_client)

            preview_id = st.session_state.get("doc_preview")
            if preview_id:
                _render_preview(api_client, preview_id)

        with tab_insights:
            _render_insights(docs)

        st.markdown("</div>", unsafe_allow_html=True)


def _render_upload_section(api_client):
    if st.session_state.pop("reset_doc_inputs", False):
        st.session_state["doc_title"] = ""
        st.session_state["doc_tags"] = ""

    st.markdown("#### Upload a document")
    st.caption("Accepted formats: PDF, Markdown, and plain text. Files stay within this local session.")

    st.markdown('<div class="doc-upload">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Click or drag a document into this panel",
        type=["txt", "md", "pdf"],
        key="file_uploader",
        label_visibility="visible",
        help="Use the button below or drop a file anywhere within the bordered panel."
    )
    title = st.text_input(
        "Document title",
        key="doc_title",
        placeholder="e.g., Week 4 Lab Notes",
        help="CourseGPT uses the title to reference this file in later chats.",
    )
    tags = st.text_input(
        "Tags",
        key="doc_tags",
        placeholder="Separate tags with commas, e.g., calculus, exam prep",
        help="Optional: add quick filters or topical keywords.",
    )

    upload_disabled = uploaded is None
    submitted = ui.button(
        "Upload document",
        key="upload_btn",
        width="stretch",
        disabled=upload_disabled,
        help="Select a file first to activate this button.",
    )

    if submitted:
        content = uploaded.read()
        try:
            doc = api_client.upload_file(
                content,
                uploaded.name,
            )
            file_info = doc.get("file", {})
            st.session_state["doc_preview"] = file_info.get("key")
            st.session_state["reset_doc_inputs"] = True
            _notify("Document uploaded!")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_search_filter():
    st.markdown("#### Search & filter")
    st.caption("Narrow the library by keyword or tag — filters update instantly.")
    qcol1, qcol2 = st.columns([3, 2])
    query = qcol1.text_input(
        "Search documents",
        key="doc_search",
        placeholder="Try “vector calculus” or “Week 2 quiz”.",
    )
    tag_filter = qcol2.text_input(
        "Filter by tag",
        key="doc_tag_filter",
        placeholder="e.g., exams",
    )
    return query, tag_filter
