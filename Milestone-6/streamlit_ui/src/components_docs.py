"""Document manager components for CourseGPT Streamlit UI.

Provides an upload area, search and tag filters, a grid of uploaded
documents, and modal viewers. Documents are stored in
`st.session_state['documents']` and manipulated through the provided
`mock_api` interface.
"""

import streamlit as st


def render_documents(mock_api):
    """Render the document manager UI."""

    # >>> FIX: Ensure EVERYTHING renders inside the Streamlit iframe
    docs_wrapper = st.container()

    with docs_wrapper:
        st.subheader("Document Manager")

        # Upload area
        with st.expander("📁 Upload Document", expanded=True):
            # Wrap uploader in a div so we can target it with CSS (.doc-upload)
            st.markdown('<div class="doc-upload">', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Drag & drop a document, or click to browse",
                type=["txt", "md", "pdf"],
                key="file_uploader",
            )
            title = st.text_input("Title", key="doc_title")
            tags = st.text_input("Tags (comma separated)", key="doc_tags")

            upload_btn = st.button("⬆️ Upload")
            st.markdown('</div>', unsafe_allow_html=True)

            if upload_btn and uploaded is not None:
                progress = st.progress(0)

                def cb(p):
                    progress.progress(min(100, int(p)))

                content = uploaded.read()
                doc = mock_api.upload_document(
                    st.session_state,
                    title,
                    tags,
                    content,
                    progress_callback=cb,
                )
                st.success(f"Uploaded: {doc['title']}")

        # Filters
        qcol1, qcol2 = st.columns([4, 1])
        query = qcol1.text_input("Search documents", key="doc_search")
        tag_filter = qcol2.text_input("Filter tag", key="doc_tag_filter")

        docs = mock_api.list_documents(st.session_state)
        if query:
            docs = [
                d
                for d in docs
                if query.lower() in (d.get("title", "") + d.get("snippet", "")).lower()
            ]
        if tag_filter:
            docs = [
                d
                for d in docs
                if tag_filter.lower() in " ".join(d.get("tags", [])).lower()
            ]

        # >>> FIX: Wrap the entire grid inside the iframe container
        grid_container = st.container()
        with grid_container:
            st.markdown('<div class="doc-grid">', unsafe_allow_html=True)

            for d in docs:
                st.markdown('<div class="doc-card">', unsafe_allow_html=True)

                # Only show title and action buttons (View / Delete)
                st.markdown(f"### {d['title']}")

                # Right-aligned action buttons
                left_col, right_col = st.columns([4, 1])
                with right_col:
                    if st.button("👁️ View", key=f"view_{d['id']}"):
                        with st.modal(f"View — {d['title']}"):
                            st.markdown(f"### {d['title']}")
                            st.markdown(f"**Tags:** {', '.join(d.get('tags', []))}")
                            st.markdown("---")
                            st.text_area("Content", value=d.get("content", ""), height=360)

                    if st.button("🗑️ Delete", key=f"del_{d['id']}"):
                        with st.modal("Confirm Delete"):
                            st.write(
                                f"Delete document '{d['title']}'? This action cannot be undone."
                            )
                            if st.button("Confirm Delete", key=f"confirm_del_{d['id']}"):
                                mock_api.delete_document(
                                    st.session_state, d["id"]
                                )
                                st.success("Deleted")

                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
