"""Compass — Paper Search Engine Web Interface (Streamlit)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_QA_PREVIEW_LENGTH = 500
_SEARCH_PREVIEW_LENGTH = 800
_MAX_TOP_K = 20

import streamlit as st

st.set_page_config(page_title="Compass", page_icon="🧭", layout="wide")

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🧭 Compass")

    from core.config import load_config

    cfg = load_config()

    # Service status
    st.subheader("Service Status")

    from core.status import check_embedding, check_llm, check_qdrant

    checks = {
        "Embedding": lambda: check_embedding(cfg),
        "LLM": lambda: check_llm(cfg),
        "Qdrant": check_qdrant,
    }

    for name, check_fn in checks.items():
        ok, info = check_fn()
        if ok:
            st.caption(f"✅ **{name}**: {info}")
        else:
            st.caption(f"❌ **{name}**: {info}")

    st.divider()

    # Quick paper count in sidebar
    from core.ingest import list_papers
    papers = list_papers()
    st.caption(f"📚 {len(papers)} papers ingested")

# ---------- Main area ----------
_paper_param = st.query_params.get("paper", "")

if _paper_param:
    # ---------- Paper detail page ----------
    _md_path = Path(_paper_param)

    # Look up paper metadata (source_url, pdf_path)
    from core.ingest import list_papers as _list_papers
    _paper_info = next((p for p in _list_papers() if p.get("markdown_path") == _paper_param), {})

    _btn_cols = st.columns([1, 1, 1, 8])
    if _btn_cols[0].button("← Back"):
        del st.query_params["paper"]
        st.rerun()
    _source_url = _paper_info.get("source_url", "")
    if _source_url:
        _btn_cols[1].link_button("🔗 Original", _source_url)
    _pdf_copy = _paper_info.get("pdf_path", "")
    if _pdf_copy and Path(_pdf_copy).exists():
        with open(_pdf_copy, "rb") as _f:
            _btn_cols[2].download_button(
                "⬇️ PDF",
                _f.read(),
                file_name=Path(_pdf_copy).name,
                mime="application/pdf",
            )

    if _md_path.exists():
        _paper_dir = _md_path.parent
        _md_content = _md_path.read_text(encoding="utf-8")

        # Render markdown, replacing image references with st.image calls
        import re as _re
        _img_pattern = _re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        _parts = _img_pattern.split(_md_content)

        # _parts alternates: text, alt, src, text, alt, src, ...
        _i = 0
        while _i < len(_parts):
            if _i + 2 < len(_parts) and _i % 3 == 0:
                # text before image
                if _parts[_i].strip():
                    st.markdown(_parts[_i])
                # alt = _parts[_i+1], src = _parts[_i+2]
                _img_src = _parts[_i + 2]
                _img_file = _paper_dir / _img_src
                if _img_file.exists():
                    st.image(str(_img_file), caption=_parts[_i + 1] or None)
                else:
                    st.markdown(f"![{_parts[_i+1]}]({_img_src})")
                _i += 3
            else:
                if _parts[_i].strip():
                    st.markdown(_parts[_i])
                _i += 1
    else:
        st.warning("Markdown file not found. Please re-import this paper to generate it.")

else:
    # ---------- Tabs (default view) ----------
    tab_ask, tab_search, tab_papers = st.tabs(["💬 Q&A", "🔍 Search", "📚 Papers"])

    # --- Q&A Tab ---
    with tab_ask:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display message history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Enter your question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                from core.retriever import search
                from core.generator import generate

                with st.spinner("Searching..."):
                    results = search(prompt, top_k=cfg["retrieval"]["top_k"])

                if not results:
                    response = "No relevant papers found. Unable to answer."
                    st.markdown(response)
                else:
                    with st.spinner("Generating answer..."):
                        answer = generate(prompt, results)

                    sources = set()
                    for item in results:
                        sources.add(item["title"])
                    source_text = ", ".join(f"\"{s}\"" for s in sources)
                    response = f"{answer}\n\n---\n**References**: {source_text}"
                    st.markdown(response)

                    # Show retrieved passages (collapsible)
                    with st.expander("View retrieved passages"):
                        for i, item in enumerate(results, 1):
                            _md_link = item.get("markdown_path", "")
                            st.markdown(
                                f"**[{i}] {item['title']}** (relevance: {item['score']:.3f})\n\n"
                                f"{item['text'][:_QA_PREVIEW_LENGTH]}{'...' if len(item['text']) > _QA_PREVIEW_LENGTH else ''}"
                            )
                            if _md_link and Path(_md_link).exists():
                                if st.button("📄 View paper", key=f"qa_view_{i}"):
                                    st.query_params["paper"] = _md_link
                                    st.rerun()
                            st.divider()

            st.session_state.messages.append({"role": "assistant", "content": response})

    # --- Search Tab ---
    with tab_search:
        search_query = st.text_input("Search query", placeholder="Enter keywords or natural language description...")
        top_k = st.slider("Number of results", 1, _MAX_TOP_K, cfg["retrieval"]["top_k"])

        if search_query:
            from core.retriever import search

            with st.spinner("Searching..."):
                results = search(search_query, top_k=top_k)

            if not results:
                st.info("No relevant results found.")
            else:
                st.caption(f"Found {len(results)} results")
                for i, item in enumerate(results, 1):
                    with st.container(border=True):
                        col1, col2 = st.columns([5, 1])
                        col1.markdown(f"### [{i}] {item['title']}")
                        col2.metric("Relevance", f"{item['score']:.3f}")
                        st.markdown(item["text"][:_SEARCH_PREVIEW_LENGTH])
                        _src_caption = f"Source: {item['source_path']} | Chunk #{item['chunk_index']}"
                        _md_link = item.get("markdown_path", "")
                        st.caption(_src_caption)
                        if _md_link and Path(_md_link).exists():
                            if st.button("📄 View full paper", key=f"search_view_{i}"):
                                st.query_params["paper"] = _md_link
                                st.rerun()

    # --- Papers Tab ---
    with tab_papers:
        from core.ingest import list_papers as _lp, remove_paper

        _papers = _lp()

        st.subheader(f"Papers ({len(_papers)})")

        if not _papers:
            st.info("No papers ingested yet. Use the sidebar to import PDFs.")
        else:
            # Sort controls
            _sort_col1, _sort_col2 = st.columns([2, 1])
            _sort_field = _sort_col1.selectbox(
                "Sort by", ["Title", "收录时间", "Chunks count"], label_visibility="collapsed"
            )
            _sort_asc = _sort_col2.toggle("Ascending", value=True)

            _sort_key_map = {
                "Title": lambda p: p["title"].lower(),
                "收录时间": lambda p: p.get("ingested_at") or "",
                "Chunks count": lambda p: p["chunks_count"],
            }
            _papers_sorted = sorted(_papers, key=_sort_key_map[_sort_field], reverse=not _sort_asc)

            for _p in _papers_sorted:
                with st.container(border=True):
                    _c1, _c2, _c3 = st.columns([5, 2, 1])
                    with _c1:
                        st.markdown(f"**{_p['title']}**")
                        _meta_parts = [f"{_p['chunks_count']} chunks"]
                        _ia = _p.get("ingested_at", "")
                        if _ia:
                            try:
                                from datetime import datetime as _dt
                                _t = _dt.fromisoformat(_ia)
                                _meta_parts.append(f"收录于 {_t.strftime('%Y-%m-%d %H:%M')}")
                            except ValueError:
                                _meta_parts.append(f"收录于 {_ia}")
                        else:
                            _meta_parts.append("收录时间: N/A")
                        _su = _p.get("source_url", "")
                        if _su:
                            _meta_parts.append(f"[source]({_su})")
                        st.caption(" · ".join(_meta_parts))
                    with _c2:
                        if _p.get("markdown_path") and Path(_p["markdown_path"]).exists():
                            if st.button("📄 View", key=f"pt_view_{_p['path']}"):
                                st.query_params["paper"] = _p["markdown_path"]
                                st.rerun()
                    with _c3:
                        if st.button("🗑️", key=f"pt_del_{_p['path']}"):
                            remove_paper(_p["path"])
                            st.rerun()
