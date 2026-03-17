"""
RAG Multi-Sources — Streamlit UI
PDF + Web + Markdown
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from rag import ingest, load_vectorstore, build_qa_chain, ask

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Multi-Sources",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 RAG Multi-Sources")
st.caption("Posez des questions sur vos PDFs, pages web et fichiers Markdown — propulsé par Groq ⚡")

# ── Session state ─────────────────────────────────────────────────────────────

if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_loaded" not in st.session_state:
    st.session_state.sources_loaded = []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Ajouter des sources")

    # PDF Upload
    st.subheader("📄 PDF")
    uploaded_files = st.file_uploader(
        "Sélectionne un ou plusieurs PDFs (max 50MB)",
        type="pdf",
        accept_multiple_files=True
    )

    # URL
    st.subheader("🌐 Page Web")
    url_input = st.text_input("Colle une URL")

    # Markdown
    st.subheader("📝 Markdown")
    uploaded_md = st.file_uploader(
        "Sélectionne un fichier Markdown",
        type=["md"],
    )

    # Bouton d'ingestion
    if st.button("🚀 Ingérer toutes les sources", type="primary"):
        sources = []
        tmp_files = []
        source_names = {}  # mapping tmp_path -> vrai nom

        # PDFs
        for uploaded_file in (uploaded_files or []):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                sources.append(tmp.name)
                tmp_files.append(tmp.name)
                source_names[tmp.name] = uploaded_file.name  # vrai nom !

        # URL
        if url_input.strip():
            sources.append(url_input.strip())
            source_names[url_input.strip()] = url_input.strip()

        # Markdown
        if uploaded_md:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
                tmp.write(uploaded_md.read())
                sources.append(tmp.name)
                tmp_files.append(tmp.name)
                source_names[tmp.name] = uploaded_md.name  # vrai nom !

        if not sources:
            st.warning("⚠️ Ajoute au moins une source !")
        else:
            with st.spinner("Ingestion en cours..."):
                try:
                    vectorstore = ingest(sources, source_names)
                    st.session_state.chain = build_qa_chain(vectorstore)
                    st.session_state.sources_loaded = list(source_names.values())
                    st.session_state.messages = []
                    for f in tmp_files:
                        os.unlink(f)
                    st.success(f"✅ {len(sources)} source(s) ingérée(s) !")
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
        # Sources chargées
        if st.session_state.sources_loaded:
            st.divider()
            st.markdown("**Sources actives :**")
            for s in st.session_state.sources_loaded:
                st.markdown(f"- `{Path(s).name}`")

        st.divider()
        st.markdown("**Modèle :** `llama-3.3-70b` via Groq ⚡")
        st.markdown("**Embeddings :** HuggingFace")
        st.markdown("**Vectorstore :** ChromaDB")

# ── Chat ──────────────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

if prompt := st.chat_input("Posez votre question..."):
    if not st.session_state.chain:
        st.warning("⚠️ Ingère d'abord au moins une source dans la barre latérale.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                result = ask(st.session_state.chain, prompt)
            st.markdown(result["answer"])
            with st.expander("📎 Sources"):
                for s in result["sources"]:
                    st.markdown(f"- {s}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })