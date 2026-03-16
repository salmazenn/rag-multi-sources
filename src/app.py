"""
RAG PDF Local — Streamlit UI
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from rag import load_and_split_pdf, build_vectorstore, load_vectorstore, build_qa_chain, ask

st.set_page_config(page_title="RAG PDF Cloud", page_icon="📄", layout="centered")
st.title("📄 RAG PDF Cloud")
st.caption("Posez des questions sur vos PDF — propulsé par Groq + HuggingFace")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

with st.sidebar:
    st.header("📂 Charger un PDF")
    st.warning("⚠️ Version de démonstration — PDF limité à 50MB")
    uploaded_file = st.file_uploader("Sélectionne un fichier PDF (max 50MB)", type="pdf")
    if uploaded_file and not st.session_state.pdf_loaded:
        with st.spinner("Ingestion en cours..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            chunks = load_and_split_pdf(tmp_path)
            vectorstore = build_vectorstore(chunks)
            st.session_state.chain = build_qa_chain(vectorstore)
            st.session_state.pdf_loaded = True
            st.session_state.messages = []
            os.unlink(tmp_path)
        st.success(f"✅ PDF chargé : {uploaded_file.name}")
    st.divider()
    st.markdown("**Modèle :** `llama3` via Ollama")
    st.markdown("**Modèle :** `llama-3.3-70b` via Groq ⚡")
    st.markdown("**Embeddings :** HuggingFace local")
    st.markdown("**Vectorstore :** ChromaDB")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

if prompt := st.chat_input("Posez votre question..."):
    if not st.session_state.chain:
        st.warning("⚠️ Chargez d'abord un PDF dans la barre latérale.")
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
