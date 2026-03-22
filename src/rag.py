"""
RAG Multi-Sources — PDF + Web + Markdown
Powered by Groq + LangChain + ChromaDB
"""

import os
import requests
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"

# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """
Tu es un assistant expert. Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.
Indique toujours la source de ta réponse.

Contexte :
{context}

Question : {question}

Réponse :"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_pdf(path: str, display_name: str = None) -> list:
    print(f"📄 Chargement PDF : {path}")
    docs = PyPDFLoader(path).load()
    # Remplace le chemin temporaire par le vrai nom
    if display_name:
        for doc in docs:
            doc.metadata["source"] = display_name
    return docs

def load_markdown(path: str) -> list:
    print(f"📝 Chargement Markdown : {path}")
    return UnstructuredMarkdownLoader(path).load()

def load_url(url: str) -> list:
    print(f"🌐 Chargement URL : {url}")
    return WebBaseLoader(url).load()

def detect_and_load(source: str, display_name: str = None) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        return load_url(source)
    elif source.endswith(".pdf"):
        return load_pdf(source, display_name or source)
    elif source.endswith(".md"):
        return load_markdown(source)
    else:
        raise ValueError(f"Source non supportée : {source}")

# ── Pipeline ──────────────────────────────────────────────────────────────────

def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  {len(chunks)} chunks créés")
    return chunks

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vectorstore(chunks: list):
    print(f"🔍 Début ingestion de {len(chunks)} chunks...")
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    embeddings = get_embeddings()
    collection = client.get_or_create_collection("langchain")
    print(f"Collection avant ajout: {collection.count()} docs")
    
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]
    
    print(f"Embedding {len(texts)} textes...")
    embed_vectors = embeddings.embed_documents(texts)
    print(f"Embeddings créés: {len(embed_vectors)}")
    
    collection.add(
        documents=texts,
        embeddings=embed_vectors,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Collection après ajout: {collection.count()} docs")
    
    return Chroma(
        client=client,
        collection_name="langchain",
        embedding_function=embeddings
    )

def load_vectorstore() -> Chroma:
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return Chroma(
        client=client,
        collection_name="langchain",
        embedding_function=get_embeddings()
    )

def build_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=True
    )

def ask(chain: RetrievalQA, question: str) -> dict:
    # Test direct du retriever
    docs = chain.retriever.get_relevant_documents(question)
    print(f"DEBUG retriever: {len(docs)} docs trouvés")
    for doc in docs:
        print(f"DEBUG metadata: {doc.metadata}")
    
    result = chain.invoke({"query": question})
    sources = list({
        doc.metadata.get("source", "Source inconnue")
        for doc in result.get("source_documents", [])
    })
    return {
        "question": question,
        "answer": result["result"].strip(),
        "sources": sources if sources else ["Sources non disponibles"]
    }

# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest(sources: list, display_names: dict = None):
    all_documents = []
    for source in sources:
        try:
            name = display_names.get(source, source) if display_names else source
            docs = detect_and_load(source, name)
            all_documents.extend(docs)
            print(f"✅ {len(docs)} pages chargées depuis : {name}")
        except Exception as e:
            print(f"❌ Erreur sur {source} : {e}")

    if not all_documents:
        raise ValueError("Aucun document chargé !")

    chunks = split_documents(all_documents)
    vectorstore = build_vectorstore(chunks)
    print(f"\n🎉 Ingestion terminée ! {len(all_documents)} documents, {len(chunks)} chunks.")
    return vectorstore  # ← retourne le vectorstore !

# ── CLI ───────────────────────────────────────────────────────────────────────

def chat():
    print("💬 Chargement du vectorstore...")
    vectorstore = load_vectorstore()
    chain = build_qa_chain(vectorstore)
    print("\n🤖 RAG Multi-Sources — prêt ! (tape 'exit' pour quitter)\n")
    while True:
        question = input("Vous : ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        result = ask(chain, question)
        print(f"\nAssistant : {result['answer']}")
        print(f"📎 Sources : {', '.join(result['sources'])}\n")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/rag.py ingest <source1> <source2> ...")
        print("  python src/rag.py chat")
        print("\nExemples:")
        print("  python src/rag.py ingest doc.pdf https://example.com notes.md")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest" and len(sys.argv) >= 3:
        ingest(sys.argv[2:])
    elif command == "chat":
        chat()
    else:
        print("Commande invalide.")
        sys.exit(1)