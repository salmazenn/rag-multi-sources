"""
RAG PDF Local — Powered by Ollama + ChromaDB
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

#from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "50"))

PROMPT_TEMPLATE = """
Tu es un assistant expert. Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse :"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def load_and_split_pdf(pdf_path: str) -> list:
    print(f"📄 Chargement de : {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  {len(chunks)} chunks créés")
    return chunks

def build_vectorstore(chunks: list) -> Chroma:
    print(f"🔍 Création des embeddings avec Ollama ({OLLAMA_MODEL})...")
    # embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    vectorstore.persist()
    print(f"💾 Vectorstore sauvegardé dans : {CHROMA_PERSIST_DIR}")
    return vectorstore

def load_vectorstore() -> Chroma:
    # embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)

def build_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    # llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

def ask(chain: RetrievalQA, question: str) -> dict:
    result = chain({"query": question})
    sources = list({
        doc.metadata.get("source", "?") + f" (p.{doc.metadata.get('page', '?')})"
        for doc in result["source_documents"]
    })
    return {
        "question": question,
        "answer": result["result"].strip(),
        "sources": sources
    }

def ingest(pdf_path: str):
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")
    chunks = load_and_split_pdf(pdf_path)
    build_vectorstore(chunks)
    print("✅ Ingestion terminée !")

def chat():
    print("💬 Chargement du vectorstore...")
    vectorstore = load_vectorstore()
    chain = build_qa_chain(vectorstore)
    print("\n🤖 RAG PDF Local — prêt ! (tape 'exit' pour quitter)\n")
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
        print("  python src/rag.py ingest <path/to/file.pdf>")
        print("  python src/rag.py chat")
        sys.exit(1)
    command = sys.argv[1]
    if command == "ingest" and len(sys.argv) == 3:
        ingest(sys.argv[2])
    elif command == "chat":
        chat()
    else:
        print("Commande invalide. Utilise 'ingest <pdf>' ou 'chat'.")
        sys.exit(1)
