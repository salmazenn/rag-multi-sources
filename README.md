---
title: RAG PDF Cloud
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 📄 RAG PDF Cloud

> Posez des questions sur vos PDFs — propulsé par Groq + HuggingFace, déployé sur HuggingFace Spaces.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Groq](https://img.shields.io/badge/Groq-llama3.3-black)
![LangChain](https://img.shields.io/badge/LangChain-0.1-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vectorstore-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)

## ✨ Fonctionnalités
- 📥 Ingestion de PDFs jusqu'à 40MB
- 🔍 Embeddings locaux via HuggingFace (all-MiniLM-L6-v2)
- 🗄️ Vectorstore persistant avec ChromaDB
- ⚡ Génération ultra-rapide via Groq (llama-3.3-70b)
- 💬 Interface chat Streamlit avec historique et sources
- 🌍 Déployé sur HuggingFace Spaces

## 🏗️ Architecture

```
PDF
 │
 ▼
PyPDFLoader ──► RecursiveCharacterTextSplitter
                          │
                          ▼
               HuggingFaceEmbeddings (all-MiniLM-L6-v2)
                          │
                          ▼
                   ChromaDB (local)
                          │
                    ┌─────┴─────┐
                    │           │
               Retriever    Groq LLM
                    │           │
                    └─────┬─────┘
                          │
                   RetrievalQA Chain
                          │
                          ▼
                       Réponse + Sources
```

## 📦 Stack technique

| Composant | Outil | Rôle |
|-----------|-------|------|
| LLM | Groq (llama-3.3-70b) | Génération de réponses |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) | Vectorisation des chunks |
| Vectorstore | ChromaDB | Stockage et recherche sémantique |
| Orchestration | LangChain | Pipeline RAG |
| UI | Streamlit | Interface utilisateur |
| Hébergement | HuggingFace Spaces | Déploiement cloud gratuit |

## ⚙️ Requirements

```
langchain==0.1.20
langchain-community==0.0.38
langchain-groq==0.1.3
chromadb==0.4.24
pypdf==4.2.0
streamlit==1.34.0
sentence-transformers==2.7.0
python-dotenv==1.0.0
numpy==1.26.4
```

## 🚀 Lancer en local

### 1. Cloner le repo
```bash
git clone https://github.com/salmazenn/rag-pdf-cloud.git
cd rag-pdf-cloud
```

### 2. Créer l'environnement virtuel
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurer la clé Groq
```bash
echo "GROQ_API_KEY=ta_clé_gsk_..." > .env
```

### 4. Lancer l'app
```bash
streamlit run src/app.py
```

Ouvre **http://localhost:8501** dans ton navigateur.

## ☁️ Déployer sur HuggingFace Spaces

### 1. Créer un Space
- Va sur **huggingface.co/new-space**
- SDK : **Docker** → Blank
- Visibility : **Public**

### 2. Connecter le repo
```bash
git remote add hf https://huggingface.co/spaces/ton-username/rag-pdf-cloud
git push hf main
```

### 3. Ajouter la clé Groq
- Space → **⋮** → **Settings** → **Variables and secrets**
- Ajoute `GROQ_API_KEY` avec ta clé `gsk_...`

### 4. Points importants
- ⚠️ HuggingFace utilise le port **7860** — pas 8501
- ⚠️ Ajouter `.streamlit/config.toml` pour les uploads
- ⚠️ Ne jamais pusher le fichier `.env` sur GitHub

## 🔭 Pistes d'amélioration
- [ ] Support multi-PDFs dans le même index
- [ ] Évaluation automatique avec RAGAS
- [ ] Reranking des chunks
- [ ] Authentification sur l'interface

## 📄 Licence
MIT
```