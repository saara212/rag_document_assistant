"""
Étape 2 : en plus de l'upload (étape 1), on génère maintenant des
embeddings pour chaque chunk et on les stocke dans FAISS. L'endpoint
/search permet de tester la recherche par similarité de sens.
La prochaine étape ajoutera la génération de réponse via OpenRouter.
"""

import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingestion import process_pdf
from embeddings import embed_texts
from vector_store import vector_store
from generation import generate_answer

app = FastAPI(title="RAG Document Assistant")

# Autorise le frontend (Streamlit) à appeler cette API depuis un autre port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG backend is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés")

    # Sauvegarde temporaire du fichier uploadé
    saved_path = os.path.join(STORAGE_DIR, file.filename)
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extraction + découpage en chunks
    chunks = process_pdf(saved_path, source_name=file.filename)

    # Génération des embeddings (un vecteur par chunk) puis stockage dans FAISS
    if chunks:
        texts = [chunk["text"] for chunk in chunks]
        vectors = embed_texts(texts)
        vector_store.add(vectors, chunks)

    return {
        "filename": file.filename,
        "num_chunks": len(chunks),
        "total_chunks_in_store": vector_store.index.ntotal,
        "preview": chunks[0]["text"][:200] if chunks else None,
    }


class SearchRequest(BaseModel):
    query: str
    top_k: int = 4


@app.delete("/reset")
async def reset_documents():
    """
    Efface tous les documents indexés (utile pour tester un PDF à la fois,
    ou repartir de zéro). Vide la base FAISS et supprime les fichiers stockés.
    """
    vector_store.clear()

    for fname in os.listdir(STORAGE_DIR):
        fpath = os.path.join(STORAGE_DIR, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

    return {"status": "ok", "message": "Base de documents réinitialisée"}


@app.post("/search")
async def search(request: SearchRequest):
    """
    Endpoint de TEST pour l'étape 2 : cherche les chunks les plus proches
    d'une question, sans encore générer de réponse (ça viendra à l'étape 4).
    """
    if vector_store.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="Aucun document n'a encore été uploadé")

    query_vector = embed_texts([request.query])[0]
    results = vector_store.search(query_vector, top_k=request.top_k)
    return {"query": request.query, "results": results}


class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    L'endpoint principal du RAG : retrouve les chunks pertinents,
    génère une réponse via OpenRouter, et renvoie la réponse + les sources
    utilisées (pour les afficher dans l'interface de chat).
    """
    if vector_store.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="Aucun document n'a encore été uploadé")

    # 1. Retrieval : on retrouve les chunks les plus pertinents
    query_vector = embed_texts([request.question])[0]
    context_chunks = vector_store.search(query_vector, top_k=request.top_k)

    # 2. Generation : on demande au modèle de répondre à partir de ce contexte
    try:
        answer = generate_answer(request.question, context_chunks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # 3. On renvoie la réponse ET les sources utilisées, séparément,
    #    pour que le frontend puisse les afficher clairement.
    sources = [
        {"source": c["source"], "page": c["page"]} for c in context_chunks
    ]

    return {
        "question": request.question,
        "answer": answer,
        "sources": sources,
    }
