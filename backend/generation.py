"""
Ce module appelle OpenRouter (un service qui donne accès à plusieurs
modèles de langage, dont certains gratuits) pour générer une réponse
en français, basée UNIQUEMENT sur les passages retrouvés dans le PDF.

C'est le cœur du "G" de RAG : Generation.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()  # charge les variables depuis le fichier .env

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Construit le prompt envoyé au modèle : on lui donne le contexte
    (les passages retrouvés) puis la question, avec des instructions
    claires pour qu'il ne réponde qu'à partir de ce contexte.
    """
    context_text = "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}"
        for c in context_chunks
    )

    prompt = f"""Tu es un assistant qui répond à des questions UNIQUEMENT à partir du contexte fourni ci-dessous, extrait de documents PDF.

Règles importantes :
- Si la réponse ne se trouve pas dans le contexte, dis clairement que tu ne trouves pas l'information dans les documents fournis.
- Ne invente jamais d'information qui n'est pas dans le contexte.
- Réponds en français, de manière claire et concise.

Contexte :
{context_text}

Question : {question}

Réponse :"""
    return prompt


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY n'est pas configurée. "
            "Copiez .env.example en .env et ajoutez votre clé."
        )

    prompt = build_prompt(question, context_chunks)

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Erreur OpenRouter ({response.status_code}): {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]
