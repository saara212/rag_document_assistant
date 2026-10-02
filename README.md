\# RAG Document Assistant



Un assistant IA qui répond à vos questions à partir de vos documents PDF,

en citant les sources utilisées. Projet de stage été 2026.



\## Fonctionnalités

\- Upload d'un ou plusieurs documents PDF

\- Questions en langage naturel via une interface de chat

\- Recherche vectorielle (FAISS) pour retrouver les passages pertinents

\- Génération de réponse via un modèle gratuit (OpenRouter)

\- Affichage des sources (fichier + numéro de page) pour chaque réponse

\- Bouton pour réinitialiser la base de documents indexés



\## Stack technique

\- \*\*Backend\*\* : FastAPI (Python)

\- \*\*Frontend\*\* : Streamlit

\- \*\*Base vectorielle\*\* : FAISS

\- \*\*Embeddings\*\* : sentence-transformers (all-MiniLM-L6-v2, local, gratuit)

\- \*\*Génération\*\* : OpenRouter API (modèle gratuit)

\- \*\*Extraction PDF\*\* : pypdf



\## Installation



\### 1. Backend



```bash

cd backend

python -m venv venv

venv\\Scripts\\activate          # Windows

\# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

```



Copiez `.env.example` en `.env` et remplissez votre clé OpenRouter

(gratuite sur https://openrouter.ai/keys) :

```

OPENROUTER\_API\_KEY=votre\_cle\_ici

OPENROUTER\_MODEL=openrouter/free

```



\### 2. Frontend



```bash

cd frontend

pip install -r requirements.txt

```



(peut utiliser le même venv que le backend)



\## Lancement



\*\*Terminal 1 — Backend :\*\*

```bash

cd backend

venv\\Scripts\\activate

uvicorn main:app --reload

```

Le serveur tourne sur http://127.0.0.1:8000

Documentation interactive (Swagger) : http://127.0.0.1:8000/docs



\*\*Terminal 2 — Frontend :\*\*

```bash

cd frontend

..\\backend\\venv\\Scripts\\activate

streamlit run app.py

```

L'interface s'ouvre automatiquement dans le navigateur sur http://localhost:8501



\## Utilisation



1\. Dans la barre latérale, uploadez un ou plusieurs PDF

2\. Attendez la confirmation ("X passages indexés")

3\. Posez vos questions dans le champ de chat

4\. Dépliez "Sources utilisées" sous chaque réponse pour voir d'où vient l'information

5\. Utilisez le bouton \*\*"🗑️ Effacer tous les documents"\*\* pour repartir de zéro

&#x20;  et tester un document à la fois



\## Endpoints de l'API (backend)



| Méthode | Route | Description |

|---|---|---|

| GET | `/` | Vérifie que le serveur tourne |

| POST | `/upload` | Upload un PDF, l'indexe (chunks + embeddings) |

| POST | `/search` | Recherche les chunks les plus pertinents pour une question |

| POST | `/chat` | Recherche + génère une réponse complète avec sources |

| DELETE | `/reset` | Réinitialise la base de documents |



\## Structure du projet



```

rag-assistant/

├── backend/

│   ├── main.py           # API FastAPI (endpoints /upload, /search, /chat, /reset)

│   ├── ingestion.py       # Extraction et découpage des PDF

│   ├── embeddings.py      # Génération des embeddings

│   ├── vector\_store.py    # Stockage et recherche FAISS

│   ├── generation.py      # Appel au modèle de langage via OpenRouter

│   ├── requirements.txt

│   └── .env.example

└── frontend/

&#x20;   ├── app.py             # Interface de chat Streamlit

&#x20;   └── requirements.txt

```



\## Limitations connues

\- La base vectorielle est stockée localement (fichiers dans `backend/storage/`),

&#x20; pas de multi-utilisateur

\- Les documents s'accumulent : utiliser le bouton de reset pour tester un

&#x20; document à la fois

\- L'extraction de texte fonctionne mieux sur des PDF textuels que sur des

&#x20; supports de slides très visuels

\- Le modèle gratuit d'OpenRouter peut avoir des temps de réponse variables

&#x20; selon la charge

