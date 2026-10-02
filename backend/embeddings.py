"""
Ce module transforme du texte en "embeddings" : des vecteurs de nombres
qui représentent le sens du texte. On utilise un modèle local et gratuit
(pas besoin de clé API pour cette étape), téléchargé une seule fois puis
mis en cache automatiquement par la librairie.
"""

from sentence_transformers import SentenceTransformer

# Nom du modèle : petit, rapide, gratuit, et suffisant pour un projet de stage.
# Il produit des vecteurs de 384 dimensions.
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # chargé une seule fois, puis réutilisé (plus rapide)


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]):
    """
    Transforme une liste de textes en une liste de vecteurs numériques.
    Utilisé à deux moments :
    - quand on ajoute des chunks de PDF à la base (à l'upload)
    - quand on transforme la question de l'utilisateur pour la comparer
    """
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
