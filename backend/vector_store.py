"""
Ce module gère la base vectorielle FAISS : stocker les embeddings des
chunks, et retrouver les plus proches d'une question posée par l'utilisateur.

FAISS lui-même ne stocke QUE des vecteurs de nombres, pas le texte original.
On garde donc en parallèle une liste "metadata" (texte + source + page)
dans le même ordre que les vecteurs, pour retrouver le texte après recherche.
Les deux sont sauvegardés sur disque pour survivre à un redémarrage du serveur.
"""

import json
import os

import faiss
import numpy as np

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
INDEX_PATH = os.path.join(STORAGE_DIR, "index.faiss")
METADATA_PATH = os.path.join(STORAGE_DIR, "metadata.json")

EMBEDDING_DIM = 384  # doit correspondre à la sortie du modèle utilisé dans embeddings.py


class VectorStore:
    def __init__(self):
        os.makedirs(STORAGE_DIR, exist_ok=True)
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            # On recharge une base déjà existante (uploads précédents)
            self.index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            # IndexFlatL2 = recherche par distance euclidienne, simple et
            # largement suffisante pour un projet de cette taille.
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
            self.metadata = []

    def add(self, vectors: np.ndarray, chunks: list[dict]) -> None:
        """Ajoute de nouveaux vecteurs + leurs métadonnées, puis sauvegarde."""
        self.index.add(vectors.astype("float32"))
        self.metadata.extend(chunks)
        self._save()

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[dict]:
        """Retourne les top_k chunks les plus proches du vecteur donné."""
        if self.index.ntotal == 0:
            return []
        query_vector = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["distance"] = float(dist)
            results.append(item)
        return results

    def _save(self) -> None:
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def clear(self) -> None:
        """Vide complètement la base : index FAISS et métadonnées."""
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.metadata = []
        self._save()


# Une seule instance partagée par toute l'application (chargée au démarrage)
vector_store = VectorStore()
