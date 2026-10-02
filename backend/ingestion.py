"""
Ce module gère l'étape 1 du RAG : transformer un PDF en morceaux de texte
("chunks") exploitables plus tard pour la recherche vectorielle.
"""

from pypdf import PdfReader


def extract_text_by_page(pdf_path: str) -> list[dict]:
    """
    Lit un PDF et retourne une liste de {"page": numéro, "text": contenu}.
    On garde le numéro de page pour pouvoir citer la source plus tard.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Découpe un texte long en morceaux de taille ~chunk_size caractères,
    avec un chevauchement pour ne pas couper une idée en plein milieu.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def process_pdf(pdf_path: str, source_name: str) -> list[dict]:
    """
    Pipeline complet : PDF -> pages -> chunks avec métadonnées.
    Chaque chunk retourné contient le texte + sa source (fichier + page),
    ce qui permettra d'afficher les sources dans l'interface de chat.
    """
    pages = extract_text_by_page(pdf_path)
    all_chunks = []
    for page in pages:
        for chunk in chunk_text(page["text"]):
            all_chunks.append({
                "text": chunk,
                "source": source_name,
                "page": page["page"],
            })
    return all_chunks
