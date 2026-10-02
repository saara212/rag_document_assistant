"""
Interface utilisateur du RAG Document Assistant.
Cette page parle au backend FastAPI (qui doit tourner en même temps,
sur http://127.0.0.1:8000) via de simples requêtes HTTP.
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="centered")
st.title("📄 RAG Document Assistant")
st.caption("Uploadez un ou plusieurs PDF, puis posez vos questions sur leur contenu.")

# --- Historique de conversation, gardé en mémoire pendant la session ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Barre latérale : upload de documents ---
with st.sidebar:
    st.header("📁 Documents")
    uploaded_files = st.file_uploader(
        "Choisissez un ou plusieurs PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            already_done = uploaded_file.name in st.session_state.get("uploaded_names", set())
            if not already_done:
                with st.spinner(f"Traitement de {uploaded_file.name}..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    try:
                        response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                        if response.status_code == 200:
                            data = response.json()
                            st.success(f"✅ {uploaded_file.name} : {data['num_chunks']} passages indexés")
                            st.session_state.setdefault("uploaded_names", set()).add(uploaded_file.name)
                        else:
                            st.error(f"Erreur : {response.json().get('detail')}")
                    except requests.exceptions.ConnectionError:
                        st.error("Impossible de contacter le serveur backend. Est-il bien lancé ?")

    st.divider()
    st.caption("Le backend doit tourner sur http://127.0.0.1:8000")

    st.divider()
    if st.button("🗑️ Effacer tous les documents", use_container_width=True):
        try:
            response = requests.delete(f"{API_URL}/reset", timeout=30)
            if response.status_code == 200:
                st.session_state.uploaded_names = set()
                st.session_state.messages = []
                st.success("Base réinitialisée ! Uploadez de nouveaux documents.")
                st.rerun()
            else:
                st.error("Erreur lors de la réinitialisation")
        except requests.exceptions.ConnectionError:
            st.error("Impossible de contacter le serveur backend.")

# --- Zone de chat principale ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources utilisées"):
                for s in message["sources"]:
                    st.markdown(f"- **{s['source']}** (page {s['page']})")

question = st.chat_input("Posez votre question sur les documents uploadés...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération de la réponse..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question, "top_k": 4},
                    timeout=120,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.markdown(data["answer"])
                    if data.get("sources"):
                        with st.expander("📚 Sources utilisées"):
                            for s in data["sources"]:
                                st.markdown(f"- **{s['source']}** (page {s['page']})")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                    })
                else:
                    error_detail = response.json().get("detail", "Erreur inconnue")
                    st.error(f"Erreur : {error_detail}")
            except requests.exceptions.ConnectionError:
                st.error("Impossible de contacter le serveur backend. Est-il bien lancé sur le port 8000 ?")
