"""
Script de test rapide pour /chat (recherche + génération de réponse).
Usage : python test_chat.py "votre question ici"
"""

import sys
import requests

if len(sys.argv) < 2:
    print('Usage : python test_chat.py "votre question"')
    sys.exit(1)

question = sys.argv[1]
url = "http://127.0.0.1:8000/chat"

response = requests.post(url, json={"question": question, "top_k": 4})

print("Code:", response.status_code)
data = response.json()

if response.status_code == 200:
    print("\n=== Réponse ===")
    print(data["answer"])
    print("\n=== Sources ===")
    for s in data["sources"]:
        print(f"- {s['source']} (page {s['page']})")
else:
    print("Erreur:", data)
