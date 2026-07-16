"""Diagnostic: show what retrieval returns for a query (run on the server)."""
from __future__ import annotations

import chromadb

import config
from rag.embeddings import embed_query
from rag.vectorstore import get_collection, query as vq

print("chromadb version :", chromadb.__version__)
print("collection count :", get_collection().count())
print("MAX_DISTANCE     :", config.MAX_DISTANCE, " TOP_K:", config.TOP_K)
print("collection metadata:", get_collection().metadata)
print()

for q in ["Give me Gulab Jamun recipe", "Gulab Jamun"]:
    print(f"QUERY: {q!r}")
    for c in vq(embed_query(q), k=8):
        keep = "KEEP" if c["distance"] <= config.MAX_DISTANCE else "DROP"
        print(f"   [{keep}] dist={c['distance']:.3f}  "
              f"{c['metadata']['recipe_name']} ({c['metadata']['country']})")
    print()
