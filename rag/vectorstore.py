"""Chroma persistent vector store helpers."""
import chromadb

import config

_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection():
    """Get (or create) the recipes collection using cosine distance."""
    return _client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    """Drop and recreate the collection (used before a fresh ingest)."""
    try:
        _client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    return get_collection()


def query(query_embedding: list[float], k: int = config.TOP_K) -> list[dict]:
    """Return top-k chunks as [{document, metadata, distance}]."""
    col = get_collection()
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return [
        {"document": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(docs, metas, dists)
    ]
