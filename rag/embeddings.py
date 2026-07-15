"""Text embeddings via Ollama (nomic-embed-text)."""
import ollama

import config

_client = ollama.Client(host=config.OLLAMA_HOST)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input."""
    vectors: list[list[float]] = []
    for text in texts:
        resp = _client.embeddings(
            model=config.EMBED_MODEL, prompt=text, keep_alive=config.KEEP_ALIVE
        )
        vectors.append(resp["embedding"])
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
