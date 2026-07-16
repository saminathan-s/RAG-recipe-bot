"""Central configuration for the RAG Recipe Bot."""
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_store"

# Drop your recipe PDF here (any *.pdf inside data/ will be ingested).
# If a specific file is preferred, set RECIPE_PDF to its path.
RECIPE_PDF = None  # e.g. DATA_DIR / "recipes.pdf"; None = ingest all PDFs in DATA_DIR

# --- Ollama models ---
EMBED_MODEL = "nomic-embed-text"
INTENT_MODEL = "qwen2.5:1.5b"      # lightweight guardrail / intent classifier
GEN_MODEL = "llama3.2:latest"          # grounded answer generation
OLLAMA_HOST = "http://localhost:11434"

# Keep models resident in memory between requests so Ollama doesn't unload/reload
# (qwen -> llama) on every turn. Big speed win. Use "-1" to keep loaded forever.
KEEP_ALIVE = "30m"

# --- Vector store ---
COLLECTION_NAME = "recipes"

# --- Chunking ---
# Each chunk = one whole recipe (possibly merged across 2 pages). Set high so a
# complete recipe is never split; only oversized prose pages fall back to splitting.
MAX_CHUNK_CHARS = 3000
CHUNK_OVERLAP = 150

# --- Retrieval ---
TOP_K = 5
# Chroma returns cosine *distance* (0 = identical, 2 = opposite). Chunks with a
# distance above this are treated as irrelevant -> triggers "not in my book".
MAX_DISTANCE = 1.0

# --- Generation ---
TEMPERATURE = 0.1                  # low = less hallucination, more grounded
MAX_TOKENS = 1024                  # hard cap on answer length (prevents runaway)
NUM_CTX = 8192

# --- Conversation memory ---
HISTORY_TURNS = 6                  # recent messages passed to the model for follow-ups
