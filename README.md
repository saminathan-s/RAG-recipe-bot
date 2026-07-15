# 🍳 RAG Recipe Bot

A guarded Retrieval-Augmented-Generation chatbot that answers **only** food/recipe
questions from a single cooking PDF. Built with Python + Streamlit + Ollama + ChromaDB.

## How it works

```
User query
  → [1] Intent guardrail (qwen2.5:1.5b)  → out-of-scope? polite rejection
  → [2] Embed (nomic-embed-text) → Chroma top-K → relevance filter
  → [3] Grounded answer (llama3.2:3b, strict "answer only from context")
  → Streamlit chat (with sources)
```

**Guardrails**
- *Layer 1 – intent gate:* off-topic queries are rejected before any retrieval.
- *Layer 2 – grounding:* the answer model may use **only** retrieved context; if
  nothing relevant is found (or the answer isn't in context) it declines instead
  of inventing. `temperature=0.1` keeps it factual.

**Chunking:** structure-aware, tuned to the *A Taste of Travel* cookbook. The
book is organised by country (each country has an intro page containing
"THE RECIPES" followed by its recipe pages). The ingester:
- detects **country intro pages** and tags the current country;
- treats an ALL-CAPS dish title (e.g. `CHOILA`, `FISH AMOK`) as the start of a recipe;
- **merges continuation pages** (a page with a `Method` but no title) into the
  recipe above it, so multi-page recipes stay whole (e.g. Fish Amok spans p22–23);
- skips front/back matter (cover, foreword, foundation pages);
- prepends `Dish` + `Country/Cuisine` to each chunk and stores them as metadata
  (`recipe_name`, `country`, `source_page`, `pages`) for citations and better
  retrieval on cuisine / location / seasonal queries.

Result on the sample PDF: **8 country intros + 23 complete recipes**. Each chunk
is one whole recipe, so no ingredients/steps are ever lost across a split.

## Setup

1. **Install Ollama** and pull the models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5:1.5b
   ollama pull llama3.2:3b
   ```
2. **Install Python deps:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Add your recipe PDF** into `data/` (e.g. `data/recipes.pdf`).
4. **Ingest** (run once, or after changing the PDF):
   ```bash
   python ingest.py
   ```
5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Configuration

All tunables live in `config.py`: model names, `TOP_K`, `MAX_DISTANCE`
(relevance threshold), chunk size/overlap, and temperature.

## Project layout

```
config.py          # models, paths, retrieval/chunking settings
prompts.py         # intent + grounded-answer + rejection templates
ingest.py          # PDF → structure-aware chunk → embed → Chroma
rag/
  embeddings.py    # nomic-embed-text via Ollama
  vectorstore.py   # Chroma persistent client + query
  intent.py        # qwen2.5:1.5b guardrail classifier (JSON)
  generate.py      # llama3.2:3b grounded streaming answer
  pipeline.py      # intent → retrieve → generate orchestration
app.py             # Streamlit chat UI
```
