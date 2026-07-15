"""Ingest recipe PDF(s): parse -> structure-aware chunk -> embed -> persist to Chroma.

Run once (or whenever the PDF changes):
    python ingest.py

Chunking strategy (tuned to "A Taste of Travel" cookbook layout):
  * The book is organised by country: each country has one INTRO page (which
    always contains the text "THE RECIPES") followed by its recipe pages.
  * Each recipe is usually one page, but long recipes overflow onto a second
    page that carries the "Method" but no title -> we MERGE those together so a
    recipe's ingredients + steps are never split (e.g. Fish Amok, p22 + p23).
  * Dish title + country are extracted and prepended to each chunk so that
    cuisine / location / season queries retrieve well.
"""
import re
import sys
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from rag.embeddings import embed_texts
from rag.vectorstore import get_collection, reset_collection

# Known countries in the cookbook. Keys are normalised (letters only, upper) so
# both spaced ("N E P A L") and contiguous ("CAMBODIA") headers match.
COUNTRY_ALIASES = {
    "NEPAL": "Nepal", "THAILAND": "Thailand", "PERU": "Peru", "INDIA": "India",
    "CAMBODIA": "Cambodia", "TANZANIA": "Tanzania", "AUSTRALIA": "Australia",
    "UNITEDKINGDOM": "United Kingdom", "UNITED": "United Kingdom",
    "KINGDOM": "United Kingdom",
}
_STOP_TITLES = {
    "INGREDIENTS", "METHOD", "DIRECTIONS", "INSTRUCTIONS",
    "THE RECIPES", "ENJOY YOUR MEAL", "RECIPE",
}


def find_pdfs() -> list[Path]:
    if config.RECIPE_PDF:
        p = Path(config.RECIPE_PDF)
        return [p] if p.exists() else []
    return sorted(config.DATA_DIR.glob("*.pdf"))


def read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text)."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)   # normalise spaces, keep newlines
        pages.append((i, text))
    return pages


# --- line/page classification helpers ---
def _norm(line: str) -> str:
    return re.sub(r"[^A-Za-z]", "", line).upper()


def _short_ratio(s: str) -> float:
    toks = s.split()
    return sum(1 for t in toks if len(t) <= 2) / len(toks) if toks else 0.0


def _find_country(lines: list[str]) -> str | None:
    for ln in lines:
        c = COUNTRY_ALIASES.get(_norm(ln))
        if c:
            return c
    return None


def _is_title_line(line: str) -> bool:
    """A contiguous ALL-CAPS dish title, e.g. 'CHOILA', 'AJI DE GALLINA'."""
    s = line.strip().rstrip(":")
    if not (3 <= len(s) <= 40):
        return False
    if s.upper() in _STOP_TITLES or _norm(s) in COUNTRY_ALIASES:
        return False
    if "(" in s or ")" in s:
        return False
    letters = [c for c in s if c.isalpha()]
    if not letters or not all(c.isupper() for c in letters):
        return False
    return _short_ratio(s) < 0.7          # exclude decorative letter-spaced lines


def _extract_title(lines: list[str]) -> str | None:
    idx = next((i for i, l in enumerate(lines) if _is_title_line(l)), None)
    if idx is None:
        return None
    parts = [lines[idx].strip().rstrip(":")]
    for l in lines[idx + 1:]:             # merge follow-on caps line (VEGETABLE / CHILLI)
        if _is_title_line(l):
            parts.append(l.strip().rstrip(":"))
        else:
            break
    return " ".join(parts).title()


def _has_recipe_signal(text: str) -> bool:
    low = text.lower()
    return (
        "method" in low
        or "ingredients" in low
        or re.search(r"^\s*1[.)]", text, re.M) is not None
    )


def build_chunks(pages: list[tuple[int, str]]) -> tuple[list[str], list[dict]]:
    """Segment pages into recipe / intro chunks (see module docstring)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.MAX_CHUNK_CHARS,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: list[str] = []
    metas: list[dict] = []
    country: str | None = None
    cur: dict | None = None

    def flush():
        nonlocal cur
        if not cur:
            return
        body = "\n".join(cur["body"]).strip()
        full = f"Dish: {cur['title']}\nCountry/Cuisine: {cur['country']}\n---\n{body}"
        parts = [full] if len(full) <= config.MAX_CHUNK_CHARS else splitter.split_text(full)
        for j, part in enumerate(parts):
            docs.append(part)
            metas.append({
                "recipe_name": cur["title"],
                "country": cur["country"],
                "kind": "recipe",
                "source_page": cur["page"],
                "pages": ",".join(str(p) for p in cur["pages"]),
                "part": j,
            })
        cur = None

    for page_no, text in pages:
        if not text.strip():
            continue
        lines = text.splitlines()

        # Country intro page (must name a known country; the foreword letter
        # also mentions "recipes" but has no country -> falls through and is
        # skipped as front matter below).
        if "the recipes" in text.lower() and _find_country(lines):
            flush()
            country = _find_country(lines)
            docs.append(f"Country/Cuisine: {country} - Introduction\n---\n{text.strip()}")
            metas.append({
                "recipe_name": f"{country} - Introduction",
                "country": country, "kind": "intro",
                "source_page": page_no, "pages": str(page_no), "part": 0,
            })
            continue

        # Front/back matter (cover, letter, foundation) -> skip.
        if not _has_recipe_signal(text):
            continue

        title = _extract_title(lines)
        if title:                              # start of a new recipe
            flush()
            cur = {"title": title, "country": country or "General",
                   "page": page_no, "pages": [page_no], "body": [text.strip()]}
        elif cur:                              # continuation of previous recipe
            cur["body"].append(text.strip())
            cur["pages"].append(page_no)
        else:                                  # recipe content with no title yet
            cur = {"title": f"{country or 'General'} recipe (p{page_no})",
                   "country": country or "General",
                   "page": page_no, "pages": [page_no], "body": [text.strip()]}

    flush()
    return docs, metas


def main() -> int:
    pdfs = find_pdfs()
    if not pdfs:
        print(f"No PDF found. Drop a recipe PDF into: {config.DATA_DIR}")
        print("(or set RECIPE_PDF in config.py)")
        return 1

    all_docs: list[str] = []
    all_metas: list[dict] = []
    for pdf in pdfs:
        print(f"Reading {pdf.name} ...")
        pages = read_pdf_pages(pdf)
        docs, metas = build_chunks(pages)
        for m in metas:
            m["source_file"] = pdf.name
        all_docs.extend(docs)
        all_metas.extend(metas)

        recipes = [m for m in metas if m["kind"] == "recipe"]
        intros = [m for m in metas if m["kind"] == "intro"]
        print(f"  -> {len(intros)} intros, {len(recipes)} recipe chunks")
        for m in metas:
            tag = "INTRO " if m["kind"] == "intro" else "recipe"
            print(f"     [{tag}] p{m['pages']:<6} {m['country']:<15} {m['recipe_name']}")

    if not all_docs:
        print("No text extracted from the PDF(s).")
        return 1

    print(f"\nEmbedding {len(all_docs)} chunks with {config.EMBED_MODEL} ...")
    vectors = embed_texts(all_docs)

    reset_collection()
    col = get_collection()
    ids = [f"chunk-{i}" for i in range(len(all_docs))]
    col.add(ids=ids, documents=all_docs, embeddings=vectors, metadatas=all_metas)
    print(f"Ingested {len(all_docs)} chunks into Chroma at {config.CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
