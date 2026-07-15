"""Orchestrates the full flow: intent gate -> retrieve -> grounded generate."""
from __future__ import annotations

import re
from typing import Iterator

import config
import prompts
from rag import intent, generate
from rag.embeddings import embed_query
from rag.vectorstore import query as vector_query

# Words that signal a follow-up referring back to the previous turn.
_FOLLOWUP = re.compile(
    r"\b(these|those|them|it|that|this|they|the recipe|recipes|above|previous|ones?|each)\b",
    re.I,
)

# Greetings / small talk -> instant canned welcome, no LLM or retrieval.
_GREETING = re.compile(
    r"^\s*(hi|hello|hey+|hiya|yo|howdy|greetings|good\s+(morning|afternoon|evening)|"
    r"namaste|thanks|thank\s*you|ty|help|what\s+can\s+you\s+do)\b[\s!.?]*$",
    re.I,
)


def _retrieval_query(user_query: str, history: list[dict] | None) -> str:
    """For follow-ups, prepend the previous user message so 'these'/'it' resolve."""
    prev_user = ""
    for m in reversed(history or []):
        if m["role"] == "user":
            prev_user = m["content"]
            break
    if prev_user and (len(user_query.split()) <= 5 or _FOLLOWUP.search(user_query)):
        return f"{prev_user}\n{user_query}"
    return user_query


def answer_stream(
    user_query: str, history: list[dict] | None = None
) -> tuple[str, Iterator[str], list[dict]]:
    """Run the pipeline.

    Returns (status, token_iterator, chunks) where status is one of:
      "rejected"   -> out of scope; iterator yields the polite rejection
      "no_context" -> in scope but nothing relevant retrieved
      "answer"     -> grounded answer streamed from the LLM
    """
    # Fast-path: greetings / small talk -> instant welcome (no model, no retrieval)
    if _GREETING.match(user_query):
        return "greeting", iter([prompts.WELCOME_MESSAGE]), []

    # Layer 1: guardrail / intent
    verdict = intent.classify(user_query)
    if not verdict["in_scope"]:
        return "rejected", iter([prompts.REJECTION_MESSAGE]), []

    # Retrieval (contextualised for follow-up questions)
    q_vec = embed_query(_retrieval_query(user_query, history))
    chunks = vector_query(q_vec, k=config.TOP_K)
    relevant = [c for c in chunks if c["distance"] <= config.MAX_DISTANCE]

    # Layer 2 guardrail: no relevant grounding -> don't hallucinate
    if not relevant:
        return "no_context", iter([prompts.NO_CONTEXT_MESSAGE]), []

    return "answer", generate.generate_stream(user_query, relevant, history), relevant
