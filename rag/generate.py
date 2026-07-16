"""Layer 2: grounded answer generation via llama3.2:3b (streaming)."""
from __future__ import annotations

from typing import Iterator

import ollama

import config
import prompts

_client = ollama.Client(host=config.OLLAMA_HOST)


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        name = c["metadata"].get("recipe_name", "Recipe")
        parts.append(f"[{i}] ({name})\n{c['document']}")
    return "\n\n".join(parts)


def generate_stream(
    query: str, chunks: list[dict], history: list[dict] | None = None
) -> Iterator[str]:
    """Yield the grounded answer token-by-token for a responsive chat UI.

    `history` is the recent chat as [{"role", "content"}] so follow-up questions
    ("give the recipe for these") can be resolved against earlier turns.
    """
    context = build_context(chunks)
    messages = [{"role": "system", "content": prompts.GEN_SYSTEM}]

    # Recent conversation for context (truncate long prior answers).
    for m in (history or [])[-config.HISTORY_TURNS:]:
        content = m["content"]
        if len(content) > 500:
            content = content[:500] + " ..."
        messages.append({"role": m["role"], "content": content})

    messages.append(
        {"role": "user", "content": prompts.GEN_USER.format(context=context, query=query)}
    )

    stream = _client.chat(
        model=config.GEN_MODEL,
        messages=messages,
        options={
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_TOKENS,
            "num_ctx": config.NUM_CTX,
        },
        keep_alive=config.KEEP_ALIVE,
        stream=True,
    )
    for part in stream:
        token = part.get("message", {}).get("content", "")
        if token:
            yield token
