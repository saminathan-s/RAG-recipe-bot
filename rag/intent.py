"""Layer 1 guardrail: intent classification via lightweight LLM (qwen2.5:1.5b)."""
from __future__ import annotations

import json
import re

import ollama

import config
import prompts

_client = ollama.Client(host=config.OLLAMA_HOST)

_FOOD_HINTS = re.compile(
    r"\b(recipe|cook|cooking|bake|baking|food|dish|meal|ingredient|cuisine|"
    r"eat|dinner|lunch|breakfast|dessert|spice|season|vegan|vegetarian)\b",
    re.I,
)


def classify(query: str) -> dict:
    """Return {"in_scope": bool, "category": str, "reason": str}.

    Uses the LLM with a strict JSON prompt; falls back to a keyword heuristic
    if the model output can't be parsed.
    """
    try:
        resp = _client.chat(
            model=config.INTENT_MODEL,
            messages=[
                {"role": "system", "content": prompts.INTENT_SYSTEM},
                {"role": "user", "content": prompts.INTENT_USER.format(query=query)},
            ],
            format="json",          # ask Ollama to constrain output to JSON
            options={"temperature": 0},
            keep_alive=config.KEEP_ALIVE,
        )
        raw = resp["message"]["content"].strip()
        data = json.loads(raw)
        return {
            "in_scope": bool(data.get("in_scope", False)),
            "category": str(data.get("category", "unknown")),
            "reason": str(data.get("reason", "")),
        }
    except Exception as e:
        # Fail safe: fall back to keyword heuristic rather than crashing.
        in_scope = bool(_FOOD_HINTS.search(query or ""))
        return {
            "in_scope": in_scope,
            "category": "heuristic",
            "reason": f"fallback classification ({e.__class__.__name__})",
        }
