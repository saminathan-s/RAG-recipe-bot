"""Streamlit chat UI for the RAG Recipe Bot."""
import streamlit as st

import config
from rag.pipeline import answer_stream
from rag.vectorstore import get_collection

st.set_page_config(page_title="Recipe Bot", page_icon="🍳", layout="centered")

# Hide the Deploy button and menu, but KEEP the running indicator -- while a
# response is generating it turns into a Stop control to cancel the run.
st.markdown(
    """
    <style>
      [data-testid="stAppDeployButton"] {display: none;}
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍳 RAG Recipe Bot")
st.caption("Ask about recipes, ingredients, cuisines, and seasonal dishes.")

# --- Sidebar: status ---
with st.sidebar:
    st.subheader("Status")
    try:
        count = get_collection().count()
    except Exception:
        count = 0
    if count == 0:
        st.error("No recipes ingested.\n\nDrop a PDF in `data/` and run:\n```\npython ingest.py\n```")
    else:
        st.success(f"{count} recipe chunks loaded")
    st.markdown(
        f"**Intent:** `{config.INTENT_MODEL}`  \n"
        f"**Answer:** `{config.GEN_MODEL}`  \n"
        f"**Embed:** `{config.EMBED_MODEL}`  \n"
        f"**Top-K:** {config.TOP_K}"
    )

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['recipe_name']}** (page {s['source_page']})")

# --- Input ---
if prompt := st.chat_input("Ask me a recipe question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prior turns (exclude the message we just appended) for follow-up context.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        # Phase 1: guardrail + retrieval (fast).
        with st.spinner("Thinking..."):
            status, stream, chunks = answer_stream(prompt, history)

        # Phase 2: stream the answer. Keep a visible spinner + typing cursor the
        # whole time so there is never a blank bubble while the model warms up.
        full = ""
        with st.spinner("Writing answer..."):
            for token in stream:
                full += token
                placeholder.markdown(full + "▌")
        placeholder.markdown(full)

        sources = []
        if status == "answer":
            seen = set()
            for c in chunks:
                key = (c["metadata"].get("recipe_name"), c["metadata"].get("source_page"))
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "recipe_name": c["metadata"].get("recipe_name", "Recipe"),
                        "source_page": c["metadata"].get("source_page", "?"),
                    })
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- **{s['recipe_name']}** (page {s['source_page']})")

    st.session_state.messages.append(
        {"role": "assistant", "content": full, "sources": sources}
    )
