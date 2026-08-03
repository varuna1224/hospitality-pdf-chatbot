"""
retrieval.py
------------
STEP 6 of the RAG pipeline: RETRIEVAL (Similarity Search)

Purpose : Pulls the most relevant chunks from the vector database
          for a given user question (e.g. "What time is check-in?",
          "Do you have a banquet hall for 200 guests?").

Technique used : Cosine/L2 similarity search (via FAISS), with an
                 optional simple keyword-boost re-rank.
"""

from embeddings import embed_query
from vector import VectorStore


def retrieve_relevant_chunks(query: str, store: VectorStore, top_k: int = 4) -> list:
    """
    Args:
        query: the user's question
        store: a populated VectorStore
        top_k: number of chunks to retrieve

    Returns:
        List of chunk dicts, most relevant first.
    """
    if store.is_empty():
        return []

    query_vector = embed_query(query)
    results = store.search(query_vector, top_k=top_k)

    # Lower L2 distance = more similar -> sort ascending
    results.sort(key=lambda r: r["score"])
    return results


def build_context_block(chunks: list) -> str:
    """
    Formats retrieved chunks into a single context string for the LLM prompt,
    tagging each chunk with its source PDF for traceability.
    """
    if not chunks:
        return "No relevant information found in the uploaded PDFs."

    blocks = []
    for c in chunks:
        blocks.append(f"[Source: {c['source']} | chunk {c['chunk_id']}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)
