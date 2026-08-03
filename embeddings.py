"""
embeddings.py
-------------
STEP 4 of the RAG pipeline: EMBEDDING MODELS

Purpose : Converts text chunks into dense vector representations.

Model used : sentence-transformers/all-MiniLM-L6-v2 (free, runs locally,
             no API key needed - good default for a hospitality FAQ bot).

You can swap this for OpenAI's "text-embedding-3-small/large" or
Google's "embedding-001" by changing get_embedding_model().
"""

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model_cache = {}


def get_embedding_model():
    """Loads (and caches) the embedding model."""
    if MODEL_NAME not in _model_cache:
        _model_cache[MODEL_NAME] = SentenceTransformer(MODEL_NAME)
    return _model_cache[MODEL_NAME]


def embed_texts(texts: list) -> np.ndarray:
    """
    Embeds a list of strings.

    Args:
        texts: list of chunk texts

    Returns:
        numpy array of shape (num_texts, embedding_dim)
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embeds a single user query."""
    return embed_texts([query])[0]
