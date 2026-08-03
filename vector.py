"""
vector.py
---------
STEP 5 of the RAG pipeline: VECTOR DATABASE

Purpose : Stores embeddings and enables fast similarity search.

Library used : FAISS (Facebook AI Similarity Search) - free, local,
               no server required. Swap for Pinecone / ChromaDB /
               Weaviate / Milvus for a hosted, multi-user deployment.
"""

import faiss
import numpy as np
import pickle
import os

from embeddings import embed_texts

INDEX_PATH = "vector_store/hospitality_index.faiss"
META_PATH = "vector_store/hospitality_meta.pkl"


class VectorStore:
    def __init__(self, dim: int = 384):
        # 384 = output dimension of all-MiniLM-L6-v2
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []  # parallel list: chunk dicts (text, source, chunk_id)

    def add_chunks(self, chunks: list):
        """
        Embeds and stores a list of chunk dicts (from chunking.py).
        """
        if not chunks:
            return  # nothing extractable - skip silently, caller shows a warning

        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)
        self.index.add(vectors)
        self.metadata.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list:
        """
        Returns the top_k most similar chunks to the query vector.
        """
        query_vector = np.expand_dims(query_vector, axis=0)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx]
            results.append({**chunk, "score": float(dist)})
        return results

    def save(self):
        os.makedirs("vector_store", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        return False

    def is_empty(self) -> bool:
        return self.index.ntotal == 0
