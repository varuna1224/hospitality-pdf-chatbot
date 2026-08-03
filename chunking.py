"""
chunking.py
-----------
STEP 3 of the RAG pipeline: CHUNKING (Text Splitter)

Purpose : Splits long extracted text into small, overlapping chunks so that
          embeddings stay meaningful and retrieval stays precise.

Library used : langchain_text_splitters.RecursiveCharacterTextSplitter
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hospitality documents (tariff cards, SOP manuals, menus) tend to have
# short structured sections, so a moderate chunk size works best.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def split_text_into_chunks(
    text: str,
    source_name: str = "document.pdf",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list:
    """
    Splits a single document's text into chunks.

    Returns:
        List[dict] -> [{"text": chunk_text, "source": source_name, "chunk_id": i}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    chunks = [
        {"text": chunk, "source": source_name, "chunk_id": idx}
        for idx, chunk in enumerate(raw_chunks)
    ]
    return chunks


def chunk_multiple_documents(documents: dict, chunk_size: int = DEFAULT_CHUNK_SIZE,
                              chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list:
    """
    Args:
        documents: {file_name: full_text} as produced by extractor.py

    Returns:
        A flat list of chunk dicts across all documents.
    """
    all_chunks = []
    for source_name, text in documents.items():
        all_chunks.extend(
            split_text_into_chunks(text, source_name, chunk_size, chunk_overlap)
        )
    return all_chunks
