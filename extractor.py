"""
extractor.py
------------
STEP 2 of the RAG pipeline: TEXT EXTRACTION

Purpose : Extracts raw text from PDF documents only.
          (Scoped to hospitality use-cases: hotel tariff sheets, SOPs,
          guest policy manuals, menu cards, banquet packages, etc.)

Library used : pypdf (PdfReader)
"""

import os
from pypdf import PdfReader


def is_pdf(file_path: str) -> bool:
    """Guard-rail: this chatbot is PDF-only."""
    return file_path.lower().endswith(".pdf")


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts raw text from a single PDF file.

    Args:
        file_path: path to a .pdf file

    Returns:
        Full extracted text (all pages concatenated with page markers).
    """
    if not is_pdf(file_path):
        raise ValueError(f"Rejected: '{file_path}' is not a PDF. This chatbot only accepts PDF files.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(file_path)
    full_text = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            full_text.append(f"\n[Page {page_num}]\n{page_text}")

    return "\n".join(full_text)


def extract_text_from_multiple_pdfs(file_paths: list) -> dict:
    """
    Extracts text from multiple PDFs.

    Returns:
        dict: {file_name: extracted_text}
    """
    results = {}
    for path in file_paths:
        try:
            results[os.path.basename(path)] = extract_text_from_pdf(path)
        except Exception as e:
            results[os.path.basename(path)] = f"[ERROR extracting text: {e}]"
    return results


if __name__ == "__main__":
    # quick manual test
    import sys
    if len(sys.argv) > 1:
        print(extract_text_from_pdf(sys.argv[1])[:1000])
    else:
        print("Usage: python extractor.py <path_to_pdf>")
