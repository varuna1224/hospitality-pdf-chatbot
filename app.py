"""
app.py
------
STEP 8 / UI: STREAMLIT

Hospitality PDF Chatbot - full pipeline:
Documents(PDF only) -> extractor.py -> chunking.py -> embeddings.py
-> vector.py -> retrieval.py -> llm.py -> Response to guest
"""

import os
import streamlit as st

from extractor import extract_text_from_pdf, is_pdf
from chunking import chunk_multiple_documents
from vector import VectorStore
from retrieval import retrieve_relevant_chunks, build_context_block
from llm import generate_answer

st.set_page_config(page_title="🏨 Hospitality PDF Chatbot", page_icon="🏨", layout="wide")

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Session state
# ---------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
    st.session_state.vector_store.load()  # load previously indexed PDFs, if any

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

# ---------------------------------------------------------------
# Sidebar: Upload PDFs (PDF only - hard restriction)
# ---------------------------------------------------------------
st.sidebar.title("🏨 Hospitality Knowledge Base")
st.sidebar.caption("Upload hotel PDFs only: tariffs, SOPs, guest policies, menus, banquet packages.")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF document(s)",
    type=["pdf"],           # hard restriction at the widget level
    accept_multiple_files=True,
)

if uploaded_files:
    new_docs = {}
    for uploaded_file in uploaded_files:
        if not is_pdf(uploaded_file.name):
            st.sidebar.error(f"❌ Rejected '{uploaded_file.name}' - only PDF files are accepted.")
            continue

        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name not in st.session_state.indexed_files:
            with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                new_docs[uploaded_file.name] = extract_text_from_pdf(save_path)

    if new_docs:
        with st.spinner("Chunking + embedding + indexing..."):
            chunks = chunk_multiple_documents(new_docs)
            st.session_state.vector_store.add_chunks(chunks)
            st.session_state.vector_store.save()
            st.session_state.indexed_files.update(new_docs.keys())
        st.sidebar.success(f"✅ Indexed: {', '.join(new_docs.keys())}")

if st.session_state.indexed_files:
    st.sidebar.markdown("**📄 Indexed documents:**")
    for fname in sorted(st.session_state.indexed_files):
        st.sidebar.markdown(f"- {fname}")

top_k = st.sidebar.slider("Chunks to retrieve (top_k)", 2, 8, 4)

if st.sidebar.button("🗑️ Clear knowledge base"):
    st.session_state.vector_store = VectorStore()
    st.session_state.indexed_files = set()
    st.session_state.chat_history = []
    st.sidebar.info("Knowledge base cleared.")

# ---------------------------------------------------------------
# Main chat window
# ---------------------------------------------------------------
st.title("🏨 Hospitality PDF Chatbot")
st.caption("Ask about check-in/out times, room tariffs, amenities, banquet packages, SOPs, menus - "
           "answered strictly from your uploaded PDF documents.")

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

user_question = st.chat_input("Ask a question about the hotel (e.g. 'What time is check-out?')")

if user_question:
    st.session_state.chat_history.append(("user", user_question))
    with st.chat_message("user"):
        st.markdown(user_question)

    if st.session_state.vector_store.is_empty():
        answer = "Please upload at least one PDF document first so I have information to answer from."
    else:
        with st.spinner("Searching documents and drafting a reply..."):
            chunks = retrieve_relevant_chunks(user_question, st.session_state.vector_store, top_k=top_k)
            context = build_context_block(chunks)
            answer = generate_answer(user_question, context)

    st.session_state.chat_history.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)
