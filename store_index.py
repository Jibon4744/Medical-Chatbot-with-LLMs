"""
store_index.py — One-time script to load PDFs, split into chunks,
generate embeddings, and save them to a local FAISS index.

Usage:
    1. Place your medical PDF files in the  data/  directory.
    2. Run:  python store_index.py
"""

from langchain_community.vectorstores import FAISS
from src.helper import load_pdf_data, split_text, download_huggingface_embeddings

FAISS_INDEX_PATH = "faiss_index"

# ── Load & chunk documents ──────────────────────────────────────────────────
print("Loading PDF files from data/ ...")
extracted_data = load_pdf_data("data/")
print(f"    Loaded {len(extracted_data)} page(s).")

text_chunks = split_text(extracted_data)
print(f"    Split into {len(text_chunks)} chunk(s).")

# ── Initialise embeddings ───────────────────────────────────────────────────
print("Downloading HuggingFace embeddings ...")
embeddings = download_huggingface_embeddings()

# ── Build & save FAISS index locally ────────────────────────────────────────
print(f"Creating FAISS index ...")
docsearch = FAISS.from_documents(
    documents=text_chunks,
    embedding=embeddings,
)

docsearch.save_local(FAISS_INDEX_PATH)
print(f"Indexing complete! FAISS index saved to '{FAISS_INDEX_PATH}/'")
