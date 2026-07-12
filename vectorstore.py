"""
vectorstore.py
--------------
Step 2 of the Drive Wise pipeline: embed brochure chunks and store them in a
local FAISS index.

Requires a Google Gemini API key set as the environment variable
GOOGLE_API_KEY (get one from https://aistudio.google.com/apikey).

Usage:
    python vectorstore.py --build   # (re)builds the index from data/*.txt or *.pdf
"""

import argparse
import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ingest import BrochureMeta, chunk_brochure

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "gemini-embedding-001"

# Map filenames in data/ -> (brand, model, version). Extend this as you add
# more brochures.
BROCHURE_REGISTRY = {
    "hyundai_creta_2026_brochure.txt": BrochureMeta(
        brand="Hyundai", model="Creta", doc_version="2026-v2.0-real"
    ),
    "tata_nexon_2026_brochure.txt": BrochureMeta(
        brand="Tata", model="Nexon", doc_version="2026-v1.0-real"
    ),
}


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Export it before running, e.g.\n"
            "  export GOOGLE_API_KEY='your-key-here'"
        )
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def build_index(data_dir: str = "data") -> FAISS:
    all_docs: List[Document] = []
    for filename, meta in BROCHURE_REGISTRY.items():
        path = Path(data_dir) / filename
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        docs = chunk_brochure(str(path), meta)
        print(f"  [ok] {filename}: {len(docs)} chunks ({meta.brand} {meta.model})")
        all_docs.extend(docs)

    if not all_docs:
        raise RuntimeError("No brochure documents found to index. Add files to data/.")

    embeddings = get_embeddings()
    store = FAISS.from_documents(all_docs, embeddings)
    store.save_local(INDEX_DIR)
    print(f"\nSaved FAISS index with {len(all_docs)} chunks to ./{INDEX_DIR}/")
    return store


def load_index() -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build/rebuild the FAISS index")
    args = parser.parse_args()

    if args.build:
        print("Building FAISS index from brochures in data/ ...")
        build_index()
    else:
        print("Nothing to do. Pass --build to create the index.")