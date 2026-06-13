"""
index_docs.py — CLI script to ingest and index local documents into ChromaDB.

Usage:
    python index_docs.py            # Index all documents in the documents/ folder
    python index_docs.py --reset    # Clear the existing index and re-index everything

Supports: .txt, .md, .pdf files
"""

import os
import sys
import glob
import shutil
import argparse

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Load environment variables
load_dotenv(override=True)

# ── Configuration ─────────────────────────────────────────────────────────────
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chroma_db")
COLLECTION_NAME = "local_documents"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Supported file extensions and their loaders
SUPPORTED_EXTENSIONS = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
}


def get_document_files(directory: str) -> list[str]:
    """Find all supported document files in the given directory."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True))
    return sorted(files)


def load_documents(file_paths: list[str]):
    """Load documents from file paths using appropriate loaders."""
    all_docs = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        loader_class = SUPPORTED_EXTENSIONS.get(ext)

        if loader_class is None:
            print(f"  [SKIP] Skipping unsupported file: {file_path}")
            continue

        try:
            if loader_class == TextLoader:
                loader = loader_class(file_path, encoding="utf-8")
            else:
                loader = loader_class(file_path)

            docs = loader.load()
            # Add source metadata (relative path for cleaner display)
            rel_path = os.path.relpath(file_path, DOCUMENTS_DIR)
            for doc in docs:
                doc.metadata["source"] = rel_path

            all_docs.extend(docs)
            print(f"  [OK] Loaded: {rel_path} ({len(docs)} page(s))")
        except Exception as e:
            print(f"  [ERR] Error loading {file_path}: {e}")

    return all_docs


def chunk_documents(documents):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    """Create or update the ChromaDB vector store with document chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vector_store


def reset_database():
    """Delete the existing ChromaDB database."""
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        print("[RESET] Existing database cleared.")
    else:
        print("[INFO] No existing database found.")


def main():
    parser = argparse.ArgumentParser(description="Index local documents into ChromaDB for RAG.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the existing ChromaDB database before indexing.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Local Document Indexer for AI Research Agent")
    print("=" * 60)

    # Ensure documents directory exists
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"\n[DIR] Created documents directory: {DOCUMENTS_DIR}")
        print("   Place your .txt, .md, or .pdf files there and run this script again.")
        sys.exit(0)

    # Find document files
    file_paths = get_document_files(DOCUMENTS_DIR)
    if not file_paths:
        print(f"\n[WARN] No supported files found in: {DOCUMENTS_DIR}")
        print("   Supported formats: .txt, .md, .pdf")
        print("   Place your files there and run this script again.")
        sys.exit(0)

    print(f"\n[SCAN] Found {len(file_paths)} file(s) in: {DOCUMENTS_DIR}")

    # Reset database if requested
    if args.reset:
        reset_database()

    # Step 1: Load documents
    print("\n[LOAD] Loading documents...")
    documents = load_documents(file_paths)
    if not documents:
        print("[ERR] No documents were loaded successfully.")
        sys.exit(1)

    # Step 2: Chunk documents
    print(f"\n[CHUNK] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_documents(documents)
    print(f"   Created {len(chunks)} chunk(s) from {len(documents)} document page(s).")

    # Step 3: Embed and store
    print("\n[EMBED] Generating embeddings and storing in ChromaDB...")
    print("   (This may take a moment on first run...)")
    vector_store = create_vector_store(chunks)

    print(f"\n[DONE] Successfully indexed {len(chunks)} chunks into ChromaDB!")
    print(f"   Database location: {CHROMA_DB_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
