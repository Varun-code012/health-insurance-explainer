"""
setup.py — Builds the ChromaDB vector database from scratch if it doesn't
already exist. Run this once before starting the app, or let app.py call
it automatically on first launch (e.g. on Streamlit Cloud where chroma_db/
doesn't exist in the repo).
"""

import os
import sys

def build_database():
    """Full pipeline: load PDFs -> chunk -> embed -> store in ChromaDB."""
    print("ChromaDB not found. Building from scratch...")
    print("This will take a few minutes on first run.\n")

    # Add src/ to path so imports work regardless of where this is called from
    src_path = os.path.join(os.path.dirname(__file__), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from store_chunks import main as run_store
    run_store()
    print("\nDatabase built successfully.")


if __name__ == "__main__":
    build_database()