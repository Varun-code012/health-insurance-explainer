"""
Day 5 — Generate embeddings for all chunks using sentence-transformers.
Loads the chunked, tagged dataset from build_chunks.py, embeds each
chunk's text, and confirms the embedding shape before moving to
ChromaDB storage (Day 6).
"""

from sentence_transformers import SentenceTransformer
from build_chunks import chunk_all_documents


def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Takes a list of chunk dicts (each with a "text" key), generates an
    embedding for each chunk's text, and attaches it under "embedding".
    Returns the same list of dicts, now with embeddings included.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


if __name__ == "__main__":
    print("=== Building chunks ===")
    chunks = chunk_all_documents()
    print(f"Total chunks: {len(chunks)}\n")

    print("=== Generating embeddings ===")
    chunks_with_embeddings = generate_embeddings(chunks)

    # Sanity check: confirm embedding shape and a sample
    sample = chunks_with_embeddings[0]
    print(f"\nSample chunk insurer: {sample['insurer']}")
    print(f"Sample chunk section_type: {sample['section_type']}")
    print(f"Embedding shape: {sample['embedding'].shape}")
    print(f"Embedding dtype: {sample['embedding'].dtype}")
    print(f"First 5 values: {sample['embedding'][:5]}")

    print(f"\nAll {len(chunks_with_embeddings)} chunks now have embeddings attached.")