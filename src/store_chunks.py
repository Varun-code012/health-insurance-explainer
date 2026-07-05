"""
Day 6 — Store all embedded, tagged chunks in a persistent ChromaDB collection.
Creates a local vector database that survives between script runs, so we
don't need to re-embed everything every time we want to search.
"""

import chromadb
from generate_embeddings import generate_embeddings
from build_chunks import chunk_all_documents


def build_chroma_collection(chunks, db_path="./chroma_db", collection_name="health_insurance_policies"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    embeddings = [chunk["embedding"].tolist() for chunk in chunks]
    metadatas = [
        {
            "insurer": chunk["insurer"],
            "section_type": chunk["section_type"],
            "source_section": chunk["source_section"],
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return collection


def main():
    print("=== Building chunks ===")
    chunks = chunk_all_documents()
    print(f"Total chunks: {len(chunks)}\n")

    print("=== Generating embeddings ===")
    chunks = generate_embeddings(chunks)

    print("\n=== Storing in ChromaDB ===")
    collection = build_chroma_collection(chunks)

    count = collection.count()
    print(f"\nChromaDB collection now contains {count} items.")
    print(f"Expected: {len(chunks)} items.")

    if count == len(chunks):
        print("✅ Counts match — storage successful.")
    else:
        print("⚠️ Counts do NOT match — something went wrong.")


if __name__ == "__main__":
    main()