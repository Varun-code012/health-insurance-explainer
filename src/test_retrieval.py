"""
Day 7 — Test retrieval quality using the 10 questions from Week 1.
For each question, embed it the same way as the chunks, query ChromaDB
for the most similar chunks, and print results to manually check:
did it return the right insurer? the right section? relevant content?

Also detects if a question names a specific insurer (e.g. "HDFC Ergo",
"Star Health", "Niva Bupa") and filters the search to that insurer only,
since otherwise a different insurer's chunk can outrank the right one
just by being more semantically similar in wording.
"""

import chromadb
from sentence_transformers import SentenceTransformer


TEST_QUESTIONS = [
    "What is the waiting period for pre-existing diseases under HDFC Ergo's my:health Medisure policy?",
    "Does Star Health's Comprehensive policy cover AYUSH treatments like Ayurveda or Homeopathy?",
    "What is excluded under cosmetic or plastic surgery in the HDFC Ergo policy?",
    "How many days after discharge can I claim post-hospitalization expenses under Star Health?",
    "Is maternity covered under Niva Bupa's Health Premia policy, and if so, what are the conditions?",
    "What is the co-payment percentage for senior citizens above 60-61 years under Star Health's policy?",
    "Compare the pre-existing disease waiting period across all three insurers - which one has the shortest?",
    "What documents do I need to submit for a reimbursement claim under HDFC Ergo?",
    "Does the Star Health policy cover air ambulance charges, and what is the maximum limit?",
    "What is the free look period for a new individual health policy, and what happens if I cancel during it?",
]


# Keywords that signal a question is asking about ONE specific insurer.
# If a question's wording matches one of these, we filter retrieval to
# only search that insurer's chunks - this is the fix for cases where a
# different insurer's chunk outranks the right one by wording similarity.
INSURER_KEYWORDS = {
    "star_health": ["star health", "star comprehensive"],
    "hdfc_ergo": ["hdfc ergo", "hdfc", "medisure"],
    "niva_bupa": ["niva bupa", "health premia"],
}


def detect_insurer_filter(question):
    """Check if the question names a specific insurer. Returns the
    insurer key to filter by, or None if no specific insurer is named
    (e.g. comparison questions across all three)."""
    question_lower = question.lower()
    for insurer, keywords in INSURER_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            return insurer
    return None


def test_retrieval(db_path="./chroma_db", collection_name="health_insurance_policies", top_k=3):
    """Run every test question through ChromaDB and print the top results."""
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)

    print(f"Collection has {collection.count()} items.\n")
    print("=" * 80)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\nQ{i}: {question}")

        insurer_filter = detect_insurer_filter(question)
        if insurer_filter:
            print(f"  [Filtering to insurer: {insurer_filter}]")
        else:
            print("  [No specific insurer detected - searching all]")
        print("-" * 80)

        query_embedding = model.encode([question])[0].tolist()

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if insurer_filter:
            query_kwargs["where"] = {"insurer": insurer_filter}

        results = collection.query(**query_kwargs)

        for rank, (doc, metadata, distance) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1
        ):
            print(f"\n  Rank {rank} (distance: {distance:.3f})")
            print(f"  Insurer: {metadata['insurer']} | Section: {metadata['section_type']} | Source: {metadata['source_section']}")
            print(f"  Text preview: {doc[:150]}...")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    test_retrieval()