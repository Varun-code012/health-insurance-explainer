"""
Day 1 (Week 3) — The core RAG function.
Question in -> retrieve relevant chunks from ChromaDB -> build a prompt
with those chunks as context -> send to Groq -> return a grounded answer
with source citations.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


# Reused from Day 7 testing - detects if a question names a specific
# insurer, so we can filter retrieval to just that insurer's chunks.
INSURER_KEYWORDS = {
    "star_health": ["star health", "star comprehensive"],
    "hdfc_ergo": ["hdfc ergo", "hdfc", "medisure"],
    "niva_bupa": ["niva bupa", "health premia"],
}


def detect_insurer_filter(question):
    """Check if the question names a specific insurer."""
    question_lower = question.lower()
    for insurer, keywords in INSURER_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            return insurer
    return None


def retrieve_chunks(question, embedding_model, collection, top_k=3):
    """Embed the question, query ChromaDB, return the top matching chunks
    along with their metadata (insurer, section_type, source_section)."""
    insurer_filter = detect_insurer_filter(question)

    query_embedding = embedding_model.encode([question])[0].tolist()

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if insurer_filter:
        query_kwargs["where"] = {"insurer": insurer_filter}

    results = collection.query(**query_kwargs)

    chunks = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "insurer": metadata["insurer"],
            "section_type": metadata["section_type"],
            "source_section": metadata["source_section"],
        })

    return chunks


def build_prompt(question, chunks):
    """Build the prompt sent to the LLM: instructions + retrieved context
    + the user's question."""
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk['insurer']} - {chunk['source_section']}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a health insurance policy assistant. Answer the user's question
using ONLY the information in the context below. Do not use any outside knowledge.

If the context does not contain enough information to answer the question,
say so clearly instead of guessing.

When you answer, mention which insurer and source section the information
came from (e.g. "According to Star Health's Section 1.G...").

Context:
{context_text}

Question: {question}

Answer:"""

    return prompt


def answer_question(question, embedding_model, collection, llm, top_k=3):
    """The full RAG pipeline: retrieve -> build prompt -> generate answer."""
    chunks = retrieve_chunks(question, embedding_model, collection, top_k=top_k)

    if not chunks:
        return "I couldn't find any relevant information to answer that question.", []

    prompt = build_prompt(question, chunks)
    response = llm.invoke(prompt)

    return response.content, chunks


if __name__ == "__main__":
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="health_insurance_policies")

    print("Connecting to Groq...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    # Test with a couple of questions
    test_questions = [
        "What is excluded under cosmetic or plastic surgery in the HDFC Ergo policy?",
        "Does the Star Health policy cover air ambulance charges, and what is the maximum limit?",
    ]

    for question in test_questions:
        print(f"\n{'=' * 80}")
        print(f"Question: {question}")
        print("-" * 80)

        answer, sources_used = answer_question(question, embedding_model, collection, llm)

        print(f"\nAnswer:\n{answer}")
        print(f"\nSources used: {[(s['insurer'], s['source_section']) for s in sources_used]}")