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


def format_citations(chunks):
    """
    Build a clean, structured citation block from the chunks that were
    actually retrieved - independent of whether the LLM's prose mentions
    them. This guarantees every answer has consistent, reliable source
    attribution, rather than relying on the LLM to remember to cite.
    """
    if not chunks:
        return ""

    insurer_display_names = {
        "star_health": "Star Health",
        "hdfc_ergo": "HDFC Ergo",
        "niva_bupa": "Niva Bupa",
    }

    lines = ["\n\n---\n**Sources:**"]
    seen = set()  # avoid listing the exact same source twice
    for chunk in chunks:
        insurer_name = insurer_display_names.get(chunk["insurer"], chunk["insurer"])
        source_key = (insurer_name, chunk["source_section"])
        if source_key in seen:
            continue
        seen.add(source_key)

        # Trim long source_section labels for readability in the citation
        section_label = chunk["source_section"]
        if len(section_label) > 70:
            section_label = section_label[:70] + "..."

        lines.append(f"- {insurer_name}: {section_label}")

    return "\n".join(lines)


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

Write a clear, direct answer. Do not include a sources or citations list in
your answer text - that will be added separately and automatically.

Context:
{context_text}

Question: {question}

Answer:"""

    return prompt


def answer_question(question, embedding_model, collection, llm, top_k=3):
    """The full RAG pipeline: retrieve -> build prompt -> generate answer
    -> append structured citations."""
    chunks = retrieve_chunks(question, embedding_model, collection, top_k=top_k)

    if not chunks:
        return "I couldn't find any relevant information to answer that question.", []

    prompt = build_prompt(question, chunks)
    response = llm.invoke(prompt)

    full_answer = response.content + format_citations(chunks)

    return full_answer, chunks


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

    # Day 2 — Edge case testing. These are deliberately tricky:
    # off-topic, info genuinely not in the docs, ambiguous, and a
    # trick question with a false premise.
    edge_case_questions = [
        "What's the best pizza topping?",  # totally off-topic
        "Does HDFC Ergo cover treatment on the moon?",  # absurd, not in docs
        "What is the exact claim settlement amount I will receive for a knee surgery?",  # not answerable - depends on specifics not in policy text
        "Does Star Health's policy have a 100% no-questions-asked refund anytime?",  # false premise - tests if it corrects misinformation or just agrees
        "Tell me about dental coverage",  # ambiguous - which insurer?
    ]

    for question in edge_case_questions:
        print(f"\n{'=' * 80}")
        print(f"EDGE CASE: {question}")
        print("-" * 80)

        answer, sources_used = answer_question(question, embedding_model, collection, llm)

        print(f"\nAnswer:\n{answer}")
        print(f"\nSources used: {[(s['insurer'], s['source_section']) for s in sources_used]}")