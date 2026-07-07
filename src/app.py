"""
Day 5 — Basic Streamlit chat UI for the Health Insurance Policy Explainer.
Wraps the existing RAG chain (rag_chain.py) in a simple chat interface:
text input, conversation history, and the model's grounded answers.
"""

import os
import sys
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag_chain import answer_question

load_dotenv()

# Auto-build ChromaDB on first launch (e.g. on Streamlit Cloud where
# chroma_db/ doesn't exist in the repo). This runs once, takes a few
# minutes, and is skipped on all subsequent runs.
if not os.path.exists("./chroma_db"):
    with st.spinner("Setting up the database for the first time... this takes a few minutes."):
        src_path = os.path.join(os.path.dirname(__file__), ".")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from store_chunks import main as build_db
        build_db()


# Cache the heavy resources (embedding model, ChromaDB connection, LLM)
# so they only load ONCE per session, not on every single message.
@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="health_insurance_policies")
    
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    
    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
)
    return embedding_model, collection, llm


st.set_page_config(page_title="Health Insurance Policy Explainer", page_icon="🏥")

# --- Sidebar: app context + insurer selector ---
with st.sidebar:
    st.header("About this assistant")
    st.markdown(
        "This assistant answers questions about three health insurance "
        "policies using RAG (Retrieval-Augmented Generation):\n\n"
        "- **Star Health** Comprehensive Insurance Policy\n"
        "- **HDFC Ergo** my:health Medisure Super Top Up\n"
        "- **Niva Bupa** Health Premia\n\n"
        "All answers are grounded in the actual policy documents, with "
        "sources cited below each response."
    )

    st.divider()

    st.subheader("Focus on one insurer (optional)")
    insurer_choice = st.radio(
        "If you select one, your question will only be answered using that insurer's policy.",
        options=["Auto-detect", "Star Health", "HDFC Ergo", "Niva Bupa"],
        index=0,
    )

    insurer_map = {
        "Star Health": "star_health",
        "HDFC Ergo": "hdfc_ergo",
        "Niva Bupa": "niva_bupa",
    }
    insurer_override = insurer_map.get(insurer_choice)  # None if "Auto-detect"

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("🏥 Health Insurance Policy Explainer")
st.caption("Ask questions about Star Health, HDFC Ergo, or Niva Bupa health insurance policies.")

embedding_model, collection, llm = load_resources()

# Initialize conversation history in Streamlit's session state, so it
# persists across reruns (Streamlit reruns the whole script on every
# interaction, so without session_state, history would reset each time).
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages in the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box at the bottom of the page
user_question = st.chat_input("Ask a question about your health insurance policy...")

if user_question:
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate and show the assistant's answer
    with st.chat_message("assistant"):
        if insurer_override:
            st.caption(f"🔒 Searching only within: **{insurer_choice}**")

        with st.spinner("Thinking..."):
            answer, sources = answer_question(
                user_question, embedding_model, collection, llm,
                insurer_override=insurer_override,
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})