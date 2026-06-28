"""
Day 5 — Basic Streamlit chat UI for the Health Insurance Policy Explainer.
Wraps the existing RAG chain (rag_chain.py) in a simple chat interface:
text input, conversation history, and the model's grounded answers.
"""

import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag_chain import answer_question

load_dotenv()


# Cache the heavy resources (embedding model, ChromaDB connection, LLM)
# so they only load ONCE per session, not on every single message.
@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="health_insurance_policies")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    return embedding_model, collection, llm


st.title("Health Insurance Policy Explainer")
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
        with st.spinner("Thinking..."):
            answer, sources = answer_question(user_question, embedding_model, collection, llm)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})