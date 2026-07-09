# 🏥 Health Insurance Policy Explainer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://health-insurance-explainer-qbfbgv9aafdw6t8yhfd7ef.streamlit.app)

A RAG (Retrieval-Augmented Generation) chatbot...
A RAG (Retrieval-Augmented Generation) chatbot that answers plain-language questions about health insurance policies — with exact source citations from the actual policy documents.

Built across three real Indian health insurance policies:
- **Star Health** Comprehensive Insurance Policy
- **HDFC Ergo** my:health Medisure Super Top Up
- **Niva Bupa** Health Premia

---

## 🎯 Problem this solves

Health insurance policy documents are dense, long, and written in legal language. Most people don't read them — and find out what's excluded only when a claim gets rejected. This tool lets anyone ask plain-language questions and get grounded, cited answers directly from the policy wording.

---

## 💬 Demo

![App Screenshot](assets/screenshot.png)

> *Ask "is pregnancy covered??" — get a clear, insurer-by-insurer breakdown with sources cited.*

---

## 🏗️ Architecture

```
PDF Policy Documents
        ↓
  load_documents.py     ← PDF loading + header/footer cleaning
        ↓
  build_chunks.py       ← Section-aware chunking (per-insurer regex logic)
        ↓
  store_chunks.py       ← sentence-transformers embeddings → ChromaDB
        ↓
  rag_chain.py          ← Retrieve chunks → build prompt → Groq LLM → answer
        ↓
  app.py                ← Streamlit chat UI
```

**Key design decisions:**
- **Section-aware chunking** (not fixed-size) — each chunk is a complete, meaningful policy clause. Critical for legal text where cutting mid-clause changes meaning.
- **Per-insurer chunking logic** — Star Health, HDFC Ergo, and Niva Bupa each use different document structures; the chunker adapts to each one rather than forcing a single regex.
- **Insurer-aware retrieval filtering** — detects the insurer named in a question and filters ChromaDB search to that insurer's chunks only, eliminating cross-insurer contamination.
- **Structured citations** — source attribution is appended programmatically after every answer, guaranteed, independent of whether the LLM mentions sources in prose.
- **Multi-insurer prompt organization** — when a question spans all three insurers, the LLM is instructed to organize the answer by insurer rather than blending facts.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--db-orange)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-all--MiniLM--L6--v2-yellow)

---

## 📊 Benchmark Results

Evaluated against a 10-question test set covering waiting periods, exclusions, benefits, claims process, and cross-insurer comparisons:

| Metric | Result |
|--------|--------|
| Retrieval accuracy (pure retrieval) | 9 / 10 |
| End-to-end accuracy (retrieval + generation) | 10 / 10 |
| Hallucination on off-topic questions | 0 / 5 tested |
| Typo tolerance | ✅ Confirmed |

---

## 🚀 Setup & Run Locally

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/Varun-code012/health-insurance-explainer.git
cd health-insurance-explainer

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# 5. Build the vector database (only needed once)
python src/store_chunks.py

# 6. Run the app
streamlit run src/app.py
```

---

## 📁 Project Structure

```
health-insurance-explainer/
├── data/
│   ├── hdfc_ergo/          ← HDFC Ergo policy PDF
│   ├── niva_bupa/          ← Niva Bupa policy PDF
│   └── star_health/        ← Star Health policy PDF
├── notebooks/
│   ├── chunking_strategy.md   ← Week 1 document analysis notes
│   └── test_questions.md      ← 10-question evaluation benchmark
├── src/
│   ├── load_documents.py      ← PDF loading and cleaning
│   ├── build_chunks.py        ← Section-aware chunking + metadata tagging
│   ├── generate_embeddings.py ← Embedding generation
│   ├── store_chunks.py        ← ChromaDB ingestion pipeline
│   ├── rag_chain.py           ← Core RAG function + prompt engineering
│   ├── test_retrieval.py      ← Retrieval evaluation script
│   └── app.py                 ← Streamlit chat UI
├── assets/
│   └── screenshot.png         ← App screenshot for README
├── requirements.txt
└── README.md
```

---

## 🔍 Sample Questions to Try

- *"What is the waiting period for pre-existing diseases under HDFC Ergo?"*
- *"Does Star Health cover air ambulance charges, and what's the limit?"*
- *"Is maternity covered under Niva Bupa's Health Premia policy?"*
- *"Compare the pre-existing disease waiting period across all three insurers"*
- *"What documents do I need to submit for a reimbursement claim under HDFC Ergo?"*

---

## 👤 Author

**Varun** — 2nd year Engineering student, Ballari, Karnataka
- GitHub: [@Varun-code012](https://github.com/Varun-code012)

---

## ⚠️ Disclaimer

This tool is for informational purposes only. Always refer to your actual policy document and consult your insurer before making any claim or coverage decisions.
