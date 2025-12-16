# Public Health Guidance Copilot

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**A transparent, deterministic Retrieval-Augmented Generation (RAG) system for grounding public health answers in official documentation.**

---

## Overview

The **Public Health Guidance Copilot** is a specialized NLP system designed to answer questions based strictly on a small, curated set of public health documents (e.g., influenza vaccination guidelines). 

Unlike standard Large Language Models (LLMs) that generate text based on pre-training, this system uses a **Retrieval-Augmented Generation (RAG)** architecture without external APIs (like OpenAI) or model fine-tuning. It retrieves relevant text chunks from local documents and presents them alongside the answer, ensuring 100% explainability and transparency.

## The Problem

Large Language Models are powerful but prone to **hallucination** generating plausible-sounding but factually incorrect information. In the domain of public health, accuracy is critical.

**This project solves that by:**
1.  **Searching official documents first:** It never answers from "memory," only from retrieved context.
2.  **Grounded Generation:** Every answer is derived directly from the provided source text.
3.  **Auditability:** The system displays the exact text chunks used to form the answer.

---

## How It Works (Architecture)

This system follows a strict **Retrieve-then-Generate** pipeline:

```mermaid
graph LR
    A[User Query] --> B(Tokenization)
    B --> C{BM25 Retrieval}
    D[Document Corpus] --> E[Chunking ~700 chars]
    E --> F[Search Index]
    F --> C
    C --> G[Top-K Relevant Chunks]
    G --> H[Response Generation]
    H --> I[Final Answer + Sources]

```

### 1. Data Pipeline **Ingestion:** Input documents are plain `.txt` files located in `data/sample_corpus/`.
* **Chunking:** Documents are split into overlapping segments of approximately **700 characters**. This ensures context is preserved across boundaries.
* **Metadata:** Each chunk retains its origin (Document ID, Position, Title).

### 2. Retrieval Engine (Sparse) We utilize **BM25**, a classic probabilistic information retrieval algorithm, rather than dense vector embeddings.

* **Why BM25?** It is fast, interpretable, and highly effective for exact keyword matching in specific domains like medical guidance.
* **Transparency:** Unlike "black box" neural embeddings, BM25 scoring is easy to debug and explain.

### 3. Query & Response The system scores all chunks against the user's query.
* The **Top-K** chunks are retrieved.
* The answer is constructed using *only* these chunks.

---

## Installation & Usage 
#### Prerequisites:  Python 3.8+, Git

### 1. Clone the Repository```bash
git clone [https://github.com/ksurya14/Public-Health-Guidance-Copilot.git](https://github.com/ksurya14/Public-Health-Guidance-Copilot.git)
cd Public-Health-Guidance-Copilot

```

### 2. Install Dependencies```bash
pip install -r requirements.txt

```

### 3. Build the IndexBefore running the app, process the documents and build the BM25 index:

```bash
# Using Makefile
make index

# OR using python directly
python src/index.py

```

### 4. Run the Interface Launch the Streamlit web app:

```bash
streamlit run app/streamlit_app.py

```

---

## Adding New Knowledge, This system is data-centric. You do not need to change code to add new information.

1. Add new `.txt` files to the `data/sample_corpus/` directory.
2. Re-run the indexing command: `make index`.
3. Restart the Streamlit app.

The system will immediately be aware of the new guidance.

---

## Project Scope & Limitations

### What this project DOES Implements a full RAG pipeline (Ingestion to Retrieval to Generation).

- Uses **sparse retrieval (BM25)** for explainability.

- Provides a UI for side-by-side verification of answers and sources.

- Runs entirely locally.

### What this project does NOT do **No Neural Training:** We do not train or fine-tune a neural network.

 **No External APIs:** We do not call OpenAI, Anthropic, or other paid APIs.

 **No "Magic":** The system does not hallucinate knowledge outside of its provided corpus.

---

## Academic Context

* **Course:** DATA 641 (Natural Language Processing)
* **Concept:** Retrieval-Augmented Generation (RAG) & Information Retrieval
* **Goal:** To demonstrate the separation of *knowledge retrieval* from *answer generation* in a safety-critical domain.

---

## Contributors
* **Team:** Parse and Conquer (Suryateja Konduri, Siva Durga Sai Prasad Buthada, Pravalika Sure, Rohan Ambati)
