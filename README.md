# Generative AI Mini Projects (Hugging Face Ecosystem)

This repository contains **three beginner-friendly Generative AI projects** built using the Hugging Face ecosystem and Python.

The goal of these projects is to understand how **LLMs, embeddings, and retrieval systems work together in modern AI applications.**

---

# Project 1: Creative Text Generation

This project demonstrates how to generate creative text using the **DistilGPT-2** language model.

## Concepts Used
- Transformer models
- Tokenization
- Autoregressive text generation
- Sampling techniques

## Techniques Applied
- Top-K Sampling
- Top-P Sampling
- Temperature scaling
- Repetition penalty

## Workflow
1. Input prompt is provided.
2. Text is tokenized.
3. Tokens are converted into embeddings.
4. The transformer model predicts the next token.
5. Tokens are generated iteratively until the sentence ends.

---

# Project 2: Sentence Embeddings & Semantic Similarity

This project demonstrates how to convert sentences into **dense numerical vectors** using Sentence Transformers.

## Model Used
`all-MiniLM-L6-v2`

## Concepts Used
- Sentence embeddings
- Semantic similarity
- Vector representations
- Cosine similarity

## Workflow
1. Sentences are provided as input.
2. Model generates embeddings for each sentence.
3. Each sentence is represented as a **384-dimensional vector**.
4. Cosine similarity is used to measure semantic similarity between sentences.

## Example Use Cases
- Semantic search
- Duplicate detection
- Recommendation systems
- Document clustering

---

# Project 3: Chat with Your PDF (RAG Pipeline)

This project builds a simple **Retrieval-Augmented Generation (RAG)** chatbot that answers questions from a PDF document.

## Technologies Used

- Hugging Face Transformers
- Sentence Transformers
- FAISS Vector Database
- PyPDF
- FLAN-T5 Language Model

## Architecture

PDF → Text Extraction → Chunking → Embeddings → Vector Store → Query Retrieval → LLM Answer Generation

## Workflow

1. Load PDF document.
2. Split text into overlapping chunks.
3. Generate embeddings for each chunk.
4. Store embeddings in a FAISS vector database.
5. Convert user query into embeddings.
6. Retrieve the most relevant chunks.
7. Pass retrieved context to the language model.
8. Generate final answer.

## Key Concepts

- Retrieval-Augmented Generation (RAG)
- Vector databases
- Embedding similarity search
- Context-based LLM responses

---

# Installation

```bash
pip install transformers torch sentence-transformers faiss-cpu pypdf scikit-learn
