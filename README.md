**📌 Problem Statement**
Real estate investment decisions rely heavily on macro-economic indicators such as interest rates, mortgage trends, and housing market data. These insights are scattered across trusted financial and real-estate sources like CNBC and Realtor.com.

Real estate analysts must manually read multiple articles, extract key facts, and summarize market conditions before presenting their findings to portfolio managers. This process is slow, repetitive, and prone to missing important information.

**🎯 Project Objective**

This project builds an AI-powered real estate research assistant that helps analysts find, extract, and summarize relevant market information from trusted sources using Retrieval-Augmented Generation (RAG).

The system allows analysts to ask natural-language questions and receive fact-based, source-grounded answers generated directly from selected financial and real-estate websites.

**🧩 How Retrieval-Augmented Generation (RAG) Works**

                         ┌─────────────────────────┐
                         │   Trusted News Sources   │
                         │ (CNBC, Realtor.com, etc.) │
                         └─────────────┬───────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │     Web Scraper          │
                         │  (WebBaseLoader)         │
                         └─────────────┬───────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │   Text Chunking          │
                         │ (Recursive Splitter)     │
                         └─────────────┬───────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │   Embedding Model        │
                         │ (HuggingFace - GTE)      │
                         └─────────────┬───────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │    Vector Database       │
                         │        (ChromaDB)        │
                         └─────────────┬───────────┘
                                       │
                    User Question ─────┘
                                       ▼
                         ┌─────────────────────────┐
                         │ Semantic Search          │
                         │ (Similarity + Scores)    │
                         └─────────────┬───────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │   Relevant Chunks        │
                         │ (Best source selected)   │
                         └─────────────┬───────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │   Prompt Builder         │
                         │ (Context + Question)     │
                         └─────────────┬───────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │      LLM (Groq / LLaMA)   │
                         │  Generates Final Answer  │
                         └─────────────┬───────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │  Answer + Sources +      │
                         │  Supporting Chunks       │
                         └─────────────────────────┘

  What This Means

Instead of letting the LLM answer from its memory, the system:

* Reads real articles

* Converts them into semantic vectors

* Finds the most relevant text

* Forces the LLM to answer using only that data

This ensures:

* No hallucinations

* Full traceability

* Enterprise-grade research reliability


**🧠 What the System Does**

The tool enables real estate analysts to:

* Ingest articles from trusted sources (e.g., CNBC, Realtor.com)

* Convert them into searchable vector embeddings

* Retrieve the most relevant information for a query

* Generate answers using only the retrieved content

* Display supporting text and source URLs for verification

This allows analysts to quickly prepare research briefs and market summaries for portfolio managers with full transparency and traceability.


<img width="1811" height="842" alt="image" src="https://github.com/user-attachments/assets/3f9f632d-be5f-40f0-97c9-361d45d6aaab" />
