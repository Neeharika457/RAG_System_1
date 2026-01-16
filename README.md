**📌 Problem Statement**
Real estate investment decisions rely heavily on macro-economic indicators such as interest rates, mortgage trends, and housing market data. These insights are scattered across trusted financial and real-estate sources like CNBC and Realtor.com.

Real estate analysts must manually read multiple articles, extract key facts, and summarize market conditions before presenting their findings to portfolio managers. This process is slow, repetitive, and prone to missing important information.

**🎯 Project Objective**

This project builds an AI-powered real estate research assistant that helps analysts find, extract, and summarize relevant market information from trusted sources using Retrieval-Augmented Generation (RAG).

The system allows analysts to ask natural-language questions and receive fact-based, source-grounded answers generated directly from selected financial and real-estate websites.

**🧩 How Retrieval-Augmented Generation (RAG) Works**

first take the doc, then split into chunks then convert them into embeddings then store in vector DB
Now user asks a query, do semantic search (Similarity score) and retrieve relavent chunks then rag_chain = {context: relevant_chunks, question: query} | prompt | llm, then we get the answer

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


    Step	  Component  	                 Action
1. Input	    URLs    	           Scrapes text from web pages.
2. Storage	  Vector               Database	Stores text chunks as searchable "meaning" vectors.
3. Query	    User Question	       Searches the database for the most relevant text snippets.
4. Analysis	  LLM (Llama 3.3)	     Reads snippets and extracts the specific answer (e.g., "6.8% on Dec 20").
5. Output     Final Response	     Provides the Answer + Source Link + Summaries of what it read.


**Phase 1: The Knowledge Ingestion (Offline)**
Before you can ask a question, the system must "learn" the documents.

* Document Loading: The WebBaseLoader pulls raw HTML/text from the CNBC URLs.

* Chunking (Splitting): The RecursiveCharacterTextSplitter breaks long articles into 800-character pieces. This ensures the LLM doesn't get overwhelmed and that search results are specific.

* Embedding: Each chunk is passed through the HuggingFaceEmbeddings model. This turns text into a long list of numbers (a vector) that represents the meaning of that text.

* Vector DB Storage: These vectors are stored in ChromaDB. This acts as a searchable "map" where chunks with similar meanings are placed physically close to each other.

**Phase 2: The Retrieval (The Search)**
When the user asks, "What was the 30-year mortgage rate?":

* Query Embedding: The system converts the user's question into the same mathematical "fingerprint" (vector) used for the chunks.

* Semantic Search: The system looks into the Vector DB to find the chunks whose vectors are most similar to the question's vector.

* Similarity Scoring: It ranks these chunks. In your code, k=10 retrieves the 10 closest matches based on their distance score.

**Phase 3: The RAG Chain (The Generation)**
This is where the "Augmented" part happens. Instead of just giving the LLM the question, you give it a "Cheat Sheet."

* Context Construction: The system takes the top retrieved chunks and pastes them together into one block of text called the context.

* The Prompt: The question and the context are inserted into your PROMPT template:

* "Answer the question using ONLY the context below: [Context] ... Question: [User Query]"

* The LLM Execution: The LLM (Llama 3.3) reads the context, finds the specific facts (like "6.8%"), and writes a natural language answer.

* The Output: The StrOutputParser cleans up the text and delivers the final answer to you.



<img width="1811" height="842" alt="image" src="https://github.com/user-attachments/assets/3f9f632d-be5f-40f0-97c9-361d45d6aaab" />
