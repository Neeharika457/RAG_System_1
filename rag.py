# RAG Architecture
# Docs -> Chunks(splitter)-> VectorDB -> Retrieval
# -> Prompt -> LLM

from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------------- CONSTANTS ----------------
CHUNK_SIZE = 800                  # FIX: smaller chunks
CHUNK_OVERLAP = 200               # FIX: overlap for fact retention
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None
# Above globals avoid re-initialization on every call

# ---------------- PROMPTS ----------------
PROMPT = PromptTemplate(
    template="""
Answer the question using ONLY the context below.
Look carefully for explicit factual statements.
If the answer is directly stated, extract it.
If it is truly missing, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

CHUNK_SUMMARY_PROMPT = PromptTemplate(
    template="""
Summarize the following text in 1–2 concise sentences.
Focus only on factual information.

Text:
{text}

Summary:
""",
    input_variables=["text"],
)

# ---------------- INITIALIZATION ----------------
def initialize_components():
    """
    Initialize LLM and Vector DB only once
    """
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=400
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )

# ---------------- INGESTION ----------------
def process_urls(urls):
    """
    Scrapes data from URLs and stores embeddings in vector DB
    """
    initialize_components()

    # Reset DB for new research session
    vector_store.reset_collection()

    loader = WebBaseLoader(
        urls,
        header_template={"User-Agent": "Mozilla/5.0"}
    )

    data = loader.load()

    # FIX: overlap added to preserve factual sentences
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    docs = splitter.split_documents(data)

    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)

    return docs

# ---------------- CHUNK SUMMARIZATION ----------------
def summarize_chunks(docs):
    """
    Generates short factual summaries for chunks
    """
    summaries = []

    summary_chain = (
        CHUNK_SUMMARY_PROMPT
        | llm
        | StrOutputParser()
    )

    for doc in docs:
        summary = summary_chain.invoke({"text": doc.page_content})
        summaries.append(summary)

    return summaries

# ---------------- SOURCE SCORING ----------------
def get_most_relevant_source(results):
    """
    Finds the source URL with lowest distance score
    """
    source_scores = defaultdict(list)

    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        source_scores[source].append(score)

    best_source = min(
        source_scores.items(),
        key=lambda item: min(item[1])
    )

    return best_source[0], min(best_source[1])

# ---------------- ANSWER GENERATION ----------------
def generate_answer(query: str):
    """
    Retrieves relevant chunks, summarizes them,
    and generates a final RAG answer
    """
    if vector_store is None:
        raise RuntimeError("Vector DB not initialized")

    # FIX: Higher recall for factual questions
    results = vector_store.similarity_search_with_score(
        query,
        k=10
    )

    if not results:
        return "I don't know", None, [], []

    # 🔥 FIX: DO NOT FILTER BY SOURCE
    # Keep top-N chunks directly
    top_docs = [doc for doc, _ in results[:6]]

    # Build context
    context = "\n\n".join(doc.page_content for doc in top_docs)

    rag_chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query)

    # Chunk summaries (preserved)
    chunk_summaries = summarize_chunks(top_docs)

    # Best source (for display only)
    best_source = top_docs[0].metadata.get("source", "unknown")

    return answer, best_source, chunk_summaries, top_docs


# THe below code is correct but doesn't generate or give best source

# def generate_answer(query):
#     if not vector_store:
#         raise RuntimeError("Vector DB not initialized")
#
#     retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#
#     rag_chain = (
#         {
#             "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
#             "question": RunnablePassthrough(),
#         }
#         | PROMPT
#         | llm
#         | StrOutputParser()
#     )
#
#     answer = rag_chain.invoke(query)
#     return answer



if __name__=="__main__":
    print("Script Started")
    urls=["https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
          "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
          ]
    process_urls(urls)

    # # Below code from https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
    # results = vector_store.similarity_search(
    #     "30 year mortgage rate",
    #     k=2
    # )
    # print(results)

    answer, source, summaries, chunks = generate_answer(
        "Tell me what was the 30 year mortgage rate along with the date?"
    )

    print("\nANSWER:\n", answer)

    print("\nMOST RELEVANT SOURCE:")
    print(source)
    # print(f"Similarity score: {score:.4f}")

    print("\nRELEVANT CHUNKS:")
    for i, doc in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:200])

    print("\nCHUNK SUMMARIES:")
    for i, summary in enumerate(summaries, 1):
        print(f"{i}. {summary}")