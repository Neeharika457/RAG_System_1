import streamlit as st
from rag import process_urls, generate_answer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Real Estate Research Tool",
    page_icon="🏠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f7f9fc;
}
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #1f2937;
}
.subtitle {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 30px;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}
.answer-box {
    background: linear-gradient(135deg, #e0f2fe, #f0f9ff);
    padding: 25px;
    border-radius: 16px;
    font-size: 18px;
    line-height: 1.6;
}
.source-link a {
    text-decoration: none;
    color: #2563eb;
    font-weight: 500;
}
.chunk-box {
    background-color: #f9fafb;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 12px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🔗 Research Sources")
    st.caption("Provide up to 3 URLs")

    urls = []
    for i in range(3):
        url = st.text_input(f"URL {i+1}")
        if url:
            urls.append(url)

    process_btn = st.button("🚀 Process URLs", use_container_width=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">Real Estate Research Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions and get fact-based answers from trusted sources</div>', unsafe_allow_html=True)

# ---------------- PROCESS URLS ----------------
if process_btn:
    if not urls:
        st.warning("Please provide at least one URL")
    else:
        with st.spinner("🔍 Scraping and indexing content..."):
            docs = process_urls(urls)
        st.success(f"✅ Processed {len(docs)} content chunks")

# ---------------- QUESTION INPUT ----------------
st.markdown("### ❓ Ask Your Question")
query = st.text_input(
    "",
    placeholder="e.g. What is the Fed’s outlook on interest rates in 2025?"
)

# ---------------- ANSWER SECTION ----------------
if query:
    with st.spinner("🧠 Analyzing sources..."):
        answer, source, summaries, chunks = generate_answer(query)

    # Answer Card
    st.markdown("## ✅ Answer")
    st.markdown(
        f'<div class="answer-box">{answer}</div>',
        unsafe_allow_html=True
    )

    # Source Card
    if source:
        st.markdown("## 🌐 Source")
        st.markdown(
            f'<div class="card source-link"><a href="{source}" target="_blank">{source}</a></div>',
            unsafe_allow_html=True
        )

    # Relevant Chunks
    st.markdown("## 📄 Relevant Chunks")
    for i, doc in enumerate(chunks, 1):
        st.markdown(
            f"""
            <div class="chunk-box">
                <strong>Chunk {i}</strong><br>
                {doc.page_content[:500]}...
            </div>
            """,
            unsafe_allow_html=True
        )
