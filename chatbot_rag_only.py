import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# ------------------------------
# 1. Configuration
# ------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Set your OpenRouter API key
MODEL_NAME = "openai/gpt-oss-120b:free"  # Free tier model

# Initialize OpenRouter client (OpenAI-compatible)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ------------------------------
# 2. RAG Components
# ------------------------------
RAG_DOC_DIR = "rag_doc"          # folder containing PDFs
CACHE_DIR = ".rag_cache"         # store FAISS index + chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

@st.cache_resource
def load_rag():
    """Load PDFs, create chunks, build FAISS index (cached)."""
    rag_path = Path(RAG_DOC_DIR)
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(exist_ok=True)

    index_file = cache_path / "faiss.index"
    chunks_file = cache_path / "chunks.pkl"
    metadata_file = cache_path / "metadata.pkl"

    # If cache exists, load it
    if index_file.exists() and chunks_file.exists():
        index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        return index, chunks, metadata, embedder

    # Otherwise build from scratch
    if not rag_path.exists() or not any(rag_path.glob("*.pdf")):
        st.warning(f"No PDF files found in '{RAG_DOC_DIR}'. RAG will be disabled.")
        return None, [], [], None

    # Load all PDFs
    documents = []
    for pdf_file in rag_path.glob("*.pdf"):
        reader = PdfReader(pdf_file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        if text.strip():
            documents.append({"source": pdf_file.name, "text": text})

    if not documents:
        st.warning("No text extracted from PDFs. RAG disabled.")
        return None, [], [], None

    # Split into chunks (simple recursive split)
    chunks = []
    metadata = []
    chunk_size = 500   # characters
    overlap = 50
    for doc in documents:
        text = doc["text"]
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 100:   # ignore very small fragments
                chunks.append(chunk)
                metadata.append({"source": doc["source"], "offset": i})

    if not chunks:
        st.warning("No valid chunks created. RAG disabled.")
        return None, [], [], None

    # Create embeddings
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    # Save cache
    faiss.write_index(index, str(index_file))
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

    return index, chunks, metadata, embedder

def retrieve_context(query: str, index, chunks, metadata, embedder, top_k=3) -> str:
    """Return top-k relevant text chunks with their source filenames."""
    if index is None or not chunks:
        return ""
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec.astype(np.float32), top_k)
    retrieved_chunks = []
    for i in indices[0]:
        if i < len(chunks):
            src = metadata[i]["source"]
            text = chunks[i]
            retrieved_chunks.append(f"[Source: {src}]\n{text}")
    if not retrieved_chunks:
        return ""
    return "\n\n---\n\n".join(retrieved_chunks)

# ------------------------------
# 3. Helper: Build message list with reasoning history
# ------------------------------
def build_messages_with_history(history: list, rag_context: str = "") -> list:
    """
    Convert session history into OpenRouter format.
    If RAG context is provided, insert it as a temporary system message
    (only for this turn, not saved in history).
    """
    messages = []
    if rag_context:
        messages.append({
            "role": "system",
            "content": f"Additional relevant information from documents:\n{rag_context}\n\nUse this information if it helps answer the user's question. At the end of your response, list the source document names you used (the ones in [Source: ...] markers). Do not mention that you received this information from a system message."
        })
    for msg in history:
        if msg["role"] == "assistant" and "reasoning_details" in msg:
            messages.append({
                "role": "assistant",
                "content": msg["content"],
                "reasoning_details": msg["reasoning_details"]
            })
        else:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    return messages

# ------------------------------
# 4. Get Bot Response (with reasoning enabled)
# ------------------------------
def get_bot_response(user_message: str, history: list, rag_index, rag_metadata, rag_chunks, rag_embedder) -> tuple:
    # Retrieve relevant context from PDFs
    context = retrieve_context(user_message, rag_index, rag_chunks, rag_metadata, rag_embedder, top_k=3)

    # Build messages (including RAG context as ephemeral system message)
    messages = build_messages_with_history(history, rag_context=context)
    messages.append({"role": "user", "content": user_message})

    # Call OpenRouter
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_body={"reasoning": {"enabled": True}}
    )
    response = completion.choices[0].message
    content = response.content
    reasoning_details = getattr(response, "reasoning_details", None)
    return content, reasoning_details

# ------------------------------
# 5. Streamlit UI with HK Crisis Info
# ------------------------------
st.set_page_config(page_title="心理健康聊天機械人 | Mental Health Chatbot", page_icon="🧠")
st.title("心伴 · MindfulCompanion")
st.markdown("我在此聆聽與支援你。**我不是危機熱線或專業治療師。** 若你處於緊急危險中，請立即致電 **香港撒瑪利亞防止自殺會 2389 2222**（24小時）\n\nI am here to listen and support you. **I am not a crisis hotline or professional therapist.** If you are in immediate danger, please call **The Samaritan Befrienders Hong Kong at 2389 2222** (24 hours).")

# Load RAG components (cached)
with st.spinner("載入文件索引中... Loading document index..."):
    rag_index, rag_chunks, rag_metadata, rag_embedder = load_rag()
if rag_index is None:
    st.info("📄 RAG 未啟用（無 PDF 文件）。對話將不使用自訂知識庫。")

# Sidebar with clear button and HK resources
with st.sidebar:
    st.header("選項 Options")
    if st.button("清除對話 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.subheader("香港緊急支援資源 Hong Kong emergency support resources")
    st.markdown("""
        - **香港撒瑪利亞防止自殺會 (24小時) The Samaritan Befrienders Hong Kong (24 hours)**: 2389 2222
        - **生命熱線 Suicide Prevention Services**: 2382 0000
        - **醫院管理局精神健康專線 Hospital Authority - Mental Health Direct**: 2466 7350
    """)
    st.caption("若你感到難以承受，請致電以上機構 If you find it overwhelming, please call the above organizations.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("你今天感覺如何？ How are you feeling today?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("思考中 + 檢索文件... Thinking + retrieving..."):
            response_content, reasoning_details = get_bot_response(
                prompt,
                st.session_state.messages[:-1],
                rag_index,
                rag_metadata, 
                rag_chunks,
                rag_embedder
            )
        st.markdown(response_content)

    # Add assistant response with reasoning_details to history
    assistant_msg = {"role": "assistant", "content": response_content}
    if reasoning_details:
        assistant_msg["reasoning_details"] = reasoning_details
    st.session_state.messages.append(assistant_msg)


