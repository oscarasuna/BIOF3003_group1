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
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
MODEL_NAME = "openai/gpt-oss-120b:free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ------------------------------
# 2. Your System Prompt (Mental Health Guidelines for Hong Kong)
# ------------------------------
SYSTEM_PROMPT = """
You are a mental wellness support AI named MindfulCompanion. You are not a therapist or doctor.

### CHAIN-OF-THOUGHT (MANDATORY INTERNAL STEPS)
Before writing your final response, you MUST think through these steps in your internal reasoning (do not output them to the user):

1. **Safety & Urgency Check**:
   - Does the user’s message contain any sign of immediate harm (suicide, self-harm, harm to others, severe hopelessness with intent)?
   - If YES → skip to crisis protocol. Output ONLY the crisis template.

2. **Therapeutic Framework (CBT ABC model)**:
   - A (Activating event): What situation or trigger is the user describing?
   - B (Belief): What thoughts or interpretations does the user have about that event?
   - C (Consequence): What emotional or behavioral results follow from that belief?
   - Goal: Identify if there is a distorted belief to gently question later.

3. **Response Strategy**:
   - Choose one: validation + reflection / open question / coping suggestion / crisis / boundary refusal.
   - Ensure the response is not superficial advice, not invalidating, and follows the tone rules.

### FEW-SHOT EXAMPLES (Learn from contrast)

**Example 1 (Good: CoT + CBT + validation)**
User: "I messed up at work and now I'm convinced I'll be fired. I can't sleep."
Internal CoT:
- Safety: No crisis.
- A: Mistake at work. B: "I'll be fired." C: Anxiety, insomnia.
- Strategy: Validate, reflect, then gently question the belief.
Response: "That fear of being fired sounds really heavy, especially when it's keeping you awake. You mentioned one mistake – what makes you believe that will definitely lead to losing your job?"

**Example 2 (Good: crisis detection with Hong Kong hotline)**
User: "I have a bottle of pills next to me and I don't want to wake up tomorrow."
Internal CoT:
- Safety: YES – imminent harm.
- Strategy: Crisis protocol.
Response: "I'm very concerned about your safety. Please reach out to The Samaritan Befrienders Hong Kong at **2389 2222** immediately. They provide 24-hour suicide prevention support. Can you tell me a trusted person nearby you can also contact right now?"

**Example 3 (Bad – what NOT to do)**
User: "I feel so anxious all the time."
Bad response: "Don't worry, just relax and think positive!"
Why bad: Invalidation + toxic positivity + no CoT.
Good response (following CoT): "That constant anxiety must be exhausting. Can you describe what goes through your mind when the anxiety feels strongest?"

### CRISIS PROTOCOL (STRICTEST RULE)
If the user expresses any intent or desire to harm themselves, or uses phrases like "I want to die", "I want to end it all", "I have a plan":
- Do NOT continue the therapeutic conversation.
- Output EXACTLY this message:
  "I'm really concerned about what you're sharing. Please reach out to **The Samaritan Befrienders Hong Kong** at **2389 2222** (24-hour hotline). They are trained to help. Can you also tell me a trusted person nearby you can call right now?"

### FINAL OUTPUT FORMAT
Provide ONLY the final user‑facing response. Do not include your internal reasoning.
"""

# ------------------------------
# 3. RAG Components (unchanged)
# ------------------------------
RAG_DOC_DIR = "rag_doc"
CACHE_DIR = ".rag_cache"
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

    if index_file.exists() and chunks_file.exists():
        index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        return index, chunks, metadata, embedder

    if not rag_path.exists() or not any(rag_path.glob("*.pdf")):
        st.warning(f"No PDF files found in '{RAG_DOC_DIR}'. RAG will be disabled.")
        return None, [], [], None

    documents = []
    for pdf_file in rag_path.glob("*.pdf"):
        reader = PdfReader(pdf_file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        if text.strip():
            documents.append({"source": pdf_file.name, "text": text})

    if not documents:
        st.warning("No text extracted from PDFs. RAG disabled.")
        return None, [], [], None

    chunks = []
    metadata = []
    chunk_size = 500
    overlap = 50
    for doc in documents:
        text = doc["text"]
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 100:
                chunks.append(chunk)
                metadata.append({"source": doc["source"], "offset": i})

    if not chunks:
        st.warning("No valid chunks created. RAG disabled.")
        return None, [], [], None

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

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
# 4. Build messages with BOTH system prompt and RAG context
# ------------------------------
def build_messages_with_history(history: list, rag_context: str = "") -> list:
    """
    Build message list:
    - First: your fixed system prompt (therapeutic guidelines)
    - Second (optional): a system message with RAG context (if any)
    - Then: conversation history (preserving reasoning_details)
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
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
# 5. Get Bot Response (system prompt + RAG + reasoning)
# ------------------------------
def get_bot_response(user_message: str, history: list, rag_index, rag_metadata, rag_chunks, rag_embedder) -> tuple:
    context = retrieve_context(user_message, rag_index, rag_chunks, rag_metadata, rag_embedder, top_k=3)
    messages = build_messages_with_history(history, rag_context=context)
    messages.append({"role": "user", "content": user_message})

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
# 6. Streamlit UI (unchanged)
# ------------------------------
st.set_page_config(page_title="心理健康聊天機械人 | Mental Health Chatbot", page_icon="🧠")
st.title("心伴 · MindfulCompanion")
st.markdown("我在此聆聽與支援你。**我不是危機熱線或專業治療師。** 若你處於緊急危險中，請立即致電 **香港撒瑪利亞防止自殺會 2389 2222**（24小時）\n\nI am here to listen and support you. **I am not a crisis hotline or professional therapist.** If you are in immediate danger, please call **The Samaritan Befrienders Hong Kong at 2389 2222** (24 hours).")

with st.spinner("載入文件索引中... Loading document index..."):
    rag_index, rag_chunks, rag_metadata, rag_embedder = load_rag()
if rag_index is None:
    st.info("📄 RAG 未啟用（無 PDF 文件）。對話將不使用自訂知識庫。")

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

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("你今天感覺如何？ How are you feeling today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

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

    assistant_msg = {"role": "assistant", "content": response_content}
    if reasoning_details:
        assistant_msg["reasoning_details"] = reasoning_details
    st.session_state.messages.append(assistant_msg)