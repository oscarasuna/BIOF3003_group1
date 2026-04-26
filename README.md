# 🧠 MindfulCompanion – Mental Health Chatbot with RAG
MindfulCompanion is a **Streamlit-based mental health chatbot** designed to provide empathetic, supportive conversations. It uses a **large language model (LLM)** via OpenRouter, with **Retrieval-Augmented Generation (RAG)** from your own PDF documents. The chatbot strictly follows a therapeutic prompt (Chain-of-Thought + CBT framework) and includes **crisis detection** with Hong Kong local resources.


> ⚠️ **Disclaimer:** This is not a substitute for professional therapy or crisis services. If you are in immediate danger, please call **The Samaritan Befrienders Hong Kong at 2389 2222**.


---

## ✨ Features

- **🧘 Therapeutic prompt** – Built‑in Chain‑of‑Thought, CBT (ABC model), validation rules, and crisis protocol.
- **📄 RAG from PDFs** – Upload your own documents (e.g., mental health guides, FAQs) and the bot will retrieve relevant chunks to answer questions.
- **🇭🇰 Hong Kong crisis resources** – Local hotlines prominently displayed.
- **🤖 OpenRouter integration** – Supports many free and paid models (e.g., `openai/gpt-oss-120b:free`).
- **💬 Multi‑turn reasoning** – Preserves conversation history and optional `reasoning_details` for models that support it.
- **⚡ Cached RAG index** – FAISS + sentence‑transformers, loads instantly after first run.
- **🖥️ Clean Streamlit UI** – Chat interface, sidebar with resources, clear conversation button.

---

In this repository, there are 4 versions of chatbot applications for experimenting:

1. chatbot_basic.py (without RAG, prompting)
2. chatbot_prompt_only.py (without RAG, with prompting)
3. chatbot_rag_only.py (with RAG, without prompting)
4. chatbot_prompt_and_rag.py (with RAG, prompting)

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/oscarasuna/BIOF3003-Assign.git
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get an OpenRouter API key
- Sign up at [openrouter.ai](openrouter.ai)
- Go to Keys and generate a key (starts with sk-or-v1-...)

### 5. Set your API key
Create a .env file in the project root:
```text
OPENROUTER_API_KEY=sk-or-v1-...
```
### 6. Run the app
```bash
streamlit run chatbot_(basic/rag_only/prompt_only/prompt_and_rag).py
```
---

##  🧪 Usage
1. Open the local URL (usually http://localhost:8501).

2. Type your message in the chat input.

3. The bot will first check for crisis keywords – if detected, it will output the HK hotline immediately.

4. Otherwise, it retrieves relevant chunks from your PDFs (if any) and generates a response using the therapeutic prompt.

5. Conversation history is maintained across turns.
---

## ⚙️ Configuration

| Variable | Description |
|----------|-------------|
| `MODEL_NAME` | OpenRouter model ID. Default: `"openai/gpt-oss-120b:free"` |
| `SYSTEM_PROMPT` | The complete therapeutic guidelines (modify with caution). |
| `RAG_DOC_DIR` | Folder containing PDFs (default: `"rag_doc"`). |
| `EMBEDDING_MODEL` | Sentence‑transformer model for embeddings (default: `"all-MiniLM-L6-v2"`). |
| `chunk_size` | Characters per text chunk (default: 500). |
| `top_k` | Number of retrieved chunks per query (default: 3). |

## 📄 License
This project is for **educational and research purposes** only. Not intended for production mental health services without professional oversight.

## 📞 Crisis Resources (Hong Kong)
| Service | Phone Number |
|---------|--------------|
| The Samaritan Befrienders HK (24h) | 2389 2222 |
| Suicide Prevention Hotline | 2382 0000 |
| Hospital Authority Mental Health Line | 2466 7350 |

_If you are outside Hong Kong, please contact your local crisis helpline._




