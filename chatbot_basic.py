import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json


# ------------------------------
# 1. Configuration
# ------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Set your OpenRouter API key
MODEL_NAME = "openai/gpt-oss-120b:free"  # Free tier model
HISTORY_FILE = "chat_history.json"

def save_conversation(messages):
    """Save conversation to a JSON file."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def load_conversation():
    """Load conversation from JSON file if exists."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

# Initialize OpenRouter client (OpenAI-compatible)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ------------------------------
# 2. Helper: Build message list with reasoning history
# ------------------------------
def build_messages_with_history(history: list) -> list:
    """
    Convert session history directly into the format required by OpenRouter.
    No system prompt is added.
    """
    messages = []
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
# 3. Get Bot Response (with reasoning enabled)
# ------------------------------
def get_bot_response(user_message: str, history: list) -> tuple:
    """
    Send conversation to OpenRouter with reasoning enabled.
    Returns (response_content, reasoning_details)
    """
    # Build full message list including system prompt and conversation history
    messages = build_messages_with_history(history)
    messages.append({"role": "user", "content": user_message})

    # Call OpenRouter with reasoning enabled
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_body={"reasoning": {"enabled": True}}
    )

    response = completion.choices[0].message
    content = response.content
    reasoning_details = getattr(response, "reasoning_details", None)  # May be None if not provided
    return content, reasoning_details

# ------------------------------
# 4. Streamlit UI with HK Crisis Info
# ------------------------------
st.set_page_config(page_title="心理健康聊天機械人 | Mental Health Chatbot", page_icon="🧠")
st.title("心伴 · MindfulCompanion")
st.markdown("我在此聆聽與支援你。**我不是危機熱線或專業治療師。** 若你處於緊急危險中，請立即致電 **香港撒瑪利亞防止自殺會 2389 2222**（24小時）\n\nI am here to listen and support you. **I am not a crisis hotline or professional therapist.** If you are in immediate danger, please call **The Samaritan Befrienders Hong Kong at 2389 2222** (24 hours).")

# Sidebar with clear button and HK resources
with st.sidebar:
    st.header("選項 Options")
    if st.button("清除對話 Clear Conversation"):
        st.session_state.messages = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
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
    st.session_state.messages = load_conversation()

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("你今天感覺如何？ How are you feeling today?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_conversation(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("思考中 Thinking..."):
            response_content, reasoning_details = get_bot_response(prompt, st.session_state.messages[:-1])
        st.markdown(response_content)

    # Add assistant response with reasoning_details to history
    assistant_msg = {"role": "assistant", "content": response_content}
    if reasoning_details:
        assistant_msg["reasoning_details"] = reasoning_details
    st.session_state.messages.append(assistant_msg)
    save_conversation(st.session_state.messages) 