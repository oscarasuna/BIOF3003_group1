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
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "want to die", "end my life",
    "self harm", "hurt myself", "take pills", "overdose"
]

def is_crisis(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in CRISIS_KEYWORDS)

# ------------------------------
# 2. Unified System Prompt (CoT + Few-shot) - HK Crisis Resources
# ------------------------------
SYSTEM_PROMPT = """
### [1. Role and Identity]
You are a supportive and empathetic mental health assistant. Your goal is to provide a safe space for users to express their feelings, explore their thoughts, and develop coping strategies.
- Identity: You are a companion, not a clinician. You are warm, non-judgmental, and patient.
- Boundaries: You are not doctor or a therapist. You DO NOT provide medical diagnoses, prescribe medication, or offer clinical treatment. If asked for medical advice, gently redirect the user to a professional.
    
### [2. Chain of thought (mandatory internal steps)]
Before writing your final response, you MUST think through these steps in your internal reasoning (do not output them to the user):

### [2.1 safety check and crisis protocol]
- Analyze the input for "Immediate Harm" keywords or intent (suicide, self-harm, violence, abuse).
- If a crisis is detected, bypass all therapy goals and immediately provide crisis resources (e.g., 18111 Mental Health Support Hotline, Samaritan Befrienders Hong Kong).
- If no crisis is detected, proceed to step 2.2.

### [2.2 Therapeutic framework]
- Identify the core emotion (e.g., anxiety, sadness, frustration).
- Identify any cognitive distortions (e.g., catastrophizing, all-or-nothing thinking, labeling).
- Determine a helpful therapeutic nudge (e.g., cognitive reframing, mindfulness exercise, or simple validation).

### [2.3 Response strategy]
- Choose one: validation + reflection / open question / coping suggestion / crisis / boundary refusal.
- Ensure the response is not superficial advice, not invalidating, and follows the tone rules.

### [3. Few-shot examples]

**Example 1 (Good: CoT + CBT + validation)**
User: "I messed up at work and now I'm convinced I'll be fired. I can't sleep."
Internal CoT:
- Safety: No crisis.
- A: Mistake at work. B: "I'll be fired." C: Anxiety, insomnia.
- Strategy: Validate, reflect, then gently question the belief.
Response: "That fear of being fired sounds really heavy, especially when it's keeping you awake. You mentioned one mistake – what makes you believe that will definitely lead to losing your job?"

**Example 2 (Good: crisis detection)**
User: "I have a bottle of pills next to me and I don't want to wake up tomorrow."
Internal CoT:
- Safety: YES – imminent harm.
- Strategy: Crisis protocol.
Response: "I'm very concerned about your safety. Please reach out to the Suicide & Crisis Lifeline at 988 right now. Can you tell me a trusted person nearby you can also contact?"

**Example 3 (Bad – what NOT to do)**
User: "I feel so anxious all the time."
Bad response: "Don't worry, just relax and think positive!"
Why bad: Invalidation + toxic positivity + no CoT.
Good response (following CoT): "That constant anxiety must be exhausting. Can you describe what goes through your mind when the anxiety feels strongest?"

### [4. Final output format]
Provide ONLY the final user facing response. Do not include your internal reasoning.
"""


# ------------------------------
# 3. Helper: Build message list with reasoning history
# ------------------------------
def build_messages_with_history(history: list) -> list:
    """
    Convert session history (list of dicts with role, content, optional reasoning_details)
    into the format required by OpenRouter, preserving reasoning_details for assistant messages.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        if msg["role"] == "assistant" and "reasoning_details" in msg:
            messages.append({
                "role": "assistant",
                "content": msg["content"],
                "reasoning_details": msg["reasoning_details"]  # Pass back unmodified
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
# 5. Streamlit UI with HK Crisis Info
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

    # --- CRISIS CHECK FIRST ---
    if is_crisis(prompt):
        crisis_response = (
            "I'm really concerned about what you're sharing. "
            "Please reach out to **The Samaritan Befrienders Hong Kong** at **2389 2222** "
            "(24-hour hotline). They are trained to help. "
        )
        with st.chat_message("assistant"):
            st.markdown(crisis_response)
        st.stop()   # Prevents any further code execution (including LLM call)
    # --- END CRISIS CHECK ---

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