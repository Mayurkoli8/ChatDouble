import streamlit as st
import faiss
import os
import json
from sentence_transformers import SentenceTransformer
from google import genai   # ✅ your original working import
from firebase_db import (
    get_user_bots, add_bot, delete_bot, update_bot, update_bot_persona,
    register_user, login_user, get_bot_file,
    save_chat_history_cloud, load_chat_history_cloud
)

# =========================================================
# 🌟 PAGE CONFIG & STYLES
# =========================================================
st.set_page_config(page_title="ChatDouble", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #0e0e10, #1c1c1f);
        color: #fff;
        font-family: 'Inter', sans-serif;
    }
    .stChatMessage {
        padding: 10px 16px;
        border-radius: 12px;
        margin: 8px 0;
        line-height: 1.5;
        font-size: 16px;
    }
    .stChatMessage.user {
        background-color: #27293d;
        border-left: 4px solid #6c63ff;
    }
    .stChatMessage.assistant {
        background-color: #191a24;
        border-left: 4px solid #ffb347;
    }
    .stSidebar {
        background: #141416 !important;
        color: white !important;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #6c63ff !important;
        color: white !important;
        border: none;
    }
    .stButton>button:hover {
        background-color: #554efc !important;
        color: white !important;
    }
    .stTextInput>div>div>input {
        background-color: #1a1b25;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🔑 Initialize Gemini (official)
# =========================================================
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
genai_client = genai.Client(api_key=api_key)
os.makedirs("chats", exist_ok=True)

# =========================================================
# 🤖 Helper: Generate Persona Description
# =========================================================
def generate_persona(text_examples):
    if not text_examples.strip():
        return ""
    prompt = f"""
Take the following example messages spoken by one person.
Write 1–2 sentences describing their tone, vibe, slang, and personality.

Example lines:
{text_examples}

Output only the description, nothing else.
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip().split("\n")[0][:240]
    except Exception as e:
        st.warning(f"Persona generation failed: {e}")
        return ""


# =========================================================
# ⚡ Cached FAISS Index Builder
# =========================================================
@st.cache_resource(show_spinner=False)
def load_faiss_index(bot_text):
    bot_lines = [line.strip() for line in bot_text.splitlines() if line.strip()]
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(bot_lines, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return embed_model, index, bot_lines


# =========================================================
# 🔐 Session Setup
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""


# =========================================================
# 📂 Upload Bot UI
# =========================================================
def show_upload_ui():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload New Chat File")

    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
    new_bot_name = st.sidebar.text_input("Bot name (e.g. John)")

    if st.sidebar.button("Upload Bot"):
        if not uploaded_file:
            st.sidebar.error("⚠️ Please upload a chat file.")
            return
        if not new_bot_name.strip():
            st.sidebar.error("⚠️ Please enter a bot name.")
            return

        raw = uploaded_file.read().decode("utf-8", "ignore")
        bot_lines = extract_bot_lines(raw, new_bot_name)

        if not bot_lines.strip():
            st.sidebar.error("Could not detect bot messages in file.")
            return

        persona = generate_persona("\n".join(bot_lines.splitlines()[:40]))
        add_bot(st.session_state.username, new_bot_name.capitalize(), bot_lines, persona=persona)

        st.sidebar.success(f"✅ {new_bot_name} added! Persona: {persona or 'Default personality'}")
        st.rerun()


# =========================================================
# 🔐 Login & Register
# =========================================================
st.sidebar.title("🔐 Login / Register")

if not st.session_state.logged_in:
    mode = st.sidebar.radio("Choose", ["Login", "Register"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button(mode):
        if mode == "Login":
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials.")
        else:
            if register_user(username, password):
                st.success("✅ Registered successfully! Please log in.")
            else:
                st.error("❌ Username already exists.")
else:
    st.sidebar.success(f"👋 Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

if not st.session_state.logged_in:
    st.stop()


# =========================================================
# 🤖 Manage Bots
# =========================================================
user = st.session_state.username
user_bots = get_user_bots(user)

st.sidebar.title("🤖 Manage My Bots")

if not user_bots:
    st.info("No bots yet. Upload one to get started.")
    show_upload_ui()
    st.stop()

for i, bot in enumerate(user_bots):
    c1, c2, c3, c4 = st.sidebar.columns([2.5, 0.8, 0.8, 0.8])
    with c1:
        st.sidebar.markdown(f"**{bot['name']}**")
    with c2:
        if st.sidebar.button("✏️", key=f"rename_{i}"):
            st.session_state.rename_bot_index = i
    with c3:
        if st.sidebar.button("🗑️", key=f"delete_{i}"):
            delete_bot(user, bot["name"])
            st.sidebar.success("✅ Bot deleted.")
            st.rerun()
    with c4:
        if st.sidebar.button("🧹", key=f"clear_{i}"):
            save_chat_history_cloud(user, bot["name"], [])
            st.sidebar.success(f"🧹 Cleared {bot['name']}'s history.")
            st.rerun()


# =========================================================
# 🧠 Select & Load Bot
# =========================================================
selected_bot = st.sidebar.selectbox("Who do you want to talk to?", [b["name"] for b in user_bots])
bot_text, persona = get_bot_file(user, selected_bot)

if not bot_text.strip():
    st.warning("⚠️ Bot chat file is empty.")
    st.stop()

embed_model, index, bot_lines = load_faiss_index(bot_text)

chat_key = f"chat_{selected_bot}_{user}"
if chat_key not in st.session_state:
    st.session_state[chat_key] = load_chat_history_cloud(user, selected_bot)

st.title(f"💬 Chat with {selected_bot}")

# Display chat history
for entry in st.session_state[chat_key]:
    st.chat_message("user").markdown(entry["user"])
    st.chat_message("assistant").markdown(entry["bot"])


# =========================================================
# 💬 Chat Input + Gemini
# =========================================================
user_input = st.chat_input(f"Talk to {selected_bot}...")

if user_input:
    query_vector = embed_model.encode([user_input])
    D, I = index.search(query_vector, k=25)

    # Choose top meaningful bot lines
    lines = []
    for idx in I[0]:
        if idx < len(bot_lines):
            line = bot_lines[idx].strip()
            if len(line.split()) > 3:
                lines.append(line)
    context = "\n".join(lines[:12])

    if len(context) > 3000:
        context = context[:3000]

    persona_block = f"Persona: {persona}\n" if persona else ""

    prompt = f"""
{persona_block}
You are {selected_bot}, a real person who has previously chatted with the user.
Use the same tone, slang, and style as before. Never act like an AI assistant.
Base your tone and vocabulary on the examples below.

Examples of how {selected_bot} talks:
{context}

User: {user_input}
{selected_bot}:
"""

    model_name = "gemini-1.5-flash" if len(context) > 1200 else "gemini-2.0-flash"

    try:
        resp = genai_client.models.generate_content_stream(
            model=model_name,
            contents=prompt
        )
        bot_reply = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk in resp:
                text = getattr(chunk, "text", "")
                if text:
                    bot_reply += text
                    placeholder.markdown(bot_reply + "▌")
            placeholder.markdown(bot_reply)
    except Exception as e:
        st.error(f"⚠️ Gemini error: {e}")
        bot_reply = "⚠️ Error generating response."
    
    st.session_state[chat_key].append({"user": user_input, "bot": bot_reply})
    save_chat_history_cloud(user, selected_bot, st.session_state[chat_key])
    st.rerun()


# =========================================================
# ⬇️ Upload UI Always at Bottom
# =========================================================
show_upload_ui()
