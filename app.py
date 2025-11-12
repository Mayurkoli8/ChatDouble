import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import os
import json
from google import genai
from firebase_db import (
    get_user_bots, add_bot, delete_bot, update_bot,
    register_user, login_user, get_bot_file,
    save_chat_history_cloud, load_chat_history_cloud
)

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
genai_client = genai.Client(api_key=api_key)

# Initialize folders for local fallback
os.makedirs("chats", exist_ok=True)

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

def show_upload_ui():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload New Chat File")

    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
    new_bot_name = st.sidebar.text_input("Bot name (e.g. John)")

    if st.sidebar.button("Upload Bot"):
        if uploaded_file is None:
            st.sidebar.error("⚠️ Please choose a file.")
            return
        if not new_bot_name.strip():
            st.sidebar.error("⚠️ Please enter a bot name.")
            return

        content = uploaded_file.read().decode("utf-8", "ignore")
        add_bot(st.session_state.username, new_bot_name.capitalize(), content)
        st.sidebar.success(f"✅ Bot '{new_bot_name}' added!")
        st.experimental_rerun()


# --- AUTH ---
st.sidebar.title("🔐 Login / Register")

if not st.session_state.logged_in:
    action = st.sidebar.radio("Choose", ["Login", "Register"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button(action):
        if action == "Login":
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome back, {username}!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid credentials.")
        else:
            if register_user(username, password):
                st.success("✅ Registered successfully! Please log in.")
            else:
                st.error("❌ Username already exists.")
else:
    st.sidebar.success(f"👋 Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()


if not st.session_state.logged_in:
    st.stop()

# --- BOTS ---
user = st.session_state.username
user_bots = get_user_bots(user)

st.sidebar.title("🤖 Your Bots")
if not user_bots:
    st.info("No bots yet. Upload one to get started.")
    show_upload_ui()
    st.stop()

selected_bot = st.sidebar.selectbox("Choose your bot", [b["name"] for b in user_bots])

# Load bot file content from Firestore
bot_text = get_bot_file(user, selected_bot)
bot_lines = [line.strip() for line in bot_text.splitlines() if line.strip()]

if not bot_lines:
    st.warning("Bot file is empty.")
    st.stop()

# Embed & FAISS
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(bot_lines, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load chat history from Firestore
chat_key = f"chat_{selected_bot}_{user}"
if chat_key not in st.session_state:
    st.session_state[chat_key] = load_chat_history_cloud(user, selected_bot)

st.title(f"💬 {selected_bot}Bot")

# Display previous messages
for entry in st.session_state[chat_key]:
    st.chat_message("user").markdown(entry["user"])
    st.chat_message("assistant").markdown(entry["bot"])

user_input = st.chat_input(f"Talk to {selected_bot}...")

if user_input:
    query_vector = embed_model.encode([user_input])
    D, I = index.search(query_vector, k=20)
    context = "\n".join(bot_lines[i] for i in I[0] if i < len(bot_lines))

    prompt = f"""
You are {selected_bot}, a person who has chatted with this user before.

Recent memory:
{context}

Now reply to the user naturally as {selected_bot}:
User: {user_input}
{selected_bot}:
"""

    try:
        resp = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        bot_reply = resp.text.strip()
    except Exception as e:
        bot_reply = "⚠️ Error generating response."

    st.session_state[chat_key].append({"user": user_input, "bot": bot_reply})
    save_chat_history_cloud(user, selected_bot, st.session_state[chat_key])
    st.experimental_rerun()


# Upload new bot option
show_upload_ui()
