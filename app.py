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

# =========================================================
# 🔑 Initialize Gemini
# =========================================================
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
genai_client = genai.Client(api_key=api_key)

# =========================================================
# 🗂️ Ensure local dirs exist (used for temp caching)
# =========================================================
os.makedirs("chats", exist_ok=True)

# =========================================================
# 🔐 Session Setup
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None

if "rename_bot_index" not in st.session_state:
    st.session_state.rename_bot_index = None


# =========================================================
# 📂 Upload Bot UI
# =========================================================
def show_upload_ui():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload New Chat File")

    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"], key="file_upload")
    new_bot_name = st.sidebar.text_input("Bot name (e.g. John)", key="bot_name_input")

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


# =========================================================
# 🔐 Authentication Panel
# =========================================================
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
                st.success(f"✅ Welcome back, {username}!")
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


# =========================================================
# ⚡ Cached FAISS builder for speed
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
# 🤖 Bot Management Panel
# =========================================================
user = st.session_state.username
user_bots = get_user_bots(user)

st.sidebar.title("🤖 Manage My Bots")

if not user_bots:
    st.info("No bots yet. Upload one to get started.")
    show_upload_ui()
    st.stop()

for i, bot in enumerate(user_bots):
    col1, col2, col3, col4 = st.sidebar.columns([2.5, 0.8, 0.8, 0.8])
    with col1:
        st.sidebar.markdown(f"**{bot['name']}**")
    with col2:
        if st.sidebar.button("✏️", key=f"rename_{i}"):
            st.session_state.rename_bot_index = i
    with col3:
        if st.sidebar.button("🗑️", key=f"delete_{i}"):
            st.session_state.confirm_delete = i
    with col4:
        if st.sidebar.button("🧹", key=f"clear_{i}"):
            st.session_state.clear_history_index = i

# Rename
if st.session_state.rename_bot_index is not None:
    i = st.session_state.rename_bot_index
    bot = user_bots[i]
    new_name = st.sidebar.text_input("New name:", value=bot["name"])
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("✅ Save"):
            update_bot(user, bot["name"], new_name)
            st.session_state.rename_bot_index = None
            st.sidebar.success(f"✅ Renamed to {new_name}")
            st.experimental_rerun()
    with col2:
        if st.sidebar.button("❌ Cancel"):
            st.session_state.rename_bot_index = None

# Delete
if st.session_state.confirm_delete is not None:
    index = st.session_state.confirm_delete
    bot = user_bots[index]
    st.sidebar.error(f"⚠️ Delete '{bot['name']}'?")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("✅ Yes"):
            delete_bot(user, bot["name"])
            st.session_state.confirm_delete = None
            st.sidebar.success("✅ Deleted successfully")
            st.experimental_rerun()
    with col2:
        if st.sidebar.button("❌ Cancel"):
            st.session_state.confirm_delete = None

# Clear chat history
if "clear_history_index" in st.session_state and st.session_state.clear_history_index is not None:
    i = st.session_state.clear_history_index
    bot = user_bots[i]
    st.sidebar.warning(f"🧹 Clear all chat history for '{bot['name']}'?")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("✅ Confirm Clear"):
            save_chat_history_cloud(user, bot["name"], [])
            st.session_state.clear_history_index = None
            st.sidebar.success(f"✅ Cleared history for {bot['name']}")
            st.experimental_rerun()
    with col2:
        if st.sidebar.button("❌ Cancel Clear"):
            st.session_state.clear_history_index = None


# =========================================================
# 🧠 Select & Load Bot
# =========================================================
selected_bot = st.sidebar.selectbox("Who do you want to talk to?", [b["name"] for b in user_bots])
bot_text = get_bot_file(user, selected_bot)

if not bot_text.strip():
    st.warning("⚠️ This bot is empty. Please upload a valid chat file.")
    st.stop()

# Load cached embeddings (fast)
embed_model, index, bot_lines = load_faiss_index(bot_text)

# Load chat history
chat_key = f"chat_{selected_bot}_{user}"
if chat_key not in st.session_state:
    st.session_state[chat_key] = load_chat_history_cloud(user, selected_bot)

st.title(f"💬 {selected_bot}Bot")

# Display past messages
for entry in st.session_state[chat_key]:
    st.chat_message("user").markdown(entry["user"])
    st.chat_message("assistant").markdown(entry["bot"])


# =========================================================
# 💬 Chat Input & Response
# =========================================================
user_input = st.chat_input(f"Talk to {selected_bot}...")

if user_input:
    query_vector = embed_model.encode([user_input])
    D, I = index.search(query_vector, k=8)
    context = "\n".join(bot_lines[i] for i in I[0] if i < len(bot_lines))

    # Trim context to 2000 chars max
    if len(context) > 2000:
        context = context[:2000]

    prompt = f"""
You are {selected_bot}, chatting with your friend.
Base your reply on these messages:
{context}

User: {user_input}
{selected_bot}:
"""

    # Stream Gemini response for faster feel
    try:
        resp = genai_client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=prompt
        )
        bot_reply = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk in resp:
                text = chunk.text or ""
                bot_reply += text
                placeholder.markdown(bot_reply + "▌")
            placeholder.markdown(bot_reply)
    except Exception:
        bot_reply = "⚠️ Error generating response."

    # Save chat history (Firestore)
    st.session_state[chat_key].append({"user": user_input, "bot": bot_reply})
    save_chat_history_cloud(user, selected_bot, st.session_state[chat_key])

    st.experimental_rerun()


# =========================================================
# ⬇️ Upload UI always visible at bottom
# =========================================================
show_upload_ui()
