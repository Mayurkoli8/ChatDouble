import streamlit as st
from ollama import Client as OllamaClient
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

## login and auth

import json

USER_FILE = "users.json"
CHAT_DIR = "chats"
os.makedirs(CHAT_DIR, exist_ok=True)

import json
##helper to load chats history

def load_chat_history(user, bot):
    path = os.path.join(CHAT_DIR, user, f"{bot.lower()}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

##helper to save chat history
def save_chat_history(user, bot, history):
    user_dir = os.path.join(CHAT_DIR, user)
    os.makedirs(user_dir, exist_ok=True)
    path = os.path.join(user_dir, f"{bot.lower()}.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

# Load users from file
if "users" not in st.session_state:
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            st.session_state.users = json.load(f)
    else:
        st.session_state.users = {}

# Auth state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# 🔁 Init session keys if not already set
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None

if "rename_bot_index" not in st.session_state:
    st.session_state.rename_bot_index = None

## upload function 
def show_upload_ui():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload New Chat File")

    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"], key="file_uploader_input")
    new_bot_name = st.sidebar.text_input("Bot name for this chat (e.g. John)", key="bot_name_input")

    if st.sidebar.button("Upload Bot"):
        user = st.session_state.username
        already_exists = any(
            bot["name"].lower() == new_bot_name.lower()
            for bot in st.session_state.users[user]["bots"]
        )

        if already_exists:
            st.sidebar.warning("⚠️ You already have a bot with this name.")
        else:
            # Save file
            save_path = f"bots/{user}_chat_{new_bot_name.lower()}.txt"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())

            # Assign bot to user
            st.session_state.users[user]["bots"].append({
                "name": new_bot_name.capitalize(),
                "file": save_path
            })

            # Save to file
            with open(USER_FILE, "w") as f:
                json.dump(st.session_state.users, f, indent=2)

            # ✅ Do NOT clear fields manually — just rerun cleanly
            st.sidebar.success(f"✅ {new_bot_name} bot added for {user}")
            st.rerun()


## Manage my bots

if st.session_state.logged_in:
    user = st.session_state.username
    user_bots = st.session_state.users[user]["bots"]

    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = None

    with st.sidebar.expander("🧰 Manage My Bots", expanded=False):
        if not user_bots:
            st.info("You haven't added any bots yet.")
        else:
            if "rename_bot_index" not in st.session_state:
                st.session_state.rename_bot_index = None

            for i, bot in enumerate(user_bots):
                col1, col2, col3, col4 = st.columns([2.5, 0.7, 0.8, 0.8])
                with col1:
                    st.markdown(f"**{bot['name']}**")
                if st.session_state.rename_bot_index is None:
                    with col2:
                        if st.button("✏️", key=f"rename_{i}"):
                            st.session_state.rename_bot_index = i
                    with col3:
                        if st.button("🗑️", key=f"delete_{i}"):
                            st.session_state.confirm_delete = i
                    with col4:
                        if st.button("🧼", key=f"clear_{i}"):
                            st.session_state.clear_history_index = i
            
        # 🔁 Rename bot UI
        if "rename_bot_index" in st.session_state and st.session_state.rename_bot_index is not None:
            i = st.session_state.rename_bot_index
            bot = user_bots[i]

            new_name = st.sidebar.text_input("New name:", value=bot["name"], key="rename_input")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ Save", key="rename_confirm"):
                    # 🛑 Prevent duplicate names
                    if any(b["name"].lower() == new_name.lower() for b in user_bots if b != bot):
                        st.sidebar.warning("⚠️ A bot with that name already exists.")
                    else:
                        old_name = bot["name"]
                        old_file = bot["file"]
                        new_file = f"bots/chat_{new_name.lower()}.txt"
                        new_history = os.path.join("chats", user, f"{new_name.lower()}.json")
                        old_history = os.path.join("chats", user, f"{old_name.lower()}.json")

                        # Rename chat file
                        if os.path.exists(old_file):
                            os.rename(old_file, new_file)

                        # Rename history file
                        if os.path.exists(old_history):
                            os.rename(old_history, new_history)

                        # Update users.json
                        st.session_state.users[user]["bots"][i]["name"] = new_name
                        st.session_state.users[user]["bots"][i]["file"] = new_file

                        with open(USER_FILE, "w") as f:
                            json.dump(st.session_state.users, f, indent=2)

                        st.session_state.rename_bot_index = None
                        st.success(f"✅ Renamed '{old_name}' to '{new_name}'")
                        st.rerun()


            with col2:
                if st.button("❌ Cancel", key="rename_cancel"):
                    st.session_state.rename_bot_index = None

        # 🧠 Confirmation outside of expander
        if st.session_state.confirm_delete is not None:
            index = st.session_state.confirm_delete
            bot = user_bots[index]

            st.sidebar.error(f"⚠️ Are you sure you want to delete '{bot['name']}'?")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ Yes", key="confirm_yes"):
                    if os.path.exists(bot["file"]):
                        os.remove(bot["file"])

                    history_path = os.path.join("chats", user, f"{bot['name'].lower()}.json")
                    if os.path.exists(history_path):
                        os.remove(history_path)

                    st.session_state.users[user]["bots"].pop(index)
                    with open(USER_FILE, "w") as f:
                        json.dump(st.session_state.users, f, indent=2)

                    st.session_state.confirm_delete = None
                    st.success(f"✅ Bot '{bot['name']}' deleted.")
                    st.rerun()

            with col2:
                if st.button("❌ Cancel", key="confirm_no"):
                    st.session_state.confirm_delete = None

        # 🔁 Confirm Clear Chat History
        if "clear_history_index" in st.session_state and st.session_state.clear_history_index is not None:
            i = st.session_state.clear_history_index
            bot = user_bots[i]

            st.error(f"🧹 Clear all chat history with '{bot['name']}'?")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ Yes", key="clear_yes"):
                    # Delete chat history file
                    history_path = os.path.join("chats", user, f"{bot['name'].lower()}.json")
                    if os.path.exists(history_path):
                        os.remove(history_path)

                    # Clear in-memory chat
                    chat_key = f"chat_{bot['name']}_{user}"
                    if chat_key in st.session_state:
                        st.session_state[chat_key] = []

                    st.success(f"✅ Chat history with '{bot['name']}' cleared.")
                    st.session_state.clear_history_index = None
                    st.rerun()

            with col2:
                if st.button("❌ Cancel", key="clear_no"):
                    st.session_state.clear_history_index = None

    



## ui for auth

st.sidebar.title("🔐 Login / Register")

if not st.session_state.logged_in:
    action = st.sidebar.radio("Choose", ["Login", "Register"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button(action):
        users = st.session_state.users

        if action == "Login":
            if username in users and users[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"✅ Welcome back, {username}!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials.")

        elif action == "Register":
            if username in users:
                st.error("❌ Username already exists.")
            else:
                users[username] = {"password": password, "bots": []}
                with open(USER_FILE, "w") as f:
                    json.dump(users, f, indent=2)
                st.session_state.users = users
                st.success("✅ Registered. Please log in.")
else:
    st.sidebar.success(f"👋 Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()


# login check (stop app if not logged in)

if not st.session_state.logged_in:
    st.warning("🔒 Please log in to access your bots.")
    st.stop()



# 🔹 Available bots and their chat files
user = st.session_state.username
user_bots = st.session_state.users[user].get("bots", [])

bot_map = {
    bot["name"]: bot["file"]
    for bot in user_bots if os.path.exists(bot["file"])
}


# 🔹 Sidebar: choose bot
st.sidebar.title("🤖 Choose Your Bot")
bot_names = list(bot_map.keys())

if not bot_names:
    st.warning("No bots found. Please upload a chat file to start.")
    show_upload_ui()
    st.stop()

selected_bot = st.sidebar.selectbox("Who do you want to talk to?", bot_names)

# 🔹 Session key per bot
bot_key = f"faiss_{selected_bot.lower()}"

# 🔹 Load FAISS memory per bot
if bot_key not in st.session_state:
    chat_path = bot_map[selected_bot]
    with open(chat_path, "r", encoding="utf-8") as f:
        bot_lines = [line.strip() for line in f.readlines() if line.strip()]

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(bot_lines, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.session_state[bot_key] = {
        "index": index,
        "lines": bot_lines,
        "embed_model": embed_model,
    }

# ✅ Load Ollama
ollama = OllamaClient()

# 🧠 Chat history per bot with diffrent sessions
user = st.session_state.username
chat_key = f"chat_{selected_bot}_{user}"

if chat_key not in st.session_state:
    st.session_state[chat_key] = load_chat_history(user, selected_bot)

# Show history
st.title(f"💬 {selected_bot}Bot")


for entry in st.session_state[chat_key]:
    st.chat_message("user").markdown(entry["user"])
    st.chat_message("assistant").markdown(entry["bot"])

# Chat input
user_input = st.chat_input(f"Say something to {selected_bot}...", key="user_input")
if user_input:
    # 🔍 Search top 20 relevant lines
    data = st.session_state[bot_key]
    query_vector = data["embed_model"].encode([user_input])
    D, I = data["index"].search(query_vector, k=20)
    top_matches = [data["lines"][i] for i in I[0]]
    context = "\n".join(top_matches)



    prompt = f"""
You are {selected_bot}, a real person who has chatted with the user before.

Recent memory:
{context}

Now reply to the user as {selected_bot} would:
User: {user_input}
{selected_bot}:
"""



    # 💬 Get response from Mistral
    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": f"You are a real person named {selected_bot}, not a chatbot. Always reply like you've done in the past messages shown. Match tone, slang, and vibe from the examples. Never make up things beyond that style."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"num_predict": 100}
    )

    bot_reply = response['message']['content'].strip()

    # 💾 Save to history (per user + bot)
    entry = {
        "user": user_input,
        "bot": bot_reply
    }
    st.session_state[chat_key].append(entry)
    save_chat_history(user, selected_bot, st.session_state[chat_key])

    
    st.rerun()


## upload bot
if st.session_state.logged_in:
    show_upload_ui()
    


## download chat history    
if st.session_state.logged_in:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Download Chat History")

    # Get user’s bots & dropdown
    user = st.session_state.username
    user_bots = st.session_state.users[user]["bots"]

    if user_bots:
        bot_names = [bot["name"] for bot in user_bots]
        export_bot = st.sidebar.selectbox("Choose a bot to export", bot_names, key="export_bot_select")

        # Load chat history from file
        chat_path = os.path.join("chats", user, f"{export_bot.lower()}.json")
        if os.path.exists(chat_path):
            with open(chat_path, "r") as f:
                history = json.load(f)
        else:
            history = []

        #  Generate download files
        # .txt format
        txt_export = "\n\n".join([f"You: {entry['user']}\n{export_bot}: {entry['bot']}" for entry in history])
    
        # .json format
        json_export = json.dumps(history, indent=2)


        #Show download buttons
        st.sidebar.download_button(
            label="⬇️ Download .txt",
            data=txt_export,
            file_name=f"{export_bot}_chat.txt",
            mime="text/plain"
        )

        st.sidebar.download_button(
            label="⬇️ Download .json",
            data=json_export,
            file_name=f"{export_bot}_chat.json",
            mime="application/json"
        )
