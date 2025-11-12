# app.py  — revised & bug-fixed (copy-paste replace)
import streamlit as st
import faiss
import os
import json
from sentence_transformers import SentenceTransformer
from google import genai   # using your working client
from firebase_db import (
    get_user_bots, add_bot, delete_bot, update_bot, update_bot_persona,
    register_user, login_user, get_bot_file,
    save_chat_history_cloud, load_chat_history_cloud
)
from datetime import datetime
import base64

# ---------------------------
# Page config + gemini init
# ---------------------------
st.set_page_config(page_title="ChatDouble", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
genai_client = genai.Client(api_key=api_key)

os.makedirs("chats", exist_ok=True)

# ---------------------------
# CSS / Styling (WhatsApp-like)
# ---------------------------
st.markdown(
    """
<style>
/* hide streamlit menu/header/footer */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }

/* page background */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg,#0b0b0d,#111118);
  color: #e6eef8;
  font-family: Inter, Arial, sans-serif;
}

/* sidebar styling */
section[data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(180deg,#0f0f12,#121217);
  padding: 16px;
  border-radius: 10px;
}

/* page-level subtle divider (replaces hr) */
.home-divider { height:1px; background:#232528; margin:18px 0; border-radius:1px; }

/* chat area */
.whatsapp-container {
  max-width: 980px;
  margin: 18px auto;
  background: transparent;
  padding: 12px;
}
.chat-header {
  display:flex; align-items:center; gap:12px;
  padding: 12px 8px; margin-bottom: 6px;
  border-radius: 10px;
}
.chat-header .title { font-size:20px; font-weight:700; color:#fff; }
.chat-window {
  background: #0f1114;
  padding: 16px;
  border-radius: 12px;
  height: 60vh;
  overflow-y: auto;
  box-shadow: 0 6px 20px rgba(0,0,0,0.6);
  display:flex;
  flex-direction:column;
}

/* messages - left = bot, right = user */
.msg {
  display: block;
  margin: 10px 0;
  max-width: 78%;
  padding: 10px 14px;
  border-radius: 18px;
  line-height: 1.4;
  font-size: 15px;
  word-wrap: break-word;
}
.msg.bot {
  background: #ffffff;
  color: #111;
  border-bottom-left-radius: 4px;
  align-self: flex-start;
  box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.msg.user {
  background: linear-gradient(90deg,#25D366,#128C7E);
  color: white;
  border-bottom-right-radius: 4px;
  align-self: flex-end;
  box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}

/* message row */
.msg-row { display:flex; gap:10px; align-items:flex-end; }
.msg-row.left { justify-content:flex-start; }
.msg-row.right { justify-content:flex-end; }

/* avatar */
.avatar {
  width:36px; height:36px; border-radius:18px;
  background:#666; display:inline-block;
  flex: 0 0 36px;
}

/* input area */
.chat-input {
  margin-top: 12px;
  display:flex;
  gap:8px;
}
input.chat-text {
  flex:1; padding:10px 12px; border-radius:12px; background:#0b0c0f; color:#fff; border:1px solid #222;
}
button.send-btn {
  background:#25D366; border:none; color:#000; padding:10px 14px; border-radius:12px; font-weight:700;
}

/* small helpers */
.small-muted { color:#9aa3b2; font-size:12px; }
.card {
  background: linear-gradient(180deg,#0f1720,#0b1014);
  padding:14px; border-radius:10px; box-shadow: 0 8px 20px rgba(0,0,0,0.5);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers: parsing, persona, FAISS
# ---------------------------
def extract_bot_lines(raw_text, bot_name):
    bot_lines = []
    name_lower = bot_name.lower()
    for line in raw_text.splitlines():
        if not line or len(line.strip()) < 2:
            continue
        l = line.strip()
        if ":" in l:
            prefix, rest = l.split(":", 1)
            if prefix.strip().lower() == name_lower:
                bot_lines.append(rest.strip())
        elif len(l.split()) > 3:
            bot_lines.append(l)
    bot_lines = [b for b in bot_lines if len(b.split()) > 1]
    return "\n".join(bot_lines)

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
        # legacy client generate_content returns a dict-like result
        resp = genai_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        # support both dict style and attribute style
        if isinstance(resp, dict):
            return resp.get('message', {}).get('content', '').strip().split("\n")[0][:240]
        else:
            # fallback for other shapes
            txt = getattr(resp, "text", None) or str(resp)
            return txt.strip().split("\n")[0][:240]
    except Exception:
        return ""

@st.cache_resource(show_spinner=False)
def load_faiss_index(bot_text):
    bot_lines = [line.strip() for line in bot_text.splitlines() if line.strip()]
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(bot_lines, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return embed_model, index, bot_lines

# ---------------------------
# Session keys + state
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if "rename_bot_index" not in st.session_state:
    st.session_state.rename_bot_index = None
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None
if "clear_history_index" not in st.session_state:
    st.session_state.clear_history_index = None
if "show_login_on_home" not in st.session_state:
    st.session_state.show_login_on_home = False

# ---------------------------
# Sidebar: Login / Manage / Buy Lollipop
# ---------------------------
with st.sidebar:
    st.markdown("<div style='margin-bottom:6px;'><img src='https://i.imgur.com/8YqzJkL.png' width=36/> <b style='font-size:18px;color:#fff'>ChatDouble</b></div>", unsafe_allow_html=True)
    st.markdown("---")

    # auth UI
    st.subheader("🔐 Login / Register")
    if not st.session_state.logged_in:
        auth_mode = st.radio("", ["Login", "Register"], index=0)
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button(auth_mode):
            if auth_mode == "Login":
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            else:
                if register_user(username, password):
                    st.success("Registered — please login.")
                else:
                    st.error("Username exists.")
    else:
        st.markdown(f"👋 Logged in as **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown("---")

    # Manage dropdown (action + bot)
    st.subheader("🧰 Manage")
    manage_action = st.selectbox("Action", ["Select", "Rename", "Delete", "Clear history"])
    bots_for_manage = [b["name"] for b in get_user_bots(st.session_state.username)] if st.session_state.logged_in else []
    manage_bot = st.selectbox("Choose bot", ["--"] + bots_for_manage)
    if manage_action != "Select" and manage_bot != "--":
        if manage_action == "Rename":
            new_name = st.text_input("New name for " + manage_bot)
            if st.button("Save rename"):
                update_bot(st.session_state.username, manage_bot, new_name)
                st.success("Renamed.")
                st.rerun()
        elif manage_action == "Delete":
            if st.button("Confirm delete"):
                delete_bot(st.session_state.username, manage_bot)
                st.success("Deleted.")
                st.rerun()
        elif manage_action == "Clear history":
            if st.button("Confirm clear"):
                save_chat_history_cloud(st.session_state.username, manage_bot, [])
                st.success("History cleared.")
                st.rerun()

    st.markdown("---")

    # Upload small form (kept here but limited to 2 bots)
    st.subheader("📂 Upload Bot (max 2)")
    up_file = st.file_uploader("", type=["txt"], key="upload_file")
    up_name = st.text_input("Bot name", key="upload_name")
    if st.button("Upload bot file"):
        if not st.session_state.logged_in:
            st.error("Please login first.")
        else:
            user_bots = get_user_bots(st.session_state.username)
            if len(user_bots) >= 2:
                st.error("You can only have 2 bots. Delete one first.")
            elif not up_file or not up_name.strip():
                st.error("Please choose a file and give bot a name.")
            else:
                raw = up_file.read().decode("utf-8", "ignore")
                bot_lines = extract_bot_lines(raw, up_name)
                if not bot_lines.strip():
                    st.warning("Couldn't confidently parse speaker lines — storing best-effort.")
                    bot_lines = "\n".join([l for l in raw.splitlines() if len(l.split())>1])
                persona = generate_persona("\n".join(bot_lines.splitlines()[:40]))
                # add_bot stores file_text and persona in Firestore
                add_bot(st.session_state.username, up_name.capitalize(), bot_lines, persona=persona)
                st.success(f"Added {up_name} — persona: {persona or '—'}")
                st.rerun()

    st.markdown("---")

    # Buy Developer Lollipop (QR + UPI)
    st.subheader("🍭 Buy developer lollipop")
    upi_id = st.secrets.get("upi_id", "") if st.secrets else ""
    upi_qr = st.secrets.get("upi_qr_base64", "") if st.secrets else ""
    if upi_qr:
        try:
            img_bytes = base64.b64decode(upi_qr)
            st.image(img_bytes, width=200)
        except Exception:
            st.write("QR (invalid base64).")
    elif st.secrets and st.secrets.get("upi_qr_url"):
        st.image(st.secrets.get("upi_qr_url"), width=200)
    else:
        st.info("Add your UPI QR (base64) to Streamlit secrets `upi_qr_base64` or `upi_qr_url`.")

    if upi_id:
        st.markdown(f"**UPI:** `{upi_id}`")
    else:
        st.write("Add your UPI id in Streamlit secrets as `upi_id`.")
    st.markdown("---")
    st.markdown("<div class='small-muted'>Tip: set `upi_id` and `upi_qr_base64` in Streamlit secrets for quick purchase.</div>", unsafe_allow_html=True)

# ---------------------------
# Home screen + inline login
# ---------------------------
def render_home():
    st.markdown("<div class='whatsapp-container card'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;align-items:center;gap:14px;'><div style='width:56px;height:56px;border-radius:12px;background:#6c63ff;display:flex;align-items:center;justify-content:center;font-weight:700'>CD</div><div><h2 style='margin:0;color:#fff'>ChatDouble</h2><div class='small-muted'>Bring your friends back to chat — private, personal bots from your chat exports.</div></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='home-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;gap:12px;'><div style='flex:1'><div class='card'><h3>How it works</h3><ul style='margin-left:1em'><li>Upload chat export (.txt)</li><li>We extract that person’s messages and create a bot</li><li>Chat — replies mimic their tone</li></ul></div></div><div style='width:320px'><div class='card'><h3>Quick Start</h3><ol><li>Register / Login (sidebar)</li><li>Upload a chat (sidebar)</li><li>Open a bot and start chatting</li></ol></div></div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    # interactive CTA that reveals inline login (works on all screens)
    if st.button("🚀 Get Started — Login or Register"):
        st.session_state.show_login_on_home = True

    if st.session_state.get("show_login_on_home"):
        st.markdown("<div class='card' style='max-width:460px;margin:20px auto;padding:16px;'>", unsafe_allow_html=True)
        st.subheader("🔐 Quick Login / Register")
        home_user = st.text_input("Username", key="home_login_user")
        home_pass = st.text_input("Password", type="password", key="home_login_pass")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login", key="home_login_btn"):
                if login_user(home_user, home_pass):
                    st.session_state.logged_in = True
                    st.session_state.username = home_user
                    st.session_state.show_login_on_home = False
                    st.success(f"Welcome back, {home_user}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        with c2:
            if st.button("Register", key="home_reg_btn"):
                if register_user(home_user, home_pass):
                    st.success("Registered — please login.")
                else:
                    st.error("Username exists.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Show home when not logged in
if not st.session_state.logged_in:
    render_home()
    st.stop()

# ---------------------------
# User is logged in — main chat UI
# ---------------------------
user = st.session_state.username
user_bots = get_user_bots(user)  # list of dicts: {"name":..., "file":...} or with persona

if not user_bots:
    st.info("You have no bots yet. Upload one in the sidebar (max 2).")
    st.stop()

# layout: left=chat main, right=bot list/controls
col_main, col_side = st.columns([2, 0.9])

with col_side:
    st.markdown("<div class='card'><b>Your bots</b></div>", unsafe_allow_html=True)
    for b in user_bots:
        st.markdown(
            f"<div style='padding:8px;margin:6px 0;border-radius:8px;background:#0d1220;display:flex;justify-content:space-between;align-items:center'><div><b>{b['name']}</b><div class='small-muted'>{b.get('persona','')}</div></div><div style='color:#9aa3b2'>bots</div></div>",
            unsafe_allow_html=True
        )

with col_main:
    st.markdown("<div class='whatsapp-container'>", unsafe_allow_html=True)
    selected_bot = st.selectbox("Select bot", [b["name"] for b in user_bots])

    # tolerant get_bot_file: allow either (text, persona) or text only
    try:
        res = get_bot_file(user, selected_bot)
        if isinstance(res, tuple) or isinstance(res, list):
            bot_text, persona = res[0], (res[1] if len(res) > 1 else "")
        else:
            bot_text, persona = res, ""
    except Exception:
        # fallback
        bot_text, persona = "", ""

    if not bot_text or not bot_text.strip():
        st.warning("This bot has no stored messages. Upload better chat data for improved replies.")
        st.stop()

    embed_model, index, bot_lines = load_faiss_index(bot_text)
    chat_key = f"chat_{selected_bot}_{user}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = load_chat_history_cloud(user, selected_bot)

    # header
    st.markdown(f"<div class='chat-header card'><div class='title'>{selected_bot}</div><div style='margin-left:auto;color:#9aa3b2'>Persona: {persona or '—'}</div></div>", unsafe_allow_html=True)

    # chat window (display history)
    st.markdown("<div class='chat-window' id='chat-window'>", unsafe_allow_html=True)
    # ensure history shows in chronological order
    for entry in st.session_state[chat_key]:
        # show user message (right)
        if entry.get("user"):
            st.markdown(
                f"<div class='msg-row right'><div class='msg user'>{entry['user']}<div class='small-muted' style='text-align:right;font-size:11px'>{entry.get('ts', datetime.now().strftime('%I:%M %p'))}</div></div></div>",
                unsafe_allow_html=True
            )
        # show bot message (left)
        if entry.get("bot"):
            st.markdown(
                f"<div class='msg-row left'><div class='msg bot'>{entry['bot']}<div class='small-muted' style='margin-top:6px'>{entry.get('ts', datetime.now().strftime('%I:%M %p'))}</div></div></div>",
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # input
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    input_col, send_col = st.columns([8,1])
    with input_col:
        user_input = st.text_input("Type a message...", key="chat_input")
    with send_col:
        send = st.button("Send", key="send_btn")

    if send and user_input:
        # retrieval
        query_vector = embed_model.encode([user_input])
        D, I = index.search(query_vector, k=25)
        lines = []
        for idx in I[0]:
            if idx < len(bot_lines):
                line = bot_lines[idx].strip()
                if len(line.split()) > 2:
                    lines.append(line)
        context = "\n".join(lines[:12])
        if len(context) > 3000:
            context = context[:3000]

        persona_block = f"Persona: {persona}\n" if persona else ""
        # correct prompt (was accidentally overwritten in previous code)
        prompt = f"""{persona_block}
You are {selected_bot}, a real person who has previously chatted with the user.
Use the same tone, slang, and style AS BEFORE. Never act like an AI assistant.
Base your tone and vocabulary on the examples below.

Examples of how {selected_bot} talks:
{context}

User: {user_input}
{selected_bot}:
"""

        # choose model: use larger model for longer context
        model_name = "gemini-1.5-flash" if len(context) > 1200 else "gemini-2.0-flash"

        try:
            # legacy client: generate_content_stream(model=..., contents=...)
            resp = genai_client.models.generate_content_stream(
                model=model_name,
                contents=prompt
            )
            # append placeholder entry so UI shows user immediately
            ts = datetime.now().strftime('%I:%M %p')
            st.session_state[chat_key].append({"user": user_input, "bot": "...thinking", "ts": ts})
            save_chat_history_cloud(user, selected_bot, st.session_state[chat_key])

            # stream and update last message in place
            placeholder = st.empty()
            accumulated = ""
            for chunk in resp:
                # support dict-like and object-like chunk shapes
                text = ""
                if isinstance(chunk, dict):
                    text = chunk.get("message", {}).get("content", "") or chunk.get("text", "") or ""
                else:
                    text = getattr(chunk, "text", "") or ""
                if not text:
                    continue
                accumulated += text
                # update session history last bot content
                if st.session_state[chat_key] and isinstance(st.session_state[chat_key][-1], dict):
                    st.session_state[chat_key][-1]["bot"] = accumulated
                else:
                    st.session_state[chat_key].append({"user": user_input, "bot": accumulated, "ts": ts})
                # render current accumulated bot reply (left bubble)
                placeholder.markdown(f"<div class='msg-row left'><div class='msg bot'>{st.session_state[chat_key][-1]['bot']}<div class='small-muted' style='margin-top:6px'>{ts}</div></div></div>", unsafe_allow_html=True)

            bot_reply = st.session_state[chat_key][-1].get("bot", "")
        except Exception as e:
            st.error(f"⚠️ Gemini error: {e}")
            bot_reply = "⚠️ Error generating response."

        # persist final history and refresh UI
        save_chat_history_cloud(user, selected_bot, st.session_state[chat_key])
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # end whatsapp-container

# EOF
