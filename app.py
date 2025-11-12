# app.py — complete copy-paste replacement
import os
import json
import base64
from datetime import datetime

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss

# Use the same import shape you used earlier:
from google import genai

# firebase_db functions you already have in project:
from firebase_db import (
    get_user_bots, add_bot, delete_bot, update_bot, update_bot_persona,
    register_user, login_user, get_bot_file,
    save_chat_history_cloud, load_chat_history_cloud
)

# ---------------------------
# Page config + Gemini client
# ---------------------------
st.set_page_config(page_title="ChatDouble", page_icon="🤖", layout="wide")
API_KEY = os.getenv("GEMINI_API_KEY") or (st.secrets.get("GEMINI_API_KEY") if st.secrets else None)
if not API_KEY:
    # app should still load if missing key — show warning later where generation happens
    genai_client = None
else:
    genai_client = genai.Client(api_key=API_KEY)

os.makedirs("chats", exist_ok=True)


# ---------------------------
# CSS: WhatsApp-like + remove streamlit header/footer
# ---------------------------
st.markdown(
    """
<style>
/* hide menu/header/footer (keeps sidebar toggle) */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }

/* app bg */
[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at top right,#0b0b0d,#111118);
  color: #eaf0ff;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}

/* top tabs container spacing */
.stApp > main > div.block-container {
  padding-top: 18px;
  padding-left: 32px;
  padding-right: 32px;
}

/* main layout wrapper */
.main-chat-container {
  max-width: 1100px;
  margin: 0 auto;
}

/* chat header */
.chat-header {
  display:flex; align-items:center; justify-content:space-between;
  padding:12px 16px; border-radius:10px; margin-bottom:10px;
  background:linear-gradient(90deg,#0f1114,#0b0c0f);
  box-shadow: 0 6px 26px rgba(0,0,0,0.6);
}
.chat-header .title { font-size:20px; font-weight:700; color:#fff; }
.chat-header .subtitle { color:#9aa3b2; font-size:13px; }

/* chat window / WhatsApp look */
/* Chat container card */
.chat-card {
  background: #0d0d11;
  border-radius: 16px;
  box-shadow: 0 8px 25px rgba(0,0,0,0.6);
  height: 75vh;              /* fixed chat height */
  display: flex;
  flex-direction: column;
  overflow: hidden;          /* keep everything inside */
  position: relative;
}

/* Scrollable area for messages */
.chat-window {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  gap: 10px;
  padding: 18px 16px 10px 16px;
  scroll-behavior: smooth;
}

/* Hide scrollbars for clean look */
.chat-window::-webkit-scrollbar {
  width: 6px;
}
.chat-window::-webkit-scrollbar-thumb {
  background: #222;
  border-radius: 10px;
}

/* Message bubbles */
.msg-row { display: flex; }
.msg.user {
  align-self: flex-end;
  background: linear-gradient(90deg,#25D366,#128C7E);
  color: #fff;
  padding: 10px 14px;
  border-radius: 18px 18px 4px 18px;
  margin-left: auto;
  max-width: 70%;
  word-wrap: break-word;
  font-size: 15px;
}
.msg.bot {
  align-self: flex-start;
  background: #fff;
  color: #111;
  padding: 10px 14px;
  border-radius: 18px 18px 18px 4px;
  margin-right: auto;
  max-width: 70%;
  word-wrap: break-word;
  font-size: 15px;
}
.ts {
  display: block;
  font-size: 10px;
  color: #999;
  margin-top: 4px;
  text-align: right;
}

/* input row */
.input-row { display:flex; gap:10px; margin-top:12px; }
input.chat-input { flex:1; padding:12px 14px; border-radius:12px; border:1px solid #202124; background:#0f1114; color:#fff; }
button.send-btn { background:#25D366; color:#000; border:none; padding:10px 14px; border-radius:10px; font-weight:700; }

/* small card */
.card { background: linear-gradient(180deg,#0f1720,#0b1014); padding:14px; border-radius:10px; box-shadow: 0 8px 20px rgba(0,0,0,0.5); color:#e6eef8; }
.small-muted { color:#9aa3b2; font-size:13px; }

</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------
# Helpers: text extraction, persona, FAISS
# ---------------------------
def extract_bot_lines(raw_text: str, bot_name: str) -> str:
    """
    Heuristic: collect lines starting with 'Name:' (case-insensitive).
    Fallback: include long lines ( > 3 words ).
    """
    bot_lines = []
    name_lower = (bot_name or "").strip().lower()
    for line in raw_text.splitlines():
        if not line or len(line.strip()) < 2:
            continue
        l = line.strip()
        if ":" in l:
            prefix, rest = l.split(":", 1)
            if prefix.strip().lower() == name_lower:
                bot_lines.append(rest.strip())
        else:
            if len(l.split()) > 3:
                bot_lines.append(l)
    bot_lines = [b for b in bot_lines if len(b.split()) > 1]
    return "\n".join(bot_lines)


def generate_persona(text_examples: str) -> str:
    """
    Ask Gemini for a short persona description.
    Keep temperature low for deterministic output.
    Tolerant if no genai client is configured.
    """
    if not text_examples or not genai_client:
        return ""
    prompt = f"""Take these example messages from a single person and write a 1-2 sentence persona description capturing their tone, slang, and typical phrases.

Examples:
{text_examples}

Return only the short persona description.
"""
    try:
        resp = genai_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            options={"temperature": 0.2, "max_output_tokens": 120}
        )
        # support dict-like and object-like responses
        if isinstance(resp, dict):
            text = resp.get("message", {}).get("content", "") or ""
        else:
            text = getattr(resp, "text", None) or str(resp)
        return text.strip().splitlines()[0][:240]
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def build_faiss_for_bot(bot_text: str):
    """
    Returns (embed_model, faiss_index, bot_lines list)
    Cached per content string.
    """
    bot_lines = [line.strip() for line in bot_text.splitlines() if line.strip()]
    if not bot_lines:
        # minimal fallback: single placeholder
        bot_lines = ["hello"]
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(bot_lines, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return embed_model, index, bot_lines


# ---------------------------
# Session state defaults
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if "show_inline_login" not in st.session_state:
    st.session_state.show_inline_login = False


# ---------------------------
# Minimal sidebar: login/logout only
# ---------------------------
with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:10px;'><div style='width:44px;height:44px;border-radius:10px;background:#6c63ff;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700'>CD</div><div><b style='font-size:16px;color:#fff'>ChatDouble</b><div class='small-muted'>Personal chatbots from exports</div></div></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🔐 Account")
    if not st.session_state.logged_in:
        mode = st.radio("", ["Login", "Register"], index=0)
        username_input = st.text_input("Username", key="sb_user")
        password_input = st.text_input("Password", type="password", key="sb_pass")
        if st.button(mode):
            if mode == "Login":
                if not username_input.strip() or not password_input.strip():
                    st.error("Enter both fields.")
                else:
                    ok = False
                    try:
                        ok = login_user(username_input, password_input)
                    except Exception as e:
                        st.error(f"Auth error: {e}")
                        ok = False
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username = username_input
                        st.success(f"Welcome, {username_input}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
            else:
                try:
                    ok = register_user(username_input, password_input)
                except Exception as e:
                    st.error(f"Register error: {e}")
                    ok = False
                if ok:
                    st.success("Registered — please login.")
                else:
                    st.error("Username exists.")
    else:
        st.markdown(f"👋 Logged in as **{st.session_state.username}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
    st.markdown("---")
    st.markdown("<div class='small-muted'>Pro tip: manage bots and upload files inside the Manage tab (no sidebar actions required).</div>", unsafe_allow_html=True)


# ---------------------------
# Tabs: Home | Chat | Manage | Buy
# ---------------------------

if not st.session_state.logged_in:
    # Unauthenticated view: show only Home
    # ----- Home tab -----
    st.markdown("<div class='main-chat-container'>", unsafe_allow_html=True)
    st.markdown("<div class='chat-header'><div class='title'>ChatDouble</div><div class='subtitle'>Bring your friends back to chat — private bots from your chat exports.</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card' style='padding:18px; margin-bottom:14px'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0;color:#fff'>How it works</h3>", unsafe_allow_html=True)
    st.markdown("<ul><li>Upload a chat export (.txt) in Manage tab</li><li>We extract that person's messages and create a bot</li><li>Chat — replies mimic their tone</li></ul>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c2:
        st.markdown("<div class='card'><h4>Your Quick Start</h4><ol><li>Register / Login (sidebar)</li><li>Upload a chat in Manage</li><li>Open Chat tab and select bot</li></ol></div>", unsafe_allow_html=True)
    with c1:
        if st.button("🚀 Get Started — Login or Register"):
            st.session_state.show_inline_login = True

    if st.session_state.show_inline_login and not st.session_state.logged_in:
        st.markdown("<div class='card' style='max-width:480px;margin-top:18px'>", unsafe_allow_html=True)
        st.subheader("Quick Login")
        h_user = st.text_input("Username", key="home_user")
        h_pass = st.text_input("Password", type="password", key="home_pass")
        cola, colb = st.columns(2)
        with cola:
            if st.button("Login", key="home_login_btn"):
                if login_user(h_user, h_pass):
                    st.session_state.logged_in = True
                    st.session_state.username = h_user
                    st.success("Logged in.")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        with colb:
            if st.button("Register", key="home_reg_btn"):
                if register_user(h_user, h_pass):
                    st.success("Registered! Now login.")
                else:
                    st.error("Username exists.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Authenticated view: hide Home, show main app
    tabs = st.tabs(["💬 Chat", "🧰 Manage Bots", "🍭 Buy Lollipop"])
# ----- Chat tab -----
    with tabs[0]:
        if not st.session_state.logged_in:
            st.warning("Please log in (sidebar) to open your bots.")
            st.stop()
    
        user = st.session_state.username
        try:
            user_bots = get_user_bots(user) or []
        except Exception as e:
            st.error(f"Firestore error: {e}")
            st.stop()
    
        if not user_bots:
            st.info("No bots found. Create one in the Manage tab.")
            st.stop()
    
        # Side-by-side layout: chat main and bot list
        col_main, col_side = st.columns([2, 0.9])
        with col_side:
            st.markdown("<div class='card'><b>Your bots</b></div>", unsafe_allow_html=True)
            for b in user_bots:
                st.markdown(f"<div style='padding:8px;margin:8px 0;border-radius:8px;background:#0d1220;'><b>{b['name']}</b><div class='small-muted'>{b.get('persona','')}</div></div>", unsafe_allow_html=True)
    
        with col_main:
            st.markdown("<div class='main-chat-container'>", unsafe_allow_html=True)
            selected_bot = st.selectbox("Select bot", [b["name"] for b in user_bots], key="chat_selected_bot")
    
            # tolerant get_bot_file (support either string or (text,persona) tuple)
            try:
                res = get_bot_file(user, selected_bot)
            except Exception as e:
                st.error(f"Could not load bot: {e}")
                st.stop()
    
            if isinstance(res, (list, tuple)):
                bot_text = res[0] if len(res) > 0 else ""
                persona = res[1] if len(res) > 1 else ""
            else:
                bot_text = res or ""
                persona = ""
    
            if not bot_text.strip():
                st.warning("This bot has no stored messages. Upload better chat data in Manage tab.")
                st.stop()
    
            embed_model, index, bot_lines = build_faiss_for_bot(bot_text)
            chat_key = f"chat_{selected_bot}_{user}"
            if chat_key not in st.session_state:
                st.session_state[chat_key] = load_chat_history_cloud(user, selected_bot) or []
    
            # header
            st.markdown(f"<div class='chat-header'><div class='title'>{selected_bot}</div><div class='subtitle'>Persona: {persona or '—'}</div></div>", unsafe_allow_html=True)

            # ---------------------------
            # Chat message area (render as a single HTML block so messages stay inside the card)
            # ---------------------------
            messages = st.session_state.get(chat_key, [])
            # build html for messages (chronological)
            msgs_html = []
            for entry in messages:
                ts = entry.get("ts", datetime.now().strftime("%I:%M %p"))
                # user on right
                if entry.get("user"):
                    safe_user = str(entry["user"]).replace("<", "&lt;").replace(">", "&gt;")
                    msgs_html.append(
                        f"<div class='msg-row'><div class='msg user'>{safe_user}<span class='ts'>{ts}</span></div></div>"
                    )
                # bot on left
                if entry.get("bot"):
                    safe_bot = str(entry["bot"]).replace("<", "&lt;").replace(">", "&gt;")
                    msgs_html.append(
                        f"<div class='msg-row'><div class='msg bot'>{safe_bot}<span class='ts'>{ts}</span></div></div>"
                    )
            
            all_html = (
                "<div class='chat-card'>"
                "<div class='chat-window' id='chat-window'>"
                + "".join(msgs_html) +
                "</div></div>"
            )
            
            st.markdown(all_html, unsafe_allow_html=True)
            
            # Auto-scroll to bottom — use document, not window.parent
            # (Streamlit executes this after the element is added)
            st.markdown("""
            <script>
            (function() {
              function scrollChatToBottom() {
                try {
                  const chatWin = document.getElementById("chat-window");
                  if (chatWin) {
                    chatWin.scrollTop = chatWin.scrollHeight;
                  }
                } catch (e) {
                  // fail silently
                  console.log("scroll error", e);
                }
              }
              // run after a short timeout to let Streamlit mount HTML
              setTimeout(scrollChatToBottom, 60);
            })();
            </script>
            """, unsafe_allow_html=True)
            

            # input row
            st.markdown("<div class='input-row'>", unsafe_allow_html=True)
            user_input = st.text_input("Type a message...", key="chat_input")
            send = st.button("Send", key="send_btn")
            st.markdown("</div>", unsafe_allow_html=True)
    
            if send and user_input.strip():
                # retrieval
                try:
                    query_vector = embed_model.encode([user_input])
                    D, I = index.search(query_vector, k=20)
                except Exception:
                    D, I = None, [[]]
                lines = []
                if I is not None:
                    for idx in I[0]:
                        if idx < len(bot_lines):
                            candidate = bot_lines[idx].strip()
                            if len(candidate.split()) > 2:
                                lines.append(candidate)
                context = "\n".join(lines[:12])
                if len(context) > 3000:
                    context = context[:3000]
    
                persona_block = f"Persona: {persona}\n\n" if persona else ""
                prompt = f"""{persona_block}You are {selected_bot}, a real person who has previously chatted with the user.
    Use the same tone, slang, and style as the examples below. Never act like an AI assistant.
    
    Examples:
    {context}
    
    User: {user_input}
    {selected_bot}:
    """
    
                # add user message to history immediately (so UI shows it)
                ts = datetime.now().strftime("%I:%M %p")
                st.session_state[chat_key].append({"user": user_input, "bot": "...thinking", "ts": ts})
                save_chat_history_cloud(user, selected_bot, st.session_state[chat_key])
                st.rerun()
    
            st.markdown("</div>", unsafe_allow_html=True)
    
    
    # ----- Manage Bots tab -----
    with tabs[1]:
        if not st.session_state.logged_in:
            st.warning("Please log in (sidebar) to manage your bots.")
            st.stop()
    
        user = st.session_state.username
        st.markdown("<div class='card'><h4>Upload chat export (.txt) — max 2 bots</h4>", unsafe_allow_html=True)
        up_file = st.file_uploader("Choose .txt file", type=["txt"], key="manage_upload")
        up_name = st.text_input("Bot name (example: John)", key="manage_name")
        if st.button("Upload bot", key="manage_upload_btn"):
            try:
                user_bots = get_user_bots(user) or []
            except Exception as e:
                st.error(f"Could not check existing bots: {e}")
                user_bots = []
            if len(user_bots) >= 2:
                st.error("You already have 2 bots. Delete one first.")
            elif (not up_file) or (not up_name.strip()):
                st.error("Please provide both file and name.")
            else:
                raw = up_file.read().decode("utf-8", "ignore")
                bot_lines = extract_bot_lines(raw, up_name)
                if not bot_lines.strip():
                    # fallback to storing longer lines
                    bot_lines = "\n".join([l for l in raw.splitlines() if len(l.split()) > 1])
                persona = generate_persona("\n".join(bot_lines.splitlines()[:40]))
                try:
                    add_bot(user, up_name.capitalize(), bot_lines, persona=persona)
                    st.success(f"Added {up_name} — persona: {persona or '—'}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload error: {e}")
    
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Manage existing bots UI
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<h4>Your bots</h4>", unsafe_allow_html=True)
        try:
            user_bots = get_user_bots(user) or []
        except Exception as e:
            st.error(f"Firestore error: {e}")
            user_bots = []
    
        for b in user_bots:
            st.markdown(f"**{b['name']}** — Persona: {b.get('persona','—')}")
            rn, dlt, clr = st.columns([1,1,1])
            with rn:
                new_name = st.text_input(f"Rename {b['name']}", key=f"rename_{b['name']}")
                if st.button("Rename", key=f"rename_btn_{b['name']}"):
                    if new_name.strip():
                        try:
                            update_bot(user, b['name'], new_name.strip())
                            st.success("Renamed.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Rename error: {e}")
                    else:
                        st.error("Enter a new name.")
            with dlt:
                if st.button("Delete", key=f"del_{b['name']}"):
                    try:
                        delete_bot(user, b['name'])
                        st.warning("Deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete error: {e}")
            with clr:
                if st.button("Clear history", key=f"clr_{b['name']}"):
                    try:
                        save_chat_history_cloud(user, b['name'], [])
                        st.success("History cleared.")
                    except Exception as e:
                        st.error(f"Clear error: {e}")
    
    
    # ----- Buy Lollipop tab -----
    with tabs[2]:
        st.markdown("<div class='card'><h4>Buy developer a lollipop 🍭</h4>", unsafe_allow_html=True)
    
        upi_id = st.secrets.get("upi_id") if st.secrets else None
        upi_qr_url = st.secrets.get("upi_qr_url") if st.secrets else None
        upi_qr_b64 = st.secrets.get("upi_qr_base64") if st.secrets else None
    
        # ✅ handle base64-encoded QR
        if upi_qr_b64:
            try:
                img_bytes = base64.b64decode(upi_qr_b64)
                st.image(img_bytes, width=220)
            except Exception:
                st.info("⚠️ Invalid base64 QR in secrets — please check your `upi_qr_base64` value.")
    
        # ✅ handle external URL QR (http/https only)
        elif upi_qr_url and isinstance(upi_qr_url, str):
            if upi_qr_url.lower().startswith("http"):
                st.image(upi_qr_url, width=220)
            else:
                st.info("⚠️ Invalid `upi_qr_url` format — must start with http/https (not a local path).")
    
        # ✅ no QR configured
        else:
            st.info("No QR configured. Add `upi_qr_url` or `upi_qr_base64` in Streamlit secrets.")
    
        # ✅ UPI ID display
        if upi_id:
            st.markdown(f"**UPI ID:** `{upi_id}`")
        else:
            st.markdown("Add `upi_id` to Streamlit secrets to show UPI ID.")
    
        st.markdown("<div class='small-muted'>Tip: add secrets via Streamlit Cloud → Settings → Secrets to show QR & UPI ID publicly in this tab.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    
# ---------------------------
# Final: keep consistent behavior
# ---------------------------
# If user pressed Send inside Chat tab, we appended "...thinking" and re-ran.
# Now, handle streaming generation (separate block that runs after rerun).
# We detect any chat entries that have bot == "...thinking" and generate for them.
def process_pending_generation():
    # Only meaningful when logged in and chat selected
    if not st.session_state.logged_in:
        return
    user = st.session_state.username
    selected_key = None
    # find any chat keys for this user that have a pending entry
    for k in list(st.session_state.keys()):
        if k.startswith("chat_") and k.endswith(f"_{user}"):
            msgs = st.session_state[k]
            if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("bot") == "...thinking":
                selected_key = k
                break
    if not selected_key:
        return

    # extract selected bot name
    # format: chat_{bot}_{user}
    try:
        parts = selected_key.split("_")
        # join middle parts as bot name might contain underscores
        bot_name = "_".join(parts[1:-1])
    except Exception:
        return

    msgs = st.session_state[selected_key]
    pending = msgs[-1]
    user_input = pending.get("user", "")
    if not user_input:
        # cleanup
        msgs[-1]["bot"] = "⚠️ No user input found."
        save_chat_history_cloud(user, bot_name, st.session_state[selected_key])
        return

    # prepare context using the bot file (if exists)
    try:
        res = get_bot_file(user, bot_name)
        if isinstance(res, (list, tuple)):
            bot_text = res[0]
            persona = res[1] if len(res) > 1 else ""
        else:
            bot_text = res or ""
            persona = ""
    except Exception:
        bot_text = ""
        persona = ""

    if not bot_text:
        pending["bot"] = "⚠️ No bot source text available."
        save_chat_history_cloud(user, bot_name, st.session_state[selected_key])
        return

    # build FAISS (fast cached)
    embed_model, index, bot_lines = build_faiss_for_bot(bot_text)
    # retrieval for extra context
    try:
        qvec = embed_model.encode([user_input])
        D, I = index.search(qvec, k=20)
    except Exception:
        I = [[]]
    lines = []
    if I is not None:
        for idx in I[0]:
            if idx < len(bot_lines):
                candidate = bot_lines[idx].strip()
                if len(candidate.split()) > 2:
                    lines.append(candidate)
    context = "\n".join(lines[:12])
    if len(context) > 3000:
        context = context[:3000]

    persona_block = f"Persona: {persona}\n\n" if persona else ""
    prompt = f"""{persona_block}You are {bot_name}, a real person who has previously chatted with the user.
Use the same tone, slang, and style as the examples below. Never act like an AI assistant.

Examples:
{context}

User: {user_input}
{bot_name}:
"""

    # generate (stream if possible)
    if not genai_client:
        pending["bot"] = "⚠️ Gemini API key not set. Add GEMINI_API_KEY to environment or Streamlit secrets."
        save_chat_history_cloud(user, bot_name, st.session_state[selected_key])
        return

    # choose model conservatively
    model_name = "gemini-2.0-flash-exp"  # general model; change if you prefer flash versions
    try:
        # use streaming if available in your genai client
        resp_iter = genai_client.models.generate_content_stream(model=model_name, contents=prompt)
    except Exception:
        # fallback to single-shot
        try:
            resp = genai_client.models.generate_content(model=model_name, contents=prompt)
            # extract text
            if isinstance(resp, dict):
                text = resp.get("message", {}).get("content", "") or ""
            else:
                text = getattr(resp, "text", None) or str(resp)
            pending["bot"] = text.strip()
            pending["ts"] = datetime.now().strftime("%I:%M %p")
            save_chat_history_cloud(user, bot_name, st.session_state[selected_key])
            return
        except Exception as e:
            pending["bot"] = f"⚠️ Error generating response: {e}"
            pending["ts"] = datetime.now().strftime("%I:%M %p")
            save_chat_history_cloud(user, bot_name, st.session_state[selected_key])
            return

    # stream handling
    accumulated = ""
    try:
        for chunk in resp_iter:
            # chunk may be dict-like or obj-like
            text = ""
            if isinstance(chunk, dict):
                text = chunk.get("message", {}).get("content", "") or chunk.get("text", "") or ""
            else:
                text = getattr(chunk, "text", "") or ""
            if not text:
                continue
            accumulated += text
            # update pending bot text in session
            st.session_state[selected_key][-1]["bot"] = accumulated
            st.session_state[selected_key][-1]["ts"] = datetime.now().strftime("%I:%M %p")
            # persist partial (optionally)
            save_chat_history_cloud(user, bot_name, st.session_state[selected_key])
        # final
        st.session_state[selected_key][-1]["bot"] = accumulated.strip()
        st.session_state[selected_key][-1]["ts"] = datetime.now().strftime("%I:%M %p")
        save_chat_history_cloud(user, bot_name, st.session_state[selected_key])
    except Exception as e:
        st.session_state[selected_key][-1]["bot"] = f"⚠️ Error generating response: {e}"
        st.session_state[selected_key][-1]["ts"] = datetime.now().strftime("%I:%M %p")
        save_chat_history_cloud(user, bot_name, st.session_state[selected_key])
        return


# run generation post-render (non-blocking style — runs during this request)
process_pending_generation()
# end of file
