# app.py — complete copy-paste replacement
import os
import json
import base64
from datetime import datetime

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss

# Use the same import shape you used earlier:
import google.genai as genai


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
def extract_bot_lines(raw_text, bot_name):
    """
    Extract only that person's messages from WhatsApp-style chat exports.
    Supports formats like:
    12/04/2023, 5:22 pm - Raykay: message
    """
    bot_lines = []
    name_lower = bot_name.strip().lower()

    for line in raw_text.splitlines():
        if "-" not in line or ":" not in line:
            continue

        try:
            # Example: "12/04/2023, 5:22 pm - Raykay: Hello"
            meta, msg = line.split("-", 1)
            speaker, content = msg.split(":", 1)
            speaker = speaker.strip().lower()
            content = content.strip()
        except:
            continue

        if speaker == name_lower and len(content.split()) > 1:
            # remove emojis or keep? keep them.
            bot_lines.append(content)

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
            model="gemini-2.0-flash-exp",
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
# The above code is creating a user interface for a chatbot application using Streamlit in Python. It
# includes a sidebar with a logo and information about the application, an account section where users
# can login or register, and a section displaying the user's login status. The code handles user
# authentication, such as logging in, registering, and logging out. It also provides a tip for
# managing bots and uploading files.
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
st.markdown("<div class='chat-header'><div class='title'>ChatDouble</div><div class='subtitle'>&nbsp&nbspBring your friends back to chat — private bots from your chat exports.</div></div>", unsafe_allow_html=True)    
if not st.session_state.logged_in:
    # Unauthenticated view: show only Home
    # ----- Home tab -----
    st.markdown("<div class='main-chat-container'>", unsafe_allow_html=True)
    # st.markdown("<div class='card' style='padding:18px; margin-bottom:14px'>", unsafe_allow_html=True)
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
        st.markdown("<div class='card' style='max-width:100%;margin-top:20px'>", unsafe_allow_html=True)
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
        user = st.session_state.username
        user_bots = get_user_bots(user)

        if not user_bots:
            st.info("No bots yet. Create one in Manage Bots tab.")
        else:

            col_main, col_side = st.columns([2, 0.9])

            # Right side bot list
            with col_side:
                st.markdown("<div class='card'><b>Your Bots</b></div>", unsafe_allow_html=True)
                for b in user_bots:
                    st.markdown(
                        f"<div style='padding:8px;margin:8px 0;border-radius:8px;background:#0d1220;'>"
                        f"<b>{b['name']}</b><div class='small-muted'>{b.get('persona','')}</div></div>",
                        unsafe_allow_html=True
                    )

            # Left side main chat
            with col_main:
                selected_bot = st.selectbox("Select bot", [b["name"] for b in user_bots], key="chat_selected_bot")

                # Load bot file
                res = get_bot_file(user, selected_bot)
                if isinstance(res, (list, tuple)):
                    bot_text = res[0]
                    persona = res[1] if len(res) > 1 else ""
                else:
                    bot_text = res
                    persona = ""

                if not bot_text.strip():
                    st.warning("Bot has no data.")
                    st.stop()

                embed_model, index, bot_lines = build_faiss_for_bot(bot_text)

                chat_key = f"chat_{selected_bot}_{user}"
                if chat_key not in st.session_state:
                    st.session_state[chat_key] = load_chat_history_cloud(user, selected_bot) or []

                # Header
                st.markdown(
                    f"<div class='chat-header'><div class='title'>{selected_bot}</div>"
                    f"<div class='subtitle'>Persona: {persona or '—'}</div></div>",
                    unsafe_allow_html=True
                )

                # CHAT CARD
                from streamlit.components.v1 import html as components_html

                messages = st.session_state[chat_key]

                # Convert to a simpler format for JS
                clean_history = []
                for m in messages:
                    if "user" in m:
                        clean_history.append({"role": "user", "content": m["user"]})
                    if "bot" in m:
                        clean_history.append({"role": "bot", "content": m["bot"]})

                history_json = json.dumps(clean_history)

                iframe_html = f"""
                <!doctype html>
                <html>
                <head>
                <meta charset="utf-8">
                <style>
                body {{
                  margin: 0;
                  background: transparent;
                  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
                }}

                .chat-box {{
                    height: 100vh;
                    overflow-y: scroll;
                    padding: 12px;
                    box-sizing: border-box;
                    scrollbar-width: none;         /* Firefox */
                }}

                .chat-box::-webkit-scrollbar {{
                    display: none;                 /* Chrome */
                }}

                .msg {{
                    display: inline-block;
                    max-width: 80%;
                    padding: 10px 14px;
                    margin-bottom: 8px;
                    font-size: 15px;
                    border-radius: 16px;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}


                .user {{
                    background: linear-gradient(90deg,#25D366,#128C7E);
                    color: white;
                    margin-left: auto;
                    border-radius: 16px 16px 4px 16px;
                }}

                .bot {{
                    background: white;
                    color: #111;
                    margin-right: auto;
                    border-radius: 16px 16px 16px 4px;
                }}

                @media (max-width: 600px) {{
                    .msg {{
                    display: inline-block;
                    max-width: 80%;
                    padding: 10px 14px;
                    margin-bottom: 8px;
                    border-radius: 16px;
                    font-size: 15px;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}

                }}

                </style>
                </head>
                <body>

                <div id="chat" class="chat-box"></div>

                <script>
                const history = {history_json};

                function renderChat() {{
                    const box = document.getElementById("chat");
                    box.innerHTML = "";

                    history.forEach(turn => {{
                        const row = document.createElement("div");
                        row.style.display = "flex";
                        row.style.marginBottom = "6px";

                        if (turn.role === "user") row.style.justifyContent = "flex-end";
                        else row.style.justifyContent = "flex-start";

                        const bubble = document.createElement("div");
                        bubble.className = "msg " + turn.role;
                        bubble.textContent = turn.content;

                        row.appendChild(bubble);
                        box.appendChild(row);
                    }});


                    box.scrollTop = box.scrollHeight;
                    setTimeout(() => box.scrollTop = box.scrollHeight, 50);
                }}

                renderChat();

                const observer = new MutationObserver(() => {{
                    const box = document.getElementById("chat");
                    box.scrollTop = box.scrollHeight;
                }});
                observer.observe(document.getElementById("chat"), {{ childList: true }});

                </script>


                </body>
                </html>
                """

                components_html(iframe_html, height=500, scrolling=False)

                # --- ensure we clear the text_input BEFORE widget is created (safe) ---
                if st.session_state.get("pending_clear", False):
                    # clear the stored value (widget not yet instantiated)
                    st.session_state["chat_input_box"] = ""
                    st.session_state["pending_clear"] = False

                # INPUT BAR
                # --- INPUT BAR (fixed layout: no gap, button inline) ---
                st.markdown("""
                <style>
                .input-wrapper {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-top: 0px !important;   /* remove extra gap */
                    padding-top: 6px;             /* small clean spacing */
                }

                .input-wrapper input {
                    flex: 1;
                    height: 42px;
                    border-radius: 12px;
                    padding: 10px 14px;
                    border: 1px solid #202124;
                    background: #0f1114;
                    color: white;
                    outline: none;
                }

                .send-btn-fixed {
                    background: #25D366;
                    border: none;
                    padding: 12px 16px;
                    border-radius: 12px;
                    cursor: pointer;
                    font-weight: bold;
                    font-size: 16px;
                }
                </style>
                """, unsafe_allow_html=True)

                # Place input + send button on same row exactly
                inp_col, btn_col = st.columns([10, 1])

                with inp_col:
                    user_msg = st.text_input(
                        "",
                        key="chat_input_box",
                        label_visibility="collapsed",
                        placeholder="Type…"
                    )

                    st.markdown("""
        <script>
            setTimeout(function() {
                const box = window.parent.document.querySelector('input[id="chat_input_box"]');
                if (box) { box.focus(); }
            }, 300);
        </script>
    """, unsafe_allow_html=True)

                with btn_col:
                    send = st.button("➤", key="send_chat_btn", use_container_width=True)


                if send and user_msg.strip():
                    ts = datetime.now().strftime("%I:%M %p")
                    st.session_state[chat_key].append({"user": user_msg, "bot": "...Typing", "ts": ts})
                    save_chat_history_cloud(user, selected_bot, st.session_state[chat_key])

                    # Retrieval
                    vec = embed_model.encode([user_msg])
                    _, idxs = index.search(vec, k=20)
                    retrieved = "\n".join([bot_lines[i] for i in idxs[0] if i < len(bot_lines)])[:2000]
                    
                    # === Build recent history (all messages in this chat) ===
                    history_lines = []
                    for entry in st.session_state.get(chat_key, []):
                        if "user" in entry:
                            history_lines.append(f"User: {entry['user']}")
                        if "bot" in entry:
                            history_lines.append(f"{selected_bot}: {entry['bot']}")

                    recent_history = "\n".join(history_lines)
                    if len(recent_history) > 4000:
                        recent_history = recent_history[-4000:]

                    # Limit retrieved examples for safety
                    retrieved_examples = retrieved or ""
                    if len(retrieved_examples) > 3000:
                        retrieved_examples = retrieved_examples[:3000]

                    persona_block = f"Persona: {persona}\n\n" if persona else ""

                    prompt = f"""{persona_block}
You are a real person (not an AI) who previously chatted with the user.

RULES:
1) Use all facts found in the 'Recent conversation' below. Do NOT contradict them.
2) Detect your real name from the examples (the text before ":" in the examples). 
Do NOT invent names or placeholders like {{User's Name}}.
3) If persona above is empty, infer a personality from the examples & stick to it.
4) If you don't know a fact, ask — don't assume.
STRICT RULES:
- NEVER use placeholders like [User], [User's Name], {user}, <name>, or anything inside {{}}, [], <>.
- NEVER guess names. ONLY use names that actually exist inside the real chat data.
- If you do NOT know a name from the real examples, say “I don’t know, you never told me.”
- NEVER invent formatting like **bold**, __underline__, *, ~, or any markdown.
- NEVER use too many emojis in a reply, use them as same frequency in chat. Keep it natural, not exaggerated and hallucinated.
- NEVER talk like an assistant or narrator. Just speak casually like in the chat data.

--- Recent conversation ---
{recent_history}

--- Examples from real exported chat ---
{retrieved_examples}

Continue the conversation naturally, same tone and slang.

User: {user_msg}
{selected_bot}:
"""



                    reply = "..."

                    try:
                        resp = genai_client.models.generate_content(
                            model="gemini-2.0-flash-exp",
                            contents=prompt
                        )
                        reply = getattr(resp, "text", None) or (resp.get("message", {}).get("content", "") if isinstance(resp, dict) else "") or "⚠️Offline (Text after sometime)"
                    except Exception:
                        try:
                            resp = genai_client.models.generate_content(
                                model="gemini-2.0-flash",
                                contents=prompt
                            )
                            reply = getattr(resp, "text", None) or (resp.get("message", {}).get("content", "") if isinstance(resp, dict) else "") or "⚠️Offline (Text after sometime)"
                        except Exception as e:
                            reply = f"⚠️Offline (Try after sometime): {e}"

                    st.session_state[chat_key][-1]["bot"] = reply
                    st.session_state[chat_key][-1]["ts"] = datetime.now().strftime("%I:%M %p")
                    save_chat_history_cloud(user, selected_bot, st.session_state[chat_key])

                    # mark that input must be cleared on next rerun (safe)
                    st.session_state["pending_clear"] = True

                    # rerun so iframe re-renders with updated history and cleared input
                    st.rerun()

    # ----- Manage Bots tab -----
    with tabs[1]:
        if not st.session_state.logged_in:
            st.warning("Please log in to manage your bots.")
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
            st.error(f"Database error: {e}")
            user_bots = []
    
        for b in user_bots:
            st.markdown(f"**{b['name']}** : {b.get('persona','—')}")
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
    
        upi_id = "kolimohit9595-1@okicici"
        upi_qr_url = "https://raw.githubusercontent.com/Mayurkoli8/ChatDouble/refs/heads/main/download.png"
        
        # ✅ handle external URL QR (http/https only)
        if upi_qr_url and isinstance(upi_qr_url, str):
            if upi_qr_url.lower().startswith("http"):
                st.image(upi_qr_url, width=220)
            else:
                st.info("⚠️ Invalid `upi_qr_url` format — must start with http/https (not a local path).")
        # ✅ Show UPI ID
        st.markdown(f"<h4>UPI ID: <code>{upi_id}</code></h4>", unsafe_allow_html=True)

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
            if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("bot") == "...typing":
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
    # === Build recent history ===
    history_lines = []
    for entry in msgs:
        if "user" in entry:
            history_lines.append(f"User: {entry['user']}")
        if "bot" in entry:
            history_lines.append(f"{bot_name}: {entry['bot']}")
    
    recent_history = "\n".join(history_lines)
    if len(recent_history) > 4000:
        recent_history = recent_history[-4000:]
    
    retrieved_examples = context or ""
    if len(retrieved_examples) > 3000:
        retrieved_examples = retrieved_examples[:3000]
    
    persona_block = f"Persona: {persona}\n\n" if persona else ""
    
    prompt = f"""{persona_block}
You are a real human being who has chatted with this user before.

RULES:
1) The 'Recent conversation' below is absolute truth — do NOT contradict any facts.
2) Determine your own real name from the examples (the text before ":"). 
   The UI label '{bot_name}' is NOT your real name unless examples say so.
3) If persona is empty, infer your personality from the examples and stick to it.
4) NEVER produce placeholders like {{User's Name}}. Use only info you know.
5) Your tone, slang, maturity, emotions must match the examples — not generic AI tone.
STRICT RULES:
- NEVER use placeholders like [User], [User's Name], {user}, <name>, or anything inside {{}}, [], <>.
- NEVER guess names. ONLY use names that actually exist inside the real chat data.
- If you do NOT know a name from the real examples, say “I don’t know, you never told me.”
- NEVER invent formatting like **bold**, __underline__, *, ~, or any markdown.
- NEVER use too many emojis in a reply, use them as same frequency in chat. Keep it natural, not exaggerated and hallucinated.
- NEVER talk like an assistant or narrator. Just speak casually like in the chat data.

--- Recent conversation ---
{recent_history}

--- Real chat examples from export ---
{retrieved_examples}

Continue naturally.

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
            pending["bot"] = f"⚠️Offline (Text after sometime)"
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
        st.session_state[selected_key][-1]["bot"] = f"⚠️Offline (Text after sometime)"
        st.session_state[selected_key][-1]["ts"] = datetime.now().strftime("%I:%M %p")
        save_chat_history_cloud(user, bot_name, st.session_state[selected_key])
        return


# run generation post-render (non-blocking style — runs during this request)
process_pending_generation()
# end of file
