import bcrypt
from firebase_config import db

# =========================================================
# 🔖 Firestore Collections
# =========================================================
USERS_COLLECTION = "users"

# =========================================================
# 👤 Authentication Functions
# =========================================================
def register_user(username: str, password: str) -> bool:
    """
    Register a new user with hashed password.
    Returns False if username already exists.
    """
    doc_ref = db.collection(USERS_COLLECTION).document(username)
    if doc_ref.get().exists:
        return False

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8", "ignore")
    doc_ref.set({"password": hashed})
    return True


def login_user(username: str, password: str) -> bool:
    if not username:
        return False
    doc = db.collection(USERS_COLLECTION).document(username).get()
    """
    Validate login credentials.
    Returns True if correct, False otherwise.
    """
    doc = db.collection(USERS_COLLECTION).document(username).get()
    if not doc.exists:
        return False

    stored = doc.to_dict().get("password")
    if not stored:
        return False

    return bcrypt.checkpw(password.encode(), stored.encode())


# =========================================================
# 🤖 Bot Management
# =========================================================
def add_bot(username: str, name: str, file_text: str, persona: str = None) -> None:
    """
    Store bot data inside Firestore:
      users/{username}/bots/{bot_name}
    Supports optional 'persona' (personality description).
    """
    bots_ref = db.collection(USERS_COLLECTION).document(username).collection("bots")
    bot_data = {
        "name": name,
        "file_text": file_text,
    }
    if persona:
        bot_data["persona"] = persona

    bots_ref.document(name.lower()).set(bot_data)


def get_user_bots(username: str):
    """
    Retrieve all bots for a given user.
    Returns a list of dicts [{name, file, persona?}, ...]
    """
    bots_ref = db.collection(USERS_COLLECTION).document(username).collection("bots").stream()
    bots = []
    for doc in bots_ref:
        data = doc.to_dict()
        bots.append({
            "name": data.get("name"),
            "file": doc.id,
            "persona": data.get("persona", "")
        })
    return bots


def get_bot_file(username: str, bot_name: str):
    """
    Get the bot's full text content and optional persona.
    Returns (file_text, persona)
    """
    doc_ref = db.collection(USERS_COLLECTION).document(username).collection("bots").document(bot_name.lower()).get()
    if doc_ref.exists:
        data = doc_ref.to_dict()
        return data.get("file_text", ""), data.get("persona", "")
    return "", ""


def update_bot(username: str, old_name: str, new_name: str, new_file_text: str = None):
    """
    Rename a bot or update its file text.
    Creates a new document and deletes the old one.
    """
    user_ref = db.collection(USERS_COLLECTION).document(username)
    old_ref = user_ref.collection("bots").document(old_name.lower())
    old_doc = old_ref.get()

    if not old_doc.exists:
        return

    data = old_doc.to_dict()
    data["name"] = new_name
    if new_file_text:
        data["file_text"] = new_file_text

    # Create new doc, then delete old
    new_ref = user_ref.collection("bots").document(new_name.lower())
    new_ref.set(data)
    old_ref.delete()


def delete_bot(username: str, bot_name: str):
    """
    Delete a bot and its data from Firestore.
    """
    db.collection(USERS_COLLECTION).document(username).collection("bots").document(bot_name.lower()).delete()


def update_bot_persona(username: str, bot_name: str, persona_text: str):
    """
    Update only the persona field for a bot.
    """
    doc_ref = db.collection(USERS_COLLECTION).document(username).collection("bots").document(bot_name.lower())
    if doc_ref.get().exists:
        doc_ref.update({"persona": persona_text})


# =========================================================
# 💬 Chat History (Cloud Stored)
# =========================================================
def save_chat_history_cloud(user: str, bot: str, history: list) -> None:
    """
    Save chat history to Firestore under:
      users/{user}/chats/{bot}
    """
    db.collection(USERS_COLLECTION).document(user).collection("chats").document(bot.lower()).set({
        "history": history
    })


def load_chat_history_cloud(user: str, bot: str) -> list:
    """
    Load chat history from Firestore.
    Returns an empty list if no history found.
    """
    doc = db.collection(USERS_COLLECTION).document(user).collection("chats").document(bot.lower()).get()
    if doc.exists:
        return doc.to_dict().get("history", [])
    return []
