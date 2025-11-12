import bcrypt
from firebase_config import db

USERS_COLLECTION = "users"

def register_user(username, password):
    doc_ref = db.collection(USERS_COLLECTION).document(username)
    if doc_ref.get().exists:
        return False
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8", "ignore")
    doc_ref.set({"password": hashed})
    return True


def login_user(username, password):
    doc = db.collection(USERS_COLLECTION).document(username).get()
    if not doc.exists:
        return False
    stored = doc.to_dict().get("password")
    if not stored:
        return False
    return bcrypt.checkpw(password.encode(), stored.encode())


def add_bot(username, name, file_text):
    """
    Store bot text directly in Firestore.
    """
    bots_ref = db.collection(USERS_COLLECTION).document(username).collection("bots")
    bots_ref.document(name.lower()).set({
        "name": name,
        "file_text": file_text
    })


def get_user_bots(username):
    bots_ref = db.collection(USERS_COLLECTION).document(username).collection("bots").stream()
    return [{"name": doc.to_dict()["name"], "file": doc.id} for doc in bots_ref]


def get_bot_file(username, bot_name):
    doc_ref = db.collection(USERS_COLLECTION).document(username).collection("bots").document(bot_name.lower()).get()
    if doc_ref.exists:
        return doc_ref.to_dict().get("file_text", "")
    return ""


def delete_bot(username, name):
    db.collection(USERS_COLLECTION).document(username).collection("bots").document(name.lower()).delete()


def update_bot(username, old_name, new_name, new_file_text=None):
    old_ref = db.collection(USERS_COLLECTION).document(username).collection("bots").document(old_name.lower())
    old_doc = old_ref.get()
    if not old_doc.exists:
        return
    data = old_doc.to_dict()
    data["name"] = new_name
    if new_file_text:
        data["file_text"] = new_file_text
    # create new doc and delete old
    new_ref = db.collection(USERS_COLLECTION).document(username).collection("bots").document(new_name.lower())
    new_ref.set(data)
    old_ref.delete()


# Chat history stored in Firestore
def save_chat_history_cloud(user, bot, history):
    db.collection(USERS_COLLECTION).document(user).collection("chats").document(bot.lower()).set({
        "history": history
    })


def load_chat_history_cloud(user, bot):
    doc = db.collection(USERS_COLLECTION).document(user).collection("chats").document(bot.lower()).get()
    if doc.exists:
        return doc.to_dict().get("history", [])
    return []
