from firebase_config import db

def register_user(username, password):
    doc_ref = db.collection("users").document(username)
    if doc_ref.get().exists:
        return False
    doc_ref.set({"password": password, "bots": []})
    return True

def login_user(username, password):
    doc = db.collection("users").document(username).get()
    return doc.exists and doc.to_dict().get("password") == password

def get_user_bots(username):
    doc = db.collection("users").document(username).get()
    if doc.exists:
        return doc.to_dict().get("bots", [])
    return []

def add_bot(username, name, file_path):
    doc_ref = db.collection("users").document(username)
    doc = doc_ref.get()
    if not doc.exists:
        return
    user_data = doc.to_dict()
    bots = user_data.get("bots", [])
    bots.append({"name": name, "file": file_path})
    doc_ref.update({"bots": bots})

def delete_bot(username, name):
    doc_ref = db.collection("users").document(username)
    doc = doc_ref.get()
    if not doc.exists:
        return
    user_data = doc.to_dict()
    bots = [bot for bot in user_data.get("bots", []) if bot["name"].lower() != name.lower()]
    doc_ref.update({"bots": bots})

def update_bot(username, old_name, new_name, new_file_path):
    doc_ref = db.collection("users").document(username)
    doc = doc_ref.get()
    if not doc.exists:
        return
    user_data = doc.to_dict()
    bots = user_data.get("bots", [])
    for bot in bots:
        if bot["name"].lower() == old_name.lower():
            bot["name"] = new_name
            bot["file"] = new_file_path
            break
    doc_ref.update({"bots": bots})
