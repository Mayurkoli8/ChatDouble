# **ChatDouble**

ChatDouble is a private AI chat platform that lets you create lifelike chatbots trained on real chat histories, like WhatsApp or Instagram exports.
Upload your chats, name your bot, and talk to an AI version of that person powered by Google Gemini.

---

## Working Prototype 
[ChatDouble](https://chatdouble.streamlit.app)


---

### Features

> Custom login system (Firebase Firestore)

> Create custom bots from your chat files (.txt)

> FAISS-based memory for realistic context recall

> Gemini-powered chat responses (no local LLMs required)

> Manage bots: rename, delete, clear chat history

> Offline storage for bots and chat history (/bots, /chats)

> Download chat history in .txt or .json

---

### Tech Stack
| Component     | Technology                                                           |
| ------------- | -------------------------------------------------------------------- |
| Frontend      | [Streamlit](https://streamlit.io)                                    |
| Vector Memory | [FAISS](https://github.com/facebookresearch/faiss)                   |
| Embeddings    | [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| LLM           | [Gemini](https://ai.google.dev) via `google-genai`                   |
| Auth & Data   | [Firebase Firestore](https://firebase.google.com/docs/firestore)     |
| Language      | Python 3.10+                                                         |


---

### Folder Structure
```
ChatDouble/
│
├── app.py                   # Main Streamlit app
├── firebase_config.py        # Firebase setup
├── firebase_db.py            # User & bot Firestore logic
│
├── bots/                     # Local bot chat files (.txt)
│   └── <user>_chat_<bot>.txt
│
├── chats/                    # Local chat history (per user)
│   └── <username>/<bot>.json
│
├── requirements.txt
└── .streamlit/
    └── secrets.toml          # API keys & Firebase config
```
---

### Setup Instructions
---
### Clone the Repository
```
git clone https://github.com/Mayurkoli8/ChatDouble.git
cd ChatDouble
```

---

### Install Dependencies
```
pip install -r requirements.txt
```

Make sure your `requirements.txt` includes:
```
streamlit
faiss-cpu
sentence-transformers
numpy
firebase-admin
bcrypt
google-genai
torch
```
---

### Firebase Setup

1. Create a Firebase project at Firebase Console
.

2. Enable Firestore Database (for user data).

3. Create a Service Account Key (⚙️ → Project Settings → Service Accounts → Generate New Key).

4. Save the JSON key, or better — paste its contents into Streamlit secrets.

---

### Configure Streamlit Secrets

Create a .streamlit/secrets.toml file:
```
GEMINI_API_KEY = "your_gemini_api_key_here"

[firebase_service_account]
type = "service_account"
project_id = "your_project_id"
private_key_id = "your_private_key_id"
private_key = "-----BEGIN PRIVATE KEY-----\nYOURKEY\n-----END PRIVATE KEY-----\n"
client_email = "firebase-adminsdk@your_project_id.iam.gserviceaccount.com"
client_id = "12345678901234567890"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk"
```

---

### Run the App
```
streamlit run app.py
```

Then open http://localhost:8501
in your browser.

---

### How It Works

1. User Registration / Login
    Credentials are stored securely (hashed) in Firestore.

2. Bot Creation
    Upload a .txt chat file.
    The file is saved locally in /bots/.
    A bot entry is added to your Firestore profile.

3. Memory Embedding
    FAISS + Sentence Transformers embed every chat line for semantic search.

4. Chatting
    When you send a message:
        FAISS retrieves the top 20 most relevant past messages.
        Gemini receives the context and generates a realistic reply.

5. Chat History
    Each user’s conversations are stored in `/chats/<username>/<bot>.json`

---

### Example File Format

Each line of your uploaded .txt file should look like this:
```
You: hey what’s up?
John: nothing much bro
You: let’s meet tomorrow
John: sure, same place?
```
Avoid timestamps or system messages for best results.

---

### Future Enhancements

> Optional Firebase Storage for cloud chat files

> Multi-device sync for chat history

> Fine-tuning mode for stronger personality capture

> Real-time streaming responses from Gemini

---

### License

MIT — Free to use and modify.

---

##  Connect With Me

I'm actively building AI, automation & networking tools.  
Reach out if you’d like to collaborate or contribute.

<div align="left">

<a href="https://github.com/mayurkoli8" target="_blank">
<img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />
</a>

<a href="https://www.linkedin.com/in/mayur-koli-484603215/" target="_blank">
<img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
</a>


<a href="https://instagram.com/mentesa.live" target="_blank">
<img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" />
</a>

<a href="mailto:kolimohit9595@gmail.com">
<img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" />
</a>

</div>

---

### Want to improve this project?
Open an issue or start a discussion — PRs welcome ⚡
