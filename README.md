# ChatDouble 🧠💬

ChatDouble is a private AI chatbot platform that lets you create custom chatbots trained on real conversations. Upload your chat history (like WhatsApp or Instagram), and ChatDouble brings your friends back to life — in chat form.

---

## 🚀 Features

- 🔐 Custom login system
- 🤖 Upload real chats to create bots
- 🔄 Multiple bots per user
- 🧠 FAISS-powered memory search
- 💬 Streamlit chat UI with Ollama + Mistral
- 🗂️ Manage bots: rename, delete, clear history
- 📁 Lightweight local storage with `users.json`

---

## 🧠 Tech Stack

- [Streamlit](https://streamlit.io) – UI & deployment
- [FAISS](https://github.com/facebookresearch/faiss) – vector search for chat memory
- [Sentence Transformers](https://www.sbert.net/) – text embeddings
- [Mistral via Ollama](https://ollama.com/) – local LLM
- Python 3.10+

---

## 📦 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/ChatDouble.git
   cd ChatDouble

2. **Install dependencies**

    ```bash
    Copy code
    pip install -r requirements.txt

3. **Run Ollama (if not running)**

    ```bash
    ollama run mistral


4. **Run the appRun the app**

    ```bash
    streamlit run app.py


📁 **Folder Structure**
```bash
ChatDouble/
├── app.py
├── users.json
├── bots/
│   └── chat_<name>.txt
├── chats/
│   └── <username>/
├── requirements.txt
└── .streamlit/
    └── config.toml
