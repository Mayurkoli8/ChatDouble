# firebase_config.py

import firebase_admin
from firebase_admin import credentials, firestore

# Path to your service account key file
cred = credentials.Certificate("chatdouble-a03c6-firebase-adminsdk-fbsvc-de00f2fc52.json")

# Initialize app if not already initialized
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()
