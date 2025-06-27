# firebase_config.py

import os
import firebase_admin
from firebase_admin import credentials, firestore

# Use environment variable to get path of the service account key
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not cred_path:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")

cred = credentials.Certificate(cred_path)

# Initialize Firebase only once
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()
