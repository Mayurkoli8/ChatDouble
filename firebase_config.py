import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

# Load Firebase credentials from Streamlit secrets
firebase_secrets = dict(st.secrets["firebase_service_account"])

cred = credentials.Certificate(firebase_secrets)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
