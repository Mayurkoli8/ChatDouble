import streamlit as st
from firebase_admin import credentials, firestore
import firebase_admin

# Load from Streamlit secrets
firebase_config = dict(st.secrets["firebase_service_account"])

cred = credentials.Certificate(firebase_config)
 
if not cred:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. Set in Streamlit secret")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
