import joblib
import streamlit as st
from preprocess import preprocess_text

# ===============================
# Load model & vectorizer (CACHED)
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("email_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# Detection function
# ===============================
def detect_email(email_text):
    email_clean = preprocess_text(email_text)
    email_vec = vectorizer.transform([email_clean])
    prediction = model.predict(email_vec)
    return prediction[0]

# ===============================
# Streamlit UI
# ===============================
st.title("📧 Email Spam / Phishing Detection")

email = st.text_area("Enter email text")

if st.button("Detect"):
    if email.strip() == "":
        st.warning("Please enter an email message")
    else:
        result = detect_email(email)
        st.success(f"Prediction: {result}")
