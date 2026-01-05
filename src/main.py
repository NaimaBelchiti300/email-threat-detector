# # Training data
# emails = [
#     ("Win money now", "spam"),
#     ("Click here to verify your account", "phishing"),
#     ("Your password has expired", "phishing"),
#     ("Meeting at 10am tomorrow", "normal"),
#     ("Project deadline is next week", "normal"),
#     ("Congratulations you won a prize", "spam")
# ]

# #  Separate text and labels
# texts = []
# labels = []

# for email, label in emails:
#     texts.append(email)
#     labels.append(label)

# #  Convert text to numbers
# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(texts)

# # Train model
# from sklearn.naive_bayes import MultinomialNB

# model = MultinomialNB()
# model.fit(X, labels)

# #  Detect function
# def detect_email(email_text):
#     email_vector = vectorizer.transform([email_text])
#     prediction = model.predict(email_vector)
#     return prediction[0]

# #  Test
# test_email = "win now "
# result = detect_email(test_email)

# print("Email:", test_email)
# print("Result:", result)
import joblib
from preprocess import preprocess_text

# Charger modèle et vectorizer
model = joblib.load("../email_model.pkl")
vectorizer = joblib.load("../vectorizer.pkl")

def detect_email(email_text):
    email_clean = preprocess_text(email_text)
    email_vec = vectorizer.transform([email_clean])
    prediction = model.predict(email_vec)
    return prediction[0]

# Mode terminal
def terminal_mode():
    print("=== Email Spam / Phishing Detection ===")
    email_text = input("Enter email text: ")
    result = detect_email(email_text)
    print("Prediction:", result)

# Mode Streamlit
def streamlit_mode():
    import streamlit as st
    st.title("Email Spam / Phishing Detection")
    email = st.text_area("Enter email")
    if st.button("Detect"):
        result = detect_email(email)
        st.write("Prediction:", result)

if __name__ == "__main__":
    mode = input("Choose mode (terminal/streamlit): ").strip().lower()
    if mode == "terminal":
        terminal_mode()
    elif mode == "streamlit":
        streamlit_mode()
    else:
        print("Invalid mode. Use 'terminal' or 'streamlit'.")
