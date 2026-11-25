
import streamlit as st
import joblib
import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load saved model and vectorizer
model_data = joblib.load("sms_spam_model.pkl")

model = model_data["model"]
tfidf = model_data["vectorizer"]

# Text preprocessing
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in ENGLISH_STOP_WORDS
    ]

    return ' '.join(words)


# Web application title
st.title("SMS Spam Detection System")

st.write(
    "Enter an SMS message below to check whether it is Spam or Ham."
)

# User input
message = st.text_area("Enter your SMS message:")

# Prediction button
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        cleaned_message = clean_text(message)

        message_tfidf = tfidf.transform([cleaned_message])

        prediction = model.predict(message_tfidf)[0]

        if prediction == 1:
            st.error("This message is SPAM!")
        else:
            st.success("This message is HAM (Not Spam).")
