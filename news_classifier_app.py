
import streamlit as st
import joblib
import pandas as pd

# Load trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# App Title
st.set_page_config(page_title="News Classifier", page_icon="📰")
st.title("📰 Fake vs Real News Classifier")
st.markdown("Paste a news article or headline to check its authenticity.")

# Text Input
user_input = st.text_area("Enter news text here:")

# Predict button
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        # Vectorize input
        input_vector = vectorizer.transform([user_input])
        # Predict
        prediction = model.predict(input_vector)[0]
        # Display result
        st.success(f"This news is classified as: **{prediction}**")
