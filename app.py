import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Web app
st.title("🧠 Emotion Detection from Text")
st.subheader("Enter a sentence and let the ML model guess your emotion!")

user_input = st.text_area("Type your sentence here:")

if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)[0]
        st.success(f"Predicted Emotion: **{prediction.upper()}**")
