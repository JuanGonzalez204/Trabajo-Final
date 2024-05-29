import streamlit as st
import joblib

# Cargar el modelo y el vectorizador desde los archivos joblib
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

st.title("Sentiment Analysis of Movie Reviews")

review = st.text_area("Enter your movie review:")

if st.button("Predict"):
    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)[0]
    st.write(f'The review is: {prediction}')
