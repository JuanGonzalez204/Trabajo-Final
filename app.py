import streamlit as st
import joblib

# Cargar el modelo y el vectorizador desde los archivos joblib
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Configurar la apariencia de la aplicación
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬")

# Estilo personalizado para el título
st.title("Sentiment Analysis of Movie Reviews")

# Sección de introducción con imagen
st.image("movie_reviews.jpg", caption="Analyzing movie reviews", use_column_width=True)

# Sección de entrada de la reseña
review = st.text_area("Enter your movie review:")

# Botón de predicción estilizado
if st.button("Predict", key="predict_button"):
    # Predicción y resultado estilizado
    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)[0]
    st.write("DEBUG: Prediction is", prediction)  # Mensaje de depuración para verificar la predicción
    st.subheader("Prediction Result:")
    if prediction == "positive":
        st.write("👍 Positive review")
    elif prediction == "negative":
        st.write("👎 Negative review")



