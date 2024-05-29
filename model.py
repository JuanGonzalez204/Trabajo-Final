import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string
import joblib

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('movie_reviews')


reviews = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]
df = pd.DataFrame(reviews, columns=['review', 'sentiment'])
df['review'] = df['review'].apply(lambda x: ' '.join(x))

# Preprocesamiento
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

df['cleaned_review'] = df['review'].apply(preprocess_text)

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorización
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Definir el modelo
model = SVC()

# Definir el grid de parámetros
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

# Grid Search
grid = GridSearchCV(model, param_grid, refit=True, verbose=2)
grid.fit(X_train_vec, y_train)

# Predicción y evaluación
y_pred = grid.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Best Parameters: {grid.best_params_}')

# Validación cruzada
scores = cross_val_score(grid.best_estimator_, X_train_vec, y_train, cv=5)
print(f'Cross-Validation Accuracy: {scores.mean()}')

# Guardar el modelo entrenado en un archivo joblib
joblib.dump(grid.best_estimator_, 'sentiment_model.joblib')

# Guardar el vectorizador en un archivo joblib
joblib.dump(vectorizer, 'vectorizer.joblib')