import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
nltk.download('stopwords')

import string
# Inicializar lematizador y lista de stop words
stop_words = set(stopwords.words('english'))
# Definir una función para preprocesar el texto
def preprocess_text(text):
  # Convertir el texto a minúsculas
  text = text.lower()
  # Eliminar signos de puntuación
  text = ''.join([char for char in text if char not in string.punctuation])
  # Eliminar stopwords
  stop_words = set(stopwords.words('english'))
  words = text.split()
  text = ' '.join([word for word in words if word not in stop_words])
  return text


def predict_proba(text):
  # Load the trained model and the vectorizer
  model = joblib.load('genre_classification_model.pkl')
  #vect_tfidf = joblib.load('tfidf_vectorizer.pkl')
  # Example input text
  documents = [text]
  
  dataTraining = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
  vect_tfidf = TfidfVectorizer(preprocessor=preprocess_text)
  X_train_tfidf = vect_tfidf.fit_transform(dataTraining['plot'])
  
  # Transform the documents using the loaded vectorizer
  X_test_tfidf = vect_tfidf.transform(documents)
  # Convert the sparse matrix to dense format if required by the model
  X_test_tfidf_dense = X_test_tfidf.toarray()
  # Make predictions
  predictions = model.predict(X_test_tfidf_dense)

  y_pred = pd.DataFrame(predictions)

  cols = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
        'Family','Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 
        'News', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
  
  y_pred.columns = cols

  row_dict = y_pred.iloc[0].to_dict()

  return row_dict