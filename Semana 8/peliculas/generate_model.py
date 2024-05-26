import warnings
warnings.filterwarnings('ignore')
# Importación librerías
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
import re
from nltk.corpus import stopwords
# Descargar stopwords de NLTK si no están disponibles
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
# Carga de datos de archivo .csv
dataTraining = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
dataTesting = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)
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

vect_tfidf = TfidfVectorizer(preprocessor=preprocess_text)
X_train_tfidf = vect_tfidf.fit_transform(dataTraining['plot'])
X_test_tfidf = vect_tfidf.transform(dataTesting['plot'])
print(X_train_tfidf.shape)
# Definición de variable de interés (y)
dataTraining['genres'] = dataTraining['genres'].map(lambda x: eval(x))
le = MultiLabelBinarizer()
y_genres = le.fit_transform(dataTraining['genres'])
# Separación de variables predictoras (X) y variable de interés (y) en set de entrenamiento y test usandola función train_test_split
X_train, X_test, y_train_genres, y_test_genres = train_test_split(X_train_tfidf, y_genres, test_size=0.33,
                                                                  random_state=42)
X_train = X_train.toarray()
X_test = X_test.toarray()
# Definición de dimensiones de salida y entrada
output_var = y_test_genres.shape[1]
print(output_var, ' output variables')
dims = X_train.shape[1]
print(dims, 'input variables')

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from livelossplot import PlotLossesKeras
# Definición de función que crea una red neuronal a partir de diferentes parámetros (nn_model_params)
# En esta función se consideran 7 parámetos a calibrar, sin embargo se pueden agregar o quitar tantos como lo consideren pertinente
def nn_model_params(optimizer ,
                    neurons,
                    batch_size,
                    epochs,
                    activation,
                    patience,
                    loss):
    K.clear_session()
    # Definición red neuronal con la función Sequential()
    model = Sequential()
    # Definición de las capas de la red con el número de neuronas y la función de activación definidos en la función nn_model_params
    model.add(Dense(neurons, input_shape=(dims,), activation=activation))
    model.add(Dropout(0.1))
    model.add(Dense(output_var, activation=activation))
    # Definición de función de perdida con parámetros definidos en la función nn_model_params
    model.compile(optimizer = optimizer, loss=loss)
    # Definición de la función EarlyStopping con parámetro definido en la función nn_model_params
    early_stopping = EarlyStopping(monitor="val_loss", patience = patience)
    # Entrenamiento de la red neuronal con parámetros definidos en la función nn_model_params
    model.fit(X_train, y_train_genres,
              validation_data = (X_test, y_test_genres),
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[early_stopping, PlotLossesKeras()],
              verbose=True
              )
    return model
model = nn_model_params(optimizer = 'adam',
                        neurons=32,
                        batch_size=16,
                        epochs=100,
                        activation='sigmoid',
                        patience=10,
                        loss='binary_crossentropy')

joblib.dump(model, 'genre_classification_model.pkl')
joblib.dump(vect_tfidf, 'tfidf_vectorizer.pkl')