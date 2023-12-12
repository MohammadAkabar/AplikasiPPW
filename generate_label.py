import nltk
import re
from nltk.corpus import stopwords
import joblib
import logging

nltk.download("punkt")
nltk.download("stopwords")

logging.basicConfig(level=logging.DEBUG)


def preprocess_text(news_text):
    # Konversi ke huruf kecil dan menghilangkan tanda baca serta angka
    lower = news_text.lower()
    puncuation = [re.sub(r'[.,()&=%:-\d]', '', token)
                  for token in lower.split()]

    # Menghilangkan kata-kata stop
    stop_words = set(stopwords.words("indonesian"))
    stopword = [token for token in puncuation if token.lower()
                not in stop_words]
    stopword = " ".join(stopword)

    return stopword


def get_label(news_text):

    # Pra-pemrosesan teks
    preprocessed_text = preprocess_text(news_text)

    # Melakukan transformasi TF-IDF pada teks yang sudah diproses
    vectorizer = joblib.load('model/tfidfvectorizer')
    x_new = vectorizer.transform([preprocessed_text]).toarray()

    # Memuat model RandomForest yang sudah ada
    model = joblib.load('model/random_forest_model')

    # Melakukan prediksi
    prediction = model.predict(x_new)
    result = prediction[0]

    logging.debug(f"Text after preprocessing: {preprocessed_text}")
    logging.debug(f"Transformed data shape: {x_new.shape}")
    logging.debug(f"Prediction array: {prediction}")

    return result
