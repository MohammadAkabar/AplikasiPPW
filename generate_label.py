import nltk
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging

nltk.download("punkt")
nltk.download("stopwords")

logging.basicConfig(level=logging.DEBUG)


def preprocess_text(news_text):
    # Konversi ke huruf kecil
    lower = news_text.lower()

    # Menghilangkan tanda baca
    punctuation = [re.sub(r'[.,()&=%:-]', '', token)
                   for token in lower.split()]

    # Menghilangkan angka
    punctuation = [re.sub(r'\d+', '', token) for token in punctuation]

    # Menghilangkan kata-kata stop
    stop_words = set(stopwords.words("indonesian"))
    stopword = [token for token in punctuation if token.lower()
                not in stop_words]
    stopword = " ".join(stopword)

    return stopword


def calculate_cosine_similarity(input_text, processed_texts):
    vectorizer = joblib.load('model/tfidfvectorizer')
    x_input = vectorizer.transform([input_text]).toarray()
    x_processed = vectorizer.transform(processed_texts).toarray()
    similarity_scores = cosine_similarity(x_input, x_processed)
    return similarity_scores


def get_label(news_text, processed_texts):
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

    # Menghitung cosine similarity
    similarity_scores = calculate_cosine_similarity(
        preprocessed_text, processed_texts)

    logging.debug(f"Text after preprocessing: {preprocessed_text}")
    logging.debug(f"Transformed data shape: {x_new.shape}")
    logging.debug(f"Prediction array: {prediction}")
    logging.debug(f"Cosine similarity scores: {similarity_scores}")

    return result
