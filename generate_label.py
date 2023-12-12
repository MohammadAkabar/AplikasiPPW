import streamlit as st
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import logging

nltk.download("punkt")
nltk.download("stopwords")

logging.basicConfig(level=logging.DEBUG)

# Load your TF-IDF vectorizer and random forest model
vectorizer = joblib.load('model/tfidfvectorizer')
model = joblib.load('model/random_forest_model')


def get_label(news_text):
    lower = news_text.lower()
    lower = news_text.split()

    puncuation = [re.sub(r'[.,()&=%:-]', '', token)
                  for token in lower]
    puncuation = [re.sub(r'\d+', '', token)
                  for token in lower]
    stop_words = set(stopwords.words("indonesian"))
    stopword = [
        puncuation for puncuation in puncuation if puncuation.lower() not in stop_words]

    stopword = " ".join(stopword)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemm = stemmer.stem(stopword)

    x_new = vectorizer.transform([stemm]).toarray()
    prediction = model.predict(x_new)
    result = prediction[0]

    logging.debug(f"Text after stemming: {stemm}")
    logging.debug(f"Transformed data shape: {x_new.shape}")
    logging.debug(f"Prediction array: {prediction}")

    return result

