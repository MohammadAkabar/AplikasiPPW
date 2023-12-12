import nltk
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import logging


nltk.download("punkt")
nltk.download("stopwords")

logging.basicConfig(level=logging.DEBUG)


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

    # tfidf_vectorizer = TfidfVectorizer()
    # data = tfidf_vectorizer.fit_transform([stemm])

    vectorizer = joblib.load('model/tfidf_vectorizer')
    model = joblib.load('model/nb_model')

    x_new = vectorizer.transform([stemm]).toarray()
    prediction = model.predict(x_new)
    result = prediction[0]

    logging.debug(f"Text after stemming: {stemm}")
    logging.debug(f"Vectorizer: {vectorizer}")
    logging.debug(f"Model: {model}")
    logging.debug(f"Transformed data shape: {x_new.shape}")
    logging.debug(f"Prediction array: {prediction}")

    return result
