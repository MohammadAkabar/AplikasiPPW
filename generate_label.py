import nltk
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import pandas as pd


nltk.download("punkt")
nltk.download("stopwords")


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

    vectorizer = joblib.load('model/tfidfvectorizer')
    model = joblib.load('model/random_forest_model')

    # x_new = vectorizer.transform([stemm]).toarray()
    # prediction = model.predict(x_new)
    # result = prediction[0]

    # Preprocess and transform input text
    x_new = vectorizer.transform([stemm]).toarray()

    # Ensure feature names are present
    feature_names = vectorizer.get_feature_names_out()
    x_new_df = pd.DataFrame(x_new, columns=feature_names)

    # Make prediction
    prediction = model.predict(x_new_df)
    result = prediction[0]

    return result
