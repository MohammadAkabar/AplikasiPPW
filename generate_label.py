
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import numpy as np

st.title("Text Analysis with Streamlit")

# Load your TF-IDF vectorizer and random forest model
summ_tfidf_vectorizer = joblib.load('model/tfidfvectorizer')
rf = joblib.load('model/random_forest_model')


def generate_label(news_text):
    new_data = news_text
    # tokenizing kalimat
    new_data = sent_tokenize(new_data)
    # Inisialisasi TfidfVectorizer
    inp_tfidf_vectorizer = TfidfVectorizer()
    # Melakukan transformasi TF-IDF pada kolom 'final_abstrak'
    inp_tfidf_matrix = inp_tfidf_vectorizer.fit_transform(new_data)
    inp_cos_sim_result = []  # untuk menyimpan hasil cosine sim akhir
    
    inp_graf = nx.Graph()  # menggunakan Graph bukan DiGraph
    inp_cos_sim = cosine_similarity(inp_tfidf_matrix)

    # inisialisasi indeks awal perulangan dari setiap hasil cosine
    for i_hasil in range(len(inp_cos_sim)):
        # inisialisasi indeks kedua perulangan dari setiap hasil cosine
        for j_hasil in range(i_hasil + 1, len(inp_cos_sim)):
            # menyimpan nilai indeks awal, indeks awal+1, hasil cosim
            inp_cos_sim_result.append(
                [i_hasil, j_hasil, inp_cos_sim[i_hasil][j_hasil]])
            inp_graf.add_edge(i_hasil, j_hasil,
                              weight=inp_cos_sim[i_hasil][j_hasil])

    # Analisis graf, closeness centrality, dll.

    # Membuat inp_summ_hasil menjadi list jika belum
    inp_summ_hasil = ["dummy_summary"]  # Ganti dengan ringkasan hasil analisis

    # Melakukan transformasi TF-IDF pada inp_summ_hasil
    summ_inp_tfidf_matrix = summ_tfidf_vectorizer.transform(inp_summ_hasil)

    inp_predict = rf.predict(summ_inp_tfidf_matrix.toarray())
    return inp_predict[0]
