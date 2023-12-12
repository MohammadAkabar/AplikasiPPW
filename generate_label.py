from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize

# Inisialisasi model RandomForest (contoh: di sini diasumsikan model_rf sudah diinisialisasi atau dimuat sebelumnya)
model_rf = RandomForestClassifier()


def get_label(news_text):
    # Tokenisasi kalimat

    nltk.download('punkt')

    new_data = sent_tokenize(news_text)

    # Inisialisasi TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    inp_tfidf_matrix = tfidf_vectorizer.fit_transform(new_data)

    # ========== cosine ==========
    inp_cos_sim_result = []  # untuk menyimpan hasil cosine sim akhir

    inp_graf = nx.DiGraph()
    inp_cos_sim = cosine_similarity(inp_tfidf_matrix)

    # inisialisasi indeks awal perulangan dari setiap hasil cosine
    for i_hasil in range(len(inp_cos_sim)):
        # inisialisasi indeks kedua perulangan dari setiap hasil cosine
        for j_hasil in range(i_hasil + 1, len(inp_cos_sim)):
            inp_cos_sim_result.append(
                [i_hasil, j_hasil, inp_cos_sim[i_hasil][j_hasil]])
            inp_graf.add_edge(i_hasil, j_hasil,
                              weight=inp_cos_sim[i_hasil][j_hasil])

    inp_summary = []  # membuat array kosong untuk hasil inp_summary

    inp_cc = nx.closeness_centrality(inp_graf)
    inp_cc = dict(sorted(inp_cc.items(),
                         key=lambda item: item[1], reverse=True))

    for key, value in inp_cc.items():
        inp_summary.append(new_data[key])

    # Menggunakan model_rf yang telah diinisialisasi sebelumnya
    summ_inp_tfidf_matrix = tfidf_vectorizer.transform(new_data)

    # Melakukan prediksi menggunakan model_rf
    result = model_rf.predict(summ_inp_tfidf_matrix.toarray())

    # Mengembalikan hasil prediksi
    return result

# Contoh penggunaan:
# news_text = "Contoh teks berita yang ingin dikategorikan."
# result = get_label(news_text)
# print(result)
