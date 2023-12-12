from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
# Menambahkan impor untuk sent_tokenize
from nltk.tokenize import sent_tokenize


def get_label(input_data, inp_tfidf_vectorizer, inp_summ_hasil, summ_tfidf_vectorizer, rf):
    # Tokenisasi kalimat
    new_data = sent_tokenize(input_data)

    # Inisialisasi TfidfVectorizer
    inp_tfidf_matrix = inp_tfidf_vectorizer.transform(new_data)

    # ========== cosine ==========
    inp_cos_sim_result = []  # untuk menyimpan hasil cosine sim akhir

    inp_graf = nx.DiGraph()
    inp_cos_sim = cosine_similarity(inp_tfidf_matrix)

    # inisialisasi indeks awal perulangan dari setiap hasil cosine
    for i_hasil in range(len(inp_cos_sim)):
        # inisialisasi indeks kedua perulangan dari setiap hasil cosine
        for j_hasil in range(i_hasil + 1, len(inp_cos_sim)):
            # if inp_cos_sim[i_hasil][j_hasil] > treshold:
            # menyimpan nilai indeks awal, indeks awal+1, hasil cosim
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

    if not isinstance(inp_summ_hasil, list):
        inp_summ_hasil = [inp_summ_hasil]

    summ_inp_tfidf_matrix = summ_tfidf_vectorizer.transform(inp_summ_hasil)

    return rf.predict(summ_inp_tfidf_matrix.toarray())
