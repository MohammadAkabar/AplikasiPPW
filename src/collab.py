from sklearn.feature_extraction.text import TfidfVectorizer
new_data = str(input("masukkan data: "))
# tokenizing kalimat
new_data = sent_tokenize(new_data)
# Inisialisasi TfidfVectorizer
inp_tfidf_vectorizer = TfidfVectorizer()
# Melakukan transformasi TF-IDF pada kolom 'final_abstrak'
inp_tfidf_matrix = inp_tfidf_vectorizer.fit_transform(new_data)
inp_cos_sim_result = []  # untuk menyimpan hasil cosine sim akhir
# graf_result = [] #untuk menyimpan hasil graf akhir
treshold = 0.04  # inisialisasi treshold

# cos_sim_now = []
inp_graf = nx.DiGraph()
inp_cos_sim = cosine_similarity(inp_tfidf_matrix)  # menjadikan tfidf ke cosine
# len(inp_cos_sim)
# inisialisasi indeks awal perulangan dari setiap hasil cosine
for i_hasil in range(len(inp_cos_sim)):

  # inisialisasi indeks kedua perulangan dari setiap hasil cosine
  for j_hasil in range(i_hasil+1, len(inp_cos_sim)):
    # if inp_cos_sim[i_hasil][j_hasil] > treshold: #cek apakah cosim dari kalimat 1 dan 2 lebih dari treshold?
      # print(f'Similairty kalimat ke - {i_hasil} : {j_hasil} = {inp_cos_sim[i_hasil][j_hasil]}')

      # menyimpan nilai indeks awal, indeks awal+1, hasil cosim
      inp_cos_sim_result.append(
          [i_hasil, j_hasil, inp_cos_sim[i_hasil][j_hasil]])
      inp_graf.add_edge(i_hasil, j_hasil,weight=inp_cos_sim[i_hasil][j_hasil]) #menyimpan nilai indeks awal, indeks awal+1, bobot=hasil cosim
# ========== inp_summary =========
inp_summary = []  # membuat array kosong untuk hasil inp_summary
# for i in range(len(graf_result)): #perulangan setiap graf result

# menjadikan closeness centrality pada setiap indeks graf result
inp_cc = nx.closeness_centrality(inp_graf)
# mengurutkan hasil closness centrality dari yang value terbesar
inp_cc = dict(sorted(inp_cc.items(), key=lambda item: item[1], reverse=True))

inp_list = list(inp_cc.keys())[:3]  # mengambil indeks 3 kalimat teratas

for key, value in inp_cc.items():
  # print((data['tokenizing'][i][key]))
  # menambahkan hasil inp_summary setiap kalimat
  inp_summary.append(new_data[key])
# inp_summary.append(current_summary) #menambahkan hasil inp_summary setiap dokumen
# ========== inp_summary =========
inp_summary = []  # membuat array kosong untuk hasil inp_summary
# for i in range(len(graf_result)): #perulangan setiap graf result

# menjadikan closeness centrality pada setiap indeks graf result
inp_cc = nx.closeness_centrality(inp_graf)
# mengurutkan hasil closness centrality dari yang value terbesar
inp_cc = dict(sorted(inp_cc.items(), key=lambda item: item[1], reverse=True))

inp_list = list(inp_cc.keys())[:3]  # mengambil indeks 3 kalimat teratas

for key, value in inp_cc.items():
  # print((data['tokenizing'][i][key]))
  # menambahkan hasil inp_summary setiap kalimat
  inp_summary.append(new_data[key])
# inp_summary.append(current_summary) #menambahkan hasil inp_summary setiap dokumen
# Membuat inp_summ_hasil menjadi list jika belum
if not isinstance(inp_summ_hasil, list):
    inp_summ_hasil = [inp_summ_hasil]

# Melakukan transformasi TF-IDF pada inp_summ_hasil
summ_inp_tfidf_matrix = summ_tfidf_vectorizer.transform(inp_summ_hasil)

inp_predict = rf.predict(summ_inp_tfidf_matrix.toarray())
inp_predict
