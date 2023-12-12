import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import altair as alt
#from streamlit_option_menu import option_menu
from joblib import load
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

showWarningOnDirectExecution = False

st.title('Portal Tugas UAS Berita PPW')

st.write("ADERISA DYTA OKVIANTI | 200411100013")
tab1, tab2, tab3, = st.tabs(["Data Mentah", "Summary","Modelling"])


# ====================== Crawling ====================
with tab1:
        st.caption('Dataset diambil dari crawling menggunakan library Beautifulsoap dari situs radarjatim.com, dengan 3 kategori yaitu: Pemilu, Lifestyle dan Sport')
        dataset = pd.read_csv("https://raw.githubusercontent.com/AderisaDyta/AplikasiPPW/main/Crawling_BeritaSatu.ipynb")
        st.dataframe(dataset)
        st.info(f"Banyak Dataset : {len(dataset)}")
        st.warning(f'Informasi Dataset')
        st.write(dataset.describe())



with tab2:
        dataset = pd.read_csv("https://raw.githubusercontent.com/AderisaDyta/AplikasiPPW/main/DataSummary")
        
        st.dataframe(dataset)
        st.info(f"Banyak Dataset : {len(dataset)}")
        st.warning(f'Informasi Dataset')
        st.write(dataset.describe())

with tab3:
        st.markdown("<h4 style='text-align: center;'>Hasil Modelling Menggunakan Random Forest</h4>", unsafe_allow_html=True)
        csv_path = 'https://raw.githubusercontent.com/AderisaDyta/AplikasiPPW/main/classification_results.csv'
        df = pd.read_csv(csv_path)
        df
        # Menampilkan gambar
        image_path = 'https://raw.githubusercontent.com/AderisaDyta/AplikasiPPW/main/Confusion-matrix%20RF.PNG'
        st.image(image_path, caption='Confusion Matrix - Random Forest', width=200)

        st.markdown("<h4 style='text-align: center;'>Hasil Modelling Menggunakan SVM</h4>", unsafe_allow_html=True)
        csv_path = 'https://raw.githubusercontent.com/AderisaDyta/AplikasiPPW/main/classification_results_svm.csv'
        df = pd.read_csv(csv_path)
        df
        # Menampilkan gambar
        image_path = 'https://raw.githubusercontent.com/AderisaDyta/AplikasiPPW/main/Confusion-matrix%20SVM.PNG'
        st.image(image_path, caption='Confusion Matrix - SVM', width=200)

        st.write(f"Dari hasil akurasi dari 2 metode diatas yaitu metode Random Forest dan SVM didapatkan akurasi yaitu 95% untuk Random Forest dan 90% untuk SVM, Jadi hasil akurasi yang paling baik ialah menggunakan metode Random Forest")

