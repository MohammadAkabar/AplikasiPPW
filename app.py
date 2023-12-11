# main.py
import streamlit as st
import time
from generate_label import get_label


def main():

    st.set_page_config(
        page_title="Aplikasi Kategori BeritaSatu", page_icon="📺")

    col1, col2 = st.columns(2)

    with col1:

        st.image("assets/gambar.jpg", use_column_width=True)

    with col2:
        st.subheader("News Classification: Aplikasi Kategori untuk Berita")
        st.caption("Nama : Aderisa Dyta Okvianti")
        st.caption("NIM : 200411100013")
        
    news_text = st.text_area(
        "Inputan Isi Berita", key="input_text", height=250)

    if st.button("Cari Kategori"):
        if news_text:
            text = get_label(news_text)
            with st.expander('Tampilkan Hasil'):
                st.write('Berita yang anda masukkan termasuk dalam kategori: ')
                if text == "pemilu":
                    st.info(text, icon="🧑‍🏫")
                    url = "https://www.google.com/search?q=berita+pemilu+hari+ini"
                    st.write(
                        'Baca juga berita terbaru terkait pemilu 🔎 [Berita pemilu hari ini](%s)'  %url)
                elif text == "sport":
                    st.info(text, icon="🚣")
                    url = "https://www.google.com/search?q=berita+sport+hari+ini"
                    st.write(
                        'Baca juga berita terbaru terkait sport 🔎 [Berita sport hari ini](%s)'  %url)
                elif text == "lifestyle":
                    st.info(text, icon="💸")
                    url = "https://www.google.com/search?q=berita+lifestyle+hari+ini"
                    st.write(
                        'Baca juga berita terbaru terkait lifestyle 🔎 [Berita lifestyle hari ini](%s)'  %url)
        else:
            time.sleep(.5)
            st.toast('Masukkan teks terlebih dahulu', icon='🤧')


if __name__ == "__main__":
    main()
