import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")  # Ganti dengan nama file datasetmu
    return df

# Fungsi untuk menampilkan data
def show_dataset(df):
    st.subheader("📌 Dataset")
    st.write(df.head())  # Menampilkan 5 data pertama

# Fungsi untuk menampilkan statistika deskriptif
def show_statistics(df):
    st.subheader("📊 Statistika Deskriptif")
    st.write(df.describe())  # Statistik ringkasan

# Fungsi untuk visualisasi data
def show_visualization(df):
    st.subheader("📈 Visualisasi Data")
    
    # Contoh visualisasi distribusi satu variabel
    selected_column = st.selectbox("Pilih Kolom untuk Histogram", df.columns)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[selected_column].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

# Fungsi utama untuk menjalankan Streamlit App
def main():
    st.title("📊 Dashboard Analisis Data")
    st.sidebar.header("Navigasi")
    menu = ["Dataset", "Statistika Deskriptif", "Visualisasi Data"]
    choice = st.sidebar.radio("Pilih Menu", menu)

    df = load_data()  # Load dataset

    if choice == "Dataset":
        show_dataset(df)
    elif choice == "Statistika Deskriptif":
        show_statistics(df)
    elif choice == "Visualisasi Data":
        show_visualization(df)

# Menjalankan aplikasi
if __name__ == "__main__":
    main()
