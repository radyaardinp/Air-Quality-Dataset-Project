import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
sns.set(style='dark')

# Judul Aplikasi
st.title("Analisis Kualitas Udara di Beijing")

# Upload Dataset
st.sidebar.header("Upload File CSV")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Menampilkan Data
    st.subheader("📊 Data Awal")
    st.write(df.head())

    # Statistik Deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.write(df.describe())

    # Visualisasi Distribusi PM2.5
    st.subheader("📌 Distribusi PM2.5")
    plt.figure(figsize=(8,5))
    sns.histplot(df["PM2.5"].dropna(), bins=30, kde=True)
    plt.xlabel("PM2.5")
    plt.ylabel("Frekuensi")
    plt.title("Distribusi PM2.5")
    st.pyplot(plt)

    # Visualisasi Tren PM2.5
    st.subheader("📌 Tren PM2.5 dari Waktu ke Waktu")
    if all(col in df.columns for col in ["year", "month", "day", "hour"]):
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        plt.figure(figsize=(10, 5))
        sns.lineplot(x="datetime", y="PM2.5", data=df)
        plt.xticks(rotation=45)
        plt.xlabel("Waktu")
        plt.ylabel("PM2.5")
        plt.title("Tren PM2.5")
        st.pyplot(plt)
    else:
        st.warning("Kolom waktu tidak tersedia dalam dataset!")

else:
    st.info("Silakan upload file CSV untuk mulai analisis.")

