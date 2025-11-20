import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# -------------------------
# Config & Theme
# -------------------------
st.set_page_config(page_title="Air Quality Dashboard - Single Page", layout="wide")
sns.set_theme(style="whitegrid")

# Colors (dark gray + blue accents)
PRIMARY = "#1f3b64"     # dark blue
SECONDARY = "#4b4f54"   # dark gray
ACCENT = "#2b8fd6"      # bright blue

# -------------------------
# Notebook path (user-uploaded)
# -------------------------
NOTEBOOK_PATH = "/mnt/data/Air Quality Analysis in Beijing.ipynb"

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_data(path="main_data.csv"):
    df = pd.read_csv(path)
    # ensure datetime
    if {"year","month","day"}.issubset(df.columns):
        df["date"] = pd.to_datetime(df[["year","month","day"]])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        # try to combine if other names
        pass

    # hour as int if exists
    if "hour" in df.columns:
        df["hour"] = df["hour"].astype(int)

    # season
    def get_season(m):
        if m in [12,1,2]:
            return "Musim Dingin"
        if m in [3,4,5]:
            return "Musim Semi"
        if m in [6,7,8]:
            return "Musim Panas"
        return "Musim Gugur"

    if "month" in df.columns:
        df["season"] = df["month"].apply(get_season)
    else:
        df["season"] = "Unknown"

    # air quality category based on PM2.5 if exists
    if "PM2.5" in df.columns:
        def categorize_pm25(pm25):
            if pd.isna(pm25):
                return "Unknown"
            elif pm25 <= 15.5:
                return "Baik"
            elif pm25 <= 55.4:
                return "Sedang"
            elif pm25 <= 150.4:
                return "Tidak Sehat"
            elif pm25 <= 250.4:
                return "Sangat Tidak Sehat"
            else:
                return "Berbahaya"
        df["air_quality_category"] = df["PM2.5"].apply(categorize_pm25)
    else:
        df["air_quality_category"] = "Unknown"

    return df

@st.cache_data
def summarize_station_insights(df, station):
    # returns a dict of computed insights for a given station
    d = {}
    sdf = df[df["station"] == station] if station != "Semua Wilayah" else df.copy()
    if len(sdf) == 0:
        return None

    # basic stats
    if "PM2.5" in sdf.columns:
        d["avg_pm25"] = float(sdf["PM2.5"].mean(skipna=True))
        d["max_pm25"] = float(sdf["PM2.5"].max(skipna=True))
        d["worst_date"] = str(sdf.loc[sdf["PM2.5"].idxmax(), "date"].date()) if not sdf["PM2.5"].isna().all() else "N/A"
    else:
        d["avg_pm25"] = None

    # trend by year (slope of yearly mean)
    if "year" in sdf.columns and "PM2.5" in sdf.columns:
        yearly = sdf.groupby("year")["PM2.5"].mean().reset_index()
        if len(yearly) >= 2:
            coeffs = np.polyfit(yearly["year"], yearly["PM2.5"], 1)
            slope = coeffs[0]
            d["yearly_trend_slope"] = float(slope)
            d["yearly_trend_direction"] = "Meningkat" if slope > 0 else ("Menurun" if slope < 0 else "Stabil")
        else:
            d["yearly_trend_direction"] = "Tidak cukup data"

    # worst season
    if "season" in sdf.columns and "PM2.5" in sdf.columns:
        season_mean = sdf.groupby("season")["PM2.5"].mean().sort_values(ascending=False)
        d["worst_season"] = season_mean.index[0] if len(season_mean) > 0 else "N/A"

    # worst hour
    if "hour" in sdf.columns and "PM2.5" in sdf.columns:
        hourly = sdf.groupby("hour")["PM2.5"].mean().sort_values(ascending=False)
        d["worst_hour"] = int(hourly.index[0]) if len(hourly) > 0 else None

    # strongest correlation with PM2.5 among available weather cols
    possible = ["TEMP","PRES","DEWP","RAIN","WSPM","TEMP_C"]
    avail = [c for c in possible if c in sdf.columns]
    if "PM2.5" in sdf.columns and len(avail) > 0:
        corr = sdf[["PM2.5"] + avail].corr()["PM2.5"].drop("PM2.5").abs().sort_values(ascending=False)
        if len(corr) > 0:
            top = corr.index[0]
            d["strongest_corr"] = (top, float(corr.iloc[0]))

    return d

# -------------------------
# Load data
# -------------------------
try:
    df = load_data("main_data.csv")
except FileNotFoundError:
    st.error("File 'main_data.csv' tidak ditemukan di working directory. Pastikan file tersedia.")
    st.stop()

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.title("Filter")
min_date = df["date"].min()
max_date = df["date"].max()

date_range = st.sidebar.date_input("Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

stations = ["Semua Wilayah"] + sorted(df["station"].unique().tolist())
selected_station = st.sidebar.selectbox("Pilih Stasiun (Wilayah)", stations)

pollutants = [c for c in ["PM2.5","PM10","SO2","NO2","CO","O3"] if c in df.columns]
selected_pollutant = st.sidebar.selectbox("Pilih Polutan", pollutants) if len(pollutants) > 0 else None

# Apply filters
start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
df_filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
if selected_station != "Semua Wilayah":
    df_filtered = df_filtered[df_filtered["station"] == selected_station]

# -------------------------
# Page Layout
# -------------------------
st.title("Air Quality Dashboard — Beijing (Single Page)")
st.markdown("---")

# Top metrics + insights headline
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    st.metric("Total Data", f"{len(df_filtered):,}")
with col2:
    if selected_pollutant:
        st.metric(f"Rata-rata {selected_pollutant}", f"{df_filtered[selected_pollutant].mean():.2f} µg/m³")
    else:
        st.metric("Rata-rata PM2.5", "N/A")
with col3:
    if "TEMP" in df_filtered.columns:
        st.metric("Rata-rata Suhu", f"{df_filtered['TEMP'].mean():.1f} °C")
    else:
        st.metric("Rata-rata Suhu", "N/A")
with col4:
    st.metric("Kategori Dominan", df_filtered["air_quality_category"].mode().iloc[0] if len(df_filtered)>0 else "N/A")

st.markdown("---")

# 1) EDA insights per wilayah (station)
st.header("Insight EDA per Wilayah")
st.write("Pilih stasiun untuk melihat insight ringkas yang dihasilkan dari EDA (trend, musim terburuk, jam terburuk, korelasi kuat).")

station_for_insight = st.selectbox("Pilih Stasiun untuk Insight", stations, index=stations.index(selected_station))
ins = summarize_station_insights(df, station_for_insight)

if ins is None:
    st.warning("Tidak ada data untuk stasiun ini di rentang yang dipilih.")
else:
    colA, colB = st.columns([2,1])
    with colA:
        st.subheader(f"Ringkasan untuk: {station_for_insight}")
        st.markdown(f"- **Rata-rata PM2.5:** {ins.get('avg_pm25', 'N/A'):.2f} µg/m³" if ins.get('avg_pm25') is not None else "- **Rata-rata PM2.5:** N/A")
        st.markdown(f"- **PM2.5 Maksimum:** {ins.get('max_pm25', 'N/A'):.2f} µg/m³ pada {ins.get('worst_date', 'N/A')}")
        st.markdown(f"- **Arah tren tahunan:** {ins.get('yearly_trend_direction', 'N/A')}")
        st.markdown(f"- **Musim terburuk:** {ins.get('worst_season', 'N/A')}")
        st.markdown(f"- **Jam rata-rata dengan polusi tertinggi:** {ins.get('worst_hour', 'N/A')}")
        if ins.get('strongest_corr'):
            var, val = ins['strongest_corr']
            st.markdown(f"- **Korelasi terkuat dengan PM2.5:** {var} (|r| = {val:.2f})")
    with colB:
        st.subheader("Statistik Singkat")
        st.write(df[df['station']==station_for_insight].describe().T[['count','mean','std']].style.format('{:.2f}'))

st.markdown("---")

# 2) Insights for each EDA question (automatically generate from data)
st.header("Insight Otomatis untuk Pertanyaan EDA")

# Pertanyaan 1: Tren kualitas udara setiap tahun di setiap wilayah
st.subheader("1. Tren kualitas udara per tahun")
if 'year' in df_filtered.columns and selected_pollutant:
    trend_df = df_filtered.groupby('year')[selected_pollutant].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(trend_df['year'], trend_df[selected_pollutant], marker='o', linewidth=2)
    ax.set_title(f"Rata-rata {selected_pollutant} per Tahun")
    ax.set_ylabel(f"{selected_pollutant} (µg/m³)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # automatic verdict
    if len(trend_df)>=2:
        slope = np.polyfit(trend_df['year'], trend_df[selected_pollutant],1)[0]
        verdict = 'Meningkat' if slope>0 else ('Menurun' if slope<0 else 'Stabil')
        st.markdown(f"**Kesimpulan:** Tren {selected_pollutant} per tahun di rentang tanggal ini cenderung **{verdict}** (slope={slope:.3f}).")
else:
    st.write("Tidak cukup data untuk menampilkan tren tahunan atau polutan tidak tersedia.")

st.markdown("---")

# Pertanyaan 2: Korelasi antara PM2.5 dan faktor cuaca
st.subheader("2. Korelasi antara polutan dan faktor cuaca")
weather_cols = [c for c in ['TEMP','PRES','DEWP','RAIN'] if c in df_filtered.columns]
if selected_pollutant and len(weather_cols)>0:
    corr_df = df_filtered[[selected_pollutant]+weather_cols].corr()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(corr_df, annot=True, fmt='.2f', ax=ax, cmap='coolwarm', cbar_kws={'label':'Korelasi'})
    ax.set_title(f"Korelasi {selected_pollutant} dengan Faktor Cuaca")
    st.pyplot(fig)

    # top correlation
    corrs = corr_df[selected_pollutant].drop(selected_pollutant).abs().sort_values(ascending=False)
    if len(corrs)>0:
        top_var = corrs.index[0]
        st.markdown(f"**Kesimpulan:** Variabel cuaca yang paling berkaitan dengan {selected_pollutant} adalah **{top_var}** (|r| = {corrs.iloc[0]:.2f}).")
else:
    st.write("Tidak ada kolom cuaca atau polutan untuk menghitung korelasi.")

st.markdown("---")

# Pertanyaan 3: Perbedaan antar wilayah
st.subheader("3. Perbedaan rata-rata polutan antar wilayah")
if selected_pollutant:
    region_avg = df_filtered.groupby('station')[selected_pollutant].mean().sort_values(ascending=False).reset_index()
    st.dataframe(region_avg.style.format({selected_pollutant: '{:.2f}'}), use_container_width=True)
    fig, ax = plt.subplots(figsize=(10,3))
    sns.barplot(data=region_avg, x=selected_pollutant, y='station', palette='Blues_r', ax=ax)
    ax.set_xlabel(f"Rata-rata {selected_pollutant} (µg/m³)")
    ax.set_ylabel("Wilayah")
    st.pyplot(fig)
    st.markdown(f"**Kesimpulan:** Wilayah dengan nilai {selected_pollutant} tertinggi: **{region_avg.loc[0,'station']}** ({region_avg.loc[0,selected_pollutant]:.2f} µg/m³).")
else:
    st.write("Pilih polutan untuk analisis per wilayah.")

st.markdown("---")

# Pertanyaan 4: Jam dengan kualitas udara terburuk
st.subheader("4. Jam dengan kualitas udara paling buruk")
if 'hour' in df_filtered.columns and selected_pollutant:
    hourly = df_filtered.groupby('hour')[selected_pollutant].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10,3))
    sns.lineplot(data=hourly, x='hour', y=selected_pollutant, marker='o', ax=ax)
    ax.set_xlabel('Jam')
    ax.set_ylabel(f'{selected_pollutant} (µg/m³)')
    ax.set_title('Rata-rata per Jam (keseluruhan wilayah yang dipilih)')
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    worst_h = int(hourly.loc[hourly[selected_pollutant].idxmax(),'hour'])
    st.markdown(f"**Kesimpulan:** Pada rentang tanggal dan wilayah ini, jam dengan rata-rata {selected_pollutant} tertinggi adalah **{worst_h}:00**.")
else:
    st.write("Data jam tidak tersedia atau polutan belum dipilih.")

st.markdown("---")

# Footer: link to original notebook and quick download
st.markdown("### Referensi dan file sumber")
st.markdown(f"- Notebook analisis (asli): `{NOTEBOOK_PATH}`")
st.markdown("- Jika ingin men-download notebook, gunakan file manager pada environment Anda atau hubungkan workspace.")

st.caption("Made by Radya Ardi — Single Page Dashboard")

# -------------------------
# End
# -------------------------
