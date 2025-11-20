import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from windrose import WindroseAxes
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi tema
st.set_page_config(
    page_title="Air Quality Dashboard",
    layout="wide"
)
sns.set(style='light')

# Menambahkan background
background_style = """
    <style>
        .stApp {
            background: linear-gradient(to bottom, #191d26, #000000);
            color: white;
        }
    </style>
"""
st.markdown(background_style, unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    file_path = "main_data.csv"
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["hour"] = df["hour"].astype(int)
    
    # Tambahkan kolom musim
    def get_season(month):
        if month in [12, 1, 2]:
            return "Musim Dingin"
        elif month in [3, 4, 5]:
            return "Musim Semi"
        elif month in [6, 7, 8]:
            return "Musim Panas"
        else:
            return "Musim Gugur"
    
    df["season"] = df["month"].apply(get_season)
    
    return df

df = load_data()

# Fungsi untuk mengkategorikan kualitas udara
def categorize_pm25(pm25):
    if pd.isna(pm25):
        return "Unknown"
    elif pm25 <= 15.5:
        return "Baik"
    elif 15.6 <= pm25 <= 55.4:
        return "Sedang"
    elif 55.5 <= pm25 <= 150.4:
        return "Tidak Sehat"
    elif 150.5 <= pm25 <= 250.4:
        return "Sangat Tidak Sehat"
    else:
        return "Berbahaya"

df["air_quality_category"] = df["PM2.5"].apply(categorize_pm25)

# Judul utama
st.markdown("""
    <style>
        .center-text {
            text-align: center;
        }
    </style>
    <h1 class="center-text" style="font-size: 50px;"> AIR QUALITY DASHBOARD </h1>
    <h3 class="center-text" style="font-size: 30px;">Beijing 2013-2017</h3>
""", unsafe_allow_html=True)
st.markdown("---")

# ========== FILTER SECTION ==========
st.sidebar.header("🎯 Filter Data")

# Filter Tanggal dengan try-exception
try:
    date_range = st.sidebar.date_input(
        "Pilih Rentang Tanggal",
        value=(df["date"].min(), df["date"].max()),
        min_value=df["date"].min(),
        max_value=df["date"].max()
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df["date"] >= pd.Timestamp(start_date)) & 
                        (df["date"] <= pd.Timestamp(end_date))]
    else:
        st.sidebar.warning("⚠️ Silakan pilih tanggal mulai dan tanggal akhir")
        df_filtered = df
except Exception as e:
    st.sidebar.error(f"Error pada filter tanggal: {e}")
    df_filtered = df

# Filter Stasiun/Wilayah
stations = ["Semua Wilayah"] + sorted(df_filtered["station"].unique().tolist())
selected_station = st.sidebar.selectbox("🏢 Pilih Stasiun", stations)

if selected_station != "Semua Wilayah":
    df_filtered = df_filtered[df_filtered["station"] == selected_station]

# Filter Jenis Polutan
pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
selected_pollutant = st.sidebar.selectbox("🌫️ Pilih Polutan untuk Analisis Detail", pollutants)

# Tampilkan info filter
st.sidebar.markdown("---")
st.sidebar.info(f"📌 Data terfilter: **{len(df_filtered):,}** dari **{len(df):,}** baris")

# ========== DASHBOARD CONTENT ==========
st.write("")

# Metrik Utama
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Total Data", f"{len(df_filtered):,}")
with col2:
    avg_pm25 = df_filtered["PM2.5"].mean()
    st.metric("🌫️ Rata-rata PM2.5", f"{avg_pm25:.2f} µg/m³")
with col3:
    avg_temp = df_filtered["TEMP"].mean()
    st.metric("🌡️ Rata-rata Suhu", f"{avg_temp:.1f}°C")
with col4:
    worst_quality = df_filtered["air_quality_category"].mode()[0] if len(df_filtered) > 0 else "N/A"
    st.metric("⚠️ Kategori Dominan", worst_quality)

st.markdown("---")

# **Kualitas Udara Berdasarkan PM2.5**
st.markdown("""
    <h3 class="center-text" style="font-size: 25px;">Kualitas Udara Berdasarkan PM2.5</h3>
""", unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns(2)

# Tabel Keterangan
with col1:
    st.markdown("""
    <h3 class="center-text" style="font-size: 20px;">Kategori Kualitas Udara</h3>
    """, unsafe_allow_html=True)
    st.markdown("""
    | PM2.5 (µg/m³) | Kategori |
    |--------------|----------|
    | 0 - 15.5    | Baik |
    | 15.6 - 55.4  | Sedang |
    | 55.5 - 150.4 | Tidak Sehat |
    | 150.5 - 250.4 | Sangat Tidak Sehat |
    | > 250.5      | Berbahaya |
    """)

# Distribusi Kualitas Udara
with col2:
    st.markdown("""
    <h3 class="center-text" style="font-size: 20px;">Distribusi Kualitas Udara per Wilayah</h3>
    """, unsafe_allow_html=True)
    region_quality_df = df_filtered.groupby(["station", "air_quality_category"]).size().reset_index(name="count")
    st.dataframe(region_quality_df, height=200)

st.markdown("---")

# Tren Polutan
st.subheader(f"📈 Tren Polusi {selected_pollutant} per Tahun dan Wilayah")
st.write("")

if selected_station == "Semua Wilayah":
    df_trend = df_filtered.groupby("year")[pollutants].mean().reset_index()
else:
    df_trend = df_filtered.groupby("year")[pollutants].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
for column in pollutants:
    ax.plot(df_trend["year"], df_trend[column], marker='o', label=column, linewidth=2)

ax.set_xlabel("Tahun", fontsize=12)
ax.set_ylabel("Konsentrasi Polutan (µg/m³)", fontsize=12)
ax.set_title(f"Tren Polutan Udara di {selected_station}", fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.markdown("---")

# Tren Polusi per Musim
st.subheader("🍃 Tren Polusi Berdasarkan Musim di Setiap Wilayah")
st.write("")

# Pilihan wilayah untuk analisis musim
season_region = st.selectbox("Pilih Wilayah untuk Analisis Musim", stations, key="season_region")

if season_region == "Semua Wilayah":
    df_season = df_filtered.groupby("season")[pollutants].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_season["season"]))
    width = 0.12
    
    for i, pollutant in enumerate(pollutants):
        ax.bar(x + i * width, df_season[pollutant], width, label=pollutant)
    
    ax.set_xlabel("Musim", fontsize=12)
    ax.set_ylabel("Konsentrasi Polutan (µg/m³)", fontsize=12)
    ax.set_title(f"Rata-rata Polutan per Musim - {season_region}", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(df_season["season"])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig)
else:
    df_season = df_filtered[df_filtered["station"] == season_region].groupby("season")[pollutants].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_season["season"]))
    width = 0.12
    
    for i, pollutant in enumerate(pollutants):
        ax.bar(x + i * width, df_season[pollutant], width, label=pollutant)
    
    ax.set_xlabel("Musim", fontsize=12)
    ax.set_ylabel("Konsentrasi Polutan (µg/m³)", fontsize=12)
    ax.set_title(f"Rata-rata Polutan per Musim - {season_region}", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(df_season["season"])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig)

st.markdown("---")

# Wind Rose Plot
st.subheader("🌬️ Analisis Arah Angin dan Polusi (Wind Rose)")
st.write("")

wind_region = st.selectbox("Pilih Wilayah untuk Wind Rose", 
                          [s for s in stations if s != "Semua Wilayah"], 
                          key="wind_region")

df_wind = df_filtered[df_filtered["station"] == wind_region].copy()

# Pastikan ada data wind direction dan wind speed
if "wd" in df_wind.columns and "WSPM" in df_wind.columns:
    df_wind_clean = df_wind.dropna(subset=["wd", "WSPM", "PM2.5"])
    
    if len(df_wind_clean) > 0:
        fig = plt.figure(figsize=(12, 5))
        
        # Wind Rose berdasarkan kecepatan angin
        ax1 = fig.add_subplot(121, projection='windrose')
        ax1.bar(df_wind_clean["wd"], df_wind_clean["WSPM"], normed=True, opening=0.8, edgecolor='white')
        ax1.set_title(f"Distribusi Arah Angin - {wind_region}", fontsize=12, fontweight='bold')
        ax1.set_legend(title="Kecepatan Angin (m/s)")
        
        # Wind Rose berdasarkan PM2.5
        ax2 = fig.add_subplot(122, projection='windrose')
        ax2.bar(df_wind_clean["wd"], df_wind_clean["WSPM"], normed=True, opening=0.8, 
                edgecolor='white', cmap=plt.cm.Reds)
        ax2.set_title(f"Arah Angin vs Kecepatan - {wind_region}", fontsize=12, fontweight='bold')
        ax2.set_legend(title="Kecepatan Angin (m/s)")
        
        st.pyplot(fig)
        
        # Analisis tambahan
        st.info(f"""
        💡 **Insight Wind Rose untuk {wind_region}:**
        - Arah angin dominan: **{df_wind_clean['wd'].mode()[0]:.0f}°**
        - Rata-rata kecepatan angin: **{df_wind_clean['WSPM'].mean():.2f} m/s**
        - Rata-rata PM2.5 saat angin dominan: **{df_wind_clean[df_wind_clean['wd'] == df_wind_clean['wd'].mode()[0]]['PM2.5'].mean():.2f} µg/m³**
        """)
    else:
        st.warning("⚠️ Tidak ada data wind direction dan wind speed yang valid untuk wilayah ini.")
else:
    st.warning("⚠️ Dataset tidak memiliki kolom 'wd' (wind direction) atau 'WSPM' (wind speed).")

st.markdown("---")

# Korelasi Cuaca dengan Polusi
st.markdown("""
    <h3 class="center-text" style="font-size: 24px;">🌦️ Korelasi antara Kondisi Cuaca dan Tingkat Polusi</h3>
""", unsafe_allow_html=True)

weather_factors = [selected_pollutant, "TEMP", "PRES", "DEWP", "RAIN"]
df_corr = df_filtered[weather_factors].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, 
            cbar_kws={'label': 'Korelasi'}, linewidths=0.5)
ax.set_title(f"Heatmap Korelasi {selected_pollutant} dengan Faktor Cuaca", fontsize=14, fontweight='bold')
st.pyplot(fig)

st.info("💡 **Note:** Nilai mendekati 0 = tidak ada korelasi, nilai mendekati ±1 = korelasi kuat (positif/negatif)")
st.markdown("---")

# Perbandingan Kualitas Udara antar Wilayah
st.markdown("""
    <h3 class="center-text" style="font-size: 24px;">🌍 Perbedaan Tingkat Polusi antara Wilayah</h3>
""", unsafe_allow_html=True)

df_region_avg = df_filtered.groupby("station")[pollutants].mean().reset_index()
st.dataframe(df_region_avg.style.format({col: "{:.2f}" for col in pollutants}), use_container_width=True)

fig, ax = plt.subplots(figsize=(12, 6))
df_region_avg.set_index(x="station",
    y=selected_pollutant,
    kind="bar")
ax.set_title("Rata-rata Polutan per Wilayah", fontsize=14, fontweight='bold')
ax.set_ylabel("Konsentrasi Polutan (µg/m³)", fontsize=12)
ax.set_xlabel("Wilayah", fontsize=12)
ax.legend(loc='best')
plt.xticks(rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')
st.pyplot(fig)

st.markdown("---")

# Jam dengan Kualitas Udara Terburuk
st.markdown("""
    <h3 class="center-text" style="font-size: 24px;">⏰ Waktu dengan Kualitas Udara Paling Buruk</h3>
""", unsafe_allow_html=True)

df_hourly = df_filtered.groupby(["hour", "station"])[selected_pollutant].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df_hourly, x="hour", y=selected_pollutant, hue="station", 
             marker="o", ax=ax, linewidth=2)
ax.set_title(f"Rata-rata {selected_pollutant} Berdasarkan Jam untuk Setiap Wilayah", 
             fontsize=14, fontweight='bold')
ax.set_ylabel(f"{selected_pollutant} (µg/m³)", fontsize=12)
ax.set_xlabel("Jam", fontsize=12)
ax.legend(title="Wilayah", loc='best')
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.markdown("---")

# Footer
st.caption("Made by Radya Ardi MC296D5X1815")
st.caption("Enhanced Dashboard with Advanced Filtering & Wind Rose Analysis")
