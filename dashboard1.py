import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from windrose import WindroseAxes
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Air Quality Dashboard",
    layout="wide"
)

sns.set(style="darkgrid")

# Background styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #1a1d24, #000000);
    color: white;
}
h1, h2, h3, h4, h5 {
    color: white;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv")

    # Combine date
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")

    # Convert hour
    df["hour"] = df["hour"].astype(int)

    # Season mapping
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


# -------------------------------------------------
# AIR QUALITY CATEGORY
# -------------------------------------------------
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


# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown("""
<h1 style="text-align:center; font-size:50px;">AIR QUALITY DASHBOARD</h1>
<h3 style="text-align:center; font-size:28px;">Beijing 2013–2017</h3>
""", unsafe_allow_html=True)
st.markdown("---")


# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("🎯 Filter Data")

# Date filter
date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    value=(df["date"].min(), df["date"].max()),
)

if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df[(df["date"] >= pd.Timestamp(start_date)) &
                     (df["date"] <= pd.Timestamp(end_date))]
else:
    df_filtered = df.copy()

# Station filter
stations = ["Semua Wilayah"] + sorted(df["station"].unique())
selected_station = st.sidebar.selectbox("🏢 Pilih Stasiun", stations)

if selected_station != "Semua Wilayah":
    df_filtered = df_filtered[df_filtered["station"] == selected_station]

# Polutan filter
pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
selected_pollutant = st.sidebar.selectbox(
    "🌫️ Polutan untuk Analisis Detail",
    pollutants
)

# Temperature filter
temp_range = st.sidebar.slider(
    "🌡️ Rentang Suhu (°C)",
    float(df["TEMP"].min()),
    float(df["TEMP"].max()),
    (float(df["TEMP"].min()), float(df["TEMP"].max()))
)
df_filtered = df_filtered[(df_filtered["TEMP"] >= temp_range[0]) &
                         (df_filtered["TEMP"] <= temp_range[1])]

# Pressure filter
pres_range = st.sidebar.slider(
    "📊 Rentang Tekanan (hPa)",
    float(df["PRES"].min()),
    float(df["PRES"].max()),
    (float(df["PRES"].min()), float(df["PRES"].max()))
)
df_filtered = df_filtered[(df_filtered["PRES"] >= pres_range[0]) &
                         (df_filtered["PRES"] <= pres_range[1])]

# Season filter
seasons = ["Semua Musim"] + sorted(df["season"].unique())
selected_season = st.sidebar.selectbox("🍂 Pilih Musim", seasons)

if selected_season != "Semua Musim":
    df_filtered = df_filtered[df_filtered["season"] == selected_season]

st.sidebar.info(f"📌 Data Terfilter: **{len(df_filtered):,}** baris")


# -------------------------------------------------
# METRICS
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Total Data", f"{len(df_filtered):,}")
with col2:
    st.metric("🌫️ Rata-rata PM2.5", f"{df_filtered['PM2.5'].mean():.2f}")
with col3:
    st.metric("🌡️ Rata-rata Suhu", f"{df_filtered['TEMP'].mean():.1f}°C")
with col4:
    st.metric("⚠️ Kategori Dominan", df_filtered["air_quality_category"].mode()[0])

st.markdown("---")


# -------------------------------------------------
# DISTRIBUSI KUALITAS UDARA
# -------------------------------------------------
st.subheader("Kualitas Udara Berdasarkan PM2.5")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Kategori PM2.5")
    st.markdown("""
| PM2.5 | Kategori |
|-------|----------|
| 0–15.5 | Baik |
| 15.6–55.4 | Sedang |
| 55.5–150.4 | Tidak Sehat |
| 150.5–250.4 | Sangat Tidak Sehat |
| >250.4 | Berbahaya |
""")

with col2:
    region_quality_df = df_filtered.groupby(
        ["station", "air_quality_category"]
    ).size().reset_index(name="count")
    st.dataframe(region_quality_df, height=220)

st.markdown("---")


# -------------------------------------------------
# TREND POLUTAN
# -------------------------------------------------
st.subheader(f"📈 Tren {selected_pollutant} per Tahun")

df_trend = df_filtered.groupby("year")[selected_pollutant].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_trend["year"], df_trend[selected_pollutant],
        marker="o", linewidth=2)
ax.set_title(f"Tren {selected_pollutant}", fontsize=14, weight="bold")
ax.set_ylabel("μg/m³")
ax.grid(alpha=0.3)
st.pyplot(fig)

st.markdown("---")


# -------------------------------------------------
# TREND PER MUSIM
# -------------------------------------------------
st.subheader("🍃 Rata-rata Polutan Berdasarkan Musim")

df_season = df_filtered.groupby("season")[pollutants].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
df_season.set_index("season")[pollutants].plot(kind="bar", ax=ax)
ax.set_title("Rata-rata Polutan per Musim", weight="bold")
plt.xticks(rotation=0)
st.pyplot(fig)

st.markdown("---")


# -------------------------------------------------
# WIND ROSE
# -------------------------------------------------
st.subheader("🌬️ Wind Rose (Arah & Kecepatan Angin)")

if selected_station == "Semua Wilayah":
    st.info("Pilih wilayah tertentu untuk melihat wind rose.")
else:
    df_w = df_filtered.dropna(subset=["wd", "WSPM"])

    if len(df_w) > 10:
        fig = plt.figure(figsize=(10, 6))
        ax = WindroseAxes.from_ax()
        ax.bar(df_w["wd"], df_w["WSPM"],
               opening=0.8, edgecolor="white")
        ax.set_legend(title="m/s")
        st.pyplot(fig)
    else:
        st.warning("Data wind direction tidak cukup.")

st.markdown("---")


# -------------------------------------------------
# KORELASI CUACA
# -------------------------------------------------
st.subheader("🌦️ Korelasi Polutan dengan Cuaca")

weather_cols = [selected_pollutant, "TEMP", "PRES", "DEWP", "RAIN"]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df_filtered[weather_cols].corr(),
            annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Heatmap Korelasi")
st.pyplot(fig)

st.markdown("---")


# -------------------------------------------------
# PERBANDINGAN WILAYAH
# -------------------------------------------------
st.subheader("🌍 Perbandingan Polusi Antar Wilayah")

df_region_avg = df_filtered.groupby("station")[pollutants].mean()

fig, ax = plt.subplots(figsize=(12, 6))
df_region_avg.plot(kind="bar", ax=ax)
ax.set_title("Per Wilayah", weight="bold")
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("---")


# -------------------------------------------------
# JAM TERBURUK
# -------------------------------------------------
st.subheader("⏰ Jam dengan Polusi Tertinggi")

df_hour = df_filtered.groupby(["hour", "station"])[selected_pollutant].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df_hour, x="hour", y=selected_pollutant,
             hue="station", marker="o", ax=ax)
ax.set_title("Rata-rata Polusi per Jam", weight="bold")
st.pyplot(fig)

# Footer
st.caption("Made by Radya Ardi")
