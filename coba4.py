import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------
# Config & Theme
# -------------------------
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

# Colors
PRIMARY = "#1f3b64"
SECONDARY = "#4b4f54"
ACCENT = "#2b8fd6"

# -------------------------
@st.cache_data
def load_data(path="main_data.csv"):
    df = pd.read_csv(path)
    
    # Ensure datetime
    if {"year","month","day"}.issubset(df.columns):
        df["date"] = pd.to_datetime(df[["year","month","day"]])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    
    # Hour as int
    if "hour" in df.columns:
        df["hour"] = df["hour"].astype(int)
    
    # Season
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
    
    # Air quality category
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
    
    return df

# -------------------------
# Load data
# -------------------------
try:
    df = load_data("main_data.csv")
except FileNotFoundError:
    st.error("File 'main_data.csv' tidak ditemukan. Pastikan file tersedia.")
    st.stop()

# -------------------------
# Page Layout
# -------------------------
st.title("🌍 Air Quality Dashboard — Beijing 2013-2017")
st.markdown("---")

# ========================
# SECTION 1: TOP METRICS (Modern Card Style)
# ========================

# Calculate AQI based on PM2.5 (simplified US EPA standard)
def calculate_aqi(pm25):
    if pd.isna(pm25):
        return 0
    if pm25 <= 12.0:
        return int((50/12.0) * pm25)
    elif pm25 <= 35.4:
        return int(50 + ((100-50)/(35.4-12.0)) * (pm25-12.0))
    elif pm25 <= 55.4:
        return int(100 + ((150-100)/(55.4-35.4)) * (pm25-35.4))
    elif pm25 <= 150.4:
        return int(150 + ((200-150)/(150.4-55.4)) * (pm25-55.4))
    elif pm25 <= 250.4:
        return int(200 + ((300-200)/(250.4-150.4)) * (pm25-150.4))
    else:
        return int(300 + ((500-300)/(500.4-250.4)) * (pm25-250.4))

def get_aqi_status(aqi):
    if aqi <= 50:
        return "BAIK", "#28a745"
    elif aqi <= 100:
        return "SEDANG", "#ffc107"
    elif aqi <= 150:
        return "TIDAK SEHAT (SENSITIF)", "#fd7e14"
    elif aqi <= 200:
        return "TIDAK SEHAT", "#dc3545"
    elif aqi <= 300:
        return "SANGAT TIDAK SEHAT", "#6f42c1"
    else:
        return "BERBAHAYA", "#8b0000"

# Calculate current values
df_curr = df.copy()

def safe_mean(series):
    return series.mean() if len(series) > 0 else 0

# ---- PM2.5 + AQI ----
if "PM2.5" in df_curr.columns:
    current_pm25 = safe_mean(df_curr["PM2.5"])
    current_aqi = calculate_aqi(current_pm25)
    status_text, status_color = get_aqi_status(current_aqi)
else:
    current_pm25 = 0
    current_aqi = 0
    status_text = "N/A"
    status_color = "#6c757d"

aqi_change = 0  # tidak ada periode pembanding

# ---- PM10 ----
current_pm10 = safe_mean(df_curr["PM10"]) if "PM10" in df_curr.columns else 0
pm10_change = 0

# ---- TEMP ----
current_temp = safe_mean(df_curr["TEMP"]) if "TEMP" in df_curr.columns else 0
temp_change = 0

# ---- DEWP (Humidity proxy)
current_humidity = safe_mean(df_curr["DEWP"]) if "DEWP" in df_curr.columns else 0
humidity_change = 0

# Display modern metrics cards
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 15px; border-left: 5px solid {status_color}'>
        <p style='color: #6c757d; font-size: 12px; margin: 0;'>Indeks Kualitas Udara (AQI)</p>
        <h1 style='font-size: 40px; margin: 10px 0; font-weight: bold;'>{current_aqi}</h1>
        <p style='color: {"#dc3545" if aqi_change > 0 else "#28a745"}; font-size: 14px; margin: 0;'>
            {"↓" if aqi_change < 0 else "↑"} {abs(aqi_change):.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='background-color: {status_color}; padding: 20px; border-radius: 15px;'>
        <p style='color: white; font-size: 12px; margin: 0; opacity: 0.9;'>Status Kualitas Udara:</p>
        <h1 style='color: white; font-size: 40px; margin: 10px 0; font-weight: bold;'>{status_text}</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Second row of metrics
col3, col4, col5, col6 = st.columns(4)

with col3:
    change_color = "#dc3545" if aqi_change > 0 else "#28a745"
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 10px; border-radius: 10px;'>
        <p style='color: #6c757d; font-size: 13px; margin: 0;'>Tingkat PM2.5</p>
        <h2 style='font-size: 32px; margin: 10px 0;'>{current_pm25:.2f} <span style='font-size: 16px;'>µg/m³</span></h2>
        <p style='color: {change_color}; font-size: 14px; margin: 0;'>
            {"↓" if aqi_change < 0 else "↑"} {abs(aqi_change):.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    pm10_color = "#dc3545" if pm10_change > 0 else "#28a745"
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 10px; border-radius: 10px;'>
        <p style='color: #6c757d; font-size: 13px; margin: 0;'>Tingkat PM10</p>
        <h2 style='font-size: 32px; margin: 10px 0;'>{current_pm10:.2f} <span style='font-size: 16px;'>µg/m³</span></h2>
        <p style='color: {pm10_color}; font-size: 14px; margin: 0;'>
            {"↓" if pm10_change < 0 else "↑"} {abs(pm10_change):.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    temp_color = "#dc3545" if temp_change > 0 else "#2b8fd6"
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 10px; border-radius: 10px;'>
        <p style='color: #6c757d; font-size: 13px; margin: 0;'>Suhu</p>
        <h2 style='font-size: 32px; margin: 10px 0;'>{current_temp:.0f}<span style='font-size: 20px;'>°C</span></h2>
        <p style='color: {temp_color}; font-size: 14px; margin: 0;'>
            {"↓" if temp_change < 0 else "↑"} {abs(temp_change):.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    humidity_color = "#2b8fd6" if humidity_change > 0 else "#dc3545"
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 10px; border-radius: 10px;'>
        <p style='color: #6c757d; font-size: 13px; margin: 0;'>Kelembaban</p>
        <h2 style='font-size: 32px; margin: 10px 0;'>{current_humidity:.2f}<span style='font-size: 16px;'>%</span></h2>
        <p style='color: {humidity_color}; font-size: 14px; margin: 0;'>
            {"↓" if humidity_change < 0 else "↑"} {abs(humidity_change):.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ========================
# SECTION 2: RINGKASAN VISUALISASI
# ========================
# Row 1: Windrose & Air Quality Category
st.subheader("Pilih Wilayah")
stations = ["Semua Wilayah"] + sorted(df["station"].dropna().unique().tolist())
selected_station = st.selectbox("Pilih Stasiun (Wilayah)", stations)

col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("🌬️Visualisasi Arah Datangnya Polusi")
    
    # Check if wind direction and speed columns exist
    if "wd" in df.columns and "WSPM" in df.columns:
        # Function to convert wind direction string to degrees
        def wind_direction_to_degrees(wd_str):
            """Convert wind direction string to degrees (0-360)"""
            direction_map = {
                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5}
            if pd.isna(wd_str):
                return np.nan
            return direction_map.get(str(wd_str).strip().upper(), np.nan)
        
        # Prepare wind data
        wind_data = df[["wd", "WSPM"]].copy()
        wind_data["wd_degrees"] = wind_data["wd"].apply(wind_direction_to_degrees)
        wind_data = wind_data.dropna()
        wind_data = wind_data[wind_data["WSPM"] >= 0]
        
        if len(wind_data) > 50:  # Need enough data points for windrose
            wind_data['wd_bin'] = (wind_data['wd_degrees'] // 22.5) * 22.5

            rose_df = wind_data.groupby('wd_bin')['WSPM'].mean().reset_index().dropna()
            
            fig = px.bar_polar(
                rose_df,
                r="WSPM",
                theta="wd_bin",
                color="WSPM",
                color_continuous_scale=px.colors.sequential.Viridis,
                title="Wind Rose (Arah & Kecepatan Angin)")
        
            fig.update_layout(
                polar=dict(
                    angularaxis=dict(direction="counterclockwise",
                                    rotation=90,
                                    tickmode='array',
                                    tickvals=[
                                        0, 22.5, 45, 67.5,
                                        90, 112.5, 135, 157.5,
                                        180, 202.5, 225, 247.5,
                                        270, 292.5, 315, 337.5],
                                    ticktext=[
                                        "N", "NNE", "NE", "ENE",
                                        "E", "ESE", "SE", "SSE",
                                        "S", "SSW", "SW", "WSW",
                                        "W", "WNW", "NW", "NNW"],),
                    radialaxis=dict(showticklabels=True)))
        
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Data arah angin tidak cukup untuk wind rose (min 50 data point). Menampilkan distribusi arah angin.")
        
            direction_counts = df["wd"].value_counts().reset_index()
            direction_counts.columns = ["direction", "count"]
            direction_counts["deg"] = direction_counts["direction"].apply(wind_direction_to_degrees)
            direction_counts = direction_counts.dropna().sort_values("deg")
            
                
            if len(direction_counts) > 0:
                fig2 = px.bar_polar(
                    direction_counts,
                    r="count",
                    theta="direction",
                    color="count",
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title="Distribusi Arah Angin"
                )
                fig2.update_layout(
                    polar=dict(
                        angularaxis=dict(direction="counterclockwise", rotation=90),
                        radialaxis=dict(showticklabels=True)
                    )
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Data arah angin tidak valid.")
    else:
        st.info("Kolom 'wd' (wind direction) atau 'WSPM' tidak tersedia dalam data.")

with col_b:
    st.subheader("🎯 Distribusi Kategori Kualitas Udara")
    
    if "air_quality_category" in df.columns:
        cat_counts = df["air_quality_category"].value_counts()
        
        # Define colors for each category
        color_map = {
            "Baik": "#28a745",
            "Sedang": "#ffc107",
            "Tidak Sehat": "#fd7e14",
            "Sangat Tidak Sehat": "#dc3545",
            "Berbahaya": "#6f42c1"
        }
        colors = [color_map.get(cat, "#6c757d") for cat in cat_counts.index]
        
        fig = px.pie(
            names=cat_counts.index,
            values=cat_counts.values,
            color=cat_counts.index,
            color_discrete_map=color_map,
            title=f"Proporsi Kategori Kualitas Udara<br><sup>Total: {len(df):,} data</sup>",
            hole=0  # Bisa diubah ke 0.4 kalau mau donut chart
        )
    
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True)
    
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Data kategori kualitas udara tidak tersedia.")

st.markdown("---")

# Row 2: Seasonal Trend per Region
st.subheader("📅 Tren dan Rata-rata PM2.5 Berdasarkan Musim")

if "season" in df.columns and "PM2.5" in df.columns:
    if selected_station == "Semua Wilayah":
        # Show comparison across all regions
        season_region = df.groupby(['season', 'station'])['PM2.5'].mean().reset_index()
        
        fig = px.bar(
            season_region,
            x="season",
            y="PM2.5",
            color="station",
            barmode="group",
            title="Perbandingan PM2.5 per Musim di Semua Wilayah",
            labels={"season": "Musim", "PM2.5": "Rata-rata PM2.5 (µg/m³)", "station": "Wilayah"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_layout(
            xaxis_title="Musim",
            yaxis_title="Rata-rata PM2.5 (µg/m³)",
            legend_title="Wilayah",
            bargap=0.2
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        season_data = df.groupby('season')['PM2.5'].mean().sort_values(ascending=False)

        fig = px.bar(
            x=season_data.index,
            y=season_data.values,
            labels={"x": "Musim", "y": "Rata-rata PM2.5 (µg/m³)"},
            title=f"Rata-rata PM2.5 per Musim di {selected_station}",
        )

        # Warna default
        colors = [ACCENT] * len(season_data)
        # Highlight musim terburuk (paling tinggi)
        colors[0] = "#dc3545"

        fig.update_traces(marker_color=colors)

        # Tambah value label di atas bar
        fig.update_traces(
            text=[f"{v:.1f}" for v in season_data.values],
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Rata-rata PM2.5 (µg/m³)",
            xaxis_title="Musim",
            uniformtext_minsize=8,
            uniformtext_mode='show'
        )

        st.plotly_chart(fig, use_container_width=True)

        worst_season = season_data.index[0]
        st.markdown(
            f"**💡 Insight:** Musim dengan PM2.5 tertinggi di {selected_station} adalah "
            f"**{worst_season}** ({season_data.iloc[0]:.2f} µg/m³)"
        )

else:
    st.info("Data musim atau PM2.5 tidak tersedia.")

st.markdown("---")

# ========================
# SECTION 3: DETAILED INSIGHTS
# ========================
st.header("🔬 Insight Detail untuk Pertanyaan EDA")

# -------------------------
# 1. Tren Tahunan
# -------------------------
st.subheader("1️⃣ Tren Kualitas Udara per Tahun")
# Pilihan polutan
pollutant_options = [col for col in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"] if col in df.columns]

selected_pollutant = st.selectbox(
    "Pilih Parameter Kualitas Udara:",
    pollutant_options,
    index=pollutant_options.index("PM2.5") if "PM2.5" in pollutant_options else 0)

if 'year' in df.columns and selected_pollutant in df.columns:
    if selected_station == "Semua Wilayah":
        # Compare all regions
        yearly_region = df.groupby(['year', 'station'])[selected_pollutant].mean().reset_index()
        
        fig = px.line(
            yearly_region,
            x="year",
            y=selected_pollutant,
            color="station",
            markers=True,
            title=f"Tren {selected_pollutant} per Tahun - Semua Wilayah",
            labels={
                "year": "Tahun",
                selected_pollutant: f"{selected_pollutant} (µg/m³)",
                "station": "Wilayah"})
        fig.update_layout(
            legend_title="Wilayah",
            hovermode="x unified",
            yaxis=dict(showgrid=True, gridwidth=0.3) )
        st.plotly_chart(fig, use_container_width=True)
    else:
        trend_df = df.groupby('year')[selected_pollutant].mean().reset_index()

        fig = px.line(
            trend_df,
            x="year",
            y=selected_pollutant,
            markers=True,
            title=f"Tren {selected_pollutant} di {selected_station}",
            labels={
                "year": "Tahun",
                selected_pollutant: f"{selected_pollutant} (µg/m³)"})

        # Warna line & fill
        fig.update_traces(
            line=dict(width=3, color=PRIMARY),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor=ACCENT + "55")

        fig.update_layout(
            yaxis=dict(showgrid=True, gridwidth=0.3),
            hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Kesimpulan tren
        if len(trend_df) >= 2:
            slope = np.polyfit(trend_df['year'], trend_df[selected_pollutant], 1)[0]
            verdict = 'Meningkat ⬆️' if slope > 0 else ('Menurun ⬇️' if slope < 0 else 'Stabil ➡️')
            st.markdown(
                f"**💡 Kesimpulan:** Tren {selected_pollutant} di {selected_station} cenderung **{verdict}** "
                f"(slope = {slope:.3f}).")
else:
    st.warning("Data tahun atau polutan tidak tersedia.")

st.markdown("---")

# -------------------------
# 2. Perbedaan Antar Wilayah
# -------------------------
st.subheader("2️⃣ Perbedaan Tingkat Polusi Antar Wilayah")

if selected_pollutant in df.columns:
    if selected_station == "Semua Wilayah":
        region_avg = df.groupby('station')[selected_pollutant].mean().sort_values(ascending=False).reset_index()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                region_avg,
                x=selected_pollutant,
                y="station",
                orientation="h",
                color=selected_pollutant,
                color_continuous_scale="RdYlGn_r",
                title=f"Perbandingan {selected_pollutant} Antar Wilayah",
                labels={
                    selected_pollutant: f"Rata-rata {selected_pollutant} (µg/m³)",
                    "station": "Wilayah" })
    
            # Styling
            fig.update_layout(
                xaxis=dict(showgrid=True, gridwidth=0.3),
                yaxis=dict(categoryorder="total descending"),
                coloraxis_colorbar=dict(title=f"{selected_pollutant} (µg/m³)"))
    
            # Add labels on bars
            fig.update_traces(
                texttemplate="%{x:.1f}",
                textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                region_avg.style.format({selected_pollutant: '{:.2f}'})
                .background_gradient(subset=[selected_pollutant], cmap='RdYlGn_r'),
                use_container_width=True,
                height=400)
        
        worst_region = region_avg.loc[0, 'station']
        worst_value = region_avg.loc[0, selected_pollutant]
        st.markdown(
            f"**💡 Kesimpulan:** Wilayah dengan {selected_pollutant} tertinggi adalah "
            f"**{worst_region}** ({worst_value:.2f} µg/m³).")
    else:
        st.info(f"Anda sedang melihat data untuk **{selected_station}** saja. Pilih 'Semua Wilayah' untuk melihat perbandingan.")
else:
    st.warning("Polutan tidak tersedia.")

st.markdown("---")

# -------------------------
# 3. Jam Terburuk
# -------------------------
st.subheader("3️⃣ Waktu dengan Kualitas Udara Paling Buruk")

if 'hour' in df.columns and selected_pollutant in df.columns:
    if selected_station == "Semua Wilayah":
        # Show all regions comparison
        hourly_region = df.groupby(['hour', 'station'])[selected_pollutant].mean().reset_index()
        
        fig = px.line(
            hourly_region,
            x="hour",
            y=selected_pollutant,
            color="station",
            markers=True,
            title=f"Rata-rata {selected_pollutant} per Jam - Semua Wilayah")
    
        fig.update_layout(
            xaxis_title="Jam",
            yaxis_title=f"{selected_pollutant} (µg/m³)",
            legend_title="Wilayah",
            xaxis=dict(dtick=2),
            template="plotly_white")
    
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Single region
        hourly = df.groupby('hour')[selected_pollutant].mean().reset_index()
        
        # Cari jam terburuk
        worst_h_idx = hourly[selected_pollutant].idxmax()
        worst_h = int(hourly.loc[worst_h_idx, 'hour'])
        worst_val = hourly.loc[worst_h_idx, selected_pollutant]
    
        fig = go.Figure()
    
        # Area fill
        fig.add_trace(go.Scatter(
            x=hourly['hour'],
            y=hourly[selected_pollutant],
            fill='tozeroy',
            mode='none',
            opacity=0.3,
            name="Area"))
    
        # Garis utama
        fig.add_trace(go.Scatter(
            x=hourly['hour'],
            y=hourly[selected_pollutant],
            mode='lines+markers',
            name=selected_station ))
    
        # Titik jam terburuk
        fig.add_trace(go.Scatter(
            x=[worst_h],
            y=[worst_val],
            mode='markers+text',
            marker=dict(size=14, color='red'),
            text=[f"Terburuk: {worst_h}:00"],
            textposition="top center",
            name="Terburuk"))
    
        fig.update_layout(
            title=f"Pola Harian {selected_pollutant} di {selected_station}",
            xaxis_title="Jam",
            yaxis_title=f"{selected_pollutant} (µg/m³)",
            xaxis=dict(dtick=2),
            template="plotly_white")
    
        st.plotly_chart(fig, use_container_width=True)
    
        st.markdown(
            f"**💡 Kesimpulan:** Jam dengan rata-rata {selected_pollutant} tertinggi di "
            f"**{selected_station}** adalah **{worst_h}:00** ({worst_val:.2f} µg/m³)." )
else:
    st.warning("Data jam atau polutan tidak tersedia.")

st.markdown("---")
st.caption("Made by Radya Ardi 💻 | Dashboard Analisis Kualitas Udara Beijing 2013-2017")
