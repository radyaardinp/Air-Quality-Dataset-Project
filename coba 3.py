import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes

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
# Sidebar filters
# -------------------------
st.sidebar.title("🔍 Filter")
min_date = df["date"].min()
max_date = df["date"].max()

date_range = st.sidebar.date_input(
    "Rentang Tanggal", 
    value=(min_date, max_date), 
    min_value=min_date, 
    max_value=max_date
)

stations = ["Semua Wilayah"] + sorted(df["station"].unique().tolist())
selected_station = st.sidebar.selectbox("Pilih Stasiun (Wilayah)", stations)

pollutants = [c for c in ["PM2.5","PM10","SO2","NO2","CO","O3"] if c in df.columns]
selected_pollutant = st.sidebar.selectbox("Pilih Polutan", pollutants) if len(pollutants) > 0 else "PM2.5"

# Apply filters
start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
df_filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
if selected_station != "Semua Wilayah":
    df_filtered = df_filtered[df_filtered["station"] == selected_station]

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
if "PM2.5" in df_filtered.columns and len(df_filtered) > 0:
    current_pm25 = df_filtered["PM2.5"].mean()
    current_aqi = calculate_aqi(current_pm25)
    status_text, status_color = get_aqi_status(current_aqi)
    
    # Calculate change (compare current period with previous period of same length)
    period_length = (end_date - start_date).days
    prev_start = start_date - pd.Timedelta(days=period_length)
    prev_end = start_date
    
    df_prev = df[(df["date"] >= prev_start) & (df["date"] < prev_end)]
    if selected_station != "Semua Wilayah":
        df_prev = df_prev[df_prev["station"] == selected_station]
    
    if len(df_prev) > 0:
        prev_aqi = calculate_aqi(df_prev["PM2.5"].mean())
        aqi_change = ((current_aqi - prev_aqi) / prev_aqi * 100) if prev_aqi > 0 else 0
    else:
        aqi_change = 0
else:
    current_pm25 = 0
    current_aqi = 0
    status_text = "N/A"
    status_color = "#6c757d"
    aqi_change = 0

# Get other metrics
if "PM10" in df_filtered.columns:
    current_pm10 = df_filtered["PM10"].mean()
    if len(df_prev) > 0 and "PM10" in df_prev.columns:
        pm10_change = ((current_pm10 - df_prev["PM10"].mean()) / df_prev["PM10"].mean() * 100)
    else:
        pm10_change = 0
else:
    current_pm10 = 0
    pm10_change = 0

if "TEMP" in df_filtered.columns:
    current_temp = df_filtered["TEMP"].mean()
    if len(df_prev) > 0 and "TEMP" in df_prev.columns:
        temp_change = ((current_temp - df_prev["TEMP"].mean()) / df_prev["TEMP"].mean() * 100)
    else:
        temp_change = 0
else:
    current_temp = 0
    temp_change = 0

if "DEWP" in df_filtered.columns:
    current_humidity = df_filtered["DEWP"].mean()
    if len(df_prev) > 0 and "DEWP" in df_prev.columns:
        humidity_change = ((current_humidity - df_prev["DEWP"].mean()) / df_prev["DEWP"].mean() * 100)
    else:
        humidity_change = 0
else:
    current_humidity = 0
    humidity_change = 0

# Display modern metrics cards
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 30px; border-radius: 10px; border-left: 5px solid {status_color}'>
        <p style='color: #6c757d; font-size: 14px; margin: 0;'>Indeks Kualitas Udara (AQI)</p>
        <h1 style='font-size: 56px; margin: 10px 0; font-weight: bold;'>{current_aqi}</h1>
        <p style='color: {"#dc3545" if aqi_change > 0 else "#28a745"}; font-size: 16px; margin: 0;'>
            {"↓" if aqi_change < 0 else "↑"} {abs(aqi_change):.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='background-color: {status_color}; padding: 30px; border-radius: 10px;'>
        <p style='color: white; font-size: 14px; margin: 0; opacity: 0.9;'>Status Kualitas Udara:</p>
        <h1 style='color: white; font-size: 42px; margin: 10px 0; font-weight: bold;'>{status_text}</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Second row of metrics
col3, col4, col5, col6 = st.columns(4)

with col3:
    change_color = "#dc3545" if aqi_change > 0 else "#28a745"
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
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
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
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
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
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
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
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
st.header("📈 Ringkasan Visualisasi")

# Row 1: Windrose & Air Quality Category
col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("🌬️ Wind Rose - Arah Datangnya Polusi")
    
    # Check if wind direction and speed columns exist
    if "wd" in df_filtered.columns and "WSPM" in df_filtered.columns:
        # Function to convert wind direction string to degrees
        def wind_direction_to_degrees(wd_str):
            """Convert wind direction string to degrees (0-360)"""
            direction_map = {
                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
            }
            if pd.isna(wd_str):
                return np.nan
            return direction_map.get(str(wd_str).strip().upper(), np.nan)
        
        # Prepare wind data
        wind_data = df_filtered[["wd", "WSPM"]].copy()
        
        # Convert wind direction to degrees
        wind_data["wd_degrees"] = wind_data["wd"].apply(wind_direction_to_degrees)
        
        # Remove NaN values and filter valid data
        wind_data = wind_data.dropna()
        wind_data = wind_data[wind_data["WSPM"] >= 0]
        
        if len(wind_data) > 50:  # Need enough data points for windrose
            try:
                fig = plt.figure(figsize=(6, 6))
                ax = WindroseAxes.from_ax(fig=fig)
                ax.bar(
                    wind_data["wd_degrees"].values, 
                    wind_data["WSPM"].values, 
                    normed=True, 
                    opening=0.8, 
                    edgecolor='white',
                    cmap=plt.cm.viridis,
                    bins=6
                )
                ax.set_legend(title="Kecepatan Angin (m/s)", loc='upper left', bbox_to_anchor=(1.1, 1))
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Tidak dapat membuat wind rose chart. Menampilkan distribusi arah angin sebagai gantinya.")
                # Fallback: show wind direction distribution as polar bar chart
                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
                
                # Count frequency of each direction
                direction_counts = df_filtered["wd"].value_counts()
                
                # Map to degrees and sort
                direction_degrees = {d: wind_direction_to_degrees(d) for d in direction_counts.index}
                sorted_dirs = sorted(direction_degrees.items(), key=lambda x: x[1] if not pd.isna(x[1]) else 999)
                
                theta = [np.deg2rad(deg) for _, deg in sorted_dirs if not pd.isna(deg)]
                counts = [direction_counts[dir_name] for dir_name, deg in sorted_dirs if not pd.isna(deg)]
                
                if len(theta) > 0:
                    width = np.deg2rad(22.5)
                    bars = ax.bar(theta, counts, width=width, bottom=0.0, alpha=0.7)
                    
                    # Color bars based on frequency
                    max_count = max(counts) if counts else 1
                    colors = plt.cm.viridis(np.array(counts) / max_count)
                    for bar, color in zip(bars, colors):
                        bar.set_facecolor(color)
                    
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    ax.set_title('Distribusi Arah Angin', pad=20)
                    st.pyplot(fig)
                else:
                    st.info("Data arah angin tidak valid.")
        else:
            st.info("Data arah angin tidak cukup untuk periode ini (minimal 50 data point diperlukan).")
    else:
        st.info("Kolom 'wd' (wind direction) atau 'WSPM' tidak tersedia dalam data.")

with col_b:
    st.subheader("🎯 Distribusi Kategori Kualitas Udara")
    
    if "air_quality_category" in df_filtered.columns:
        cat_counts = df_filtered["air_quality_category"].value_counts()
        
        # Define colors for each category
        color_map = {
            "Baik": "#28a745",
            "Sedang": "#ffc107",
            "Tidak Sehat": "#fd7e14",
            "Sangat Tidak Sehat": "#dc3545",
            "Berbahaya": "#6f42c1"
        }
        colors = [color_map.get(cat, "#6c757d") for cat in cat_counts.index]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            cat_counts.values, 
            labels=cat_counts.index, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax.set_title(f"Proporsi Kategori Kualitas Udara\n(Total: {len(df_filtered):,} data)")
        st.pyplot(fig)
    else:
        st.info("Data kategori kualitas udara tidak tersedia.")

st.markdown("---")

# Row 2: Seasonal Trend per Region
st.subheader("📅 Tren dan Rata-rata PM2.5 Berdasarkan Musim")

if "season" in df_filtered.columns and "PM2.5" in df_filtered.columns:
    if selected_station == "Semua Wilayah":
        # Show comparison across all regions
        season_region = df_filtered.groupby(['season', 'station'])['PM2.5'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(
            data=season_region, 
            x='season', 
            y='PM2.5', 
            hue='station',
            palette='Set2',
            ax=ax
        )
        ax.set_xlabel('Musim', fontsize=12)
        ax.set_ylabel('Rata-rata PM2.5 (µg/m³)', fontsize=12)
        ax.set_title('Perbandingan PM2.5 per Musim di Semua Wilayah', fontsize=14, fontweight='bold')
        ax.legend(title='Wilayah', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    else:
        # Show single region seasonal trend
        season_data = df_filtered.groupby('season')['PM2.5'].mean().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(season_data.index, season_data.values, color=ACCENT, edgecolor='black', linewidth=1.2)
        
        # Highlight worst season
        max_idx = season_data.values.argmax()
        bars[max_idx].set_color('#dc3545')
        
        ax.set_xlabel('Musim', fontsize=12)
        ax.set_ylabel('Rata-rata PM2.5 (µg/m³)', fontsize=12)
        ax.set_title(f'Rata-rata PM2.5 per Musim di {selected_station}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        st.pyplot(fig)
        
        worst_season = season_data.index[0]
        st.markdown(f"**💡 Insight:** Musim dengan PM2.5 tertinggi di {selected_station} adalah **{worst_season}** ({season_data.iloc[0]:.2f} µg/m³)")
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

if 'year' in df_filtered.columns and selected_pollutant in df_filtered.columns:
    if selected_station == "Semua Wilayah":
        # Compare all regions
        yearly_region = df_filtered.groupby(['year', 'station'])[selected_pollutant].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        for station in yearly_region['station'].unique():
            station_data = yearly_region[yearly_region['station'] == station]
            ax.plot(station_data['year'], station_data[selected_pollutant], 
                   marker='o', linewidth=2, label=station)
        
        ax.set_xlabel('Tahun', fontsize=12)
        ax.set_ylabel(f'{selected_pollutant} (µg/m³)', fontsize=12)
        ax.set_title(f'Tren {selected_pollutant} per Tahun - Semua Wilayah', fontsize=14, fontweight='bold')
        ax.legend(title='Wilayah', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    else:
        # Single region
        trend_df = df_filtered.groupby('year')[selected_pollutant].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(trend_df['year'], trend_df[selected_pollutant], 
               marker='o', linewidth=2.5, color=PRIMARY, markersize=8)
        ax.fill_between(trend_df['year'], trend_df[selected_pollutant], alpha=0.3, color=ACCENT)
        ax.set_xlabel('Tahun', fontsize=12)
        ax.set_ylabel(f'{selected_pollutant} (µg/m³)', fontsize=12)
        ax.set_title(f'Tren {selected_pollutant} di {selected_station}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        if len(trend_df) >= 2:
            slope = np.polyfit(trend_df['year'], trend_df[selected_pollutant], 1)[0]
            verdict = 'Meningkat ⬆️' if slope > 0 else ('Menurun ⬇️' if slope < 0 else 'Stabil ➡️')
            st.markdown(f"**💡 Kesimpulan:** Tren {selected_pollutant} di {selected_station} cenderung **{verdict}** (slope = {slope:.3f}).")
else:
    st.warning("Data tahun atau polutan tidak tersedia.")

st.markdown("---")

# -------------------------
# 2. Korelasi Cuaca
# -------------------------
st.subheader("2️⃣ Korelasi antara Polutan dan Faktor Cuaca")

weather_cols = [c for c in ['TEMP','PRES','DEWP','RAIN','WSPM'] if c in df_filtered.columns]
if selected_pollutant in df_filtered.columns and len(weather_cols) > 0:
    corr_df = df_filtered[[selected_pollutant] + weather_cols].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, fmt='.2f', ax=ax, 
               cmap='coolwarm', cbar_kws={'label':'Koefisien Korelasi'},
               linewidths=0.5, linecolor='white')
    ax.set_title(f'Korelasi {selected_pollutant} dengan Faktor Cuaca\n({selected_station})', 
                fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    # Find strongest correlation
    corrs = corr_df[selected_pollutant].drop(selected_pollutant).abs().sort_values(ascending=False)
    if len(corrs) > 0:
        top_var = corrs.index[0]
        top_val = corr_df.loc[top_var, selected_pollutant]
        direction = "positif" if top_val > 0 else "negatif"
        st.markdown(f"**💡 Kesimpulan:** Faktor cuaca yang paling berkaitan dengan {selected_pollutant} adalah **{top_var}** dengan korelasi **{direction}** (r = {top_val:.2f}).")
else:
    st.warning("Data cuaca atau polutan tidak tersedia.")

st.markdown("---")

# -------------------------
# 3. Perbedaan Antar Wilayah
# -------------------------
st.subheader("3️⃣ Perbedaan Rata-rata Polutan Antar Wilayah")

if selected_pollutant in df_filtered.columns:
    if selected_station == "Semua Wilayah":
        region_avg = df_filtered.groupby('station')[selected_pollutant].mean().sort_values(ascending=False).reset_index()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(data=region_avg, x=selected_pollutant, y='station', 
                             palette='RdYlGn_r', ax=ax, edgecolor='black', linewidth=1)
            ax.set_xlabel(f'Rata-rata {selected_pollutant} (µg/m³)', fontsize=12)
            ax.set_ylabel('Wilayah', fontsize=12)
            ax.set_title(f'Perbandingan {selected_pollutant} Antar Wilayah', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars.patches):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.1f}',
                       ha='left', va='center', fontsize=10, fontweight='bold')
            
            st.pyplot(fig)
        
        with col2:
            st.dataframe(
                region_avg.style.format({selected_pollutant: '{:.2f}'})
                .background_gradient(subset=[selected_pollutant], cmap='RdYlGn_r'),
                use_container_width=True,
                height=400
            )
        
        worst_region = region_avg.loc[0, 'station']
        worst_value = region_avg.loc[0, selected_pollutant]
        st.markdown(f"**💡 Kesimpulan:** Wilayah dengan {selected_pollutant} tertinggi adalah **{worst_region}** ({worst_value:.2f} µg/m³).")
    else:
        st.info(f"Anda sedang melihat data untuk **{selected_station}** saja. Pilih 'Semua Wilayah' untuk melihat perbandingan.")
else:
    st.warning("Polutan tidak tersedia.")

st.markdown("---")

# -------------------------
# 4. Jam Terburuk
# -------------------------
st.subheader("4️⃣ Jam dengan Kualitas Udara Paling Buruk")

if 'hour' in df_filtered.columns and selected_pollutant in df_filtered.columns:
    if selected_station == "Semua Wilayah":
        # Show all regions comparison
        hourly_region = df_filtered.groupby(['hour', 'station'])[selected_pollutant].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        for station in hourly_region['station'].unique():
            station_data = hourly_region[hourly_region['station'] == station]
            ax.plot(station_data['hour'], station_data[selected_pollutant], 
                   marker='o', linewidth=2, label=station, alpha=0.7)
        
        ax.set_xlabel('Jam', fontsize=12)
        ax.set_ylabel(f'{selected_pollutant} (µg/m³)', fontsize=12)
        ax.set_title(f'Rata-rata {selected_pollutant} per Jam - Semua Wilayah', fontsize=14, fontweight='bold')
        ax.legend(title='Wilayah', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    else:
        # Single region
        hourly = df_filtered.groupby('hour')[selected_pollutant].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(hourly['hour'], hourly[selected_pollutant], alpha=0.3, color=ACCENT)
        ax.plot(hourly['hour'], hourly[selected_pollutant], 
               marker='o', linewidth=2.5, color=PRIMARY, markersize=6)
        
        # Highlight worst hour
        worst_h_idx = hourly[selected_pollutant].idxmax()
        worst_h = int(hourly.loc[worst_h_idx, 'hour'])
        worst_val = hourly.loc[worst_h_idx, selected_pollutant]
        ax.plot(worst_h, worst_val, 'ro', markersize=12, label=f'Terburuk: {worst_h}:00')
        
        ax.set_xlabel('Jam', fontsize=12)
        ax.set_ylabel(f'{selected_pollutant} (µg/m³)', fontsize=12)
        ax.set_title(f'Pola Harian {selected_pollutant} di {selected_station}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(0, 24, 2))
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        st.markdown(f"**💡 Kesimpulan:** Jam dengan rata-rata {selected_pollutant} tertinggi di {selected_station} adalah **{worst_h}:00** ({worst_val:.2f} µg/m³).")
else:
    st.warning("Data jam atau polutan tidak tersedia.")

st.markdown("---")
st.caption("Made by Radya Ardi 💻 | Dashboard Analisis Kualitas Udara Beijing 2013-2017")
