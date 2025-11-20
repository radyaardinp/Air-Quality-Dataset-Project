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
st.markdown(f"**Wilayah terpilih:** {selected_station} | **Polutan:** {selected_pollutant}")
st.markdown("---")

# ========================
# SECTION 1: TOP METRICS
# ========================
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    st.metric("📊 Total Data", f"{len(df_filtered):,}")
with col2:
    if selected_pollutant and selected_pollutant in df_filtered.columns:
        avg_val = df_filtered[selected_pollutant].mean()
        st.metric(f"🌫️ Rata-rata {selected_pollutant}", f"{avg_val:.2f} µg/m³")
    else:
        st.metric("Rata-rata", "N/A")
with col3:
    if "TEMP" in df_filtered.columns:
        st.metric("🌡️ Rata-rata Suhu", f"{df_filtered['TEMP'].mean():.1f} °C")
    else:
        st.metric("Suhu", "N/A")
with col4:
    if "air_quality_category" in df_filtered.columns and len(df_filtered)>0:
        dom_cat = df_filtered["air_quality_category"].mode().iloc[0]
        st.metric("🏷️ Kategori Dominan", dom_cat)
    else:
        st.metric("Kategori", "N/A")

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
        # Remove NaN values
        wind_data = df_filtered[["wd", "WSPM", selected_pollutant]].dropna()
        
        if len(wind_data) > 0:
            fig = plt.figure(figsize=(6, 6))
            ax = WindroseAxes.from_ax(fig=fig)
            ax.bar(
                wind_data["wd"], 
                wind_data["WSPM"], 
                normed=True, 
                opening=0.8, 
                edgecolor='white',
                cmap=plt.cm.viridis
            )
            ax.set_legend(title="Kecepatan Angin (m/s)")
            st.pyplot(fig)
        else:
            st.info("Data arah angin tidak tersedia untuk periode ini.")
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
