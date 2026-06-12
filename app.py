import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import warnings
import io
warnings.filterwarnings('ignore')

# ============================================================
# 1. FUNGSI LOAD DATA (DENGAN DETEKSI TAHUN)
# ============================================================

def load_discharge_data(uploaded_file):
    """
    Membaca file CSV dengan format: Year, Month, Date, 06:00, 12:00, 18:00
    """
    # Baca file CSV dari uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Cek kolom yang tersedia
    available_columns = df.columns.tolist()
    
    # Deteksi apakah ada kolom Year
    has_year_column = 'Year' in available_columns or 'year' in available_columns
    
    # Transformasi dari wide ke long format
    records = []
    month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
                 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    
    for _, row in df.iterrows():
        # Ambil tahun
        if has_year_column:
            year = int(row['Year'])
        else:
            st.warning("Kolom 'Year' tidak ditemukan! Menggunakan tahun 2024 sebagai default. Tambahkan kolom 'Year' untuk akurasi yang lebih baik.")
            year = 2024
        
        month_name = row['Month']
        month_num = month_map[month_name]
        date = int(row['Date'])
        
        # Buat tanggal dengan tahun yang sesuai
        try:
            full_date = pd.to_datetime(f"{year}-{month_num:02d}-{date:02d}")
        except:
            continue
        
        # Tambahkan 3 pengukuran
        for hour, col_name in [('06:00', '06:00'), ('12:00', '12:00'), ('18:00', '18:00')]:
            if col_name in available_columns:
                value = row[col_name]
                if pd.notna(value):
                    datetime_str = f"{full_date.strftime('%Y-%m-%d')} {hour}"
                    records.append({
                        'datetime': pd.to_datetime(datetime_str),
                        'discharge': float(value),
                        'year': year,
                        'month': month_num,
                        'date': date
                    })
    
    if len(records) == 0:
        raise ValueError("Tidak ada data yang valid. Periksa format file CSV Anda.")
    
    df_long = pd.DataFrame(records)
    return df_long


# ============================================================
# 2. FUNGSI BFI YANG DIPERBAIKI
# ============================================================

def calculate_bfi(daily_flow, block_len=5, tp_factor=0.9):
    """
    Menghitung BFI dari daily flow series
    Method: Institute of Hydrology (UK) Baseflow Separation Method
    """
    x = daily_flow.values
    n = len(x)
    dates = daily_flow.index
    
    if n == 0:
        raise ValueError("Tidak ada data")
    
    # Block minimum method
    n_cols = int(np.ceil(n / block_len))
    y = np.full((block_len, n_cols), np.nan)
    y.flat[:n] = x
    
    idx_min_in_block = np.nanargmin(y, axis=0)
    idx_min_global = idx_min_in_block + (np.arange(n_cols) * block_len)
    valid_mask = idx_min_global < n
    idx_min_global = idx_min_global[valid_mask]
    block_min = x[idx_min_global]
    
    # Determine turning points
    if len(block_min) >= 3:
        block_mid = block_min[1:-1]
        cv_mod = tp_factor * block_mid
        is_tp_mid = (cv_mod <= block_min[2:]) & (cv_mod <= block_min[:-2])
        is_tp = np.concatenate([[False], is_tp_mid, [False]])
        tp_idx = idx_min_global[is_tp]
        tp_values = block_min[is_tp]
        
        if len(tp_idx) >= 2:
            f = interp1d(tp_idx, tp_values, kind='linear', 
                        bounds_error=False, fill_value=np.nan)
            baseflow = f(np.arange(n))
            baseflow = pd.Series(baseflow).ffill().bfill().values
        else:
            baseflow = np.full(n, np.nanmean(block_min))
    else:
        baseflow = np.full(n, np.nanmean(block_min))
    
    # Hitung BFI
    total_flow = np.nansum(x)
    total_baseflow = np.nansum(baseflow)
    bfi_overall = total_baseflow / total_flow if total_flow > 0 else 0
    
    bfi_daily = np.where(x > 0, baseflow / x, 0)
    bfi_daily = np.where(np.isfinite(bfi_daily), bfi_daily, 0)
    
    result = pd.DataFrame({
        'date': dates,
        'daily_mean_flow': x,
        'baseflow': baseflow,
        'bfi_daily': bfi_daily
    })
    
    return result, bfi_overall


def process_bfi(df_long, block_len=5, tp_factor=0.9):
    """
    Proses lengkap dari data 3x/hari ke BFI
    """
    # Konversi ke daily mean
    df_long['date_only'] = df_long['datetime'].dt.date
    daily_means = df_long.groupby('date_only')['discharge'].mean()
    daily_means.index = pd.to_datetime(daily_means.index)
    
    # Isi missing days
    full_index = pd.date_range(start=daily_means.index.min(), 
                               end=daily_means.index.max(), 
                               freq='D')
    daily_means = daily_means.reindex(full_index)
    
    missing_count = daily_means.isna().sum()
    if missing_count > 0:
        daily_means = daily_means.interpolate(method='linear', limit=3)
        daily_means = daily_means.ffill().bfill()
    
    # Hitung BFI
    result, bfi_overall = calculate_bfi(daily_means, block_len, tp_factor)
    
    # Tambahkan informasi tahun dan bulan
    result['year'] = result['date'].dt.year
    result['month'] = result['date'].dt.month
    
    years_list = sorted(result['year'].unique())
    
    # Hitung BFI per tahun
    yearly_bfi = result.groupby('year').agg({
        'bfi_daily': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'daily_mean_flow': 'sum',
        'baseflow': 'sum'
    }).round(4)
    
    yearly_bfi.columns = ['bfi_mean', 'bfi_median', 'bfi_std', 'bfi_min', 'bfi_max', 'days_count', 'total_flow', 'total_baseflow']
    yearly_bfi['bfi_overall'] = yearly_bfi['total_baseflow'] / yearly_bfi['total_flow']
    yearly_bfi = yearly_bfi.reset_index()
    
    metadata = {
        'n_days': len(daily_means),
        'n_measurements': len(df_long),
        'date_range': f"{daily_means.index.min().strftime('%Y-%m-%d')} to {daily_means.index.max().strftime('%Y-%m-%d')}",
        'start_year': daily_means.index.min().year,
        'end_year': daily_means.index.max().year,
        'total_years': len(years_list),
        'years': years_list,
        'yearly_bfi': yearly_bfi.to_dict('records'),
        'block_len': block_len,
        'tp_factor': tp_factor,
        'bfi_overall': bfi_overall,
        'missing_days': missing_count,
        'total_flow': result['daily_mean_flow'].sum(),
        'total_baseflow': result['baseflow'].sum()
    }
    
    return result, bfi_overall, metadata


# ============================================================
# 3. FUNGSI EKSPORT KE EXCEL
# ============================================================

def export_to_excel(result, metadata):
    """Export hasil ke Excel dalam memory buffer"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Daily data
        result.to_excel(writer, sheet_name='Daily Data', index=False)
        
        # Summary
        summary = pd.DataFrame({
            'Parameter': [
                'Overall BFI (Period)',
                'Mean Daily BFI', 
                'Median Daily BFI',
                'Std Daily BFI', 
                'Min BFI', 
                'Max BFI',
                'Q1 (25th percentile)',
                'Q3 (75th percentile)',
                'Total Flow (Qt)',
                'Total Baseflow (Qb)',
                'Qb/Qt Ratio',
                'Number of Days',
                'Missing Days (interpolated)',
                'Block Length (days)',
                'Turning Point Factor',
                'Analysis Period',
                'Total Years Analyzed'
            ],
            'Value': [
                f"{metadata['bfi_overall']:.4f}",
                f"{result['bfi_daily'].mean():.4f}",
                f"{result['bfi_daily'].median():.4f}",
                f"{result['bfi_daily'].std():.4f}",
                f"{result['bfi_daily'].min():.4f}",
                f"{result['bfi_daily'].max():.4f}",
                f"{result['bfi_daily'].quantile(0.25):.4f}",
                f"{result['bfi_daily'].quantile(0.75):.4f}",
                f"{metadata['total_flow']:.2f}",
                f"{metadata['total_baseflow']:.2f}",
                f"{metadata['total_baseflow'] / metadata['total_flow']:.4f}",
                metadata['n_days'],
                metadata['missing_days'],
                metadata['block_len'],
                metadata['tp_factor'],
                metadata['date_range'],
                metadata['total_years']
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Yearly BFI Summary
        yearly_df = pd.DataFrame(metadata['yearly_bfi'])
        yearly_df.to_excel(writer, sheet_name='Yearly BFI Analysis', index=False)
        
        # Monthly summary
        monthly = result.groupby(['year', 'month']).agg({
            'bfi_daily': ['mean', 'std', 'min', 'max', 'count'],
            'daily_mean_flow': 'sum',
            'baseflow': 'sum'
        }).round(4)
        monthly.to_excel(writer, sheet_name='Monthly Summary')
        
        # Metadata
        pd.DataFrame([metadata]).to_excel(writer, sheet_name='Metadata', index=False)
    
    output.seek(0)
    return output


# ============================================================
# 4. KONFIGURASI HALAMAN & TEMA
# ============================================================

st.set_page_config(
    page_title="HydroFlow BFI Analytics",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #06B6D4;
    }
    .info-card {
        background: rgba(59, 130, 246, 0.1);
        border-radius: 12px;
        padding: 15px;
        border-left: 4px solid #06B6D4;
    }
    .formula-card {
        background: rgba(16, 185, 129, 0.1);
        border-radius: 12px;
        padding: 15px;
        border-left: 4px solid #10B981;
        font-family: monospace;
    }
    .stButton > button {
        background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(6, 182, 212, 0.3);
    }
    .step-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# 5. SIDEBAR PANEL
# ============================================================

with st.sidebar:
    st.markdown("## 💧 Control Center")
    st.markdown("---")
    
    st.markdown("### 📂 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload CSV Discharge Data",
        type=["csv"],
        help="Format: Year, Month, Date, 06:00, 12:00, 18:00"
    )
    
    st.markdown("### ⚙️ Hyperparameters")
    block_len = st.slider(
        "Block Length (Days)",
        min_value=3,
        max_value=10,
        value=5,
        help="Panjang blok untuk mencari nilai minimum"
    )
    
    tp_factor = st.slider(
        "Turning Point Factor",
        min_value=0.5,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Faktor untuk menentukan turning points"
    )
    
    st.markdown("---")
    run_analysis = st.button("🚀 Run Advanced Analysis", use_container_width=True)
    
    st.markdown("---")
    
    # Tab informasi di sidebar
    st.markdown("### 📚 Information Menu")
    info_tab = st.radio(
        "Pilih Informasi:",
        ["📖 Cara Penggunaan", "🔬 Teori & Metode", "⚙️ Hyperparameters", "📐 Formula & Persamaan"]
    )


# ============================================================
# 6. FUNGSI INFORMASI (TAB)
# ============================================================

def show_usage_guide():
    """Menampilkan panduan penggunaan aplikasi"""
    st.markdown("## 📖 Panduan Penggunaan Aplikasi")
    
    st.markdown("""
    ### Langkah-langkah Menggunakan Web App:
    
    <div class='step-card'>
    <strong>Step 1: Persiapan Data</strong><br>
    Siapkan file CSV dengan format berikut:
    <ul>
    <li>Kolom <strong>Year</strong>: Tahun (contoh: 2023, 2024)</li>
    <li>Kolom <strong>Month</strong>: Nama bulan (Jan, Feb, Mar, ..., Dec)</li>
    <li>Kolom <strong>Date</strong>: Tanggal (1-31)</li>
    <li>Kolom <strong>06:00, 12:00, 18:00</strong>: Nilai debit pada jam tersebut</li>
    </ul>
    </div>
    
    <div class='step-card'>
    <strong>Step 2: Upload File</strong><br>
    Upload file CSV melalui panel <strong>Control Center</strong> di sebelah kiri.
    </div>
    
    <div class='step-card'>
    <strong>Step 3: Atur Hyperparameters</strong><br>
    Sesuaikan nilai <strong>Block Length</strong> dan <strong>Turning Point Factor</strong> sesuai kebutuhan.
    </div>
    
    <div class='step-card'>
    <strong>Step 4: Jalankan Analisis</strong><br>
    Klik tombol <strong>"Run Advanced Analysis"</strong> untuk memproses data.
    </div>
    
    <div class='step-card'>
    <strong>Step 5: Interpretasi Hasil</strong><br>
    Dashboard akan menampilkan:
    <ul>
    <li>Periode analisis (tahun berapa sampai tahun berapa)</li>
    <li>Metrik utama BFI, Total Flow, Baseflow</li>
    <li>Tabel dan chart BFI per tahun</li>
    <li>Hydrograph separation (Qt vs Qb)</li>
    <li>Distribusi BFI harian</li>
    <li>Download report Excel lengkap</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Contoh data
    with st.expander("📋 Contoh Format File CSV", expanded=False):
        sample_data = pd.DataFrame({
            'Year': [2023, 2023, 2023, 2024, 2024],
            'Month': ['Jan', 'Jan', 'Jan', 'Jan', 'Jan'],
            'Date': [1, 2, 3, 4, 5],
            '06:00': [10.5, 10.2, 9.8, 8.5, 8.2],
            '12:00': [12.3, 11.9, 10.5, 9.2, 8.9],
            '18:00': [11.8, 11.5, 10.1, 8.8, 8.5]
        })
        st.dataframe(sample_data, use_container_width=True)


def show_theory_method():
    """Menampilkan teori dan metode BFI"""
    st.markdown("## 🔬 Teori & Metode Base Flow Index (BFI)")
    
    st.markdown("""
    ### Apa itu Base Flow Index (BFI)?
    
    **Base Flow Index (BFI)** adalah rasio antara aliran dasar (*baseflow*) dengan total aliran sungai (*total flow*). 
    BFI merupakan indikator penting dalam hidrologi yang menunjukkan kontribusi air tanah terhadap aliran sungai.
    
    ### Klasifikasi Nilai BFI:
    
    | Nilai BFI | Klasifikasi | Karakteristik |
    |-----------|-------------|----------------|
    | 0.8 - 1.0 | Sangat Tinggi | Didominasi aliran air tanah |
    | 0.6 - 0.8 | Tinggi | Kontribusi air tanah signifikan |
    | 0.4 - 0.6 | Sedang | Keseimbangan air tanah dan limpasan |
    | 0.2 - 0.4 | Rendah | Didominasi limpasan permukaan |
    | 0.0 - 0.2 | Sangat Rendah | Limpasan permukaan dominan |
    
    ### Metode yang Digunakan: Institute of Hydrology (UK) Method
    
    Metode ini dikembangkan oleh Institute of Hydrology, UK (sekarang CEH) dan merupakan metode standar untuk 
    pemisahan aliran dasar (*baseflow separation*).
    
    ### Asumsi Dasar:
    
    1. **Aliran sungai terdiri dari dua komponen**:
       - *Quick flow* (limpasan permukaan) - respons cepat terhadap hujan
       - *Baseflow* (aliran dasar) - respons lambat dari air tanah
    
    2. **Turning points** adalah titik-titik minimum yang merepresentasikan kondisi aliran dasar
    
    3. **Interpolasi linear antar turning points** menghasilkan kurva aliran dasar
    
    ### Karakteristik Metode:
    
    - **Keunggulan**: Objektif, dapat direproduksi, cocok untuk data harian
    - **Keterbatasan**: Bergantung pada pemilihan parameter (block length dan tp factor)
    - **Aplikasi**: Analisis hidrologi, pengelolaan DAS, studi perubahan iklim
    """, unsafe_allow_html=True)


def show_hyperparameters():
    """Menampilkan penjelasan hyperparameters"""
    st.markdown("## ⚙️ Penjelasan Hyperparameters")
    
    st.markdown("""
    ### 1. Block Length (Days)
    
    **Definisi:** Panjang blok (dalam hari) yang digunakan untuk mencari nilai minimum aliran.
    
    **Cara Kerja:**
    - Data aliran harian dibagi menjadi beberapa blok dengan panjang = Block Length
    - Di setiap blok, dicari nilai aliran minimum
    - Nilai minimum ini menjadi kandidat *turning points*
    
    **Pengaruh Parameter:**
    - **Nilai kecil (3-5 hari)**: Lebih sensitif, menghasilkan lebih banyak turning points, BFI cenderung lebih rendah
    - **Nilai besar (7-10 hari)**: Lebih halus, menghasilkan lebih sedikit turning points, BFI cenderung lebih tinggi
    
    **Rekomendasi:**
    - Data harian: 5 hari (default)
    - Data dengan variasi tinggi: 3-4 hari
    - Data dengan variasi rendah: 6-7 hari
    
    ### 2. Turning Point Factor
    
    **Definisi:** Faktor pengali untuk menentukan apakah suatu titik minimum merupakan *turning point*.
    
    **Cara Kerja:**
    
    Jika cv_mod <= block_min[i+1] DAN cv_mod <= block_min[i-1]
    maka titik tersebut adalah turning point
    
    Dimana: cv_mod = tp_factor x block_min[i]
    
    **Pengaruh Parameter:**
    - **Nilai kecil (0.5-0.7)**: Lebih ketat, lebih sedikit turning points, BFI lebih tinggi
    - **Nilai besar (0.8-1.0)**: Lebih longgar, lebih banyak turning points, BFI lebih rendah
    
    **Rekomendasi:**
    - Default: 0.9 (standar Institute of Hydrology)
    - Untuk data dengan noise tinggi: 0.7-0.8
    - Untuk data yang halus: 0.9-1.0
    
    ### Interaksi Kedua Parameter:
    
    Block Length dan Turning Point Factor bekerja bersama-sama:
    - **Block Length** mengontrol seberapa sering kita mencari titik minimum
    - **Turning Point Factor** mengontrol seberapa ketat kita memilih turning points
    
    ### Tips Optimasi:
    
    1. Mulai dengan default (block_len=5, tp_factor=0.9)
    2. Jika BFI terlalu rendah -> tingkatkan block_len atau turunkan tp_factor
    3. Jika BFI terlalu tinggi -> turunkan block_len atau tingkatkan tp_factor
    4. Bandingkan dengan metode manual atau literatur untuk validasi
    """, unsafe_allow_html=True)


def show_formulas():
    """Menampilkan formula dan persamaan"""
    st.markdown("## 📐 Formula & Persamaan")
    
    st.markdown("""
    ### Formula Dasar BFI
    
    <div class='formula-card'>
    <strong>BFI (Base Flow Index)</strong><br>
    <span style='font-size:20px; font-family:monospace'>
    BFI = Qb / Qt
    </span><br><br>
    Dimana:<br>
    Qb = Total Baseflow (aliran dasar)<br>
    Qt = Total Flow (total aliran sungai)
    </div>
    
    ### Daily BFI
    
    <div class='formula-card'>
    <strong>BFI Harian</strong><br>
    <span style='font-size:18px; font-family:monospace'>
    BFI_daily = baseflow_t / discharge_t
    </span><br><br>
    Dimana:<br>
    baseflow_t = nilai baseflow pada hari ke-t<br>
    discharge_t = nilai debit pada hari ke-t
    </div>
    
    ### Block Minimum Method
    
    <div class='formula-card'>
    <strong>Pencarian Minimum per Blok</strong><br>
    <span style='font-size:16px; font-family:monospace'>
    m_j = min(q_i) untuk i = (j-1)*L + 1 sampai j*L
    </span><br><br>
    Dimana:<br>
    m_j = nilai minimum pada blok ke-j<br>
    L = Block Length<br>
    q_i = debit harian ke-i
    </div>
    
    ### Turning Point Condition
    
    <div class='formula-card'>
    <strong>Kondisi Turning Point</strong><br>
    <span style='font-size:16px; font-family:monospace'>
    TP = True jika: (a * m_j) <= m_{j-1} DAN (a * m_j) <= m_{j+1}
    </span><br><br>
    Dimana:<br>
    m_j = nilai minimum blok ke-j<br>
    a = Turning Point Factor (tp_factor)<br>
    TP = Turning Point
    </div>
    
    ### Interpolasi Baseflow
    
    <div class='formula-card'>
    <strong>Interpolasi Linear antar Turning Points</strong><br>
    <span style='font-size:16px; font-family:monospace'>
    baseflow(t) = baseflow(tp_k) + [baseflow(tp_{k+1}) - baseflow(tp_k)] * (t - t_k)/(t_{k+1} - t_k)
    </span><br><br>
    Dimana:<br>
    tp_k = turning point ke-k<br>
    t = waktu antara dua turning points
    </div>
    
    ### Interpretasi Hasil
    
    | Rentang BFI | Interpretasi |
    |-------------|--------------|
    | 0.8 - 1.0 | Aliran didominasi air tanah (akuifer) |
    | 0.6 - 0.8 | Kontribusi air tanah signifikan |
    | 0.4 - 0.6 | Keseimbangan air tanah dan limpasan |
    | 0.2 - 0.4 | Didominasi limpasan permukaan |
    | 0.0 - 0.2 | Aliran permukaan mendominasi |
    
    ### Referensi:
    
    1. Institute of Hydrology (1980). Low Flow Studies. Report No. 1, Wallingford, UK.
    2. Nathan, R.J. and McMahon, T.A. (1990). Evaluation of automated techniques for base flow and recession analyses.
    3. Smakhtin, V.U. (2001). Low flow hydrology: a review.
    """, unsafe_allow_html=True)


# ============================================================
# 7. MAIN DASHBOARD DISPLAY
# ============================================================

st.title("📊 HydroFlow BFI Dashboard")
st.caption("Advanced Automated Baseflow Separation Framework | Multi-Year Data Support with Annual BFI Analysis")

# Tampilkan informasi berdasarkan pilihan di sidebar
if info_tab == "📖 Cara Penggunaan":
    show_usage_guide()
    st.markdown("---")
elif info_tab == "🔬 Teori & Metode":
    show_theory_method()
    st.markdown("---")
elif info_tab == "⚙️ Hyperparameters":
    show_hyperparameters()
    st.markdown("---")
elif info_tab == "📐 Formula & Persamaan":
    show_formulas()
    st.markdown("---")

# Main analysis section
if uploaded_file and run_analysis:
    with st.spinner("🔄 Processing hydrological time-series data..."):
        try:
            # Load dan proses data
            df_long = load_discharge_data(uploaded_file)
            result, bfi_overall, metadata = process_bfi(df_long, block_len, tp_factor)
            
            # ========== SECTION 1: PERIODE ANALISIS ==========
            st.markdown("### 📅 Analysis Period Information")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f"<div class='info-card'>"
                    f"<strong>📆 Start Year</strong><br>"
                    f"<span style='font-size:24px; font-weight:bold; color:#06B6D4'>{metadata['start_year']}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"<div class='info-card'>"
                    f"<strong>📅 End Year</strong><br>"
                    f"<span style='font-size:24px; font-weight:bold; color:#F59E0B'>{metadata['end_year']}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    f"<div class='info-card'>"
                    f"<strong>📊 Total Years</strong><br>"
                    f"<span style='font-size:24px; font-weight:bold; color:#10B981'>{metadata['total_years']} tahun</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col4:
                st.markdown(
                    f"<div class='info-card'>"
                    f"<strong>📈 Total Days</strong><br>"
                    f"<span style='font-size:24px; font-weight:bold; color:#3B82F6'>{metadata['n_days']} hari</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            # ========== SECTION 2: METRIK UTAMA ==========
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Klasifikasi BFI
                if bfi_overall >= 0.8:
                    bfi_class = "Sangat Tinggi"
                    bfi_color = "#06B6D4"
                elif bfi_overall >= 0.6:
                    bfi_class = "Tinggi"
                    bfi_color = "#10B981"
                elif bfi_overall >= 0.4:
                    bfi_class = "Sedang"
                    bfi_color = "#F59E0B"
                elif bfi_overall >= 0.2:
                    bfi_class = "Rendah"
                    bfi_color = "#EF4444"
                else:
                    bfi_class = "Sangat Rendah"
                    bfi_color = "#8B5CF6"
                    
                st.markdown(
                    f"<div class='metric-card'><h3 style='margin:0; color:#94A3B8'>Overall BFI</h3>"
                    f"<h1 style='margin:0; color:{bfi_color}'>{bfi_overall:.4f}</h1>"
                    f"<p style='margin:0; color:#94A3B8; font-size:12px'>Klasifikasi: {bfi_class}</p>"
                    f"<p style='margin:0; color:#94A3B8; font-size:11px'>Periode {metadata['start_year']}-{metadata['end_year']}</p></div>",
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"<div class='metric-card'><h3 style='margin:0; color:#94A3B8'>Total Flow (Qt)</h3>"
                    f"<h1 style='margin:0; color:#3B82F6'>{metadata['total_flow']:.1f}</h1></div>",
                    unsafe_allow_html=True
                )
            
            with col3:
                st.markdown(
                    f"<div class='metric-card'><h3 style='margin:0; color:#94A3B8'>Baseflow (Qb)</h3>"
                    f"<h1 style='margin:0; color:#10B981'>{metadata['total_baseflow']:.1f}</h1></div>",
                    unsafe_allow_html=True
                )
            
            with col4:
                st.markdown(
                    f"<div class='metric-card'><h3 style='margin:0; color:#94A3B8'>Missing Days</h3>"
                    f"<h1 style='margin:0; color:#F59E0B'>{metadata['missing_days']}</h1></div>",
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            # ========== SECTION 3: YEARLY BFI ANALYSIS ==========
            st.subheader("📊 Yearly BFI Analysis - Per Tahun")
            
            # Tabel BFI per tahun
            yearly_df = pd.DataFrame(metadata['yearly_bfi'])
            
            # Format tampilan tabel
            display_yearly = yearly_df[['year', 'bfi_mean', 'bfi_median', 'bfi_std', 'bfi_min', 'bfi_max', 'bfi_overall']].copy()
            display_yearly.columns = ['Tahun', 'Mean BFI', 'Median BFI', 'Std BFI', 'Min BFI', 'Max BFI', 'BFI Overall']
            display_yearly = display_yearly.round(4)
            
            st.dataframe(display_yearly, use_container_width=True)
            
            # Bar chart BFI per tahun
            yearly_fig = go.Figure()
            
            yearly_fig.add_trace(go.Bar(
                x=yearly_df['year'],
                y=yearly_df['bfi_overall'],
                name='BFI Overall',
                marker_color='#3B82F6',
                text=yearly_df['bfi_overall'].round(4),
                textposition='auto',
                hovertemplate='<b>Tahun %{x}</b><br>BFI: %{y:.4f}<extra></extra>'
            ))
            
            yearly_fig.add_trace(go.Scatter(
                x=yearly_df['year'],
                y=yearly_df['bfi_mean'],
                name='Mean BFI',
                line=dict(color='#F59E0B', width=3, dash='dash'),
                mode='lines+markers',
                marker=dict(size=8, color='#F59E0B'),
                hovertemplate='<b>Tahun %{x}</b><br>Mean BFI: %{y:.4f}<extra></extra>'
            ))
            
            # Tambahkan garis rata-rata keseluruhan
            yearly_fig.add_hline(
                y=bfi_overall,
                line_dash="dot",
                line_color="#EF4444",
                annotation_text=f"Overall BFI: {bfi_overall:.4f}",
                annotation_position="top right"
            )
            
            yearly_fig.update_layout(
                template="plotly_dark",
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Tahun",
                yaxis_title="Nilai BFI",
                title="Perbandingan BFI per Tahun",
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(0,0,0,0.5)'
                )
            )
            
            yearly_fig.update_yaxes(range=[0, 1])
            
            st.plotly_chart(yearly_fig, use_container_width=True)
            
            # Informasi tren BFI
            if len(yearly_df) > 1:
                bfi_trend = yearly_df['bfi_overall'].iloc[-1] - yearly_df['bfi_overall'].iloc[0]
                trend_icon = "📈" if bfi_trend > 0 else "📉"
                trend_color = "#10B981" if bfi_trend > 0 else "#EF4444"
                trend_text = "meningkat" if bfi_trend > 0 else "menurun"
                
                st.markdown(
                    f"<div class='info-card' style='text-align:center'>"
                    f"<strong>{trend_icon} Tren BFI {trend_text}</strong><br>"
                    f"Dari <strong>{yearly_df['year'].iloc[0]}</strong> (BFI: {yearly_df['bfi_overall'].iloc[0]:.4f}) "
                    f"ke <strong>{yearly_df['year'].iloc[-1]}</strong> (BFI: {yearly_df['bfi_overall'].iloc[-1]:.4f})<br>"
                    f"<span style='color:{trend_color}; font-size:18px; font-weight:bold'>Perubahan: {bfi_trend:+.4f}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            # ========== SECTION 4: YEARLY COMPARISON CHART ==========
            st.subheader("📊 Perbandingan Qt vs Qb per Tahun")
            
            comparison_fig = go.Figure()
            
            comparison_fig.add_trace(go.Bar(
                x=yearly_df['year'],
                y=yearly_df['total_flow'],
                name='Total Flow (Qt)',
                marker_color='#3B82F6',
                text=yearly_df['total_flow'].round(1),
                textposition='outside'
            ))
            
            comparison_fig.add_trace(go.Bar(
                x=yearly_df['year'],
                y=yearly_df['total_baseflow'],
                name='Baseflow (Qb)',
                marker_color='#10B981',
                text=yearly_df['total_baseflow'].round(1),
                textposition='outside'
            ))
            
            comparison_fig.update_layout(
                template="plotly_dark",
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Tahun",
                yaxis_title="Volume",
                title="Perbandingan Total Flow dan Baseflow per Tahun",
                barmode='group',
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            st.markdown("---")
            
            # ========== SECTION 5: HYDROGRAPH ==========
            st.subheader("📈 Hydrograph Separation Analysis")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Discharge vs Baseflow Over Time", "Daily Baseflow Index (BFI)"),
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4]
            )
            
            # Hydrograph plot
            fig.add_trace(
                go.Scatter(
                    x=result['date'],
                    y=result['daily_mean_flow'],
                    name='Qt (Daily Mean Flow)',
                    fill='tozeroy',
                    line=dict(color='#3B82F6', width=1),
                    fillcolor='rgba(59, 130, 246, 0.3)'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=result['date'],
                    y=result['baseflow'],
                    name='Qb (Baseflow)',
                    fill='tozeroy',
                    line=dict(color='#10B981', width=2),
                    fillcolor='rgba(16, 185, 129, 0.2)'
                ),
                row=1, col=1
            )
            
            # Daily BFI plot
            fig.add_trace(
                go.Scatter(
                    x=result['date'],
                    y=result['bfi_daily'],
                    name='Daily BFI',
                    line=dict(color='#F59E0B', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(245, 158, 11, 0.2)'
                ),
                row=2, col=1
            )
            
            # Add horizontal line for mean BFI
            fig.add_hline(
                y=result['bfi_daily'].mean(),
                line_dash="dash",
                line_color="#EF4444",
                annotation_text=f"Mean BFI = {result['bfi_daily'].mean():.3f}",
                row=2, col=1
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=700,
                showlegend=True,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title={
                    'text': f"BFI Analysis Results (Overall BFI = {bfi_overall:.4f})",
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Discharge", row=1, col=1)
            fig.update_yaxes(title_text="BFI", range=[0, 1], row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ========== SECTION 6: STATISTIK ==========
            st.subheader("📊 Statistical Summary")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Mean Daily BFI", f"{result['bfi_daily'].mean():.4f}")
            with stat_col2:
                st.metric("Median Daily BFI", f"{result['bfi_daily'].median():.4f}")
            with stat_col3:
                st.metric("Std Daily BFI", f"{result['bfi_daily'].std():.4f}")
            with stat_col4:
                st.metric("Qb/Qt Ratio", f"{metadata['total_baseflow']/metadata['total_flow']:.4f}")
            
            st.markdown("---")
            
            # ========== SECTION 7: DISTRIBUSI ==========
            st.subheader("📊 Distribution of Daily BFI")
            
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(
                x=result['bfi_daily'],
                nbinsx=30,
                marker_color='#3B82F6',
                opacity=0.7,
                name='Frequency'
            ))
            
            hist_fig.add_vline(
                x=result['bfi_daily'].mean(),
                line_dash="dash",
                line_color="#EF4444",
                annotation_text=f"Mean: {result['bfi_daily'].mean():.3f}"
            )
            
            hist_fig.add_vline(
                x=result['bfi_daily'].median(),
                line_dash="dash",
                line_color="#F59E0B",
                annotation_text=f"Median: {result['bfi_daily'].median():.3f}"
            )
            
            hist_fig.update_layout(
                template="plotly_dark",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="BFI Value",
                yaxis_title="Frequency",
                showlegend=True
            )
            
            st.plotly_chart(hist_fig, use_container_width=True)
            
            st.markdown("---")
            
            # ========== SECTION 8: PREVIEW & DOWNLOAD ==========
            st.subheader("📋 Processed Data Ledger")
            st.dataframe(result.head(50), use_container_width=True)
            
            # Download button
            excel_file = export_to_excel(result, metadata)
            st.download_button(
                label="🟢 Download Complete Excel Report",
                data=excel_file,
                file_name="BFI_Complete_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # Metadata expander
            with st.expander("ℹ️ Processing Metadata"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Data Info:**")
                    st.write(f"- Total Days: {metadata['n_days']}")
                    st.write(f"- Measurements: {metadata['n_measurements']}")
                    st.write(f"- Missing Days: {metadata['missing_days']}")
                    st.write(f"- Years: {', '.join(map(str, metadata['years']))}")
                    st.write(f"- Start Year: {metadata['start_year']}")
                    st.write(f"- End Year: {metadata['end_year']}")
                    st.write(f"- Total Years: {metadata['total_years']}")
                with col2:
                    st.write("**Parameters:**")
                    st.write(f"- Block Length: {metadata['block_len']} days")
                    st.write(f"- TP Factor: {metadata['tp_factor']}")
                    st.write(f"- Date Range: {metadata['date_range']}")
                    st.write(f"- Overall BFI: {metadata['bfi_overall']:.4f}")
                
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            st.info("Pastikan format CSV Anda sesuai: Kolom 'Year', 'Month', 'Date', '06:00', '12:00', '18:00'")
            
else:
    if not run_analysis and not uploaded_file:
        st.info("💡 Silakan unggah file data discharge Anda di panel sebelah kiri dan klik 'Run Advanced Analysis' untuk memulai visualisasi interaktif.")
    
    # Tampilkan contoh format file
    with st.expander("📋 Lihat Contoh Format File CSV", expanded=False):
        st.markdown("""
        **Format file CSV yang diharapkan (dengan kolom Year):**
        
        | Year | Month | Date | 06:00 | 12:00 | 18:00 |
        |------|-------|------|-------|-------|-------|
        | 2023 | Jan   | 1    | 10.5  | 12.3  | 11.8  |
        | 2023 | Jan   | 2    | 10.2  | 11.9  | 11.5  |
        | 2023 | Jan   | 3    | 9.8   | 10.5  | 10.1  |
        | 2024 | Jan   | 4    | 8.5   | 9.2   | 8.8   |
        
        **Keterangan:**
        - **Year**: Tahun (contoh: 2023, 2024) - WAJIB ADA
        - Month: Nama bulan (Jan, Feb, Mar, ..., Dec)
        - Date: Tanggal (1-31)
        - 06:00, 12:00, 18:00: Nilai debit pada jam tersebut
        """)