import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import warnings
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
import os
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
                        'date': date,
                        'hour': hour
                    })
    
    if len(records) == 0:
        raise ValueError("Tidak ada data yang valid. Periksa format file CSV Anda.")
    
    df_long = pd.DataFrame(records)
    return df_long


# ============================================================
# 2. FUNGSI EDA (EXPLORATORY DATA ANALYSIS)
# ============================================================

def perform_eda(df_long, result):
    """
    Melakukan Exploratory Data Analysis dan menghasilkan berbagai statistik
    """
    eda_results = {}
    
    # 1. Statistik dasar data harian
    eda_results['daily_stats'] = {
        'mean': result['daily_mean_flow'].mean(),
        'median': result['daily_mean_flow'].median(),
        'std': result['daily_mean_flow'].std(),
        'var': result['daily_mean_flow'].var(),
        'skewness': result['daily_mean_flow'].skew(),
        'kurtosis': result['daily_mean_flow'].kurt(),
        'cv': result['daily_mean_flow'].std() / result['daily_mean_flow'].mean() if result['daily_mean_flow'].mean() > 0 else 0,
        'q1': result['daily_mean_flow'].quantile(0.25),
        'q3': result['daily_mean_flow'].quantile(0.75),
        'iqr': result['daily_mean_flow'].quantile(0.75) - result['daily_mean_flow'].quantile(0.25)
    }
    
    # 2. Nilai maksimum dan minimum dengan waktu (dari data harian)
    max_flow = result.loc[result['daily_mean_flow'].idxmax()]
    min_flow = result.loc[result['daily_mean_flow'].idxmin()]
    
    eda_results['extreme_values'] = {
        'max': {
            'value': max_flow['daily_mean_flow'],
            'date': max_flow['date'],
            'year': max_flow['date'].year,
            'month': max_flow['date'].strftime('%B'),
            'month_num': max_flow['date'].month,
            'day': max_flow['date'].day,
            'bfi': max_flow['bfi_daily']
        },
        'min': {
            'value': min_flow['daily_mean_flow'],
            'date': min_flow['date'],
            'year': min_flow['date'].year,
            'month': min_flow['date'].strftime('%B'),
            'month_num': min_flow['date'].month,
            'day': min_flow['date'].day,
            'bfi': min_flow['bfi_daily']
        }
    }
    
    # 3. Analisis per jam (dari data original)
    hourly_avg = df_long.groupby('hour')['discharge'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
    eda_results['hourly_analysis'] = hourly_avg.to_dict('records')
    
    # 4. Analisis per bulan
    monthly_avg = result.groupby(result['date'].dt.month)['daily_mean_flow'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
    monthly_avg.columns = ['month', 'mean', 'median', 'std', 'min', 'max']
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: month_names[x-1])
    eda_results['monthly_analysis'] = monthly_avg.to_dict('records')
    
   # 5. Boxplot statistics per tahun
    yearly_box_stats = result.groupby('year')['daily_mean_flow'].agg([
    'min', 
    lambda x: x.quantile(0.25), 
    'median', 
    lambda x: x.quantile(0.75), 
    'max', 
    'mean', 
    'std'
    ]).reset_index()
    yearly_box_stats.columns = ['year', 'min', 'q1', 'median', 'q3', 'max', 'mean', 'std']
    
    # 6. Seasonal analysis (musiman)
    result['season'] = result['date'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    seasonal_avg = result.groupby('season')['daily_mean_flow'].agg(['mean', 'median', 'std', 'min', 'max', 'count']).reset_index()
    eda_results['seasonal_analysis'] = seasonal_avg.to_dict('records')
    
    # 7. Cumulative distribution (Flow Duration Curve)
    sorted_flows = np.sort(result['daily_mean_flow'].values)
    cumulative_prob = np.arange(1, len(sorted_flows)+1) / len(sorted_flows)
    eda_results['cumulative_distribution'] = {
        'flows': sorted_flows.tolist(),
        'probabilities': cumulative_prob.tolist()
    }
    
    # 8. Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    eda_results['percentiles'] = {f'p{p}': result['daily_mean_flow'].quantile(p/100) for p in percentiles}
    
    # 9. Trend analysis (perubahan tahun ke tahun)
    yearly_trend = result.groupby('year')['daily_mean_flow'].mean().reset_index()
    eda_results['yearly_trend'] = yearly_trend.to_dict('records')
    
    return eda_results


# ============================================================
# 3. FUNGSI BFI YANG DIPERBAIKI
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
# 4. FUNGSI EKSPORT KE EXCEL (DENGAN EDA)
# ============================================================

def export_to_excel(result, metadata, eda_results):
    """Export hasil ke Excel termasuk EDA"""
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
        
        # EDA: Daily Statistics
        daily_stats_df = pd.DataFrame([eda_results['daily_stats']])
        daily_stats_df.to_excel(writer, sheet_name='EDA_Daily_Stats', index=False)
        
        # EDA: Extreme Values
        extreme_df = pd.DataFrame({
            'Type': ['Maximum Flow', 'Minimum Flow'],
            'Value': [eda_results['extreme_values']['max']['value'], eda_results['extreme_values']['min']['value']],
            'Date': [eda_results['extreme_values']['max']['date'], eda_results['extreme_values']['min']['date']],
            'Year': [eda_results['extreme_values']['max']['year'], eda_results['extreme_values']['min']['year']],
            'Month': [eda_results['extreme_values']['max']['month'], eda_results['extreme_values']['min']['month']],
            'BFI_at_that_time': [eda_results['extreme_values']['max']['bfi'], eda_results['extreme_values']['min']['bfi']]
        })
        extreme_df.to_excel(writer, sheet_name='EDA_Extreme_Values', index=False)
        
        # EDA: Hourly Analysis
        hourly_df = pd.DataFrame(eda_results['hourly_analysis'])
        hourly_df.to_excel(writer, sheet_name='EDA_Hourly_Analysis', index=False)
        
        # EDA: Monthly Analysis
        monthly_df = pd.DataFrame(eda_results['monthly_analysis'])
        monthly_df.to_excel(writer, sheet_name='EDA_Monthly_Analysis', index=False)
        
        # EDA: Seasonal Analysis
        seasonal_df = pd.DataFrame(eda_results['seasonal_analysis'])
        seasonal_df.to_excel(writer, sheet_name='EDA_Seasonal_Analysis', index=False)
        
        # EDA: Percentiles
        percentiles_df = pd.DataFrame([eda_results['percentiles']])
        percentiles_df.to_excel(writer, sheet_name='EDA_Percentiles', index=False)
        
        # Metadata
        pd.DataFrame([metadata]).to_excel(writer, sheet_name='Metadata', index=False)
    
    output.seek(0)
    return output


# ============================================================
# 5. FUNGSI VISUALISASI EDA
# ============================================================

def create_eda_visualizations(result, df_long, eda_results):
    """Membuat berbagai grafik untuk EDA"""
    
    # 1. Boxplot per tahun
    fig_box = go.Figure()
    years = sorted(result['year'].unique())
    for year in years:
        year_data = result[result['year'] == year]['daily_mean_flow']
        fig_box.add_trace(go.Box(
            y=year_data,
            name=str(year),
            boxmean='sd',
            marker_color='#3B82F6'
        ))
    fig_box.update_layout(
        template="plotly_dark",
        title="Distribusi Debit per Tahun (Boxplot)",
        xaxis_title="Tahun",
        yaxis_title="Debit",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 2. Rata-rata per jam
    hourly_df = pd.DataFrame(eda_results['hourly_analysis'])
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(
        x=hourly_df['hour'],
        y=hourly_df['mean'],
        error_y=dict(type='data', array=hourly_df['std']),
        marker_color='#3B82F6',
        name='Rata-rata Debit'
    ))
    fig_hourly.update_layout(
        template="plotly_dark",
        title="Rata-rata Debit per Jam Pengukuran",
        xaxis_title="Jam",
        yaxis_title="Debit Rata-rata",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 3. Rata-rata per bulan
    monthly_df = pd.DataFrame(eda_results['monthly_analysis'])
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(
        x=monthly_df['month_name'],
        y=monthly_df['mean'],
        mode='lines+markers',
        line=dict(color='#F59E0B', width=3),
        marker=dict(size=10, color='#F59E0B'),
        name='Rata-rata Debit'
    ))
    fig_monthly.add_trace(go.Scatter(
        x=monthly_df['month_name'],
        y=monthly_df['median'],
        mode='lines+markers',
        line=dict(color='#10B981', width=2, dash='dash'),
        marker=dict(size=8, color='#10B981'),
        name='Median Debit'
    ))
    fig_monthly.update_layout(
        template="plotly_dark",
        title="Variasi Debit per Bulan",
        xaxis_title="Bulan",
        yaxis_title="Debit",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 4. Seasonal analysis
    seasonal_df = pd.DataFrame(eda_results['seasonal_analysis'])
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_df['season'] = pd.Categorical(seasonal_df['season'], categories=season_order, ordered=True)
    seasonal_df = seasonal_df.sort_values('season')
    
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Bar(
        x=seasonal_df['season'],
        y=seasonal_df['mean'],
        error_y=dict(type='data', array=seasonal_df['std']),
        marker_color='#8B5CF6',
        name='Rata-rata Debit'
    ))
    fig_seasonal.update_layout(
        template="plotly_dark",
        title="Analisis Musiman (Seasonal Analysis)",
        xaxis_title="Musim",
        yaxis_title="Debit Rata-rata",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 5. Flow Duration Curve
    flows = np.array(eda_results['cumulative_distribution']['flows'])
    probs = np.array(eda_results['cumulative_distribution']['probabilities'])
    
    fig_fdc = go.Figure()
    fig_fdc.add_trace(go.Scatter(
        x=probs * 100,
        y=flows,
        mode='lines',
        line=dict(color='#06B6D4', width=3),
        fill='tozeroy',
        fillcolor='rgba(6, 182, 212, 0.2)',
        name='Flow Duration Curve'
    ))
    fig_fdc.update_layout(
        template="plotly_dark",
        title="Flow Duration Curve (Kurva Durasi Aliran)",
        xaxis_title="Persentase Waktu Terlampaui (%)",
        yaxis_title="Debit",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 6. Histogram distribusi
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=result['daily_mean_flow'],
        nbinsx=30,
        marker_color='#3B82F6',
        opacity=0.7,
        name='Frekuensi'
    ))
    fig_hist.add_vline(
        x=result['daily_mean_flow'].mean(),
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Mean: {result['daily_mean_flow'].mean():.2f}"
    )
    fig_hist.add_vline(
        x=result['daily_mean_flow'].median(),
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"Median: {result['daily_mean_flow'].median():.2f}"
    )
    fig_hist.update_layout(
        template="plotly_dark",
        title="Distribusi Frekuensi Debit Harian",
        xaxis_title="Debit",
        yaxis_title="Frekuensi",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 7. Trend Analysis (Line chart)
    yearly_trend = pd.DataFrame(eda_results['yearly_trend'])
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=yearly_trend['year'],
        y=yearly_trend['daily_mean_flow'],
        mode='lines+markers',
        line=dict(color='#EF4444', width=3),
        marker=dict(size=10, color='#EF4444'),
        name='Rata-rata Debit Tahunan'
    ))
    fig_trend.update_layout(
        template="plotly_dark",
        title="Tren Debit Tahunan",
        xaxis_title="Tahun",
        yaxis_title="Rata-rata Debit",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 8. Scatter plot BFI vs Debit
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=result['daily_mean_flow'],
        y=result['bfi_daily'],
        mode='markers',
        marker=dict(size=5, color='#10B981', opacity=0.6),
        name='Data Point'
    ))
    fig_scatter.update_layout(
        template="plotly_dark",
        title="Hubungan Debit vs BFI",
        xaxis_title="Debit",
        yaxis_title="BFI",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return {
        'boxplot': fig_box,
        'hourly': fig_hourly,
        'monthly': fig_monthly,
        'seasonal': fig_seasonal,
        'fdc': fig_fdc,
        'histogram': fig_hist,
        'trend': fig_trend,
        'scatter': fig_scatter
    }


# ============================================================
# 6. FUNGSI EKSPORT KE PDF
# ============================================================

def create_pdf_report(result, metadata, eda_results, eda_figs):
    """Membuat laporan PDF lengkap"""
    
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "BFI_Report.pdf")
    
    # Buat PDF dengan ReportLab
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, 
                           topMargin=1*cm, bottomMargin=1*cm,
                           leftMargin=1.5*cm, rightMargin=1.5*cm)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#06B6D4'),
        alignment=TA_CENTER,
        spaceAfter=20
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#3B82F6'),
        spaceAfter=10,
        spaceBefore=15
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )
    
    story = []
    
    # Title
    story.append(Paragraph("HydroFlow BFI Analysis Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Summary Information
    story.append(Paragraph("1. Executive Summary", heading_style))
    
    # BFI Classification
    if metadata['bfi_overall'] >= 0.8:
        bfi_class = "Sangat Tinggi"
        bfi_color = "#06B6D4"
    elif metadata['bfi_overall'] >= 0.6:
        bfi_class = "Tinggi"
        bfi_color = "#10B981"
    elif metadata['bfi_overall'] >= 0.4:
        bfi_class = "Sedang"
        bfi_color = "#F59E0B"
    elif metadata['bfi_overall'] >= 0.2:
        bfi_class = "Rendah"
        bfi_color = "#EF4444"
    else:
        bfi_class = "Sangat Rendah"
        bfi_color = "#8B5CF6"
    
    summary_data = [
        ["Parameter", "Value"],
        ["Overall BFI", f"{metadata['bfi_overall']:.4f} ({bfi_class})"],
        ["Total Flow (Qt)", f"{metadata['total_flow']:.2f}"],
        ["Total Baseflow (Qb)", f"{metadata['total_baseflow']:.2f}"],
        ["Qb/Qt Ratio", f"{metadata['total_baseflow']/metadata['total_flow']:.4f}"],
        ["Analysis Period", metadata['date_range']],
        ["Total Years", str(metadata['total_years'])],
        ["Total Days", str(metadata['n_days'])],
        ["Missing Days", str(metadata['missing_days'])]
    ]
    
    summary_table = Table(summary_data, colWidths=[4*cm, 8*cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#3B82F6')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (1, -1), 6),
        ('TOPPADDING', (0, 0), (1, -1), 6),
        ('GRID', (0, 0), (1, -1), 1, colors.HexColor('#334155'))
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # Extreme Values
    story.append(Paragraph("2. Extreme Values Analysis", heading_style))
    
    max_val = eda_results['extreme_values']['max']
    min_val = eda_results['extreme_values']['min']
    
    extreme_data = [
        ["Type", "Value", "Date", "BFI at Event"],
        ["Maximum Flow", f"{max_val['value']:.2f}", max_val['date'].strftime('%d %B %Y'), f"{max_val['bfi']:.4f}"],
        ["Minimum Flow", f"{min_val['value']:.2f}", min_val['date'].strftime('%d %B %Y'), f"{min_val['bfi']:.4f}"]
    ]
    
    extreme_table = Table(extreme_data, colWidths=[3*cm, 3*cm, 5*cm, 4*cm])
    extreme_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (3, 0), colors.HexColor('#EF4444')),
        ('TEXTCOLOR', (0, 0), (3, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (3, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (3, -1), 9),
        ('GRID', (0, 0), (3, -1), 1, colors.HexColor('#334155'))
    ]))
    story.append(extreme_table)
    story.append(Spacer(1, 15))
    
    # Statistical Summary
    story.append(Paragraph("3. Statistical Summary", heading_style))
    
    stats_data = [
        ["Statistic", "Value"],
        ["Mean", f"{eda_results['daily_stats']['mean']:.4f}"],
        ["Median", f"{eda_results['daily_stats']['median']:.4f}"],
        ["Standard Deviation", f"{eda_results['daily_stats']['std']:.4f}"],
        ["Coefficient of Variation", f"{eda_results['daily_stats']['cv']:.4f}"],
        ["Skewness", f"{eda_results['daily_stats']['skewness']:.4f}"],
        ["Kurtosis", f"{eda_results['daily_stats']['kurtosis']:.4f}"],
        ["Q1 (25th percentile)", f"{eda_results['daily_stats']['q1']:.4f}"],
        ["Q3 (75th percentile)", f"{eda_results['daily_stats']['q3']:.4f}"],
        ["IQR", f"{eda_results['daily_stats']['iqr']:.4f}"]
    ]
    
    stats_table = Table(stats_data, colWidths=[5*cm, 7*cm])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#10B981')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (1, -1), 9),
        ('GRID', (0, 0), (1, -1), 1, colors.HexColor('#334155'))
    ]))
    story.append(stats_table)
    
    story.append(PageBreak())
    
    # Yearly BFI Analysis
    story.append(Paragraph("4. Yearly BFI Analysis", heading_style))
    
    yearly_df = pd.DataFrame(metadata['yearly_bfi'])
    yearly_display = yearly_df[['year', 'bfi_mean', 'bfi_median', 'bfi_overall']].head(20)
    
    yearly_data = [["Tahun", "Mean BFI", "Median BFI", "BFI Overall"]] + \
                  yearly_display.values.tolist()
    
    yearly_table = Table(yearly_data, colWidths=[2.5*cm, 3*cm, 3*cm, 3*cm])
    yearly_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (3, 0), colors.HexColor('#F59E0B')),
        ('TEXTCOLOR', (0, 0), (3, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (3, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (3, -1), 8),
        ('GRID', (0, 0), (3, -1), 0.5, colors.HexColor('#334155'))
    ]))
    story.append(yearly_table)
    
    story.append(Spacer(1, 20))
    
    # Monthly and Seasonal Summary
    story.append(Paragraph("5. Monthly & Seasonal Analysis", heading_style))
    
    monthly_df = pd.DataFrame(eda_results['monthly_analysis'])
    monthly_display = monthly_df[['month_name', 'mean', 'median']].head(12)
    
    monthly_data = [["Bulan", "Rata-rata", "Median"]] + monthly_display.values.tolist()
    
    monthly_table = Table(monthly_data, colWidths=[3*cm, 3.5*cm, 3.5*cm])
    monthly_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (2, 0), colors.HexColor('#8B5CF6')),
        ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (2, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (2, -1), 9),
        ('GRID', (0, 0), (2, -1), 0.5, colors.HexColor('#334155'))
    ]))
    story.append(monthly_table)
    
    story.append(Spacer(1, 20))
    
    # Hourly Analysis
    story.append(Paragraph("6. Hourly Analysis", heading_style))
    
    hourly_df = pd.DataFrame(eda_results['hourly_analysis'])
    
    hourly_data = [["Jam", "Rata-rata", "Median", "Min", "Max"]] + \
                  [[h['hour'], f"{h['mean']:.2f}", f"{h['median']:.2f}", f"{h['min']:.2f}", f"{h['max']:.2f}"] 
                   for h in eda_results['hourly_analysis']]
    
    hourly_table = Table(hourly_data, colWidths=[2*cm, 3*cm, 3*cm, 3*cm, 3*cm])
    hourly_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (4, 0), colors.HexColor('#06B6D4')),
        ('TEXTCOLOR', (0, 0), (4, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (4, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (4, -1), 8),
        ('GRID', (0, 0), (4, -1), 0.5, colors.HexColor('#334155'))
    ]))
    story.append(hourly_table)
    
    story.append(PageBreak())
    
    # Percentiles
    story.append(Paragraph("7. Flow Percentiles", heading_style))
    
    percentiles_data = [["Percentile", "Value"]] + \
                       [[p, f"{v:.4f}"] for p, v in eda_results['percentiles'].items()]
    
    percentiles_table = Table(percentiles_data, colWidths=[4*cm, 8*cm])
    percentiles_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#3B82F6')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (1, -1), 9),
        ('GRID', (0, 0), (1, -1), 0.5, colors.HexColor('#334155'))
    ]))
    story.append(percentiles_table)
    
    story.append(Spacer(1, 15))
    
    # Method and Parameters
    story.append(Paragraph("8. Method & Parameters", heading_style))
    
    method_data = [
        ["Parameter", "Value"],
        ["Method", "Institute of Hydrology (UK) Method"],
        ["Block Length", f"{metadata['block_len']} days"],
        ["Turning Point Factor", f"{metadata['tp_factor']}"],
        ["Data Source", f"{metadata['n_measurements']} measurements from {metadata['date_range']}"]
    ]
    
    method_table = Table(method_data, colWidths=[5*cm, 7*cm])
    method_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#10B981')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (1, -1), 9),
        ('GRID', (0, 0), (1, -1), 0.5, colors.HexColor('#334155'))
    ]))
    story.append(method_table)
    
    # Build PDF
    doc.build(story)
    
    # Read PDF file
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    
    # Cleanup
    os.remove(pdf_path)
    os.rmdir(temp_dir)
    
    return pdf_data


# ============================================================
# 7. KONFIGURASI HALAMAN & TEMA
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
    .extreme-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
        border-radius: 12px;
        padding: 15px;
        border-left: 4px solid #EF4444;
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
    .eda-container {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# 8. SIDEBAR PANEL
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
# 9. FUNGSI INFORMASI (TAB)
# ============================================================

def show_usage_guide():
    """Menampilkan panduan penggunaan aplikasi"""
    st.markdown("## 📖 Panduan Penggunaan Aplikasi")
    
    st.markdown("""
    ### Langkah-langkah Menggunakan Web App:
    
    **Step 1: Persiapan Data**
    Siapkan file CSV dengan format berikut:
    - Kolom **Year**: Tahun (contoh: 2023, 2024)
    - Kolom **Month**: Nama bulan (Jan, Feb, Mar, ..., Dec)
    - Kolom **Date**: Tanggal (1-31)
    - Kolom **06:00, 12:00, 18:00**: Nilai debit pada jam tersebut
    
    **Step 2: Upload File**
    Upload file CSV melalui panel Control Center di sebelah kiri.
    
    **Step 3: Atur Hyperparameters**
    Sesuaikan nilai Block Length dan Turning Point Factor sesuai kebutuhan.
    
    **Step 4: Jalankan Analisis**
    Klik tombol "Run Advanced Analysis" untuk memproses data.
    
    **Step 5: Eksplorasi Hasil**
    - Lihat periode analisis dan metrik utama
    - Eksplorasi grafik EDA (Boxplot, Flow Duration Curve, dll)
    - Download laporan Excel atau PDF
    """)
    
    # Contoh data
    with st.expander("📋 Contoh Format File CSV", expanded=False):
        sample_data = pd.DataFrame({
            'Year': [2023, 2023, 2023, 2023, 2023],
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
    """, unsafe_allow_html=True)


def show_hyperparameters():
    """Menampilkan penjelasan hyperparameters"""
    st.markdown("## ⚙️ Penjelasan Hyperparameters")
    
    st.markdown("""
    ### 1. Block Length (Days)
    
    **Definisi:** Panjang blok (dalam hari) yang digunakan untuk mencari nilai minimum aliran.
    
    **Pengaruh Parameter:**
    - **Nilai kecil (3-5 hari)**: Lebih sensitif, BFI cenderung lebih rendah
    - **Nilai besar (7-10 hari)**: Lebih halus, BFI cenderung lebih tinggi
    
    ### 2. Turning Point Factor
    
    **Definisi:** Faktor pengali untuk menentukan apakah suatu titik minimum merupakan *turning point*.
    
    **Pengaruh Parameter:**
    - **Nilai kecil (0.5-0.7)**: Lebih ketat, BFI lebih tinggi
    - **Nilai besar (0.8-1.0)**: Lebih longgar, BFI lebih rendah
    
    ### Tips Optimasi:
    
    1. Mulai dengan default (block_len=5, tp_factor=0.9)
    2. Jika BFI terlalu rendah -> tingkatkan block_len atau turunkan tp_factor
    3. Jika BFI terlalu tinggi -> turunkan block_len atau tingkatkan tp_factor
    """, unsafe_allow_html=True)


def show_formulas():
    """Menampilkan formula dan persamaan"""
    st.markdown("## 📐 Formula & Persamaan")
    
    st.markdown("""
    ### Formula Dasar BFI
    
    **BFI = Qb / Qt**
    
    Dimana:
    - Qb = Total Baseflow (aliran dasar)
    - Qt = Total Flow (total aliran sungai)
    
    ### Daily BFI
    
    **BFI_daily = baseflow_t / discharge_t**
    
    ### Interpretasi Hasil
    
    | Rentang BFI | Interpretasi |
    |-------------|--------------|
    | 0.8 - 1.0 | Aliran didominasi air tanah |
    | 0.6 - 0.8 | Kontribusi air tanah signifikan |
    | 0.4 - 0.6 | Keseimbangan air tanah dan limpasan |
    | 0.2 - 0.4 | Didominasi limpasan permukaan |
    | 0.0 - 0.2 | Aliran permukaan mendominasi |
    """, unsafe_allow_html=True)


# ============================================================
# 10. MAIN DASHBOARD DISPLAY
# ============================================================

st.title("📊 HydroFlow BFI Dashboard")
st.caption("Advanced Automated Baseflow Separation Framework | Multi-Year Data Support with EDA & PDF Export")

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
            
            # Perform EDA
            eda_results = perform_eda(df_long, result)
            
            # Create EDA visualizations
            eda_figs = create_eda_visualizations(result, df_long, eda_results)
            
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
            
            # ========== SECTION 2: EXTREME VALUES (MAKSIMUM & MINIMUM) ==========
            st.markdown("### 🌊 Extreme Values Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_val = eda_results['extreme_values']['max']
                st.markdown(
                    f"""
                    <div class='extreme-card'>
                    <h3 style='color:#EF4444; margin:0'>🔴 DEBIT MAKSIMUM</h3>
                    <h1 style='margin:10px 0; color:#EF4444'>{max_val['value']:.2f}</h1>
                    <p><strong>📅 Tanggal:</strong> {max_val['date'].strftime('%d %B %Y')}</p>
                    <p><strong>📆 Tahun:</strong> {max_val['year']}</p>
                    <p><strong>📊 Bulan:</strong> {max_val['month']}</p>
                    <p><strong>💧 BFI saat kejadian:</strong> {max_val['bfi']:.4f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                min_val = eda_results['extreme_values']['min']
                st.markdown(
                    f"""
                    <div class='extreme-card' style='border-left-color:#10B981'>
                    <h3 style='color:#10B981; margin:0'>🔵 DEBIT MINIMUM</h3>
                    <h1 style='margin:10px 0; color:#10B981'>{min_val['value']:.2f}</h1>
                    <p><strong>📅 Tanggal:</strong> {min_val['date'].strftime('%d %B %Y')}</p>
                    <p><strong>📆 Tahun:</strong> {min_val['year']}</p>
                    <p><strong>📊 Bulan:</strong> {min_val['month']}</p>
                    <p><strong>💧 BFI saat kejadian:</strong> {min_val['bfi']:.4f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            # ========== SECTION 3: METRIK UTAMA ==========
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
            
            # ========== SECTION 4: YEARLY BFI ANALYSIS ==========
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
            
            # ========== SECTION 5: YEARLY COMPARISON CHART ==========
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
            
            # ========== SECTION 6: EDA VISUALIZATIONS ==========
            st.subheader("📈 Exploratory Data Analysis (EDA)")
            
            # Pilih jenis visualisasi EDA
            eda_option = st.selectbox(
                "Pilih Visualisasi EDA:",
                ["Boxplot per Tahun", "Flow Duration Curve", "Distribusi Debit", 
                 "Variasi Bulanan", "Analisis Per Jam", "Analisis Musiman", "Tren Tahunan", "Scatter Plot Debit vs BFI"]
            )
            
            if eda_option == "Boxplot per Tahun":
                st.plotly_chart(eda_figs['boxplot'], use_container_width=True)
                st.caption("Boxplot menunjukkan distribusi, median, kuartil, dan outlier debit per tahun")
            elif eda_option == "Flow Duration Curve":
                st.plotly_chart(eda_figs['fdc'], use_container_width=True)
                st.caption("Flow Duration Curve - menunjukkan persentase waktu dimana debit tertentu terlampaui")
            elif eda_option == "Distribusi Debit":
                st.plotly_chart(eda_figs['histogram'], use_container_width=True)
                st.caption(f"Distribusi debit harian (Mean: {eda_results['daily_stats']['mean']:.2f}, Median: {eda_results['daily_stats']['median']:.2f})")
            elif eda_option == "Variasi Bulanan":
                st.plotly_chart(eda_figs['monthly'], use_container_width=True)
                st.caption("Variasi debit rata-rata dan median per bulan")
            elif eda_option == "Analisis Per Jam":
                st.plotly_chart(eda_figs['hourly'], use_container_width=True)
                st.caption("Perbandingan debit rata-rata pada jam pengukuran 06:00, 12:00, dan 18:00")
            elif eda_option == "Analisis Musiman":
                st.plotly_chart(eda_figs['seasonal'], use_container_width=True)
                st.caption("Analisis musiman: Winter (Des-Feb), Spring (Mar-Mei), Summer (Jun-Agu), Fall (Sep-Nov)")
            elif eda_option == "Tren Tahunan":
                st.plotly_chart(eda_figs['trend'], use_container_width=True)
                st.caption("Tren rata-rata debit dari tahun ke tahun")
            elif eda_option == "Scatter Plot Debit vs BFI":
                st.plotly_chart(eda_figs['scatter'], use_container_width=True)
                st.caption("Hubungan antara nilai debit dengan BFI (Baseflow Index)")
            
            st.markdown("---")
            
            # ========== SECTION 7: STATISTICAL SUMMARY ==========
            st.subheader("📊 Statistical Summary")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Mean Daily Flow", f"{eda_results['daily_stats']['mean']:.2f}")
            with stat_col2:
                st.metric("Median Daily Flow", f"{eda_results['daily_stats']['median']:.2f}")
            with stat_col3:
                st.metric("Std Deviation", f"{eda_results['daily_stats']['std']:.2f}")
            with stat_col4:
                st.metric("CV", f"{eda_results['daily_stats']['cv']:.3f}")
            
            st.markdown("---")
            
            # ========== SECTION 8: PERCENTILES ==========
            st.subheader("📊 Flow Percentiles")
            
            percentiles_df = pd.DataFrame([eda_results['percentiles']]).T
            percentiles_df.columns = ['Value']
            percentiles_df.index = [f"{i}%" for i in [1, 5, 10, 25, 50, 75, 90, 95, 99]]
            st.dataframe(percentiles_df, use_container_width=True)
            
            st.markdown("---")
            
            # ========== SECTION 9: HYDROGRAPH ==========
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
            
            # ========== SECTION 10: DISTRIBUSI BFI ==========
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
            
            # ========== SECTION 11: PREVIEW & DOWNLOAD ==========
            st.subheader("📋 Processed Data Ledger")
            st.dataframe(result.head(50), use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Download Excel
                excel_file = export_to_excel(result, metadata, eda_results)
                st.download_button(
                    label="📊 Download Excel Report (with EDA)",
                    data=excel_file,
                    file_name="BFI_Complete_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col2:
                # Download PDF
                with st.spinner("📄 Generating PDF report..."):
                    pdf_data = create_pdf_report(result, metadata, eda_results, eda_figs)
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_data,
                        file_name="BFI_Analysis_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            
            # Metadata expander
            with st.expander("ℹ️ Processing Metadata & Additional Statistics"):
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
                    st.write("**Statistical Moments:**")
                    st.write(f"- Mean: {eda_results['daily_stats']['mean']:.4f}")
                    st.write(f"- Median: {eda_results['daily_stats']['median']:.4f}")
                    st.write(f"- Std Dev: {eda_results['daily_stats']['std']:.4f}")
                    st.write(f"- Skewness: {eda_results['daily_stats']['skewness']:.4f}")
                    st.write(f"- Kurtosis: {eda_results['daily_stats']['kurtosis']:.4f}")
                    st.write(f"- CV: {eda_results['daily_stats']['cv']:.4f}")
                
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.info("Pastikan format CSV Anda sesuai: Kolom 'Year', 'Month', 'Date', '06:00', '12:00', '18:00'")
            
else:
    if not run_analysis and uploaded_file:
        st.info("👆 Klik 'Run Advanced Analysis' untuk memulai analisis")
    elif not uploaded_file:
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
        | 2023 | Jan   | 4    | 8.5   | 9.2   | 8.8   |
        
        **Keterangan:**
        - **Year**: Tahun (contoh: 2023, 2024) - WAJIB ADA
        - Month: Nama bulan (Jan, Feb, Mar, ..., Dec)
        - Date: Tanggal (1-31)
        - 06:00, 12:00, 18:00: Nilai debit pada jam tersebut
        
        **Fitur yang tersedia:**
        - ✅ Analisis BFI Multi-tahun
        - ✅ Nilai Maksimum & Minimum dengan waktu kejadian (tanggal)
        - ✅ Exploratory Data Analysis (8 jenis visualisasi)
        - ✅ Boxplot per tahun, Flow Duration Curve, Distribusi Debit
        - ✅ Analisis per jam, bulan, musiman, dan tren tahunan
        - ✅ Download laporan Excel (lengkap dengan sheet EDA)
        - ✅ Download laporan PDF
        """)
