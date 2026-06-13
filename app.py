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
    eda_results['yearly_box_stats'] = yearly_box_stats.to_dict('records')
    
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
# 3. FUNGSI BFI STANDARD (INSTITUTE OF HYDROLOGY UK)
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
    Proses lengkap dari data 3x/hari ke BFI menggunakan metode IH UK
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
# 4. FUNGSI ALGORITMA BFI ALTERNATIF (7 ALGORITMA)
# ============================================================

def calculate_bfi_one_parameter(daily_flow, k=0.925):
    """
    One-parameter Algorithm (Chapman & Maxwell, 1996)
    
    Parameters:
    -----------
    daily_flow : pandas Series
        Data debit harian
    k : float, default=0.925
        Filter parameter (recession constant)
    
    Returns:
    --------
    result : DataFrame
        Hasil perhitungan dengan kolom baseflow dan bfi
    bfi_overall : float
        Nilai BFI keseluruhan
    """
    x = daily_flow.values
    n = len(x)
    dates = daily_flow.index
    
    baseflow = np.zeros(n)
    baseflow[0] = x[0]
    
    for i in range(1, n):
        # b(i) = (k/(2-k)) * (b(i-1) + q(i) - q(i-1))
        baseflow[i] = (k / (2 - k)) * (baseflow[i-1] + x[i] - x[i-1])
        
        # Constraint: b(i) <= q(i)
        baseflow[i] = min(baseflow[i], x[i])
        # Constraint: b(i) >= 0
        baseflow[i] = max(baseflow[i], 0)
    
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
        'bfi_daily': bfi_daily,
        'algorithm': 'One-parameter (Chapman & Maxwell, 1996)'
    })
    
    return result, bfi_overall


def calculate_bfi_boughton(daily_flow, C=0.15, k=0.925):
    """
    Boughton Two-parameter Algorithm (Boughton, 1993; Chapman & Maxwell, 1996)
    
    Parameters:
    -----------
    daily_flow : pandas Series
        Data debit harian
    C : float, default=0.15
        Parameter that allows the shape of the separation to be altered
    k : float, default=0.925
        Recession constant
    
    Returns:
    --------
    result : DataFrame
        Hasil perhitungan dengan kolom baseflow dan bfi
    bfi_overall : float
        Nilai BFI keseluruhan
    """
    x = daily_flow.values
    n = len(x)
    dates = daily_flow.index
    
    baseflow = np.zeros(n)
    baseflow[0] = x[0]
    
    for i in range(1, n):
        # b(i) = (k/(1+C)) * b(i-1) + (C/(1+C)) * q(i)
        baseflow[i] = (k / (1 + C)) * baseflow[i-1] + (C / (1 + C)) * x[i]
        
        # Constraint: b(i) <= q(i)
        baseflow[i] = min(baseflow[i], x[i])
        # Constraint: b(i) >= 0
        baseflow[i] = max(baseflow[i], 0)
    
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
        'bfi_daily': bfi_daily,
        'algorithm': 'Boughton Two-parameter (Boughton, 1993)'
    })
    
    return result, bfi_overall


def calculate_bfi_ihacres(daily_flow, C=0.15, k=0.925, alpha=0.5):
    """
    IHACRES Three-parameter Algorithm (Jakeman & Hornbarger, 1993)
    
    Parameters:
    -----------
    daily_flow : pandas Series
        Data debit harian
    C : float, default=0.15
        Parameter for baseflow separation
    k : float, default=0.925
        Recession constant
    alpha : float, default=0.5
        Additional filter parameter
    
    Returns:
    --------
    result : DataFrame
        Hasil perhitungan dengan kolom baseflow dan bfi
    bfi_overall : float
        Nilai BFI keseluruhan
    """
    x = daily_flow.values
    n = len(x)
    dates = daily_flow.index
    
    baseflow = np.zeros(n)
    baseflow[0] = x[0]
    
    for i in range(1, n):
        # qb(i) = (k/(1+C)) * qb(i-1) + (C/(1+C)) * (q(i) + alpha * q(i-1))
        baseflow[i] = (k / (1 + C)) * baseflow[i-1] + (C / (1 + C)) * (x[i] + alpha * x[i-1])
        
        # Constraint: b(i) <= q(i)
        baseflow[i] = min(baseflow[i], x[i])
        # Constraint: b(i) >= 0
        baseflow[i] = max(baseflow[i], 0)
    
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
        'bfi_daily': bfi_daily,
        'algorithm': 'IHACRES Three-parameter (Jakeman & Hornbarger, 1993)'
    })
    
    return result, bfi_overall


def calculate_bfi_lyne_hollick(daily_flow, alpha=0.925, passes=3):
    """
    Lynie & Hollick Algorithm (Lyne & Hollick 1979; Nathan & McMahon, 1990)
    
    Parameters:
    -----------
    daily_flow : pandas Series
        Data debit harian
    alpha : float, default=0.925
        Filter parameter (0.925 recommended for daily stream data)
    passes : int, default=3
        Number of filter passes (recommended 3)
    
    Returns:
    --------
    result : DataFrame
        Hasil perhitungan dengan kolom baseflow dan bfi
    bfi_overall : float
        Nilai BFI keseluruhan
    """
    x = daily_flow.values.copy()
    n = len(x)
    dates = daily_flow.index
    
    # Quick flow filter
    qf = np.zeros(n)
    qf[0] = x[0]
    
    for _ in range(passes):
        for i in range(1, n):
            # qf(i) = alpha * qf(i-1) + (1-alpha)/2 * (q(i) - q(i-1))
            qf[i] = alpha * qf[i-1] + ((1 - alpha) / 2) * (x[i] - x[i-1])
            # Constraint: qf(i) >= 0
            qf[i] = max(qf[i], 0)
        
        # Reverse pass for the next iteration
        if _ < passes - 1:
            qf = qf[::-1]
            x = x[::-1]
    
    # Baseflow = original flow - quick flow
    baseflow = x - qf
    baseflow = np.maximum(baseflow, 0)
    
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
        'bfi_daily': bfi_daily,
        'algorithm': 'Lyne & Hollick (1979)'
    })
    
    return result, bfi_overall


def calculate_bfi_chapman(daily_flow, alpha=0.925):
    """
    Chapman Algorithm (Chapman, 1991; Mau & Winter, 1997)
    
    Parameters:
    -----------
    daily_flow : pandas Series
        Data debit harian
    alpha : float, default=0.925
        Filter parameter
    
    Returns:
    --------
    result : DataFrame
        Hasil perhitungan dengan kolom baseflow dan bfi
    bfi_overall : float
        Nilai BFI keseluruhan
    """
    x = daily_flow.values
    n = len(x)
    dates = daily_flow.index
    
    qf = np.zeros(n)
    qf[0] = x[0]
    
    for i in range(1, n):
        # qf(i) = (3*alpha-1)/(3-alpha) * qf(i-1) + (2/(3-alpha)) * (q(i) - alpha*q(i-1))
        denominator = 3 - alpha
        qf[i] = ((3 * alpha - 1) / denominator) * qf[i-1] + (2 / denominator) * (x[i] - alpha * x[i-1])
        qf[i] = max(qf[i], 0)
    
    # Baseflow = original flow - quick flow
    baseflow = x - qf
    baseflow = np.maximum(baseflow, 0)
    
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
        'bfi_daily': bfi_daily,
        'algorithm': 'Chapman (1991)'
    })
    
    return result, bfi_overall


def calculate_bfi_ewma(daily_flow, alpha=0.9):
    """
    EWMA Filter (Tularam & Ilahee, 2008)
    Exponential Smoothing Method of Base Flow Separation
    
    Parameters:
    -----------
    daily_flow : pandas Series
        Data debit harian
    alpha : float, default=0.9
        Smoothing parameter (0 < alpha < 1)
    
    Returns:
    --------
    result : DataFrame
        Hasil perhitungan dengan kolom baseflow dan bfi
    bfi_overall : float
        Nilai BFI keseluruhan
    """
    x = daily_flow.values
    n = len(x)
    dates = daily_flow.index
    
    baseflow = np.zeros(n)
    baseflow[0] = x[0]
    
    for i in range(1, n):
        # b(i) = alpha * q(i) + (1-alpha) * b(i-1)
        baseflow[i] = alpha * x[i] + (1 - alpha) * baseflow[i-1]
        
        # Constraint: b(i) <= q(i)
        baseflow[i] = min(baseflow[i], x[i])
    
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
        'bfi_daily': bfi_daily,
        'algorithm': 'EWMA Filter (Tularam & Ilahee, 2008)'
    })
    
    return result, bfi_overall


def calculate_bfi_comparison(df_long, block_len=5, tp_factor=0.9, algorithm_params=None):
    """
    Menjalankan semua algoritma BFI untuk perbandingan
    
    Parameters:
    -----------
    df_long : DataFrame
        Data long format dari load_discharge_data
    block_len : int
        Block length untuk metode IH UK
    tp_factor : float
        Turning point factor untuk metode IH UK
    algorithm_params : dict
        Parameter untuk masing-masing algoritma
    
    Returns:
    --------
    comparison_results : dict
        Hasil perbandingan semua algoritma
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
    
    # Set default parameters if not provided
    if algorithm_params is None:
        algorithm_params = {
            'one_param': {'k': 0.925},
            'boughton': {'C': 0.15, 'k': 0.925},
            'ihacres': {'C': 0.15, 'k': 0.925, 'alpha': 0.5},
            'lyne_hollick': {'alpha': 0.925, 'passes': 3},
            'chapman': {'alpha': 0.925},
            'ewma': {'alpha': 0.9}
        }
    
    comparison_results = {}
    
    # 1. Institute of Hydrology UK (existing method)
    result_ih, bfi_ih = calculate_bfi(daily_means, block_len, tp_factor)
    comparison_results['Institute of Hydrology (UK)'] = {
        'result': result_ih,
        'bfi': bfi_ih,
        'params': {'block_len': block_len, 'tp_factor': tp_factor}
    }
    
    # 2. One-parameter
    result_op, bfi_op = calculate_bfi_one_parameter(
        daily_means, 
        k=algorithm_params['one_param']['k']
    )
    comparison_results['One-parameter (Chapman & Maxwell, 1996)'] = {
        'result': result_op,
        'bfi': bfi_op,
        'params': algorithm_params['one_param']
    }
    
    # 3. Boughton Two-parameter
    result_bg, bfi_bg = calculate_bfi_boughton(
        daily_means,
        C=algorithm_params['boughton']['C'],
        k=algorithm_params['boughton']['k']
    )
    comparison_results['Boughton Two-parameter (Boughton, 1993)'] = {
        'result': result_bg,
        'bfi': bfi_bg,
        'params': algorithm_params['boughton']
    }
    
    # 4. IHACRES Three-parameter
    result_ihacres, bfi_ihacres = calculate_bfi_ihacres(
        daily_means,
        C=algorithm_params['ihacres']['C'],
        k=algorithm_params['ihacres']['k'],
        alpha=algorithm_params['ihacres']['alpha']
    )
    comparison_results['IHACRES Three-parameter (Jakeman & Hornbarger, 1993)'] = {
        'result': result_ihacres,
        'bfi': bfi_ihacres,
        'params': algorithm_params['ihacres']
    }
    
    # 5. Lyne & Hollick
    result_lh, bfi_lh = calculate_bfi_lyne_hollick(
        daily_means,
        alpha=algorithm_params['lyne_hollick']['alpha'],
        passes=algorithm_params['lyne_hollick']['passes']
    )
    comparison_results['Lyne & Hollick (1979)'] = {
        'result': result_lh,
        'bfi': bfi_lh,
        'params': algorithm_params['lyne_hollick']
    }
    
    # 6. Chapman
    result_ch, bfi_ch = calculate_bfi_chapman(
        daily_means,
        alpha=algorithm_params['chapman']['alpha']
    )
    comparison_results['Chapman (1991)'] = {
        'result': result_ch,
        'bfi': bfi_ch,
        'params': algorithm_params['chapman']
    }
    
    # 7. EWMA
    result_ewma, bfi_ewma = calculate_bfi_ewma(
        daily_means,
        alpha=algorithm_params['ewma']['alpha']
    )
    comparison_results['EWMA Filter (Tularam & Ilahee, 2008)'] = {
        'result': result_ewma,
        'bfi': bfi_ewma,
        'params': algorithm_params['ewma']
    }
    
    return comparison_results


# ============================================================
# 5. FUNGSI EKSPORT KE EXCEL (DENGAN EDA)
# ============================================================

def export_to_excel(result, metadata, eda_results, algorithm_info=None):
    """Export hasil ke Excel termasuk EDA dan informasi algoritma"""
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
        
        # Algorithm Info
        if algorithm_info:
            algo_df = pd.DataFrame([algorithm_info])
            algo_df.to_excel(writer, sheet_name='Algorithm_Info', index=False)
        
        # Metadata
        pd.DataFrame([metadata]).to_excel(writer, sheet_name='Metadata', index=False)
        
        # KRA Analysis
        if 'kra_results' in locals():
            # KRA per tahun
            kra_yearly_df = pd.DataFrame(kra_results['yearly_kra'])
            kra_yearly_df.to_excel(writer, sheet_name='KRA_Per_Tahun', index=False)
            
            # KRA Overall
            kra_overall_df = pd.DataFrame({
                'Parameter': ['KRA Keseluruhan', 'Kategori', 'Deskripsi'],
                'Value': [kra_results['overall_kra'], kra_results['kra_category'], kra_results['kra_desc']]
            })
            kra_overall_df.to_excel(writer, sheet_name='KRA_Overall', index=False)
            
        # KRA 10 Tahunan jika ada
        if 'decadal_kra' in kra_results:
            kra_decadal_df = pd.DataFrame(kra_results['decadal_kra'])
            kra_decadal_df.to_excel(writer, sheet_name='KRA_10_Tahunan', index=False)
    output.seek(0)
    return output


# ============================================================
# 6. FUNGSI VISUALISASI EDA
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
# 7. FUNGSI VISUALISASI PERBANDINGAN ALGORITMA
# ============================================================

def create_algorithm_comparison_chart(comparison_results):
    """Membuat chart perbandingan BFI dari berbagai algoritma"""
    
    algorithms = list(comparison_results.keys())
    bfi_values = [comparison_results[alg]['bfi'] for alg in algorithms]
    
    colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#06B6D4', '#F97316']
    
    fig = go.Figure()
    
    # Bar chart
    fig.add_trace(go.Bar(
        x=algorithms,
        y=bfi_values,
        marker_color=colors[:len(algorithms)],
        text=[f'{v:.4f}' for v in bfi_values],
        textposition='auto',
        name='BFI Value'
    ))
    
    # Add horizontal line at mean
    mean_bfi = np.mean(bfi_values)
    fig.add_hline(
        y=mean_bfi,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Mean BFI: {mean_bfi:.4f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        template="plotly_dark",
        title="Perbandingan BFI dari Berbagai Algoritma",
        xaxis_title="Algoritma",
        yaxis_title="Nilai BFI",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis={'tickangle': 45, 'tickfont': {'size': 10}}
    )
    
    fig.update_yaxes(range=[0, max(bfi_values) * 1.1])
    
    return fig


def create_hydrograph_comparison(comparison_results, days_limit=365):
    """Membuat hydrograph perbandingan untuk beberapa algoritma"""
    
    fig = make_subplots(
        rows=len(comparison_results), cols=1,
        subplot_titles=list(comparison_results.keys()),
        vertical_spacing=0.05,
        shared_xaxes=True
    )
    
    for idx, (alg_name, alg_data) in enumerate(comparison_results.items(), 1):
        result = alg_data['result']
        display_data = result.tail(days_limit)
        
        fig.add_trace(
            go.Scatter(
                x=display_data['date'],
                y=display_data['daily_mean_flow'],
                name='Qt' if idx == 1 else None,
                line=dict(color='#3B82F6', width=1),
                showlegend=(idx == 1)
            ),
            row=idx, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=display_data['date'],
                y=display_data['baseflow'],
                name='Qb' if idx == 1 else None,
                line=dict(color='#10B981', width=2),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.2)',
                showlegend=(idx == 1)
            ),
            row=idx, col=1
        )
        
        # Add BFI annotation
        fig.add_annotation(
            x=0.98, y=0.95,
            xref=f"x{idx}", yref=f"y{idx}",
            text=f"BFI = {alg_data['bfi']:.4f}",
            showarrow=False,
            font=dict(size=10, color="#F59E0B"),
            bgcolor='rgba(0,0,0,0.5)',
            row=idx, col=1
        )
    
    fig.update_layout(
        template="plotly_dark",
        height=350 * len(comparison_results),
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_text="Perbandingan Hydrograph Antar Algoritma"
    )
    
    fig.update_xaxes(title_text="Date", row=len(comparison_results), col=1)
    
    return fig


def create_algorithm_comparison_table(comparison_results):
    """Membuat tabel perbandingan semua algoritma"""
    
    table_data = []
    for alg_name, alg_data in comparison_results.items():
        result = alg_data['result']
        params = alg_data.get('params', {})
        
        # Format parameters string
        params_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        
        table_data.append({
            'Algoritma': alg_name,
            'BFI': f"{alg_data['bfi']:.4f}",
            'Mean BFI Harian': f"{result['bfi_daily'].mean():.4f}",
            'Median BFI': f"{result['bfi_daily'].median():.4f}",
            'Std BFI': f"{result['bfi_daily'].std():.4f}",
            'Parameter': params_str if params_str else '-'
        })
    
    return pd.DataFrame(table_data)


# ============================================================
# 8. FUNGSI EKSPORT KE PDF
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
    elif metadata['bfi_overall'] >= 0.6:
        bfi_class = "Tinggi"
    elif metadata['bfi_overall'] >= 0.4:
        bfi_class = "Sedang"
    elif metadata['bfi_overall'] >= 0.2:
        bfi_class = "Rendah"
    else:
        bfi_class = "Sangat Rendah"
    
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
    
    # KRA Section (if data available)
    if 'yearly_kra_df' in locals() or 'yearly_kra_df' in dir():
        story.append(PageBreak())
        story.append(Paragraph("9. Koefisien Regim Aliran (KRA)", heading_style))
        story.append(Paragraph("Berdasarkan Peraturan Menteri Kehutanan No. 61 Tahun 2014", normal_style))
        story.append(Spacer(1, 10))
        
    # KRA Overall
    kra_summary = [
        ["Parameter", "Nilai"],
        ["KRA Keseluruhan", f"{overall_kra:.2f}"],
        ["Kategori", kra_category],
        ["Deskripsi", kra_desc]
    ]
        
    kra_table = Table(kra_summary, colWidths=[5*cm, 8*cm])
    kra_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#8B5CF6')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (1, -1), 10),
         ('GRID', (0, 0), (1, -1), 1, colors.HexColor('#334155'))
    ]))
    story.append(kra_table)
    story.append(Spacer(1, 15))
        
    # Tabel KRA per Tahun
    story.append(Paragraph("KRA per Tahun", heading_style))
        
    yearly_display = yearly_kra_df[['Tahun', 'Qmax (Musim Hujan)', 'Qmin (Musim Kemarau)', 'KRA', 'Kategori']].head(15)
    yearly_data = [["Tahun", "Qmax", "Qmin", "KRA", "Kategori"]] + yearly_display.values.tolist()
        
    yearly_table = Table(yearly_data, colWidths=[2*cm, 2.5*cm, 2.5*cm, 2*cm, 3.5*cm])
    yearly_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (4, 0), colors.HexColor('#F59E0B')),
        ('TEXTCOLOR', (0, 0), (4, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (4, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (4, -1), 8),
        ('GRID', (0, 0), (4, -1), 0.5, colors.HexColor('#334155'))
    ]))
    story.append(yearly_table)
        
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
# 9. FUNGSI KOEFISIEN REGIM ALIRAN (KRA)
# ============================================================

def classify_kra(kra_value):
    """
    Mengklasifikasikan nilai Koefisien Regim Aliran (KRA) berdasarkan Permenhut No. 61 Tahun 2014
    
    Parameters:
    -----------
    kra_value : float
        Nilai Koefisien Regim Aliran
    
    Returns:
    --------
    category : str
        Kategori KRA
    description : str
        Deskripsi kategori
    color : str
        Warna untuk visualisasi
    """
    if kra_value <= 20:
        return "Sangat Rendah (SR)", "Sangat Rendah", "#10B981"
    elif kra_value <= 50:
        return "Rendah (R)", "Rendah", "#3B82F6"
    elif kra_value <= 80:
        return "Sedang (S)", "Sedang", "#F59E0B"
    elif kra_value <= 110:
        return "Tinggi (T)", "Tinggi", "#EF4444"
    else:
        return "Sangat Tinggi (ST)", "Sangat Tinggi", "#8B5CF6"


def calculate_kra_from_yearly_data(result):
    """
    Menghitung Koefisien Regim Aliran (KRA) per tahun berdasarkan data debit maksimum dan minimum tahunan
    
    Formula: KRA = Qmax / Qmin
    Qmax = debit maksimum pada musim penghujan
    Qmin = debit minimum pada musim kemarau
    
    Parameters:
    -----------
    result : DataFrame
        Data hasil BFI dengan kolom date, daily_mean_flow
    
    Returns:
    --------
    yearly_kra : DataFrame
        Data KRA per tahun
    overall_kra : float
        Nilai KRA keseluruhan (rata-rata 10 tahunan)
    kra_category : str
        Kategori KRA keseluruhan
    """
    
    # Definisikan musim di Indonesia (berdasarkan bulan)
    # Musim Hujan: November - Maret (bulan 11,12,1,2,3)
    # Musim Kemarau: April - Oktober (bulan 4,5,6,7,8,9,10)
    result['month'] = result['date'].dt.month
    result['season'] = result['month'].apply(lambda x: 'Rainy' if x in [11, 12, 1, 2, 3] else 'Dry')
    
    yearly_kra_data = []
    
    for year in result['year'].unique():
        year_data = result[result['year'] == year]
        
        # Cari Qmax pada musim hujan
        rainy_data = year_data[year_data['season'] == 'Rainy']
        if len(rainy_data) > 0:
            qmax = rainy_data['daily_mean_flow'].max()
            qmax_date = rainy_data.loc[rainy_data['daily_mean_flow'].idxmax(), 'date']
        else:
            qmax = year_data['daily_mean_flow'].max()
            qmax_date = year_data.loc[year_data['daily_mean_flow'].idxmax(), 'date']
        
        # Cari Qmin pada musim kemarau
        dry_data = year_data[year_data['season'] == 'Dry']
        if len(dry_data) > 0:
            qmin = dry_data['daily_mean_flow'].min()
            qmin_date = dry_data.loc[dry_data['daily_mean_flow'].idxmin(), 'date']
        else:
            qmin = year_data['daily_mean_flow'].min()
            qmin_date = year_data.loc[year_data['daily_mean_flow'].idxmin(), 'date']
        
        # Hitung KRA tahunan
        if qmin > 0:
            kra = qmax / qmin
        else:
            kra = np.nan
        
        # Klasifikasi KRA
        category, desc, color = classify_kra(kra)
        
        yearly_kra_data.append({
            'Tahun': year,
            'Qmax (Musim Hujan)': qmax,
            'Tanggal Qmax': qmax_date,
            'Qmin (Musim Kemarau)': qmin,
            'Tanggal Qmin': qmin_date,
            'KRA': kra,
            'Kategori': category,
            'Deskripsi': desc,
            'Warna': color
        })
    
    yearly_kra_df = pd.DataFrame(yearly_kra_data)
    
    # Hitung KRA 10 tahunan (rata-rata Qmax dan Qmin 10 tahunan)
    if len(yearly_kra_df) >= 1:
        # Metode 1: Rata-rata KRA per tahun
        overall_kra_avg = yearly_kra_df['KRA'].mean()
        
        # Metode 2: KRA dari rata-rata Qmax dan Qmin (sesuai Permenhut)
        avg_qmax = yearly_kra_df['Qmax (Musim Hujan)'].mean()
        avg_qmin = yearly_kra_df['Qmin (Musim Kemarau)'].mean()
        overall_kra_ratio = avg_qmax / avg_qmin if avg_qmin > 0 else np.nan
        
        # Gunakan metode yang lebih konservatif
        overall_kra = overall_kra_ratio if not np.isnan(overall_kra_ratio) else overall_kra_avg
    else:
        overall_kra = np.nan
    
    # Klasifikasi KRA keseluruhan
    kra_category, kra_desc, kra_color = classify_kra(overall_kra)
    
    return yearly_kra_df, overall_kra, kra_category, kra_desc, kra_color


def calculate_kra_10year_moving_average(result, window=10):
    """
    Menghitung KRA dengan interval waktu 10 tahunan (moving average)
    
    Parameters:
    -----------
    result : DataFrame
        Data hasil BFI dengan kolom date, daily_mean_flow
    window : int
        Jangka waktu dalam tahun (default 10)
    
    Returns:
    --------
    decadal_kra : DataFrame
        Data KRA per dekade
    """
    
    decadal_data = []
    years = sorted(result['year'].unique())
    
    for i in range(len(years) - window + 1):
        start_year = years[i]
        end_year = years[i + window - 1]
        
        period_data = result[(result['year'] >= start_year) & (result['year'] <= end_year)]
        
        # Definisikan musim
        period_data['month'] = period_data['date'].dt.month
        period_data['season'] = period_data['month'].apply(lambda x: 'Rainy' if x in [11, 12, 1, 2, 3] else 'Dry')
        
        # Cari Qmax pada musim hujan dalam periode
        rainy_data = period_data[period_data['season'] == 'Rainy']
        if len(rainy_data) > 0:
            qmax = rainy_data['daily_mean_flow'].max()
            qmax_date = rainy_data.loc[rainy_data['daily_mean_flow'].idxmax(), 'date']
        else:
            qmax = period_data['daily_mean_flow'].max()
            qmax_date = period_data.loc[period_data['daily_mean_flow'].idxmax(), 'date']
        
        # Cari Qmin pada musim kemarau dalam periode
        dry_data = period_data[period_data['season'] == 'Dry']
        if len(dry_data) > 0:
            qmin = dry_data['daily_mean_flow'].min()
            qmin_date = dry_data.loc[dry_data['daily_mean_flow'].idxmin(), 'date']
        else:
            qmin = period_data['daily_mean_flow'].min()
            qmin_date = period_data.loc[period_data['daily_mean_flow'].idxmin(), 'date']
        
        if qmin > 0:
            kra = qmax / qmin
        else:
            kra = np.nan
        
        category, desc, color = classify_kra(kra)
        
        decadal_data.append({
            'Periode': f"{start_year}-{end_year}",
            'Start Year': start_year,
            'End Year': end_year,
            'Qmax (Musim Hujan)': qmax,
            'Tanggal Qmax': qmax_date,
            'Qmin (Musim Kemarau)': qmin,
            'Tanggal Qmin': qmin_date,
            'KRA': kra,
            'Kategori': category
        })
    
    return pd.DataFrame(decadal_data)


def create_kra_visualizations(yearly_kra_df, overall_kra, kra_category, kra_color):
    """Membuat visualisasi untuk Koefisien Regim Aliran"""
    
    # 1. Bar chart KRA per tahun
    fig_kra_bar = go.Figure()
    
    fig_kra_bar.add_trace(go.Bar(
        x=yearly_kra_df['Tahun'],
        y=yearly_kra_df['KRA'],
        marker_color=yearly_kra_df['Warna'],
        text=yearly_kra_df['KRA'].round(2),
        textposition='auto',
        hovertemplate='<b>Tahun %{x}</b><br>KRA: %{y:.2f}<br>Kategori: %{customdata}<extra></extra>',
        customdata=yearly_kra_df['Kategori']
    ))
    
    # Tambahkan garis nilai overall KRA
    fig_kra_bar.add_hline(
        y=overall_kra,
        line_dash="dash",
        line_color=kra_color,
        annotation_text=f"Overall KRA: {overall_kra:.2f} ({kra_category})",
        annotation_position="top right"
    )
    
    fig_kra_bar.update_layout(
        template="plotly_dark",
        title="Koefisien Regim Aliran (KRA) per Tahun",
        xaxis_title="Tahun",
        yaxis_title="Nilai KRA",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 2. Line chart Qmax dan Qmin per tahun
    fig_q_comparison = go.Figure()
    
    fig_q_comparison.add_trace(go.Scatter(
        x=yearly_kra_df['Tahun'],
        y=yearly_kra_df['Qmax (Musim Hujan)'],
        mode='lines+markers',
        name='Qmax (Musim Hujan)',
        line=dict(color='#EF4444', width=2),
        marker=dict(size=8, color='#EF4444')
    ))
    
    fig_q_comparison.add_trace(go.Scatter(
        x=yearly_kra_df['Tahun'],
        y=yearly_kra_df['Qmin (Musim Kemarau)'],
        mode='lines+markers',
        name='Qmin (Musim Kemarau)',
        line=dict(color='#3B82F6', width=2),
        marker=dict(size=8, color='#3B82F6')
    ))
    
    fig_q_comparison.update_layout(
        template="plotly_dark",
        title="Perbandingan Qmax (Musim Hujan) vs Qmin (Musim Kemarau)",
        xaxis_title="Tahun",
        yaxis_title="Debit (m³/detik)",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01)
    )
    
    return fig_kra_bar, fig_q_comparison


def add_kra_to_pdf_report(story, styles, heading_style, normal_style, yearly_kra_df, overall_kra, kra_category, kra_desc, decadal_kra_df=None):
    """Menambahkan section KRA ke laporan PDF"""
    
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    
    story.append(PageBreak())
    story.append(Paragraph("9. Koefisien Regim Aliran (KRA)", heading_style))
    story.append(Paragraph("Berdasarkan Peraturan Menteri Kehutanan No. 61 Tahun 2014", normal_style))
    story.append(Spacer(1, 10))
    
    # KRA Overall
    kra_summary = [
        ["Parameter", "Nilai"],
        ["KRA Keseluruhan", f"{overall_kra:.2f}"],
        ["Kategori", kra_category],
        ["Deskripsi", kra_desc],
        ["Interpretasi", f"Koefisien Regim Aliran {kra_desc.lower()} mengindikasikan fluktuasi debit antara musim hujan dan kemarau"]
    ]
    
    kra_table = Table(kra_summary, colWidths=[5*cm, 8*cm])
    kra_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#8B5CF6')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, -1), 10),
        ('GRID', (0, 0), (1, -1), 1, colors.HexColor('#334155'))
    ]))
    story.append(kra_table)
    story.append(Spacer(1, 15))
    
    # Tabel KRA per Tahun
    story.append(Paragraph("KRA per Tahun", heading_style))
    
    yearly_display = yearly_kra_df[['Tahun', 'Qmax (Musim Hujan)', 'Qmin (Musim Kemarau)', 'KRA', 'Kategori']].head(20)
    yearly_data = [["Tahun", "Qmax", "Qmin", "KRA", "Kategori"]] + yearly_display.values.tolist()
    
    yearly_table = Table(yearly_data, colWidths=[2*cm, 2.5*cm, 2.5*cm, 2*cm, 3.5*cm])
    yearly_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (4, 0), colors.HexColor('#F59E0B')),
        ('TEXTCOLOR', (0, 0), (4, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (4, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (4, -1), 8),
        ('GRID', (0, 0), (4, -1), 0.5, colors.HexColor('#334155'))
    ]))
    story.append(yearly_table)
    
    # KRA 10 Tahunan jika ada
    if decadal_kra_df is not None and len(decadal_kra_df) > 0:
        story.append(Spacer(1, 15))
        story.append(Paragraph("KRA Interval 10 Tahunan", heading_style))
        
        decadal_display = decadal_kra_df[['Periode', 'Qmax (Musim Hujan)', 'Qmin (Musim Kemarau)', 'KRA', 'Kategori']]
        decadal_data = [["Periode", "Qmax", "Qmin", "KRA", "Kategori"]] + decadal_display.values.tolist()
        
        decadal_table = Table(decadal_data, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2*cm, 3.5*cm])
        decadal_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (4, 0), colors.HexColor('#06B6D4')),
            ('TEXTCOLOR', (0, 0), (4, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (4, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (4, -1), 8),
            ('GRID', (0, 0), (4, -1), 0.5, colors.HexColor('#334155'))
        ]))
        story.append(decadal_table)
        
        
# ============================================================
# 9. KONFIGURASI HALAMAN & TEMA
# ============================================================

st.set_page_config(
    page_title="HydroFlow BFI Analytics - Multi Algorithm",
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
# 10. SIDEBAR PANEL
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
    
    st.markdown("### 🔬 Algorithm Selection")
    
    # Pilihan algoritma
    algorithm_choice = st.selectbox(
        "Pilih Algoritma Baseflow Separation:",
        [
            "Institute of Hydrology (UK) - Default",
            "One-parameter (Chapman & Maxwell, 1996)",
            "Boughton Two-parameter (Boughton, 1993)",
            "IHACRES Three-parameter (Jakeman & Hornbarger, 1993)",
            "Lyne & Hollick (1979)",
            "Chapman (1991)",
            "EWMA Filter (Tularam & Ilahee, 2008)",
            "COMPARE ALL ALGORITHMS"
        ],
        help="Pilih algoritma pemisahan aliran dasar yang akan digunakan"
    )
    
    st.markdown("### ⚙️ Hyperparameters")
    
    # Parameter untuk IH UK (default)
    block_len = st.slider(
        "Block Length (Days)",
        min_value=3,
        max_value=10,
        value=5,
        help="Panjang blok untuk mencari nilai minimum (khusus metode IH UK)"
    )
    
    tp_factor = st.slider(
        "Turning Point Factor",
        min_value=0.5,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Faktor untuk menentukan turning points (khusus metode IH UK)"
    )
    
    st.markdown("### 🎛️ Algorithm Parameters")
    
    # Default parameters
    algorithm_params = {
        'one_param': {'k': 0.925},
        'boughton': {'C': 0.15, 'k': 0.925},
        'ihacres': {'C': 0.15, 'k': 0.925, 'alpha': 0.5},
        'lyne_hollick': {'alpha': 0.925, 'passes': 3},
        'chapman': {'alpha': 0.925},
        'ewma': {'alpha': 0.9}
    }
    
    # Parameter sliders berdasarkan algoritma yang dipilih
    if "One-parameter" in algorithm_choice:
        algorithm_params['one_param']['k'] = st.slider(
            "Parameter k (recession constant)",
            min_value=0.5, max_value=0.99, value=0.925, step=0.005,
            help="Filter parameter - recession constant"
        )
    elif "Boughton" in algorithm_choice:
        algorithm_params['boughton']['C'] = st.slider(
            "Parameter C",
            min_value=0.05, max_value=0.5, value=0.15, step=0.01,
            help="Parameter yang mengontrol bentuk pemisahan"
        )
        algorithm_params['boughton']['k'] = st.slider(
            "Parameter k (recession constant)",
            min_value=0.5, max_value=0.99, value=0.925, step=0.005
        )
    elif "IHACRES" in algorithm_choice:
        algorithm_params['ihacres']['C'] = st.slider(
            "Parameter C",
            min_value=0.05, max_value=0.5, value=0.15, step=0.01
        )
        algorithm_params['ihacres']['k'] = st.slider(
            "Parameter k (recession constant)",
            min_value=0.5, max_value=0.99, value=0.925, step=0.005
        )
        algorithm_params['ihacres']['alpha'] = st.slider(
            "Parameter alpha",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05
        )
    elif "Lyne" in algorithm_choice:
        algorithm_params['lyne_hollick']['alpha'] = st.slider(
            "Parameter alpha",
            min_value=0.5, max_value=0.99, value=0.925, step=0.005,
            help="Recommended value for daily stream data is 0.925"
        )
        algorithm_params['lyne_hollick']['passes'] = st.slider(
            "Number of passes",
            min_value=1, max_value=5, value=3, step=1,
            help="Recommended to be applied in three passes"
        )
    elif "Chapman" in algorithm_choice:
        algorithm_params['chapman']['alpha'] = st.slider(
            "Parameter alpha",
            min_value=0.5, max_value=0.99, value=0.925, step=0.005
        )
    elif "EWMA" in algorithm_choice:
        algorithm_params['ewma']['alpha'] = st.slider(
            "Smoothing parameter alpha",
            min_value=0.5, max_value=0.99, value=0.9, step=0.01,
            help="Exponential smoothing weight (0 < alpha < 1)"
        )
    
    st.markdown("---")
    run_analysis = st.button("🚀 Run Advanced Analysis", use_container_width=True)
    
    st.markdown("---")
    
    # Tab informasi di sidebar
    st.markdown("### 📚 Information Menu")
    info_tab = st.radio(
        "Pilih Informasi:",
        ["📖 Cara Penggunaan", "🔬 Teori & Metode", "⚙️ Hyperparameters", "📐 Formula & Persamaan", "🔄 Perbandingan Algoritma"]
    )


# ============================================================
# 11. FUNGSI INFORMASI (TAB)
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
    
    **Step 3: Pilih Algoritma**
    Pilih salah satu dari 7 algoritma baseflow separation yang tersedia:
    - Institute of Hydrology (UK) - Default
    - One-parameter (Chapman & Maxwell, 1996)
    - Boughton Two-parameter (Boughton, 1993)
    - IHACRES Three-parameter (Jakeman & Hornbarger, 1993)
    - Lyne & Hollick (1979)
    - Chapman (1991)
    - EWMA Filter (Tularam & Ilahee, 2008)
    - COMPARE ALL ALGORITHMS
    
    **Step 4: Atur Parameter**
    Sesuaikan parameter algoritma sesuai kebutuhan.
    
    **Step 5: Jalankan Analisis**
    Klik tombol "Run Advanced Analysis" untuk memproses data.
    
    **Step 6: Eksplorasi Hasil**
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
    
    ### Algoritma yang Tersedia:
    
    1. **Institute of Hydrology (UK) Method** - Metode standar menggunakan block minimum dan turning points
    2. **One-parameter (Chapman & Maxwell, 1996)** - Metode sederhana dengan satu parameter k
    3. **Boughton Two-parameter (Boughton, 1993)** - Dua parameter C dan k untuk fleksibilitas lebih
    4. **IHACRES Three-parameter (Jakeman & Hornbarger, 1993)** - Tiga parameter untuk akurasi tinggi
    5. **Lyne & Hollick (1979)** - Filter dengan tiga pass, parameter alpha=0.925
    6. **Chapman (1991)** - Perbaikan dari algoritma Lyne & Hollick
    7. **EWMA Filter (Tularam & Ilahee, 2008)** - Exponential smoothing method
    """, unsafe_allow_html=True)


def show_hyperparameters():
    """Menampilkan penjelasan hyperparameters"""
    st.markdown("## ⚙️ Penjelasan Hyperparameters & Parameter Algoritma")
    
    st.markdown("""
    ### Parameter Umum:
    
    **Block Length (Days)** - untuk metode IH UK:
    - Panjang blok untuk mencari nilai minimum aliran
    - Nilai kecil (3-5 hari): Lebih sensitif, BFI cenderung lebih rendah
    - Nilai besar (7-10 hari): Lebih halus, BFI cenderung lebih tinggi
    
    **Turning Point Factor** - untuk metode IH UK:
    - Faktor pengali untuk menentukan turning points
    - Nilai kecil (0.5-0.7): Lebih ketat, BFI lebih tinggi
    - Nilai besar (0.8-1.0): Lebih longgar, BFI lebih rendah
    
    ### Parameter Algoritma Lainnya:
    
    **Parameter k (recession constant)**:
    - Menunjukkan laju resesi aliran
    - Nilai umum: 0.85-0.95 untuk data harian
    
    **Parameter C**:
    - Mengontrol bentuk pemisahan baseflow
    - Nilai umum: 0.1-0.3
    
    **Parameter alpha**:
    - Smoothing parameter untuk filter
    - Lyne & Hollick: 0.925 (rekomendasi)
    - EWMA: 0.9 (default)
    
    ### Tips Optimasi:
    
    1. Mulai dengan nilai default masing-masing algoritma
    2. Bandingkan hasil antar algoritma menggunakan mode "COMPARE ALL ALGORITHMS"
    3. Validasi dengan data tracer jika tersedia
    4. Pilih algoritma yang paling sesuai dengan karakteristik DAS Anda
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
    
    ### Formula Setiap Algoritma:
    
    **1. Institute of Hydrology (UK) Method:**
    - Block minimum method + interpolasi linear antar turning points
    
    **2. One-parameter (Chapman & Maxwell, 1996):**
    - b(i) = (k/(2-k)) * (b(i-1) + q(i) - q(i-1))
    - Constraint: b(i) <= q(i)
    
    **3. Boughton Two-parameter (Boughton, 1993):**
    - b(i) = (k/(1+C)) * b(i-1) + (C/(1+C)) * q(i)
    
    **4. IHACRES Three-parameter (Jakeman & Hornbarger, 1993):**
    - b(i) = (k/(1+C)) * b(i-1) + (C/(1+C)) * (q(i) + alpha * q(i-1))
    
    **5. Lyne & Hollick (1979):**
    - qf(i) = alpha * qf(i-1) + ((1-alpha)/2) * (q(i) - q(i-1))
    - Baseflow = q - qf
    
    **6. Chapman (1991):**
    - qf(i) = (3*alpha-1)/(3-alpha) * qf(i-1) + (2/(3-alpha)) * (q(i) - alpha*q(i-1))
    
    **7. EWMA Filter (Tularam & Ilahee, 2008):**
    - b(i) = alpha * q(i) + (1-alpha) * b(i-1)
    """, unsafe_allow_html=True)


def show_algorithm_comparison_info():
    """Menampilkan informasi perbandingan algoritma"""
    st.markdown("## 🔄 Perbandingan Algoritma")
    
    st.markdown("""
    ### Karakteristik Setiap Algoritma:
    
    | Algoritma | Jumlah Parameter | Kompleksitas | Akurasi | Kecepatan |
    |-----------|-----------------|--------------|---------|-----------|
    | IH UK | 2 (block_len, tp_factor) | Medium | Tinggi | Cepat |
    | One-parameter | 1 (k) | Rendah | Sedang | Sangat Cepat |
    | Boughton | 2 (C, k) | Rendah | Tinggi | Cepat |
    | IHACRES | 3 (C, k, alpha) | Medium | Sangat Tinggi | Cepat |
    | Lyne & Hollick | 1 (alpha, passes) | Medium | Tinggi | Sedang |
    | Chapman | 1 (alpha) | Medium | Tinggi | Cepat |
    | EWMA | 1 (alpha) | Rendah | Sedang | Sangat Cepat |
    
    ### Rekomendasi Penggunaan:
    
    - **Data dengan variasi rendah**: One-parameter atau EWMA cukup
    - **Data dengan variasi sedang**: Boughton atau IH UK
    - **Data dengan variasi tinggi**: IHACRES atau Lyne & Hollick
    - **Penelitian ilmiah**: IHACRES atau Lyne & Hollick
    - **Analisis cepat**: EWMA atau One-parameter
    - **Validasi dengan tracer**: Boughton atau IHACRES (dapat dikalibrasi)
    
    ### Interpretasi Hasil Perbandingan:
    
    Jika BFI antar algoritma:
    - **Sangat dekat (< 0.05)**: Algoritma konsisten, hasil reliable
    - **Cukup dekat (0.05-0.1)**: Ada variasi, perlu validasi tambahan
    - **Jauh (> 0.1)**: Pilih algoritma berdasarkan karakteristik DAS
    """, unsafe_allow_html=True)


# ============================================================
# 12. MAIN DASHBOARD DISPLAY
# ============================================================

st.title("📊 HydroFlow BFI Dashboard")
st.caption("Advanced Automated Baseflow Separation Framework | 7 Algorithms | EDA | Multi-Year Analysis")

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
elif info_tab == "🔄 Perbandingan Algoritma":
    show_algorithm_comparison_info()
    st.markdown("---")

# Main analysis section
if uploaded_file and run_analysis:
    with st.spinner("🔄 Processing hydrological time-series data..."):
        try:
            # Load data
            df_long = load_discharge_data(uploaded_file)
            
            # Pilih algoritma berdasarkan pilihan user
            if algorithm_choice == "Institute of Hydrology (UK) - Default":
                result, bfi_overall, metadata = process_bfi(df_long, block_len, tp_factor)
                current_algorithm = "Institute of Hydrology (UK)"
                algorithm_info = {
                    'Algorithm': current_algorithm,
                    'Block Length': block_len,
                    'Turning Point Factor': tp_factor
                }
                
            elif algorithm_choice == "One-parameter (Chapman & Maxwell, 1996)":
                # Konversi ke daily mean
                df_long['date_only'] = df_long['datetime'].dt.date
                daily_means = df_long.groupby('date_only')['discharge'].mean()
                daily_means.index = pd.to_datetime(daily_means.index)
                full_index = pd.date_range(start=daily_means.index.min(), end=daily_means.index.max(), freq='D')
                daily_means = daily_means.reindex(full_index)
                if daily_means.isna().sum() > 0:
                    daily_means = daily_means.interpolate(method='linear', limit=3).ffill().bfill()
                
                result, bfi_overall = calculate_bfi_one_parameter(daily_means, **algorithm_params['one_param'])
                current_algorithm = "One-parameter (Chapman & Maxwell, 1996)"
                algorithm_info = {
                    'Algorithm': current_algorithm,
                    'Parameter k': algorithm_params['one_param']['k']
                }
                metadata = {
                    'n_days': len(daily_means), 'n_measurements': len(df_long),
                    'date_range': f"{daily_means.index.min().strftime('%Y-%m-%d')} to {daily_means.index.max().strftime('%Y-%m-%d')}",
                    'start_year': daily_means.index.min().year, 'end_year': daily_means.index.max().year,
                    'total_years': len(daily_means.index.year.unique()), 'years': sorted(daily_means.index.year.unique()),
                    'block_len': block_len, 'tp_factor': tp_factor, 'bfi_overall': bfi_overall,
                    'missing_days': daily_means.isna().sum(), 'total_flow': result['daily_mean_flow'].sum(),
                    'total_baseflow': result['baseflow'].sum()
                }
                # Add year and month columns for consistency
                result['year'] = result['date'].dt.year
                result['month'] = result['date'].dt.month
                
            elif algorithm_choice == "Boughton Two-parameter (Boughton, 1993)":
                df_long['date_only'] = df_long['datetime'].dt.date
                daily_means = df_long.groupby('date_only')['discharge'].mean()
                daily_means.index = pd.to_datetime(daily_means.index)
                full_index = pd.date_range(start=daily_means.index.min(), end=daily_means.index.max(), freq='D')
                daily_means = daily_means.reindex(full_index)
                if daily_means.isna().sum() > 0:
                    daily_means = daily_means.interpolate(method='linear', limit=3).ffill().bfill()
                
                result, bfi_overall = calculate_bfi_boughton(daily_means, **algorithm_params['boughton'])
                current_algorithm = "Boughton Two-parameter (Boughton, 1993)"
                algorithm_info = {
                    'Algorithm': current_algorithm,
                    'Parameter C': algorithm_params['boughton']['C'],
                    'Parameter k': algorithm_params['boughton']['k']
                }
                metadata = {
                    'n_days': len(daily_means), 'n_measurements': len(df_long),
                    'date_range': f"{daily_means.index.min().strftime('%Y-%m-%d')} to {daily_means.index.max().strftime('%Y-%m-%d')}",
                    'start_year': daily_means.index.min().year, 'end_year': daily_means.index.max().year,
                    'total_years': len(daily_means.index.year.unique()), 'years': sorted(daily_means.index.year.unique()),
                    'block_len': block_len, 'tp_factor': tp_factor, 'bfi_overall': bfi_overall,
                    'missing_days': daily_means.isna().sum(), 'total_flow': result['daily_mean_flow'].sum(),
                    'total_baseflow': result['baseflow'].sum()
                }
                result['year'] = result['date'].dt.year
                result['month'] = result['date'].dt.month
                
            elif algorithm_choice == "IHACRES Three-parameter (Jakeman & Hornbarger, 1993)":
                df_long['date_only'] = df_long['datetime'].dt.date
                daily_means = df_long.groupby('date_only')['discharge'].mean()
                daily_means.index = pd.to_datetime(daily_means.index)
                full_index = pd.date_range(start=daily_means.index.min(), end=daily_means.index.max(), freq='D')
                daily_means = daily_means.reindex(full_index)
                if daily_means.isna().sum() > 0:
                    daily_means = daily_means.interpolate(method='linear', limit=3).ffill().bfill()
                
                result, bfi_overall = calculate_bfi_ihacres(daily_means, **algorithm_params['ihacres'])
                current_algorithm = "IHACRES Three-parameter (Jakeman & Hornbarger, 1993)"
                algorithm_info = {
                    'Algorithm': current_algorithm,
                    'Parameter C': algorithm_params['ihacres']['C'],
                    'Parameter k': algorithm_params['ihacres']['k'],
                    'Parameter alpha': algorithm_params['ihacres']['alpha']
                }
                metadata = {
                    'n_days': len(daily_means), 'n_measurements': len(df_long),
                    'date_range': f"{daily_means.index.min().strftime('%Y-%m-%d')} to {daily_means.index.max().strftime('%Y-%m-%d')}",
                    'start_year': daily_means.index.min().year, 'end_year': daily_means.index.max().year,
                    'total_years': len(daily_means.index.year.unique()), 'years': sorted(daily_means.index.year.unique()),
                    'block_len': block_len, 'tp_factor': tp_factor, 'bfi_overall': bfi_overall,
                    'missing_days': daily_means.isna().sum(), 'total_flow': result['daily_mean_flow'].sum(),
                    'total_baseflow': result['baseflow'].sum()
                }
                result['year'] = result['date'].dt.year
                result['month'] = result['date'].dt.month
                
            elif algorithm_choice == "Lyne & Hollick (1979)":
                df_long['date_only'] = df_long['datetime'].dt.date
                daily_means = df_long.groupby('date_only')['discharge'].mean()
                daily_means.index = pd.to_datetime(daily_means.index)
                full_index = pd.date_range(start=daily_means.index.min(), end=daily_means.index.max(), freq='D')
                daily_means = daily_means.reindex(full_index)
                if daily_means.isna().sum() > 0:
                    daily_means = daily_means.interpolate(method='linear', limit=3).ffill().bfill()
                
                result, bfi_overall = calculate_bfi_lyne_hollick(daily_means, **algorithm_params['lyne_hollick'])
                current_algorithm = "Lyne & Hollick (1979)"
                algorithm_info = {
                    'Algorithm': current_algorithm,
                    'Parameter alpha': algorithm_params['lyne_hollick']['alpha'],
                    'Number of passes': algorithm_params['lyne_hollick']['passes']
                }
                metadata = {
                    'n_days': len(daily_means), 'n_measurements': len(df_long),
                    'date_range': f"{daily_means.index.min().strftime('%Y-%m-%d')} to {daily_means.index.max().strftime('%Y-%m-%d')}",
                    'start_year': daily_means.index.min().year, 'end_year': daily_means.index.max().year,
                    'total_years': len(daily_means.index.year.unique()), 'years': sorted(daily_means.index.year.unique()),
                    'block_len': block_len, 'tp_factor': tp_factor, 'bfi_overall': bfi_overall,
                    'missing_days': daily_means.isna().sum(), 'total_flow': result['daily_mean_flow'].sum(),
                    'total_baseflow': result['baseflow'].sum()
                }
                result['year'] = result['date'].dt.year
                result['month'] = result['date'].dt.month
                
            elif algorithm_choice == "Chapman (1991)":
                df_long['date_only'] = df_long['datetime'].dt.date
                daily_means = df_long.groupby('date_only')['discharge'].mean()
                daily_means.index = pd.to_datetime(daily_means.index)
                full_index = pd.date_range(start=daily_means.index.min(), end=daily_means.index.max(), freq='D')
                daily_means = daily_means.reindex(full_index)
                if daily_means.isna().sum() > 0:
                    daily_means = daily_means.interpolate(method='linear', limit=3).ffill().bfill()
                
                result, bfi_overall = calculate_bfi_chapman(daily_means, **algorithm_params['chapman'])
                current_algorithm = "Chapman (1991)"
                algorithm_info = {
                    'Algorithm': current_algorithm,
                    'Parameter alpha': algorithm_params['chapman']['alpha']
                }
                metadata = {
                    'n_days': len(daily_means), 'n_measurements': len(df_long),
                    'date_range': f"{daily_means.index.min().strftime('%Y-%m-%d')} to {daily_means.index.max().strftime('%Y-%m-%d')}",
                    'start_year': daily_means.index.min().year, 'end_year': daily_means.index.max().year,
                    'total_years': len(daily_means.index.year.unique()), 'years': sorted(daily_means.index.year.unique()),
                    'block_len': block_len, 'tp_factor': tp_factor, 'bfi_overall': bfi_overall,
                    'missing_days': daily_means.isna().sum(), 'total_flow': result['daily_mean_flow'].sum(),
                    'total_baseflow': result['baseflow'].sum()
                }
                result['year'] = result['date'].dt.year
                result['month'] = result['date'].dt.month
                
            elif algorithm_choice == "EWMA Filter (Tularam & Ilahee, 2008)":
                df_long['date_only'] = df_long['datetime'].dt.date
                daily_means = df_long.groupby('date_only')['discharge'].mean()
                daily_means.index = pd.to_datetime(daily_means.index)
                full_index = pd.date_range(start=daily_means.index.min(), end=daily_means.index.max(), freq='D')
                daily_means = daily_means.reindex(full_index)
                if daily_means.isna().sum() > 0:
                    daily_means = daily_means.interpolate(method='linear', limit=3).ffill().bfill()
                
                result, bfi_overall = calculate_bfi_ewma(daily_means, **algorithm_params['ewma'])
                current_algorithm = "EWMA Filter (Tularam & Ilahee, 2008)"
                algorithm_info = {
                    'Algorithm': current_algorithm,
                    'Smoothing parameter alpha': algorithm_params['ewma']['alpha']
                }
                metadata = {
                    'n_days': len(daily_means), 'n_measurements': len(df_long),
                    'date_range': f"{daily_means.index.min().strftime('%Y-%m-%d')} to {daily_means.index.max().strftime('%Y-%m-%d')}",
                    'start_year': daily_means.index.min().year, 'end_year': daily_means.index.max().year,
                    'total_years': len(daily_means.index.year.unique()), 'years': sorted(daily_means.index.year.unique()),
                    'block_len': block_len, 'tp_factor': tp_factor, 'bfi_overall': bfi_overall,
                    'missing_days': daily_means.isna().sum(), 'total_flow': result['daily_mean_flow'].sum(),
                    'total_baseflow': result['baseflow'].sum()
                }
                result['year'] = result['date'].dt.year
                result['month'] = result['date'].dt.month
                
            elif algorithm_choice == "COMPARE ALL ALGORITHMS":
                # Jalankan semua algoritma untuk perbandingan
                comparison_results = calculate_bfi_comparison(df_long, block_len, tp_factor, algorithm_params)
                
                # Tampilkan chart perbandingan
                st.subheader("📊 Perbandingan Semua Algoritma")
                st.caption("Hasil BFI dari 7 algoritma baseflow separation yang berbeda")
                
                comparison_chart = create_algorithm_comparison_chart(comparison_results)
                st.plotly_chart(comparison_chart, use_container_width=True)
                
                # Tampilkan tabel perbandingan
                comparison_table = create_algorithm_comparison_table(comparison_results)
                st.dataframe(comparison_table, use_container_width=True)
                
                # Tampilkan hydrograph perbandingan
                st.subheader("📈 Hydrograph Perbandingan (3 Algoritma Terbaik)")
                
                # Pilih 3 algoritma dengan performa terbaik berdasarkan BFI
                sorted_by_bfi = sorted(comparison_results.items(), key=lambda x: x[1]['bfi'], reverse=True)
                top_3_algorithms = dict(sorted_by_bfi[:3])
                
                # Buat figure perbandingan hydrograph
                fig_compare = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=list(top_3_algorithms.keys()),
                    vertical_spacing=0.1,
                    shared_xaxes=True
                )
                
                for idx, (alg_name, alg_data) in enumerate(top_3_algorithms.items(), 1):
                    result_alg = alg_data['result']
                    display_data = result_alg.tail(365)  # 1 tahun terakhir
                    
                    fig_compare.add_trace(
                        go.Scatter(
                            x=display_data['date'],
                            y=display_data['daily_mean_flow'],
                            name='Qt' if idx == 1 else None,
                            line=dict(color='#3B82F6', width=1),
                            showlegend=(idx == 1)
                        ),
                        row=idx, col=1
                    )
                    
                    fig_compare.add_trace(
                        go.Scatter(
                            x=display_data['date'],
                            y=display_data['baseflow'],
                            name='Qb' if idx == 1 else None,
                            line=dict(color='#10B981', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(16, 185, 129, 0.2)',
                            showlegend=(idx == 1)
                        ),
                        row=idx, col=1
                    )
                    
                    # Tambahkan anotasi BFI
                    fig_compare.add_annotation(
                        x=0.98, y=0.95,
                        xref=f"x{idx}", yref=f"y{idx}",
                        text=f"BFI = {alg_data['bfi']:.4f}",
                        showarrow=False,
                        font=dict(size=10, color="#F59E0B"),
                        bgcolor='rgba(0,0,0,0.5)',
                        row=idx, col=1
                    )
                
                fig_compare.update_layout(
                    template="plotly_dark",
                    height=900,
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_text="Perbandingan Hydrograph - 3 Algoritma dengan BFI Tertinggi"
                )
                
                fig_compare.update_xaxes(title_text="Date", row=3, col=1)
                fig_compare.update_yaxes(title_text="Discharge", row=1, col=1)
                fig_compare.update_yaxes(title_text="Discharge", row=2, col=1)
                fig_compare.update_yaxes(title_text="Discharge", row=3, col=1)
                
                st.plotly_chart(fig_compare, use_container_width=True)
                
                st.info("💡 Mode perbandingan menampilkan hasil dari semua algoritma. Pilih algoritma spesifik dari menu dropdown untuk analisis lebih detail.")
                
                # Gunakan hasil dari algoritma default (IH UK) untuk sisa dashboard
                result, bfi_overall, metadata = process_bfi(df_long, block_len, tp_factor)
                current_algorithm = "Institute of Hydrology (UK) - (Default untuk dashboard)"
                algorithm_info = {
                    'Algorithm': 'Comparison Mode',
                    'Algorithms compared': list(comparison_results.keys()),
                    'BFI values': {k: v['bfi'] for k, v in comparison_results.items()}
                }
                result['year'] = result['date'].dt.year
                result['month'] = result['date'].dt.month
                
                # Set flag untuk mode perbandingan
                is_comparison_mode = True
            
            # Perform EDA (hanya jika tidak dalam mode perbandingan atau setelah dapat result)
            if algorithm_choice != "COMPARE ALL ALGORITHMS":
                eda_results = perform_eda(df_long, result)
                eda_figs = create_eda_visualizations(result, df_long, eda_results)
                
                # Tampilkan informasi algoritma yang digunakan
                st.info(f"🔬 **Algoritma yang digunakan:** {current_algorithm}")
                
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
                
                # Hitung BFI per tahun untuk ditampilkan
                yearly_bfi = result.groupby('year').agg({
                    'bfi_daily': ['mean', 'median', 'std', 'min', 'max'],
                    'daily_mean_flow': 'sum',
                    'baseflow': 'sum'
                }).round(4)
                yearly_bfi.columns = ['bfi_mean', 'bfi_median', 'bfi_std', 'bfi_min', 'bfi_max', 'total_flow', 'total_baseflow']
                yearly_bfi['bfi_overall'] = yearly_bfi['total_baseflow'] / yearly_bfi['total_flow']
                yearly_bfi = yearly_bfi.reset_index()
                
                # Format tampilan tabel
                display_yearly = yearly_bfi[['year', 'bfi_mean', 'bfi_median', 'bfi_std', 'bfi_min', 'bfi_max', 'bfi_overall']].copy()
                display_yearly.columns = ['Tahun', 'Mean BFI', 'Median BFI', 'Std BFI', 'Min BFI', 'Max BFI', 'BFI Overall']
                display_yearly = display_yearly.round(4)
                
                st.dataframe(display_yearly, use_container_width=True)
                
                # Bar chart BFI per tahun
                yearly_fig = go.Figure()
                
                yearly_fig.add_trace(go.Bar(
                    x=yearly_bfi['year'],
                    y=yearly_bfi['bfi_overall'],
                    name='BFI Overall',
                    marker_color='#3B82F6',
                    text=yearly_bfi['bfi_overall'].round(4),
                    textposition='auto',
                    hovertemplate='<b>Tahun %{x}</b><br>BFI: %{y:.4f}<extra></extra>'
                ))
                
                yearly_fig.add_trace(go.Scatter(
                    x=yearly_bfi['year'],
                    y=yearly_bfi['bfi_mean'],
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
                if len(yearly_bfi) > 1:
                    bfi_trend = yearly_bfi['bfi_overall'].iloc[-1] - yearly_bfi['bfi_overall'].iloc[0]
                    trend_icon = "📈" if bfi_trend > 0 else "📉"
                    trend_color = "#10B981" if bfi_trend > 0 else "#EF4444"
                    trend_text = "meningkat" if bfi_trend > 0 else "menurun"
                    
                    st.markdown(
                        f"<div class='info-card' style='text-align:center'>"
                        f"<strong>{trend_icon} Tren BFI {trend_text}</strong><br>"
                        f"Dari <strong>{yearly_bfi['year'].iloc[0]}</strong> (BFI: {yearly_bfi['bfi_overall'].iloc[0]:.4f}) "
                        f"ke <strong>{yearly_bfi['year'].iloc[-1]}</strong> (BFI: {yearly_bfi['bfi_overall'].iloc[-1]:.4f})<br>"
                        f"<span style='color:{trend_color}; font-size:18px; font-weight:bold'>Perubahan: {bfi_trend:+.4f}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                st.markdown("---")
                
                # ========== SECTION 5: YEARLY COMPARISON CHART ==========
                st.subheader("📊 Perbandingan Qt vs Qb per Tahun")
                
                comparison_fig = go.Figure()
                
                comparison_fig.add_trace(go.Bar(
                    x=yearly_bfi['year'],
                    y=yearly_bfi['total_flow'],
                    name='Total Flow (Qt)',
                    marker_color='#3B82F6',
                    text=yearly_bfi['total_flow'].round(1),
                    textposition='outside'
                ))
                
                comparison_fig.add_trace(go.Bar(
                    x=yearly_bfi['year'],
                    y=yearly_bfi['total_baseflow'],
                    name='Baseflow (Qb)',
                    marker_color='#10B981',
                    text=yearly_bfi['total_baseflow'].round(1),
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
                
                # ========== SECTION 11: KOEFISIEN REGIM ALIRAN (KRA) ==========
                st.markdown("---")
                st.subheader("🌊 Koefisien Regim Aliran (KRA)")
                st.caption("Berdasarkan Peraturan Menteri Kehutanan No. 61 Tahun 2014")
                
                # Hitung KRA
                yearly_kra_df, overall_kra, kra_category, kra_desc, kra_color = calculate_kra_from_yearly_data(result)
                
                # Tampilkan KRA Overall
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(
                        f"<div class='info-card' style='text-align:center'>"
                        f"<strong>📊 KRA Keseluruhan</strong><br>"
                        f"<span style='font-size:28px; font-weight:bold; color:{kra_color}'>{overall_kra:.2f}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"<div class='info-card' style='text-align:center'>"
                        f"<strong>🏷️ Kategori</strong><br>"
                        f"<span style='font-size:18px; font-weight:bold; color:{kra_color}'>{kra_category}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with col3:
                    st.markdown(
                        f"<div class='info-card' style='text-align:center'>"
                        f"<strong>📈 Deskripsi</strong><br>"
                        f"<span style='font-size:16px'>{kra_desc}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with col4:
                    st.markdown(
                        f"<div class='info-card' style='text-align:center'>"
                        f"<strong>📅 Periode Analisis</strong><br>"
                        f"<span style='font-size:16px'>{metadata['start_year']} - {metadata['end_year']}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                st.markdown("---")
                
                # Tabel KRA per Tahun
                st.subheader("📋 Tabel KRA per Tahun")
                
                # Format tabel untuk ditampilkan
                display_kra = yearly_kra_df[['Tahun', 'Qmax (Musim Hujan)', 'Tanggal Qmax', 'Qmin (Musim Kemarau)', 'Tanggal Qmin', 'KRA', 'Kategori']].copy()
                display_kra['Qmax (Musim Hujan)'] = display_kra['Qmax (Musim Hujan)'].round(2)
                display_kra['Qmin (Musim Kemarau)'] = display_kra['Qmin (Musim Kemarau)'].round(2)
                display_kra['KRA'] = display_kra['KRA'].round(2)
                
                st.dataframe(display_kra, use_container_width=True)
                
                # Visualisasi KRA
                kra_fig_bar, kra_fig_q = create_kra_visualizations(yearly_kra_df, overall_kra, kra_category, kra_color)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(kra_fig_bar, use_container_width=True)
                with col2:
                    st.plotly_chart(kra_fig_q, use_container_width=True)
                
                # KRA 10 Tahunan (Moving Average)
                st.subheader("📊 KRA Interval 10 Tahunan")
                
                if metadata['total_years'] >= 10:
                    decadal_kra_df = calculate_kra_10year_moving_average(result, window=10)
                    
                    if len(decadal_kra_df) > 0:
                        display_decadal = decadal_kra_df[['Periode', 'Qmax (Musim Hujan)', 'Qmin (Musim Kemarau)', 'KRA', 'Kategori']].copy()
                        display_decadal['Qmax (Musim Hujan)'] = display_decadal['Qmax (Musim Hujan)'].round(2)
                        display_decadal['Qmin (Musim Kemarau)'] = display_decadal['Qmin (Musim Kemarau)'].round(2)
                        display_decadal['KRA'] = display_decadal['KRA'].round(2)
                        
                        st.dataframe(display_decadal, use_container_width=True)
                        
                        # Bar chart KRA 10 tahunan
                        fig_decadal = go.Figure()
                        fig_decadal.add_trace(go.Bar(
                            x=decadal_kra_df['Periode'],
                            y=decadal_kra_df['KRA'],
                            marker_color='#06B6D4',
                            text=decadal_kra_df['KRA'].round(2),
                            textposition='auto'
                        ))
                        fig_decadal.update_layout(
                            template="plotly_dark",
                            title="Koefisien Regim Aliran (KRA) - Interval 10 Tahunan",
                            xaxis_title="Periode",
                            yaxis_title="Nilai KRA",
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_decadal, use_container_width=True)
                    else:
                        st.info("Data tidak mencukupi untuk analisis KRA 10 tahunan")
                else:
                    st.info(f"Data hanya mencakup {metadata['total_years']} tahun. Minimal 10 tahun diperlukan untuk analisis KRA interval 10 tahunan.")
                
                # Informasi tentang formula KRA
                with st.expander("ℹ️ Tentang Koefisien Regim Aliran (KRA)"):
                    st.markdown("""
                    ### Formula Koefisien Regim Aliran (KRA)
                    
                    **KRA = Qmax / Qmin**
                    
                    Dimana:
                    - **Qmax**: Debit aliran sungai maksimum pada musim penghujan (m³/detik)
                    - **Qmin**: Debit aliran sungai minimum pada musim kemarau (m³/detik)
                    
                    ### Kategori Nilai KRA (Permenhut No. 61 Tahun 2014)
                    
                    | Nilai KRA | Kategori |
                    |-----------|----------|
                    | ≤ 20 | Sangat Rendah (SR) |
                    | 20 < KRA ≤ 50 | Rendah (R) |
                    | 50 < KRA ≤ 80 | Sedang (S) |
                    | 80 < KRA ≤ 110 | Tinggi (T) |
                    | > 110 | Sangat Tinggi (ST) |
                    
                    ### Interpretasi:
                    - **Semakin tinggi nilai KRA**, semakin besar fluktuasi antara debit musim hujan dan musim kemarau
                    - **Semakin rendah nilai KRA**, semakin stabil aliran sungai sepanjang tahun
                    """)
                
                # Simpan KRA results untuk export ke Excel
                kra_results = {
                    'yearly_kra': yearly_kra_df.to_dict('records'),
                    'overall_kra': overall_kra,
                    'kra_category': kra_category,
                    'kra_desc': kra_desc
                }
                
                if metadata['total_years'] >= 10:
                    kra_results['decadal_kra'] = decadal_kra_df.to_dict('records')
                # ========== SECTION 11: PREVIEW & DOWNLOAD ==========
                st.subheader("📋 Processed Data Ledger")
                st.dataframe(result.head(50), use_container_width=True)
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download Excel
                    excel_file = export_to_excel(result, metadata, eda_results, algorithm_info)
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
                        st.write("**Algorithm Info:**")
                        for k, v in algorithm_info.items():
                            st.write(f"- {k}: {v}")
                
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
        - ✅ 7 Algoritma Baseflow Separation
        - ✅ Perbandingan semua algoritma
        - ✅ Nilai Maksimum & Minimum dengan waktu kejadian
        - ✅ Exploratory Data Analysis (8 jenis visualisasi)
        - ✅ Download laporan Excel & PDF
        """)
