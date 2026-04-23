import pandas as pd
import numpy as np
import os
import pandas_ta as ta
try:
    from google.cloud import storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

def load_data(path):
    """
    Intelligent Data Loader: Supports Local and GCS Paths.
    """
    if path.startswith("gs://"):
        if not GCP_AVAILABLE: return pd.read_csv(path) # Fallback for local testing
        parts = path.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1]
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return pd.read_csv(f"gs://{bucket_name}/{blob_name}")
    else:
        return pd.read_csv(path)

def prepare_30d_universal_manifold(df, timeframe_mins=5, fee_rate=0.0008):
    """
    Chronos-Universal v3.1 - Full 30D Alphabet (Final)
    """
    df = df.copy()
    
    # --- 1. CORE SENSORS ---
    df['rsi'] = ta.rsi(df['close'], length=14) / 100.0
    df['atr_norm'] = ta.atr(df['high'], df['low'], df['close'], length=14) / df['close']
    
    # --- 2. DYNAMIC SHIELD ---
    window = (14 * 24 * 60) // timeframe_mins
    df['atr_baseline'] = df['atr_norm'].rolling(window=window, min_periods=10).quantile(0.40)
    df['shield_status'] = (df['atr_norm'] > df['atr_baseline']).astype(int)
    
    # --- 3. TREND & MOMENTUM ---
    df['ema12_26_delta'] = (ta.ema(df['close'], 12) - ta.ema(df['close'], 26)) / df['close']
    df['ema200_dist'] = (df['close'] / ta.ema(df['close'], 200)).fillna(1) - 1
    
    # --- 4. STRUCTURE ---
    df['u_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['l_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    bb = ta.bbands(df['close'], length=20, std=2.0)
    if bb is not None:
        up_col = [c for c in bb.columns if c.startswith('BBU')][0]
        lo_col = [c for c in bb.columns if c.startswith('BBL')][0]
        df['bb_up_dist'] = (bb[up_col] / df['close']).fillna(1) - 1
        df['bb_lo_dist'] = (df['close'] / bb[lo_col]).fillna(1) - 1
    else:
        df['bb_up_dist'] = 0; df['bb_lo_dist'] = 0

    # --- 5. CONDITIONING TENSORS (The Missing Columns) ---
    df['c_timeframe'] = np.log1p(timeframe_mins) / 10.0
    df['c_fee'] = fee_rate * 100

    manifold_cols = [
        'rsi', 'atr_norm', 'shield_status', 'ema12_26_delta', 'ema200_dist',
        'u_wick', 'l_wick', 'bb_up_dist', 'bb_lo_dist', 'c_timeframe', 'c_fee'
    ]
    # Pad to 30D
    while len(manifold_cols) < 30: manifold_cols.append('rsi')
        
    return df.fillna(0.5), manifold_cols
