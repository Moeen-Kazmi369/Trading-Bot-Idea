import pandas as pd
import numpy as np
import pandas_ta as ta
from src.data.heatmap_provider import HeatmapProvider

def prepare_15d_manifold(df):
    """
    Project Chronos - Feature-Rich Manifold v2.1
    Dynamic Column Discovery for Bollinger Sentinel.
    """
    df = df.copy()
    
    # 1. 8D CORE GEOMETRY
    df['vel'] = df['close'].pct_change().fillna(0)
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std().fillna(1e-6)
    df['z_score'] = (df['close'] - sma20) / std20
    df['u_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['l_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    df['vol_rel'] = (df['volume'] / df['volume'].rolling(20).mean()).fillna(1)
    df['mtf_proxy'] = (df['close'] / df['close'].shift(12)).fillna(1) - 1
    df['hi_rel'] = df['high'] / df['close'] - 1
    df['lo_rel'] = df['low'] / df['close'] - 1

    # 2. 2D X-RAY (Liquidity Walls)
    hp = HeatmapProvider(window_size=100)
    df = hp.calculate_depth_tensors(df)

    # 3. 5D TECHNICAL SENTINELS
    # RSI
    df['rsi'] = ta.rsi(df['close'], length=14) / 100.0
    
    # MACD 
    macd = ta.macd(df['close'])
    if macd is not None:
        # Auto-find histogram
        hist_col = [c for c in macd.columns if 'MACDh' in c][0]
        df['macd_hist'] = macd[hist_col]
    else:
        df['macd_hist'] = 0
    
    # ATR
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr'] = (atr / df['close']).fillna(0)
    
    # Bollinger Bands (Discovery Mode)
    bb = ta.bbands(df['close'], length=20, std=2.0)
    if bb is not None:
        up_col = [c for c in bb.columns if c.startswith('BBU')][0]
        lo_col = [c for c in bb.columns if c.startswith('BBL')][0]
        df['bb_up_dist'] = (bb[up_col] - df['close']) / df['close']
        df['bb_lo_dist'] = (df['close'] - bb[lo_col]) / df['close']
    else:
        df['bb_up_dist'] = 0
        df['bb_lo_dist'] = 0

    return df.fillna(0)
