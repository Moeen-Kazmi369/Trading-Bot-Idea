import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

def detect_pois(symbol, timeframe):
    data_path = f"data/raw/{symbol}_{timeframe}.csv"
    if not os.path.exists(data_path): return
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate Batch Metrics (50 candle lookback)
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(window=50).mean()
    df['rolling_low'] = df['low'].shift(1).rolling(window=50).min()
    
    results = {
        'Power Candle': [],
        'Vector Run': [],
        'Spring': []
    }
    
    for i in range(50, len(df)):
        # 1. Power Candle Detection
        # Rule: Single body > 4x Batch Average
        if df.iloc[i]['body'] > 4.0 * df.iloc[i]['avg_body'] and df.iloc[i]['close'] > df.iloc[i]['open']:
            results['Power Candle'].append({
                'time': df.iloc[i]['timestamp'],
                'price': df.iloc[i]['close'],
                'size': f"{df.iloc[i]['body']:.2f}"
            })

        # 2. Vector Run Detection (3 consecutive strong green candles)
        if i > 2:
            prev3 = df.iloc[i-2:i+1]
            if all(prev3['close'] > prev3['open']) and prev3['body'].sum() > 5.0 * df.iloc[i]['avg_body']:
                results['Vector Run'].append({
                    'time': df.iloc[i]['timestamp'],
                    'price': df.iloc[i]['close'],
                    'size': f"{prev3['body'].sum():.2f}"
                })
        
        # 3. Spring Detection (Liquidity Sweep + Reversal)
        if i > 1:
            sweep_candle = df.iloc[i-1]
            reversal_candle = df.iloc[i]
            if sweep_candle['low'] <= df.iloc[i-1]['rolling_low'] and reversal_candle['close'] > sweep_candle['high']:
                if reversal_candle['body'] > 2.5 * df.iloc[i]['avg_body']:
                    results['Spring'].append({
                        'time': df.iloc[i]['timestamp'],
                        'price': df.iloc[i]['close'],
                        'size': f"{reversal_candle['body']:.2f}"
                    })

    return results

def report_samples():
    data = detect_pois("BTCUSDT", "5m")
    
    table = Table(title="POI VERIFICATION SAMPLES (BTC 5m)")
    table.add_column("Type", style="cyan")
    table.add_column("Timestamp (UTC)", style="magenta")
    table.add_column("Price (Exit)", style="green")
    table.add_column("Move Size", style="yellow")
    
    for poi_type, samples in data.items():
        # Get first 2 samples from the middle of the dataset for cleaner visuals
        valid_samples = samples[len(samples)//2 : len(samples)//2 + 2]
        for s in valid_samples:
            table.add_row(poi_type, str(s['time']), str(s['price']), s['size'])
            
    console.print(table)

if __name__ == "__main__":
    report_samples()
