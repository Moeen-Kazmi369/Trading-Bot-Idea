import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

def analyze_pre_trend(df, start_idx, lookback=50):
    """
    Analyzes the 'Silence' before the 'Starting Point' at start_idx.
    """
    if start_idx < lookback: return None
    
    # The Pre-Trend Window (The Squeeze)
    pre_window = df.iloc[start_idx - lookback : start_idx]
    
    # The Trigger Candle
    trigger = df.iloc[start_idx]
    
    metrics = {
        'compression': pre_window['high'].max() - pre_window['low'].min(),
        'avg_vol': pre_window['volume'].mean(),
        'vol_std': pre_window['volume'].std(),
        'volatility_drop': pre_window['body'].std() / pre_window['body'].mean(),
        'trigger_strength': trigger['body'] / pre_window['body'].mean(),
        'trigger_vol_surge': trigger['volume'] / pre_window['volume'].mean()
    }
    
    return metrics

def discover_trend_starts(symbol, timeframe):
    path = f"data/raw/{symbol}_{timeframe}.csv"
    if not os.path.exists(path): return []
    
    df = pd.read_csv(path)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['body'] = abs(df['close'] - df['open'])
    
    starts = []
    
    # Define a 'Trend Start' as a breakout from a 50-bar channel with high volume
    for i in range(50, len(df) - 50):
        prev_50 = df.iloc[i-50:i]
        upper = prev_50['high'].max()
        lower = prev_50['low'].min()
        
        # Bullish Start Condition: Close > Upper Channel + Volume > 2x Avg
        if df.iloc[i]['close'] > upper and df.iloc[i]['volume'] > 2 * prev_50['volume'].mean():
            # Validate if it actually 'Started' a trend (Next 20 candles stay above)
            future = df.iloc[i+1 : i+21]
            if future['close'].min() > lower:
                # This is a Valid Starting Point! Now audit the BEFORE data
                pre_trend_metrics = analyze_pre_trend(df, i)
                starts.append({
                    'time': df.iloc[i]['timestamp'],
                    'price': df.iloc[i]['close'],
                    'metrics': pre_trend_metrics,
                    'result': future['close'].max() - df.iloc[i]['close'] # Profit potential
                })
                
    return starts

def report():
    # Test on 15m for clear trends
    results = discover_trend_starts("BTCUSDT", "15m")
    
    table = Table(title="TREND START AUDIT (15m BTC)")
    table.add_column("Starting Point (UTC)", style="magenta")
    table.add_column("Pre-Trend Squeeze", style="cyan")
    table.add_column("Wait Volume Avg", style="yellow")
    table.add_column("Trigger Power", style="green")
    table.add_column("Trend Success", style="bold white")
    
    # Sort by success to see the 'Best' starts
    sorted_res = sorted(results, key=lambda x: x['result'], reverse=True)
    
    for r in sorted_res[:10]:
        m = r['metrics']
        table.add_row(
            str(r['time']),
            f"{m['volatility_drop']:.2f}",
            f"{int(m['avg_vol'])}",
            f"{m['trigger_strength']:.1f}x",
            f"${r['result']:.2f}"
        )
            
    console.print(table)

if __name__ == "__main__":
    report()
