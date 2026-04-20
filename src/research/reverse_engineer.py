import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

def load_last_week(symbol, timeframe):
    """Loads the last 7 days of data to find the most recent trends."""
    path = f"data/raw/{symbol}_{timeframe}.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    end_time = df['timestamp'].max()
    start_time = end_time - pd.Timedelta(days=7)
    df = df[df['timestamp'] >= start_time].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def find_uptrends(df, window, min_gain_pct):
    """
    Finds starting points of uptrends. 
    A starting point is a local low that leads to a significant % gain.
    """
    trends = []
    for i in range(window, len(df) - window * 2):
        # Check if current candle is a local minimum (trough)
        if df['low'].iloc[i] == df['low'].iloc[i-window : i+window+1].min():
            
            # Find the peak in the subsequent period
            future = df.iloc[i+1 : i + window * 4]
            peak_idx = future['high'].idxmax()
            peak_price = future['high'].max()
            trough_price = df['low'].iloc[i]
            
            gain = (peak_price - trough_price) / trough_price * 100
            
            if gain >= min_gain_pct:
                trends.append({
                    'start_idx': i,
                    'end_idx': peak_idx,
                    'start_time': df['timestamp'].iloc[i],
                    'end_time': df['timestamp'].iloc[peak_idx],
                    'start_price': trough_price,
                    'end_price': peak_price,
                    'gain_pct': gain
                })
                
    # Deduplicate (if multiple troughs map to the same peak run, keep the lowest starting point)
    unique_trends = []
    seen_peaks = set()
    for t in sorted(trends, key=lambda x: x['start_price']):
        if t['end_time'] not in seen_peaks:
            unique_trends.append(t)
            # block nearby peaks to avoid duplicates of the same move
            for j in range(-5, 6):
                seen_peaks.add(df.iloc[min(len(df)-1, max(0, t['end_idx'] + j))]['timestamp'])
                
    return unique_trends

def analyze_pre_trend(df, start_idx, lookback):
    """
    Analyzes the data *before* the starting point to mathematically define the pattern.
    """
    if start_idx < lookback: return None
    
    # The period exactly before the starting point
    pre_df = df.iloc[start_idx - lookback : start_idx]
    
    # 1. Consolidation / Compression
    price_range = pre_df['high'].max() - pre_df['low'].min()
    price_range_pct = price_range / pre_df['close'].mean() * 100
    avg_body_pct = abs(pre_df['close'] - pre_df['open']).mean() / pre_df['close'].mean() * 100
    
    # 2. Volume Behavior
    avg_vol = pre_df['volume'].mean()
    vol_std_pct = (pre_df['volume'].std() / avg_vol * 100) if avg_vol > 0 else 0
    
    # 3. Last 3 Candles Action (What happened *right* prior to launch?)
    last3 = pre_df.iloc[-3:]
    red_count = len(last3[last3['close'] < last3['open']])
    
    return {
        'price_range_pct': price_range_pct,
        'avg_body_pct': avg_body_pct,
        'vol_std_pct': vol_std_pct,
        'red_candle_count': red_count
    }

def main():
    # Settings: (Timeframe, Local Window size, Minimum Trend Gain %)
    configs = [
        ('1m', 15, 0.4),   # Need 0.4% move on 1m to call it a trend
        ('5m', 12, 1.0),   # Need 1.0% move on 5m
        ('15m', 10, 2.0)   # Need 2.0% move on 15m
    ]
    
    console.print("\n[bold]========== REVERSE ENGINEERING TRENDS (Last 7 Days) ==========[/bold]")
    
    for tf, window, prom in configs:
        console.print(f"\n[bold cyan]► Processing {tf} Timeframe...[/bold cyan]")
        df = load_last_week("BTCUSDT", tf)
        if df is None: continue
        
        trends = find_uptrends(df, window, prom)
        console.print(f"Found [bold green]{len(trends)}[/bold green] confirmed uptrend starting points.")
        
        lookback = window * 2 # Lookback amount depends on timeframe window
        features = []
        for t in trends:
            feat = analyze_pre_trend(df, t['start_idx'], lookback=lookback)
            if feat:
                features.append(feat)
                
        if features:
            f_df = pd.DataFrame(features)
            
            table = Table(title=f"Common 'Pre-Trend' Patterns (Based on {len(features)} successful {tf} trends)")
            table.add_column("Metric to find 'The Cause'")
            table.add_column("Average Behavior", style="green")
            table.add_column("Min Range", style="yellow")
            table.add_column("Max Range", style="yellow")
            
            table.add_row("Compression Box % (High-Low)", f"{f_df['price_range_pct'].mean():.2f}%", f"{f_df['price_range_pct'].min():.2f}%", f"{f_df['price_range_pct'].max():.2f}%")
            table.add_row("Avg Candle Body Size %", f"{f_df['avg_body_pct'].mean():.3f}%", f"{f_df['avg_body_pct'].min():.3f}%", f"{f_df['avg_body_pct'].max():.3f}%")
            table.add_row("Red Candles in last 3 bars", f"{f_df['red_candle_count'].mean():.1f} out of 3", f"{f_df['red_candle_count'].min()}", f"{f_df['red_candle_count'].max()}")
            
            console.print(table)
            
            console.print("\n[bold]Top 3 most explosive Starting Points (Check on chart):[/bold]")
            for st in sorted(trends, key=lambda x: x['gain_pct'], reverse=True)[:3]:
                # Print in PKT 
                start_pkt = st['start_time'].tz_convert('Asia/Karachi')
                console.print(f"  Start: [magenta]{start_pkt}[/magenta] | Price: ${st['start_price']:,.2f} | Trend Gain: {st['gain_pct']:.2f}%")
                
if __name__ == "__main__":
    main()
