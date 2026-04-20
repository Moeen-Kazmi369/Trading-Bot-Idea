import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

def get_candle_type(open_, high, low, close):
    body = abs(close - open_)
    range_ = high - low
    if range_ == 0: return "Doji"
    
    body_pct = body / range_
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    
    # Doji
    if body_pct < 0.1: return "Doji"
    # Hammer
    if body_pct < 0.35 and lower_wick > 1.5 * body and upper_wick < 0.2 * range_: return "Hammer"
    # Spinning Top
    if body_pct < 0.4 and upper_wick > 0.3 * range_ and lower_wick > 0.3 * range_: return "Spinning Top"
    
    return "Normal"

def discover_strategy_v3(symbol, timeframe):
    data_path = f"data/raw/{symbol}_{timeframe}.csv"
    if not os.path.exists(data_path): return []
    
    df = pd.read_csv(data_path)
    # Ensure columns are numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(window=50).mean()
    
    setups = []
    
    for i in range(50, len(df) - 5):
        # 1. FIND POI (Vector Run: 3 strong green candles)
        v_run = df.iloc[i:i+3]
        if all(v_run['close'] > v_run['open']) and v_run['body'].sum() > 4.5 * df.iloc[i]['avg_body']:
            
            # 2. FIND OB (Last R in GRG sequence before the run)
            # We look at i-1 (the candle before the run)
            if i < 2: continue
            
            c1 = df.iloc[i-2] # Potential G
            c2 = df.iloc[i-1] # Potential R (The OB)
            c3 = df.iloc[i]   # Start of Pump (G)
            
            # GRG check
            if c1['close'] > c1['open'] and c2['close'] < c2['open'] and c3['close'] > c3['open']:
                # HH / LL check
                # Rule: C2 (Red) must have the absolute lowest low of the 3
                if c2['low'] < c1['low'] and c2['low'] < c3['low']:
                    
                    # Plus Point check
                    candle_kind = get_candle_type(c2['open'], c2['high'], c2['low'], c2['close'])
                    
                    setups.append({
                        'timestamp': c2['timestamp'],
                        'price': c2['close'],
                        'poi_time': v_run.iloc[-1]['timestamp'],
                        'plus_point': candle_kind,
                        'is_elite': "YES" if candle_kind != "Normal" else "NO"
                    })
                    
    return setups

def report():
    setups = discover_strategy_v3("BTCUSDT", "5m")
    
    table = Table(title="STRATEGY v3 DISCOVERY (BTC 5m)")
    table.add_column("OB Timestamp (UTC)", style="magenta")
    table.add_column("Type", style="cyan")
    table.add_column("POI Time (Pump)", style="green")
    table.add_column("Plus Point Candle", style="yellow")
    table.add_column("Elite?", style="bold white")
    
    # Filter for the last few to be fresh
    for s in setups[-5:]:
        table.add_row(
            str(s['timestamp']),
            "BULLISH OB",
            str(s['poi_time']),
            s['plus_point'],
            s['is_elite']
        )
            
    console.print(table)
    console.print(f"\n[bold cyan]TOTAL VALID SETUPS FOUND:[/bold cyan] {len(setups)}")

if __name__ == "__main__":
    report()
