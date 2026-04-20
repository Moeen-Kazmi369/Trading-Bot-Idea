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
    if body_pct < 0.15: return "Doji"
    if body_pct < 0.4 and lower_wick > 1.3 * body and upper_wick < 0.3 * range_: return "Hammer"
    if body_pct < 0.45 and upper_wick > 0.3 * range_ and lower_wick > 0.3 * range_: return "Spinning Top"
    return "Normal"

def run_dynamic_root_strategy(symbol, timeframe):
    data_path = f"data/raw/{symbol}_{timeframe}.csv"
    if not os.path.exists(data_path): return []
    
    df = pd.read_csv(data_path)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(window=50).mean()
    
    setups = []
    
    # Loop through data to find Momentum Matches (Windows of 1-5)
    for i in range(50, len(df) - 5):
        found_momentum = False
        momentum_start_idx = -1
        momentum_end_idx = -1
        
        # Check window sizes 1 to 5 starting at i
        for win in range(1, 6):
            window_df = df.iloc[i:i+win]
            avg_b = df.iloc[i]['avg_body']
            
            # Bullish Momentum Check
            if all(window_df['close'] > window_df['open']):
                if window_df['body'].sum() > 4.5 * avg_b:
                    found_momentum = True
                    momentum_start_idx = i
                    break
        
        if found_momentum:
            # Backtrack to find the VERY LAST RED candle at the base
            # Check up to 3 candles before the pump start
            for back_idx in range(momentum_start_idx, momentum_start_idx - 5, -1):
                if back_idx < 1: break
                
                c1 = df.iloc[back_idx - 1] # Prev
                c2 = df.iloc[back_idx]     # Potential OB
                c3 = df.iloc[back_idx + 1] # Next
                
                # BULLISH GRG CHECK (C1 G -> C2 R -> C3 G)
                if c2['close'] < c2['open']: # It is RED
                    if c1['close'] > c1['open'] and c3['close'] > c3['open']: # GRG Pattern
                        # The 2 Strict Approved Rules
                        if c2['low'] > c1['low'] and c3['low'] > c2['low']:
                            candle_kind = get_candle_type(c2['open'], c2['high'], c2['low'], c2['close'])
                            setups.append({
                                'time': c2['timestamp'],
                                'type': 'BULLISH (Demand)',
                                'zone': f"{c2['high']} - {c2['low']}",
                                'plus_point': candle_kind,
                                'pump_time': df.iloc[momentum_start_idx]['timestamp']
                            })
                            break # Found the last red, move to next area

    return setups

def report():
    results = run_dynamic_root_strategy("BTCUSDT", "5m")
    
    table = Table(title="DYNAMIC ROOT STRATEGY (Approved Logic - BTC 5m)")
    table.add_column("OB Timestamp (UTC)", style="magenta")
    table.add_column("Type", style="cyan")
    table.add_column("OB Zone (H-L)", style="yellow")
    table.add_column("Plus Point", style="green")
    table.add_column("Pump Start Time", style="bold white")
    
    for s in results[-10:]:
        table.add_row(str(s['time']), s['type'], s['zone'], s['plus_point'], str(s['pump_time']))
        
    console.print(table)
    console.print(f"\n[bold cyan]TOTAL DYNAMIC SETUPS FOUND:[/bold cyan] {len(results)}")

if __name__ == "__main__":
    report()
