import os
import pandas as pd
import random
from src.strategies.order_block import OrderBlockDetector
from rich.console import Console

console = Console()

def get_samples_fast():
    data_path = "data/raw/"
    file = "BTCUSDT_5m.csv"
    if not os.path.exists(os.path.join(data_path, file)):
        console.print("[red]Data not found.[/red]")
        return
        
    df = pd.read_csv(os.path.join(data_path, file))
    # Standardize timestamp to strings for display
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    detector = OrderBlockDetector(df)
    bull_obs = detector.find_bullish_order_blocks()
    bear_obs = detector.find_bearish_order_blocks()
    
    samples_found = 0
    # Search from the middle of the dataset for fresh samples
    start_point = len(bull_obs) // 2 
    
    for i in range(start_point, len(bull_obs)):
        ob = bull_obs[i]
        start_idx = ob['index'] + 1
        z_hh, z_ll = ob['zone_hh'], ob['zone_ll']
        entry = z_hh + (0.03 * ob['height'])
        
        # Check next 20 candles
        for j in range(start_idx, min(start_idx + 20, len(df))):
            if df.iloc[j]['low'] <= entry and df.iloc[j]['low'] > z_ll:
                console.print(f"\n[bold green]SAMPLE BULLISH TRADE (BTC 5m)[/bold green]")
                console.print(f"Pattern Time: {df.iloc[ob['index']]['timestamp']} UTC")
                console.print(f"Entry Price: {entry:.2f}")
                console.print(f"Pattern High: {z_hh:.2f} | Pattern Low: {z_ll:.2f}")
                samples_found += 1
                break
        if samples_found >= 3: break

    samples_found = 0
    for i in range(start_point, len(bear_obs)):
        ob = bear_obs[i]
        start_idx = ob['index'] + 1
        z_hh, z_ll = ob['zone_hh'], ob['zone_ll']
        entry = z_ll - (0.03 * ob['height'])
        
        for j in range(start_idx, min(start_idx + 20, len(df))):
            if df.iloc[j]['high'] >= entry and df.iloc[j]['high'] < z_hh:
                console.print(f"\n[bold red]SAMPLE BEARISH TRADE (BTC 5m)[/bold red]")
                console.print(f"Pattern Time: {df.iloc[ob['index']]['timestamp']} UTC")
                console.print(f"Entry Price: {entry:.2f}")
                console.print(f"Pattern High: {z_hh:.2f} | Pattern Low: {z_ll:.2f}")
                samples_found += 1
                break
        if samples_found >= 2: break

if __name__ == "__main__":
    get_samples_fast()
