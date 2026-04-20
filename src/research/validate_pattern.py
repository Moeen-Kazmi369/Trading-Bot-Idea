import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

def load_data(symbol, timeframe, days=7):
    path = f"data/raw/{symbol}_{timeframe}.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if days is not None:
        end_time = df['timestamp'].max()
        start_time = end_time - pd.Timedelta(days=days)
        df = df[df['timestamp'] >= start_time].copy()
        df.reset_index(drop=True, inplace=True)
    return df

def test_pattern_validity(df, timeframe, window, max_compression_pct, max_body_pct, target_gain_pct):
    """
    Scans the data for the 'Cause' we reverse-engineered:
    1. Pre-trend compression (tight range)
    2. Low volatility (small bodies)
    3. Final spark (2 out of 3 recent candles are red)
    It then records if a trend actually started from that point.
    """
    signals = []
    
    # Analyze the data with our newly found parameters
    for i in range(window * 2, len(df) - window * 4):
        # The 'Before' data
        pre_df = df.iloc[i - window * 2 : i]
        
        # Condition 1: Compression Box
        price_range = pre_df['high'].max() - pre_df['low'].min()
        price_range_pct = price_range / pre_df['close'].mean() * 100
        
        if price_range_pct > max_compression_pct:
            continue
            
        # Condition 2: Low Volatility (Small bodies)
        avg_body_pct = abs(pre_df['close'] - pre_df['open']).mean() / pre_df['close'].mean() * 100
        if avg_body_pct > max_body_pct:
            continue
            
        # Condition 3: Final Spark (Liquidity Grab / 2 of 3 red)
        last3 = pre_df.iloc[-3:]
        red_count = len(last3[last3['close'] < last3['open']])
        if red_count < 2:
            continue
            
        # >> WE FOUND A MATHMATCHICAL "STARTING POINT" SETUP <<
        
        # Let's test if it actually started a trend
        entry_price = df.iloc[i]['close']
        cushion = entry_price * 0.003 # 0.3% stop loss for validation
        stop_price = df.iloc[i]['low'] - cushion 
        target_price = entry_price * (1 + target_gain_pct / 100)
        
        future = df.iloc[i+1 : i + window * 4]
        
        success = False
        max_reached = entry_price
        for _, row in future.iterrows():
            if row['low'] < stop_price:
                break # Stopped out before trend began
            if row['high'] > max_reached:
                max_reached = row['high']
            if row['high'] >= target_price:
                success = True
                
        # Deduplication hack: ignore signals that are basically the same cluster
        if len(signals) > 0 and (df.iloc[i]['timestamp'] - signals[-1]['time']).total_seconds() < window * 60:
            # Update previous signal if this one is lower
            if entry_price < signals[-1]['price']:
               signals[-1]['price'] = entry_price
               signals[-1]['success'] = success or signals[-1]['success']
               signals[-1]['max_gain_pct'] = max((max_reached - entry_price) / entry_price * 100, signals[-1]['max_gain_pct'])
            continue
            
        signals.append({
            'time': df.iloc[i]['timestamp'],
            'price': entry_price,
            'success': success,
            'max_gain_pct': (max_reached - entry_price) / entry_price * 100
        })
        
    return signals

def run():
    # Parameters derived from our Reverse-Engineering averages (slightly relaxed to catch setups)
    # (Timeframe, Window, Max Compression %, Max Body %, Target Trend %)
    configs = [
        ('1m', 15, 0.8, 0.08, 0.4),  # Averages were ~0.5% comp, 0.04% body
        ('5m', 12, 1.5, 0.15, 1.0),  # Averages were ~1.0% comp, 0.09% body
        ('15m', 10, 2.0, 0.20, 2.0)  # Averages were ~1.4% comp, 0.13% body
    ]
    
    console.print("\n[bold]========== VALIDATING THE REVERSE-ENGINEERED PATTERN ==========[/bold]")
    
    for tf, win, comp, body, target in configs:
        console.print(f"\n[bold cyan]► Testing Discovered Math on {tf} (Last 7 Days)...[/bold cyan]")
        df = load_data("BTCUSDT", tf, days=7)
        if df is None: continue
        
        signals = test_pattern_validity(df, tf, win, comp, body, target)
        
        if not signals:
            console.print("No valid starting points detected.")
            continue
            
        total = len(signals)
        wins = sum([1 for s in signals if s['success']])
        win_rate = wins / total * 100 if total > 0 else 0
        
        table = Table(title=f"{tf} Forward Test Results")
        table.add_column("Measurement", style="cyan")
        table.add_column("Result", style="bold white")
        
        table.add_row("Total Predicted Starting Points", str(total))
        table.add_row(f"Successful Trends Caught (>{target}% Gain)", f"[green]{wins}[/green]")
        table.add_row("Validation (Win Rate)", f"{win_rate:.2f}%")
        
        avg_mfe = sum([s['max_gain_pct'] for s in signals]) / total
        table.add_row("Avg Potential Trend Peak", f"[yellow]{avg_mfe:.2f}%[/yellow]")
        
        console.print(table)
        
        if total > 0:
            console.print("[bold]Latest Predicted Starting Point:[/bold]")
            last_sig = signals[-1]
            status = "[green]SUCCESS[/green]" if last_sig['success'] else "[red]FAILED[/red]"
            local_time = last_sig['time'].tz_convert('Asia/Karachi')
            console.print(f"  Time: {local_time} | Entry: ${last_sig['price']:,.2f} | Result: {status} (Max {last_sig['max_gain_pct']:.2f}% gain)")

if __name__ == "__main__":
    run()
