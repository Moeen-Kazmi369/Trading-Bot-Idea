import os
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

def optimized_audit():
    data_path = "data/raw/"
    files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    
    table = Table(title="STRATEGY VERIFICATION AUDIT (ALL TIMEFRAMES)")
    table.add_column("Coin/TF", style="cyan")
    table.add_column("Bullish (GRG)", style="green")
    table.add_column("Bearish (RGR)", style="red")
    table.add_column("Total Found", style="yellow")
    table.add_column("Executed (Passed Filter)", style="magenta")

    grand_bull = 0
    grand_bear = 0
    grand_exec = 0

    for file in files:
        df = pd.read_csv(os.path.join(data_path, file))
        if len(df) < 100: continue
        
        # Vectorized Pattern Detection
        close = df['close'].values
        open_ = df['open'].values
        high = df['high'].values
        low = df['low'].values
        
        is_green = close > open_
        is_red = close < open_
        
        # Bullish Pattern: Green(i-2), Red(i-1), Green(i)
        # Conditions: 
        # 1. Pattern: Green, Red, Green
        # 2. C2.Low > C1.Low AND C3.Low > C2.Low
        # 3. C3.High > MaxHigh(last 20)
        
        bull_pattern = (is_green[:-2]) & (is_red[1:-1]) & (is_green[2:])
        low_condition = (low[1:-1] > low[:-2]) & (low[2:] > low[1:-1])
        
        # 20-candle high check (vectorized rolling max)
        # This is a bit complex for pure numpy but we can approximate for audit or use pandas rolling
        rolling_high = df['high'].shift(1).rolling(window=20).max().values[2:]
        breakout_condition = high[2:] > rolling_high
        
        bull_masks = bull_pattern & low_condition & breakout_condition
        bull_indices = np.where(bull_masks)[0] + 2 # Adjust for slicing
        
        # Bearish Pattern: Red(i-2), Green(i-1), Red(i)
        bear_pattern = (is_red[:-2]) & (is_green[1:-1]) & (is_red[2:])
        high_condition = (high[1:-1] < high[:-2]) & (high[2:] < high[1:-1])
        rolling_low = df['low'].shift(1).rolling(window=20).min().values[2:]
        breakdown_condition = low[2:] < rolling_low
        
        bear_masks = bear_pattern & high_condition & breakdown_condition
        bear_indices = np.where(bear_masks)[0] + 2
        
        # Executed trades (approx based on our 3% rule logic)
        # For audit, we'll assume a % pass the "Gemini" filter (roughly 10-15%)
        # But we'll count the verified setups from the backtest runner logs if possible
        # Or just show the raw pattern hits as the user requested
        
        b_count = len(bull_indices)
        r_count = len(bear_indices)
        
        grand_bull += b_count
        grand_bear += r_count
        
        table.add_row(
            file.replace(".csv", ""), 
            str(b_count), 
            str(r_count), 
            str(b_count + r_count),
            "Scanning..." # This is specific to the Hybrid logic
        )

    console.print(table)
    console.print(f"\n[bold green]GRAND TOTAL BULLISH:[/bold green] {grand_bull}")
    console.print(f"[bold red]GRAND TOTAL BEARISH:[/bold red] {grand_bear}")
    console.print(f"[bold yellow]TOTAL PATTERN MATCHES:[/bold yellow] {grand_bull + grand_bear}")

if __name__ == "__main__":
    optimized_audit()
