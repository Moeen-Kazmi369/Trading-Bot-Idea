import os
import pandas as pd
from src.strategies.order_block import OrderBlockDetector
from rich.console import Console
from rich.table import Table

console = Console()

def run_audit():
    data_path = "data/raw/"
    files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    
    total_bullish = 0
    total_bearish = 0
    total_setup_attempts = 0 # Those that passed our 20-candle breakout rule
    
    table = Table(title="STRATEGY VERIFICATION AUDIT (ALL TIMEFRAMES)")
    table.add_column("Coin", style="cyan")
    table.add_column("TF", style="magenta")
    table.add_column("Bullish OBs", style="green")
    table.add_column("Bearish OBs", style="red")
    table.add_column("Total Pattern Hits", style="yellow")

    for file in files:
        symbol = file.split("_")[0]
        interval = file.split("_")[1].replace(".csv", "")
        
        df = pd.read_csv(os.path.join(data_path, file))
        detector = OrderBlockDetector(df)
        
        # We detect the "Raw" OBs before the Hybrid/Gemini filtering
        bull_obs = detector.find_bullish_order_blocks()
        bear_obs = detector.find_bearish_order_blocks()
        
        b_count = len(bull_obs)
        r_count = len(bear_obs)
        total_pattern = b_count + r_count
        
        total_bullish += b_count
        total_bearish += r_count
        total_setup_attempts += total_pattern
        
        # Only show a few as summary if list is too long, but we keep total
        if total_pattern > 0:
            table.add_row(symbol, interval, str(b_count), str(r_count), str(total_pattern))

    console.print(table)
    console.print(f"\n[bold green]GRAND TOTAL BULLISH OBs FOUND:[/bold green] {total_bullish}")
    console.print(f"[bold red]GRAND TOTAL BEARISH OBs FOUND:[/bold red] {total_bearish}")
    console.print(f"[bold yellow]GRAND TOTAL PATTERN MATCHS (GRG/RGR):[/bold yellow] {total_setup_attempts}")

if __name__ == "__main__":
    run_audit()
