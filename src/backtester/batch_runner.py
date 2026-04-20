import os
import pandas as pd
from src.strategies.order_block import OrderBlockDetector
from src.strategies.compression_accumulator import CompressionAccumulator
from src.backtester.hybrid_engine import HybridEngine
from rich.console import Console
from rich.table import Table

console = Console()

class BatchRunner:
    def __init__(self, data_path="data/raw/"):
        self.data_path = data_path
        self.results = []

    def run_all(self, interval="1m"):
        files = [f for f in os.listdir(self.data_path) if f.endswith(f"_{interval}.csv")]
        console.print(f"[bold yellow]Found {len(files)} coins for {interval} batch testing...[/bold yellow]")
        
        for file in files:
            symbol = file.split("_")[0]
            console.print(f"[cyan]Testing {symbol}...[/cyan]")
            
            try:
                df = pd.read_csv(os.path.join(self.data_path, file))
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None) # Make naive for simplicity in simulation
                if len(df) < 100: continue
                
                # 1. Detect OBs
                detector = OrderBlockDetector(df)
                bull_obs = detector.find_bullish_order_blocks()
                bear_obs = detector.find_bearish_order_blocks()
                
                # 2. Detect Accumulator Signals
                accumulator = CompressionAccumulator(df)
                signals = accumulator.find_signals()
                
                # 3. Run Hybrid Engine
                engine = HybridEngine(df, bull_obs, bear_obs)
                trades_df = engine.run(signals)
                
                if isinstance(trades_df, str): # "No trades taken"
                    self.results.append({'symbol': symbol, 'trades': 0, 'wins': 0, 'pnl': 0, 'win_rate': 0})
                else:
                    wins = len(trades_df[trades_df['result'] == 'WIN'])
                    self.results.append({
                        'symbol': symbol,
                        'trades': len(trades_df),
                        'wins': wins,
                        'pnl': trades_df['pnl'].sum(),
                        'win_rate': (wins / len(trades_df)) * 100 if len(trades_df) > 0 else 0
                    })
            except Exception as e:
                console.print(f"[bold red]Failed to process {symbol}: {e}[/bold red]")
                
        self._show_master_report()

    def _show_master_report(self):
        table = Table(title="GLOBAL STRESS TEST (6 MONTHS, TOP 20 COINS)")
        table.add_column("Symbol", style="cyan")
        table.add_column("Trades", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Total PnL", justify="right", style="bold green")
        
        total_pnl = 0
        for r in self.results:
            pnl_color = "green" if r['pnl'] > 0 else "red"
            table.add_row(
                r['symbol'], 
                str(r['trades']), 
                f"{r['win_rate']:.2f}%", 
                f"[{pnl_color}]{r['pnl']:.2f}[/{pnl_color}]"
            )
            total_pnl += r['pnl']
            
        console.print(table)
        console.print(f"\n[bold green]FINAL AGGREGATE PNL: {total_pnl:.2f} UNITS[/bold green]")

if __name__ == "__main__":
    runner = BatchRunner()
    
    console.print("[bold magenta]=== STARTING 5-MINUTE STRESS TEST ===[/bold magenta]")
    runner.run_all(interval="5m")
    
    runner.results = [] # Reset for next run
    console.print("\n[bold magenta]=== STARTING 15-MINUTE STRESS TEST ===[/bold magenta]")
    runner.run_all(interval="15m")
