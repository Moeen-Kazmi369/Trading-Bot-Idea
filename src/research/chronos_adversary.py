import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

class ProjectChronosAdversary:
    """
    Project Chronos - Adversarial Data Harvester
    Goal: Extract 'Trap Snapshots'—moments that look quiet but FAIL to trend.
    This provides the 'Negative' training data for the LLM.
    """
    
    def __init__(self, symbol="BTCUSDT", timeframe="5m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_path = f"data/raw/{symbol}_{timeframe}.csv"
        self.output_dir = "research/chronos_dataset"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_df(self):
        if not os.path.exists(self.data_path): return None
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def harvest_traps(self, fail_threshold_pct=0.5, lookback=50):
        df = self.load_df()
        if df is None: return
        
        future_lookahead = 48
        df['future_max_gain'] = df['high'].shift(-future_lookahead).rolling(window=future_lookahead).max()
        df['gain_pct'] = (df['future_max_gain'] - df['close']) / df['close'] * 100
        
        # A Trap: Price is very quiet (compression), but gain is < 0.5% (failure)
        df['compression'] = (df['high'].rolling(window=lookback).max() - df['low'].rolling(window=lookback).min()) / df['close']
        
        # Threshold for 'Quiet' is based on the top 20% most compressed moments
        compression_threshold = df['compression'].quantile(0.2)
        
        traps = df[(df['compression'] <= compression_threshold) & (df['gain_pct'] <= fail_threshold_pct)].copy()
        
        # Downsample traps to match the number of positive samples (around 413)
        if len(traps) > 420:
            traps = traps.sample(n=420, random_state=42)
            
        console.print(f"\n[bold red]PROJECT CHRONOS:[/bold red] Captured [bold yellow]{len(traps)}[/bold yellow] Market Traps (Negative Samples).")
        
        for idx in traps.index:
            if idx < lookback: continue
            pre_data = df.iloc[idx - lookback : idx].copy()
            snapshot_file = f"{self.output_dir}/trap_{idx}.csv"
            pre_data[['open', 'high', 'low', 'close', 'volume']].to_csv(snapshot_file, index=False)
            
        return traps

if __name__ == "__main__":
    adversary = ProjectChronosAdversary("BTCUSDT", "5m")
    summary = adversary.harvest_traps()
    
    if summary is not None:
        console.print(f"\n[bold green]ADVERSARIAL DATA READY:[/bold green] All {len(summary)} Trap Snapshots saved.")
        console.print("[bold cyan]PHASE 1 COMPLETE.[/bold cyan] We now have a balanced 'Success vs. Failure' dataset.")
