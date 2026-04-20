import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

class ProjectChronosHarvester:
    """
    Project Chronos - Frontier Level Data Harvester
    Goal: Extract 'Snapshots' of the market state immediately BEFORE major trends.
    These snapshots will be used for LLM Pattern Discovery.
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

    def harvest_snapshots(self, trend_threshold_pct=2.0, lookback=50):
        df = self.load_df()
        if df is None: return
        
        # Calculate Future Returns to find the 'Success Cases'
        # Check the next 4 hours (48 bars on 5m) for a significant move
        future_lookahead = 48
        df['future_max_gain'] = df['high'].shift(-future_lookahead).rolling(window=future_lookahead).max()
        df['gain_pct'] = (df['future_max_gain'] - df['close']) / df['close'] * 100
        
        # Filter for the 'Starting Points' of major moves
        # A starting point: Gain > Threshold AND it's a local minimum in the last 20 candles
        starts = df[(df['gain_pct'] >= trend_threshold_pct) & 
                    (df['low'] == df['low'].rolling(window=20).min())].copy()
        
        console.print(f"\n[bold green]PROJECT CHRONOS:[/bold green] Discovered [bold cyan]{len(starts)}[/bold cyan] major Trend Launchpads.")
        
        snapshots = []
        for idx in starts.index:
            if idx < lookback: continue
            
            # The Pre-Trend Data Manifold
            pre_data = df.iloc[idx - lookback : idx].copy()
            
            # Feature extraction for the manifold
            metrics = {
                'timestamp': df.iloc[idx]['timestamp'],
                'launch_price': df.iloc[idx]['close'],
                'gain_potential': df.iloc[idx]['gain_pct'],
                'volatility_compression': (pre_data['high'].max() - pre_data['low'].min()) / df.iloc[idx]['close'],
                'relative_volume_avg': pre_data['volume'].mean(),
                'price_velocity': (pre_data['close'].iloc[-1] - pre_data['close'].iloc[0]) / pre_data['close'].iloc[0]
            }
            
            # Store raw price action for the LLM
            snapshot_file = f"{self.output_dir}/launchpad_{idx}.csv"
            pre_data[['open', 'high', 'low', 'close', 'volume']].to_csv(snapshot_file, index=False)
            
            snapshots.append(metrics)
            
        return pd.DataFrame(snapshots)

if __name__ == "__main__":
    harvester = ProjectChronosHarvester("BTCUSDT", "5m")
    summary = harvester.harvest_snapshots()
    
    if summary is not None:
        table = Table(title="CHRONOS DATASET SUMMARY")
        table.add_column("Launch Time", style="magenta")
        table.add_column("Potential Gain", style="green")
        table.add_column("Pre-Trend Squeeze", style="cyan")
        table.add_column("Price Velocity", style="yellow")
        
        for _, row in summary.iloc[-10:].iterrows():
            table.add_row(
                str(row['timestamp']),
                f"{row['gain_potential']:.2f}%",
                f"{row['volatility_compression']:.4f}",
                f"{row['price_velocity']:.4f}"
            )
            
        console.print(table)
        console.print(f"\n[bold yellow]DATA READY:[/bold yellow] All {len(summary)} Launchpad Snapshots saved to research/chronos_dataset/")
