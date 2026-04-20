import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

def calculate_anomaly_score(df: pd.DataFrame) -> float:
    """
    FRONTIER QUANTITATIVE RESEARCH - V2 SENSITIVITY UPGRADE
    Detects Launchpads using Shadow Absorption, VSA Compression, and Momentum Sparks.
    """
    if len(df) < 20: return 0.0

    df = df.copy()
    # Basic candle math
    df['body'] = abs(df['close'] - df['open'])
    df['spread'] = df['high'] - df['low']
    df['lower_wick'] = np.where(df['close'] > df['open'], 
                                 df['open'] - df['low'], 
                                 df['close'] - df['low'])
    
    # Baselines
    df['vol_sma'] = df['volume'].rolling(window=14).mean()
    
    # 1. Shadow Absorption (40 pts)
    df['wick_to_body'] = df['lower_wick'] / (df['body'] + 1e-9)
    df['vol_surge'] = df['volume'] / (df['vol_sma'] + 1e-9)
    df['abs_score'] = np.where((df['wick_to_body'] > 1.5) & (df['vol_surge'] > 1.5), 40, 0)

    # 2. VSA Compression (30 pts)
    df['spread_change'] = df['spread'] / (df['spread'].shift(1) + 1e-9)
    df['vol_change'] = df['volume'] / (df['volume'].shift(1) + 1e-9)
    df['vsa_score'] = np.where((df['vol_change'] > 1.2) & (df['spread_change'] < 0.8), 30, 0)

    # 3. Momentum Spark (30 pts)
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    upper_band = rolling_mean + (1.5 * rolling_std)
    df['spark'] = np.where(df['close'] > upper_band, 30, 0)

    # Latest signals sum
    recent_signals = df.tail(5)
    peak_absorption = recent_signals['abs_score'].max()
    peak_vsa = recent_signals['vsa_score'].max()
    peak_spark = recent_signals['spark'].max()
    trend_bonus = 10 if df['close'].iloc[-1] > df['close'].iloc[-3] else 0
    
    total_score = peak_absorption + peak_vsa + peak_spark + trend_bonus
    return float(np.clip(total_score, 0, 100))

class ChronosValidationLab:
    def __init__(self, dataset_dir="research/chronos_dataset"):
        self.dataset_dir = dataset_dir
        
    def validate(self):
        launch_files = [f for f in os.listdir(self.dataset_dir) if f.startswith("launchpad_")]
        trap_files = [f for f in os.listdir(self.dataset_dir) if f.startswith("trap_")]
        
        console.print("[bold cyan]LAB:[/bold cyan] Validating v2 'Shadow Absorption' formula...")
        
        pos_scores = [calculate_anomaly_score(pd.read_csv(os.path.join(self.dataset_dir, f))) for f in launch_files]
        neg_scores = [calculate_anomaly_score(pd.read_csv(os.path.join(self.dataset_dir, f))) for f in trap_files]
        
        avg_pos, avg_neg = np.mean(pos_scores), np.mean(neg_scores)
        
        # Detection Threshold: We'll set it at 70 pts
        threshold = 70
        pos_detections = sum(1 for s in pos_scores if s >= threshold)
        neg_detections = sum(1 for s in neg_scores if s >= threshold)
        
        table = Table(title="CHRONOS VALIDATION v2 - SHADOW ABSORPTION")
        table.add_column("Metric", style="magenta")
        table.add_column("Result", style="bold white")
        table.add_row("Avg Score for Trend Launches", f"[green]{avg_pos:.2f}%[/green]")
        table.add_row("Avg Score for Market Traps", f"[red]{avg_neg:.2f}%[/red]")
        table.add_row(f"Recall Rate (Score >= {threshold})", f"{(pos_detections/len(launch_files)*100):.1f}%")
        table.add_row(f"False Alarm Rate (Score >= {threshold})", f"{(neg_detections/len(trap_files)*100):.1f}%")
        
        console.print(table)
        
        if (pos_detections/len(launch_files)) > (neg_detections/len(trap_files)) * 2:
            console.print("[bold green]FRONTIER VALIDATION SUCCESSFUL:[/bold green] Signal-to-Noise ratio is healthy.")
        else:
            console.print("[bold yellow]REFINEMENT NEEDED:[/bold yellow] Pattern is still catching too much noise.")

if __name__ == "__main__":
    lab = ChronosValidationLab()
    lab.validate()
