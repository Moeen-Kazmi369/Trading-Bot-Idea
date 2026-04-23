import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from src.neural.transformer_brain import UniversalOracleV2
from src.neural.universal_manifold import prepare_30d_universal_manifold
from rich.console import Console
from rich.table import Table

console = Console()

class UniversalAuditRunner:
    """
    Project Chronos - Triple-Timeframe Verdict (v2.0)
    Validates Universality across Micro, Tactical, and Macro grids.
    """
    def __init__(self, model_path="models/universal_oracle_snap_300.pth", initial_balance=400.00):
        self.seq_len = 50
        self.brain = UniversalOracleV2(feature_dim=30, hidden_dim=256)
        
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path))
            console.print(f"[bold green]SENTINEL v2.0:[/bold green] Universal Oracle Loaded.")
        
        self.brain.eval()
        if torch.cuda.is_available(): self.brain.to("cuda")
        
        self.initial_balance = initial_balance
        self.fee_rate = 0.0008
        self.conviction_gate = 0.60
        
        self.target_cols = [
            'rsi', 'stoch_k', 'stoch_d', 'willr', 'mfi', 'cci', 'uo', 'bop', 'ao',
            'ema8_dist', 'ema21_dist', 'ema50_dist', 'ema200_dist', 'macd_hist', 'adx', 'ichi_span_a',
            'atr_norm', 'bb_up_dist', 'bb_lo_dist', 'u_wick', 'l_wick', 'vol_rel',
            'c_timeframe', 'c_fee'
        ]
        # Padding for 30D
        while len(self.target_cols) < 30: self.target_cols.append('rsi')

    async def run_audit(self, timeframe_mins):
        console.print(f"[bold cyan]AUDIT {timeframe_mins}m:[/bold cyan] Scanning Grid (Gate 60%)...")
        
        df_raw = pd.read_csv("data/raw/BTCUSDT_5m.csv")
        # Resample logic
        df_tf = df_raw.iloc[::(timeframe_mins//5)].copy() if timeframe_mins > 5 else df_raw.copy()
        df, _ = prepare_30d_universal_manifold(df_tf, timeframe_mins=timeframe_mins, fee_rate=self.fee_rate)
        
        features = df[self.target_cols].values
        balance = self.initial_balance
        trades = 0
        wins = 0
        
        # Audit Window: Adaptive to available history (Max 1000 bars for speed)
        simulation_len = min(1000, len(df) - self.seq_len - 15)
        start_idx = len(df) - simulation_len
        
        for i in range(start_idx, len(df) - 10):
            seq = features[i-self.seq_len:i]
            cond = np.array([[df.iloc[i]['c_timeframe'], df.iloc[i]['c_fee'] * 100]])
            
            seq_t = torch.from_numpy(seq).float().unsqueeze(0)
            cond_t = torch.from_numpy(cond).float()
            if torch.cuda.is_available(): seq_t, cond_t = seq_t.to("cuda"), cond_t.to("cuda")
            
            with torch.no_grad():
                probs = self.brain(seq_t, cond_t)
            
            p_long = probs[0, 1].item()
            p_short = probs[0, 2].item()
            
            if p_long > self.conviction_gate or p_short > self.conviction_gate:
                action = 1 if p_long > p_short else 2
                px = float(df.iloc[i]['close'])
                exit_px = float(df.iloc[i+3]['close']) # 3-bar hold
                
                roi = (exit_px - px) / px if action == 1 else (px - exit_px) / px
                total_fee = balance * self.fee_rate * 2
                pnl = (balance * roi) - total_fee
                balance += pnl
                trades += 1
                if pnl > 0: wins += 1
        
        return balance, trades, wins

if __name__ == "__main__":
    import asyncio
    runner = UniversalAuditRunner()
    
    table = Table(title="PROJECT CHRONOS: TRIPLE-TIMEFRAME VERDICT")
    table.add_column("Timeframe", style="cyan")
    table.add_column("Trades", style="green")
    table.add_column("Final Balance", style="yellow")
    table.add_column("ROI %", style="magenta")

    for tf in [5, 60, 240]:
        bal, tr, wn = asyncio.run(runner.run_audit(tf))
        roi = ((bal - 400)/400)*100
        table.add_row(f"{tf}m", str(tr), f"${bal:.2f}", f"{roi:+.2f}%")
        
    console.print(table)
