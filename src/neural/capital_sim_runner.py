import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from src.neural.transformer_brain import HybridSentinel, prepare_15d_manifold
from rich.console import Console
from rich.table import Table

console = Console()

class HybridSimRunner:
    """
    Project Chronos - Final Hybrid Verdict ($400)
    Implements the Phase 11.3 Volatility Shield + Forensic Dashboard.
    """
    def __init__(self, symbols, model_path="models/transformer_hybrid_v17.pth", initial_balance=400.00):
        self.symbols = symbols
        self.seq_len = 50
        self.brain = HybridSentinel(feature_dim=15, seq_len=self.seq_len, hidden_dim=128)
        
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path))
            console.print(f"[bold green]SENTINEL v1.7.1:[/bold green] Hybrid Oracle Active (Capital: ${initial_balance}).")
        
        self.brain.eval()
        if torch.cuda.is_available(): self.brain.to("cuda")
        
        self.balance = initial_balance
        self.start_balance = initial_balance
        self.fee_rate = 0.0008
        self.conviction_gate = 0.60 # The Hunter Setting
        
        self.target_cols = [
            'vel', 'vol_rel', 'u_wick', 'l_wick', 'z_score', 'mtf_proxy', 'hi_rel', 'lo_rel',
            'buy_wall_prox', 'sell_wall_prox', 'rsi', 'macd_hist', 'atr', 'bb_up_dist', 'bb_lo_dist'
        ]

    async def run_verdict(self):
        console.print(f"[bold cyan]PROJECT CHRONOS:[/bold cyan] Executing Final $400 Shielded Verdict...")
        
        total_trades = 0
        wins = 0
        losses = 0
        shield_pass_count = 0
        total_commission = 0
        
        for sym in self.symbols:
            df = pd.read_csv(f"data/raw/{sym}_5m.csv")
            df = prepare_15d_manifold(df)
            features = df[self.target_cols].values
            
            # Audit Window: Last 3 Months (~25,000 bars)
            start_idx = len(df) - 25000
            
            for i in range(start_idx, len(df) - 10):
                row = df.iloc[i]
                
                # --- THE VOLATILITY SHIELD (Permission Gate) ---
                if HybridSentinel.verify_shield(row):
                    shield_pass_count += 1
                    
                    # --- NEURAL TRIGGER (Strike Decision) ---
                    seq = features[i-self.seq_len:i]
                    seq_tensor = torch.from_numpy(np.array(seq)).float().unsqueeze(0)
                    if torch.cuda.is_available(): seq_tensor = seq_tensor.to("cuda")
                    
                    with torch.no_grad():
                        probs = self.brain(seq_tensor)
                    
                    p_long = probs[0, 1].item()
                    p_short = probs[0, 2].item()
                    
                    if p_long > self.conviction_gate or p_short > self.conviction_gate:
                        action = 1 if p_long > p_short else 2
                        price = float(df.iloc[i]['close'])
                        exit_px = float(df.iloc[i+6]['close']) 
                        
                        raw_roi = (exit_px - price) / price if action == 1 else (price - exit_px) / price
                        
                        entry_fee = self.balance * self.fee_rate
                        exit_fee = self.balance * (1 + raw_roi) * self.fee_rate
                        total_fee = entry_fee + exit_fee
                        
                        trade_pnl = (self.balance * raw_roi) - total_fee
                        self.balance += trade_pnl
                        
                        total_trades += 1
                        total_commission += total_fee
                        if trade_pnl > 0: wins += 1
                        else: losses += 1
                        
            console.print(f"  {sym} Verdict Concluded | Current Balance: ${self.balance:.2f}")

        roi_total = ((self.balance - self.start_balance) / self.start_balance) * 100
        
        table = Table(title="PROJECT CHRONOS: FINAL HYBRID REPORT")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Initial Capital", f"${self.start_balance:.2f}")
        table.add_row("Shield-Pass Opportunities", str(shield_pass_count))
        table.add_row("Total Trades Executed", str(total_trades))
        table.add_row("Win/Loss Ratio", f"{wins}/{losses}")
        table.add_row("Net Profit/Loss", f"${(self.balance - self.start_balance):+.2f}")
        table.add_row("Final ROI %", f"{roi_total:+.2f}%")
        
        console.print(table)

if __name__ == "__main__":
    import asyncio
    runner = HybridSimRunner(["BTCUSDT", "ETHUSDT"])
    asyncio.run(runner.run_verdict())
