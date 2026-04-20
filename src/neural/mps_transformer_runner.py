import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from src.neural.transformer_brain import ChronosTransformer, prepare_transformer_df
from rich.console import Console
from rich.table import Table

console = Console()

class MPSTransformerRunner:
    def __init__(self, symbols, model_path="models/transformer_xray_oracle_v1.pth"):
        self.symbols = symbols
        self.seq_len = 50
        self.brain = ChronosTransformer(feature_dim=10, seq_len=self.seq_len, num_heads=5, hidden_dim=160)
        
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path))
            console.print(f"[bold green]MPS v1.6.1:[/bold green] X-Ray Grid ready with {len(symbols)} assets.")
        
        self.brain.eval()
        if torch.cuda.is_available(): self.brain.to("cuda")
        
        self.fee = 0.0008
        self.target_cols = [
            'vel', 'vol_rel', 'u_wick', 'l_wick', 'z_score', 
            'mtf_proxy', 'hi_rel', 'lo_rel', 
            'buy_wall_prox', 'sell_wall_prox'
        ]

    async def run_simulation(self):
        console.print(f"[bold cyan]PHASE 9:[/bold cyan] Executing X-Ray Sequence Audit (Gate: 30%)...")
        
        total_pnl = []
        hits = 0
        total_signals = 0
        
        for sym in self.symbols:
            df = pd.read_csv(f"data/raw/{sym}_5m.csv")
            df = prepare_transformer_df(df)
            feature_matrix = df[self.target_cols].values
            
            sim_len = 2500
            start_idx = len(df) - sim_len
            
            for i in range(start_idx, len(df) - 10):
                seq = feature_matrix[i-self.seq_len:i]
                seq_tensor = torch.from_numpy(np.array(seq)).float().unsqueeze(0)
                if torch.cuda.is_available(): seq_tensor = seq_tensor.to("cuda")
                
                with torch.no_grad():
                    logits = self.brain(seq_tensor, return_logits=True)
                    probs = torch.softmax(logits, dim=-1)
                    
                long_logit = logits[0, 1].item()
                short_logit = logits[0, 2].item()
                
                # Lowered Conviction Gate to 30% to capture Neural Leans
                if probs[0, 1] > 0.30 or probs[0, 2] > 0.30:
                    action = 1 if long_logit > short_logit else 2
                    
                    price = float(df.iloc[i]['close'])
                    exit_px = float(df.iloc[i+6]['close']) 
                    
                    raw_pnl = (exit_px - price) / price if action == 1 else (price - exit_px) / price
                    net_pnl = raw_pnl - self.fee
                    
                    total_pnl.append(net_pnl)
                    total_signals += 1
                    if net_pnl > 0: hits += 1
                    
                if (i - start_idx) % 500 == 0:
                    console.print(f"  {sym} Audit: {i-start_idx}/{sim_len} | Signals: {total_signals}")

        table = Table(title="MPS v1.6.1 X-RAY AUDIT")
        table.add_column("Metric", style="cyan")
        table.add_column("Result", style="green")
        
        win_rate = (hits / total_signals * 100) if total_signals > 0 else 0
        mean_pnl = np.mean(total_pnl) if len(total_pnl) > 0 else 0
        
        table.add_row("Total Temporal Signals", f"{total_signals:,}")
        table.add_row("Mean Net P&L", f"{mean_pnl*100:+.4f}%")
        table.add_row("X-Ray Win Rate", f"{win_rate:.2f}%")
        
        console.print(table)

if __name__ == "__main__":
    import asyncio
    runner = MPSTransformerRunner(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    asyncio.run(runner.run_simulation())
