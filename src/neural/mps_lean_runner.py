import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from src.neural.transformer_brain import ChronosTransformer, prepare_transformer_df
from rich.console import Console
from rich.table import Table

console = Console()

class MPSLeanRunner:
    """
    MPS v1.7 - Neural Lean Auditor
    Bypasses Softmax Conviction Gating to measure the raw directional bias.
    """
    def __init__(self, symbols, model_path="models/transformer_xray_oracle_v1.pth"):
        self.symbols = symbols
        self.seq_len = 50
        self.brain = ChronosTransformer(feature_dim=10, seq_len=self.seq_len, num_heads=5, hidden_dim=160)
        
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path))
            console.print(f"[bold green]MPS v1.7:[/bold green] Row Neural Lean Grid ready.")
        
        self.brain.eval()
        if torch.cuda.is_available(): self.brain.to("cuda")
        
        self.fee = 0.0008
        self.target_cols = ['vel', 'vol_rel', 'u_wick', 'l_wick', 'z_score', 'mtf_proxy', 'hi_rel', 'lo_rel', 'buy_wall_prox', 'sell_wall_prox']

    async def run_simulation(self):
        console.print(f"[bold cyan]PHASE 10:[/bold cyan] Executing Raw Neural Lean Audit...")
        
        total_pnl = []
        hits = 0
        total_signals = 0
        
        for sym in self.symbols:
            df = prepare_transformer_df(pd.read_csv(f"data/raw/{sym}_5m.csv"))
            fm = df[self.target_cols].values
            
            # We audit a high-intensity window
            start_idx = len(df) - 2000 
            
            for i in range(start_idx, len(df) - 10):
                seq = fm[i-self.seq_len:i]
                seq_tensor = torch.from_numpy(np.array(seq)).float().unsqueeze(0)
                if torch.cuda.is_available(): seq_tensor = seq_tensor.to("cuda")
                
                with torch.no_grad():
                    logits = self.brain(seq_tensor, return_logits=True)
                
                # NO GATING: We take the best directional lean (Index 1 or 2)
                # We compare Index 0 (STAY) vs others
                long_lean = logits[0, 1].item()
                short_lean = logits[0, 2].item()
                stay_lean = logits[0, 0].item()
                
                # If ANY movement is preferred over STAY (even if stay is 100 on softmax)
                # We force the bot to reveal its "Alpha Lean"
                action = 1 if long_lean > short_lean else 2
                
                price = float(df.iloc[i]['close'])
                exit_px = float(df.iloc[i+6]['close']) 
                net_pnl = ((exit_px - price) / price if action == 1 else (price - exit_px) / price) - self.fee
                
                total_pnl.append(net_pnl)
                total_signals += 1
                if net_pnl > 0: hits += 1
                
            console.print(f"  {sym} Audit Complete | LEAN WR: {hits/total_signals*100:.1f}%")

        table = Table(title="MPS v1.7 RAW LEAN VERDICT")
        table.add_row("Total Lean Signals", f"{total_signals:,}")
        table.add_row("X-Ray Lean Win Rate", f"{hits / total_signals * 100:.2f}%")
        table.add_row("Mean Net Lean P&L", f"{np.mean(total_pnl)*100:+.4f}%")
        console.print(table)

if __name__ == "__main__":
    import asyncio
    runner = MPSLeanRunner(["BTCUSDT", "ETHUSDT"])
    asyncio.run(runner.run_simulation())
