import os
import pandas as pd
import numpy as np
import torch
import time
from src.neural.imitation_learner import PolicyNetwork
from rich.console import Console
from rich.table import Table

console = Console()

class MPSRunner:
    def __init__(self, symbols, model_path="models/evolved_brain_v1.pth"):
        self.symbols = symbols
        self.brain = PolicyNetwork(input_dim=5, output_dim=4)
        if torch.cuda.is_available(): self.brain.to("cuda")
        self.brain.load_state_dict(torch.load(model_path))
        self.brain.eval()
        
        self.datafeeds = {}
        self.results_path = "research/mps_audit_500.csv"
        self.fee_friction = 0.0008
        
        for sym in symbols:
            path = f"data/raw/{sym}_5m.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                self.datafeeds[sym] = df.tail(20000).reset_index(drop=True)
                
        console.print(f"[bold green]MPS v1.3:[/bold green] Grid initialized with {len(self.datafeeds)} assets.")

    async def run_simulation(self):
        console.print("[bold cyan]PHASE 4:[/bold cyan] Initiating Neural Logit Deep-Dive...")
        
        start_time = time.time()
        total_bars = 0
        all_results = []
        
        # Determine the shared length for synchronized simulation
        max_idx = min([len(df) for df in self.datafeeds.values()]) - 10 
        
        for idx in range(50, max_idx):
            for sym, df in self.datafeeds.items():
                total_bars += 1
                window = df.iloc[idx-5:idx]
                price = float(df.iloc[idx]['close'])
                
                state = [
                    (window['close'].iloc[-1] / window['close'].iloc[0]) - 1,
                    (window['volume'].iloc[-1] / window['volume'].mean()) - 1,
                    (window['high'].max() - window['low'].min()) / window['close'].mean(),
                    0, 0
                ]
                
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
                if torch.cuda.is_available(): state_tensor = state_tensor.to("cuda")
                
                with torch.no_grad():
                    # Pass through the Sequential block UP TO the final logit layer (Linear 2)
                    # self.net[4] is the last Linear layer. Softmax is self.net[5]
                    # We take net[:5] to get the 4 raw logits
                    logits = self.brain.net[:5](state_tensor).cpu().numpy()[0]
                
                # 3. Logit & Magnitude Extraction
                long_logit = logits[1]
                short_logit = logits[2]
                lean_action = 1 if long_logit > short_logit else 2
                magnitude = abs(long_logit - short_logit)
                
                # 4. Result Solving
                exit_px = float(df.iloc[idx+6]['close']) # 30 min outcome
                raw_pnl = (exit_px - price) / price if lean_action == 1 else (price - exit_px) / price
                net_pnl = raw_pnl - self.fee_friction
                
                all_results.append({
                    'symbol': sym,
                    'timestamp': df.iloc[idx]['timestamp'],
                    'direction': "LONG" if lean_action == 1 else "SHORT",
                    'magnitude': magnitude,
                    'net_pnl': net_pnl
                })
            
            if idx % 1000 == 0:
                elapsed = time.time() - start_time
                bps = total_bars / elapsed
                console.print(f"  [dim]Progress: {idx}/{max_idx} | Speed: {bps:.0f} Bars/Sec | Signals: {len(all_results)}[/dim]")

        df_audit = pd.DataFrame(all_results)
        df_audit.to_csv(self.results_path, index=False)
        self.display_summary(df_audit, time.time() - start_time, total_bars)

    def display_summary(self, df, elapsed, total_bars):
        table = Table(title="MPS v1.3 AUDIT RESULTS")
        table.add_column("Metric", style="cyan")
        table.add_column("Result", style="white")
        table.add_row("Total Signals Captured", f"{len(df):,}")
        table.add_row("Mean Net P&L", f"{df['net_pnl'].mean()*100:+.4f}%")
        table.add_row("Raw Hit Rate", f"{(df[df['net_pnl'] > 0].shape[0] / len(df) * 100):.2f}%")
        console.print("\n", table)

if __name__ == "__main__":
    import asyncio
    top_4_coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    runner = MPSRunner(top_4_coins)
    asyncio.run(runner.run_simulation())
