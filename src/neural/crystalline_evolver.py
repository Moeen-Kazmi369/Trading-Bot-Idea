import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import random
from src.neural.transformer_brain import ChronosTransformer, prepare_transformer_df
from rich.console import Console

console = Console()

class CrystallineEvolver:
    """
    Project Chronos - Phase 8.2 Deep Cooling Forge
    Implements Manifold Dropout + DSR Multi-Sensory Training
    """
    def __init__(self, symbols, model_path="models/transformer_deep_oracle_v1.pth"):
        self.symbols = symbols
        self.seq_len = 50
        self.brain = ChronosTransformer(feature_dim=8, seq_len=self.seq_len)
        if torch.cuda.is_available(): self.brain.to("cuda")
        
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path))
            console.print(f"[bold green]CRYSTAL:[/bold green] Deep cooling starting from {model_path}")
            
        self.optimizer = optim.Adam(self.brain.parameters(), lr=5e-6) # Micro-LR for precision
        self.datafeeds = {sym: prepare_transformer_df(pd.read_csv(f"data/raw/{sym}_5m.csv")) for sym in symbols}
        self.target_cols = ['vel', 'vol_rel', 'u_wick', 'l_wick', 'z_score', 'mtf_proxy', 'hi_rel', 'lo_rel']
        
        # Cooling Parameters
        self.beta_start = 5.0
        self.beta_end = 0.5
        self.max_gens = 500

    async def evolve(self, start_gen=150, run_gens=100):
        console.print(f"[bold cyan]PHASE 8.2:[/bold cyan] Deep Cooling Sprint (Gens {start_gen}-{start_gen+run_gens})")
        
        for gen in range(start_gen, start_gen + run_gens):
            beta = max(self.beta_end, self.beta_start - (gen * (self.beta_start - self.beta_end) / self.max_gens))
            
            self.brain.train()
            batch_log_probs = []
            batch_rewards = []
            batch_entropies = []
            trades_won = 0
            total_trades = 0
            
            for _ in range(128):
                sym = random.choice(self.symbols)
                df = self.datafeeds[sym]
                idx = random.randint(self.seq_len + 1, len(df) - 10)
                
                seq = df[self.target_cols].iloc[idx-self.seq_len:idx].values
                
                # --- Manifold Dropout (15%) ---
                if random.random() < 0.15:
                    drop_col = random.randint(0, 7)
                    seq[:, drop_col] = 0 # Force multi-sensory synergy
                
                seq_tensor = torch.from_numpy(np.array(seq)).float().unsqueeze(0)
                if torch.cuda.is_available(): seq_tensor = seq_tensor.to("cuda")
                
                logits = self.brain(seq_tensor, return_logits=True)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                
                dist = torch.distributions.Categorical(logits=logits)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)
                
                action = action_idx.item()
                if action in [1, 2]:
                    total_trades += 1
                    price = float(df.iloc[idx]['close'])
                    exit_px = float(df.iloc[idx+6]['close']) 
                    raw_pnl = (exit_px - price) / price if action == 1 else (price - exit_px) / price
                    net_pnl = raw_pnl - 0.0008
                    
                    if net_pnl > 0: trades_won += 1
                    
                    # HER Relay
                    if net_pnl < 0:
                        counter_action = 2 if action == 1 else 1
                        counter_log_prob = torch.log(probs[0, counter_action] + 1e-8)
                        batch_log_probs.append(counter_log_prob)
                        batch_rewards.append(abs(net_pnl) * 0.5)
                        
                    batch_log_probs.append(log_prob)
                    batch_rewards.append(net_pnl)
                    batch_entropies.append(entropy)

            if len(batch_log_probs) < 5: continue
            
            # --- Update Step ---
            rewards = torch.tensor(batch_rewards, dtype=torch.float32)
            if torch.cuda.is_available(): rewards = rewards.to("cuda")
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            loss = 0
            for lp, r, ent in zip(batch_log_probs, rewards, batch_entropies):
                loss -= (lp.reshape(-1) * r.reshape(-1)) + (beta * ent.reshape(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 0.5)
            self.optimizer.step()
            
            if (gen + 1) % 10 == 0:
                win_rate = (trades_won / total_trades * 100) if total_trades > 0 else 0
                console.print(f"  Gen {gen+1} | Beta: {beta:.2f} | WinRate: {win_rate:.1f}% | Neural ROI: {rewards.mean():+.4f}")

        torch.save(self.brain.state_dict(), "models/transformer_deep_oracle_v1.pth")
        console.print("\n[bold green]SENSORY ANNEALING COMPLETE.[/bold green] Oracle v1.5 persisted.")

if __name__ == "__main__":
    import asyncio
    evolver = CrystallineEvolver(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    asyncio.run(evolver.evolve(start_gen=150, run_gens=150)) # Deep Cooling to Gen 300
