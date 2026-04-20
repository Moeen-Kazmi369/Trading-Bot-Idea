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

class AnnealingTransformerEvolver:
    """
    Project Chronos - Phase 8.1 Annealing Forge
    Implements Linear Entropy Decay + Hindsight Experience Replay (HER)
    """
    def __init__(self, symbols, model_path="models/transformer_deep_oracle_v1.pth"):
        self.symbols = symbols
        self.seq_len = 50
        self.brain = ChronosTransformer(feature_dim=8, seq_len=self.seq_len)
        if torch.cuda.is_available(): self.brain.to("cuda")
        
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path))
            console.print(f"[bold green]FORGE:[/bold green] Annealing starting from {model_path}")
            
        self.optimizer = optim.Adam(self.brain.parameters(), lr=1e-5)
        self.datafeeds = {sym: prepare_transformer_df(pd.read_csv(f"data/raw/{sym}_5m.csv")) for sym in symbols}
        self.target_cols = ['vel', 'vol_rel', 'u_wick', 'l_wick', 'z_score', 'mtf_proxy', 'hi_rel', 'lo_rel']
        
        # Annealing Config
        self.beta_start = 5.0
        self.beta_end = 0.5
        self.max_gens = 500

    async def evolve(self, current_gen_offset=0, run_gens=50):
        console.print(f"[bold cyan]PHASE 8.1:[/bold cyan] Cooling from Gen {current_gen_offset}...")
        
        for gen in range(current_gen_offset, current_gen_offset + run_gens):
            # Calculate Current Beta (Linear Decay)
            beta = max(self.beta_end, self.beta_start - (gen * (self.beta_start - self.beta_end) / self.max_gens))
            
            self.brain.train()
            batch_log_probs = []
            batch_rewards = []
            batch_entropies = []
            
            for _ in range(128):
                sym = random.choice(self.symbols)
                df = self.datafeeds[sym]
                idx = random.randint(self.seq_len + 1, len(df) - 10)
                
                seq = df[self.target_cols].iloc[idx-self.seq_len:idx].values
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
                    # Simulation
                    price = float(df.iloc[idx]['close'])
                    exit_px = float(df.iloc[idx+6]['close']) 
                    raw_pnl = (exit_px - price) / price if action == 1 else (price - exit_px) / price
                    net_pnl = raw_pnl - 0.0008
                    
                    # --- Hindsight Experience Replay (HER) ---
                    if net_pnl < 0: # If trade lost, relabel the counter-action as correct
                        counter_action = 2 if action == 1 else 1
                        counter_log_prob = torch.log(probs[0, counter_action] + 1e-8)
                        
                        # Injecting the "Mistake Map"
                        batch_log_probs.append(counter_log_prob)
                        batch_rewards.append(abs(net_pnl) * 0.5) # Reward the logic of doing the opposite
                    
                    batch_log_probs.append(log_prob)
                    batch_rewards.append(net_pnl)
                    batch_entropies.append(entropy)
            
            if len(batch_log_probs) < 5: continue
            
            rewards = torch.tensor(batch_rewards, dtype=torch.float32)
            if torch.cuda.is_available(): rewards = rewards.to("cuda")
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            loss = 0
            for lp, r, ent in zip(batch_log_probs, rewards, batch_entropies):
                # Enforce scalar alignment
                loss -= (lp.reshape(-1) * r.reshape(-1)) + (beta * ent.reshape(-1))
                
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
            self.optimizer.step()
            
            if (gen + 1) % 10 == 0:
                console.print(f"  Gen {gen+1} | Beta: {beta:.2f} | Neural ROI: {rewards.mean():+.4f} | Samples: {len(batch_rewards)}")

        torch.save(self.brain.state_dict(), "models/transformer_deep_oracle_v1.pth")
        console.print("\n[bold green]FORGE STEP COMPLETE.[/bold green] Oracle cooled.")

if __name__ == "__main__":
    import asyncio
    evolver = AnnealingTransformerEvolver(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    asyncio.run(evolver.evolve(current_gen_offset=100, run_gens=50))
