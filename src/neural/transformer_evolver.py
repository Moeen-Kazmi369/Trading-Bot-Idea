import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import random
import time
from src.neural.transformer_brain import ChronosTransformer, prepare_transformer_df
from rich.console import Console

console = Console()

class TransformerEvolver:
    def __init__(self, symbols, model_path="models/transformer_seed_v1.pth"):
        self.symbols = symbols
        self.brain = ChronosTransformer(feature_dim=8, seq_len=20)
        if torch.cuda.is_available(): self.brain.to("cuda")
        
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path))
            console.print(f"[bold green]TRL v1.3:[/bold green] Seeding with Forced Exploration ({model_path})")
            
        self.optimizer = optim.Adam(self.brain.parameters(), lr=1e-5)
        self.datafeeds = {sym: prepare_transformer_df(pd.read_csv(f"data/raw/{sym}_5m.csv")) for sym in symbols}
        self.fee = 0.0008
        self.target_cols = ['vel', 'vol_rel', 'u_wick', 'l_wick', 'z_score', 'mtf_proxy', 'hi_rel', 'lo_rel']

    async def evolve_attention(self, generations=10):
        console.print(f"[bold cyan]PHASE 7:[/bold cyan] Initiating Epsilon-Greedy Evolution...")
        
        epsilon = 0.3 # 30% Forced exploration
        
        for gen in range(generations):
            self.brain.train()
            batch_log_probs = []
            batch_rewards = []
            
            for _ in range(128):
                sym = random.choice(self.symbols)
                df = self.datafeeds[sym]
                idx = random.randint(50, len(df) - 10)
                
                seq = df[self.target_cols].iloc[idx-20:idx].values
                price = float(df.iloc[idx]['close'])
                seq_tensor = torch.from_numpy(np.array(seq)).float().unsqueeze(0)
                if torch.cuda.is_available(): seq_tensor = seq_tensor.to("cuda")
                
                # 1. Prediction
                logits = self.brain(seq_tensor, return_logits=True)
                
                # 2. Epsilon-Greedy Action Selection
                if random.random() < epsilon:
                    action = random.choice([1, 2]) # Specifically force Long/Short exploration
                    # Create a dummy log_prob for the random action
                    log_prob = torch.log(torch.softmax(logits, dim=-1))[0, action]
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action_idx = dist.sample()
                    log_prob = dist.log_prob(action_idx)
                    action = action_idx.item()
                
                if action in [1, 2]:
                    exit_idx = idx + 6
                    exit_px = float(df.iloc[exit_idx]['close'])
                    raw_pnl = (exit_px - price) / price if action == 1 else (price - exit_px) / price
                    net_pnl = raw_pnl - self.fee
                    
                    batch_log_probs.append(log_prob)
                    batch_rewards.append(net_pnl)
            
            # 3. Training Step
            if len(batch_log_probs) < 5: continue 
            
            rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
            if torch.cuda.is_available(): rewards_tensor = rewards_tensor.to("cuda")
            
            if len(rewards_tensor) > 1:
                rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
                
            loss = 0
            for lp, r in zip(batch_log_probs, rewards_tensor):
                loss -= lp * r
                
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
            self.optimizer.step()
            
            console.print(f"  Gen {gen+1}/{generations} | Samples: {len(batch_rewards)} | Mean R: {rewards_tensor.mean():+.4f}")
            
        torch.save(self.brain.state_dict(), "models/transformer_oracle_v1.pth")
        console.print("\n[bold green]EVOLUTION COMPLETE:[/bold green] Oracle v1 persists.")

if __name__ == "__main__":
    import asyncio
    top_4 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    evolver = TransformerEvolver(top_4)
    asyncio.run(evolver.evolve_attention(generations=10))
