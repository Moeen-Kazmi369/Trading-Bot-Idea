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

class XRLEvolver:
    """
    Phase 10: X-Ray Reinforcement Learning (X-RL)
    Forces the Sentinel to integrate Liquidity Walls into Action Logic.
    """
    def __init__(self, symbols, model_path="models/transformer_xray_v1.pth"):
        self.symbols = symbols
        self.seq_len = 50
        # 10D / 5 Heads / 160 Hidden
        self.brain = ChronosTransformer(feature_dim=10, seq_len=self.seq_len, num_heads=5, hidden_dim=160)
        if torch.cuda.is_available(): self.brain.to("cuda")
        
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path))
            console.print(f"[bold green]X-RL:[/bold green] Training the Sentinel from {model_path}")
            
        self.optimizer = optim.Adam(self.brain.parameters(), lr=1e-5)
        self.datafeeds = {sym: prepare_transformer_df(pd.read_csv(f"data/raw/{sym}_5m.csv")) for sym in symbols}
        self.target_cols = [
            'vel', 'vol_rel', 'u_wick', 'l_wick', 'z_score', 
            'mtf_proxy', 'hi_rel', 'lo_rel', 
            'buy_wall_prox', 'sell_wall_prox'
        ]

    async def evolve(self, generations=100):
        console.print(f"[bold cyan]PHASE 10:[/bold cyan] Initiating 10D X-Ray Reinforcement (X-RL)...")
        
        # We use a 30% Force Exploration to BREAK the "Stay Only" bias
        epsilon = 0.3 
        
        for gen in range(generations):
            self.brain.train()
            batch_log_probs = []
            batch_rewards = []
            
            for _ in range(128):
                sym = random.choice(self.symbols)
                df = self.datafeeds[sym]
                idx = random.randint(self.seq_len + 1, len(df) - 10)
                
                seq = df[self.target_cols].iloc[idx-self.seq_len:idx].values
                seq_tensor = torch.from_numpy(np.array(seq)).float().unsqueeze(0)
                if torch.cuda.is_available(): seq_tensor = seq_tensor.to("cuda")
                
                # 1. Prediction (Logits for Unfiltered Drive)
                logits = self.brain(seq_tensor, return_logits=True)
                
                # 2. Force Exploration of Actions [1, 2]
                if random.random() < epsilon:
                    action = random.choice([1, 2])
                    probs = torch.softmax(logits, dim=-1)
                    log_prob = torch.log(probs[0, action] + 1e-8)
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action_idx = dist.sample()
                    log_prob = dist.log_prob(action_idx)
                    action = action_idx.item()
                
                if action in [1, 2]:
                    price = float(df.iloc[idx]['close'])
                    exit_px = float(df.iloc[idx+6]['close']) 
                    raw_pnl = (exit_px - price) / price if action == 1 else (price - exit_px) / price
                    net_pnl = raw_pnl - 0.0008
                    
                    batch_log_probs.append(log_prob)
                    batch_rewards.append(net_pnl)
            
            # --- Update Step ---
            if len(batch_log_probs) < 5: continue
            
            rewards = torch.tensor(batch_rewards, dtype=torch.float32)
            if torch.cuda.is_available(): rewards = rewards.to("cuda")
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            loss = 0
            for lp, r in zip(batch_log_probs, rewards):
                loss -= (lp.reshape(-1) * r.reshape(-1))
                
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
            self.optimizer.step()
            
            if (gen + 1) % 10 == 0:
                console.print(f"  Gen {gen+1}/{generations} | Neural Delta: {rewards.mean():+.4f} | Samples: {len(batch_rewards)}")

        torch.save(self.brain.state_dict(), "models/transformer_xray_oracle_v1.pth")
        console.print("\n[bold green]X-RL COMPLETE.[/bold green] Sentinel is now a Hunter.")

if __name__ == "__main__":
    import asyncio
    evolver = XRLEvolver(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    asyncio.run(evolver.evolve(generations=100))
