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

class DeepTransformerEvolver:
    def __init__(self, symbols, model_path="models/transformer_oracle_v1.pth"):
        self.symbols = symbols
        self.seq_len = 50 # Deep Horizon
        self.brain = ChronosTransformer(feature_dim=8, seq_len=self.seq_len)
        if torch.cuda.is_available(): self.brain.to("cuda")
        
        # Load Existing Alpha if available
        if os.path.exists(model_path):
            state = torch.load(model_path)
            # Handle weight shape mismatch if seq_len changed
            try:
                self.brain.load_state_dict(state)
            except:
                console.print("[bold yellow]TRL:[/bold yellow] Re-initializing Attention Heuristics for 50-bar Horizon.")
            
        self.optimizer = optim.Adam(self.brain.parameters(), lr=5e-6) # Micro-LR for Deep Evolution
        self.datafeeds = {sym: prepare_transformer_df(pd.read_csv(f"data/raw/{sym}_5m.csv")) for sym in symbols}
        self.target_cols = ['vel', 'vol_rel', 'u_wick', 'l_wick', 'z_score', 'mtf_proxy', 'hi_rel', 'lo_rel']

    async def evolve(self, generations=1000):
        console.print(f"[bold cyan]PHASE 8:[/bold cyan] Initiating 1,000-Gen Deep Evolution (Horizon: 50 | Entropy: 5x)")
        
        for gen in range(generations):
            self.brain.train()
            batch_log_probs = []
            batch_rewards = []
            batch_entropies = []
            
            # Sampling Episodes
            for _ in range(64):
                sym = random.choice(self.symbols)
                df = self.datafeeds[sym]
                idx = random.randint(self.seq_len + 1, len(df) - 10)
                
                seq = df[self.target_cols].iloc[idx-self.seq_len:idx].values
                seq_tensor = torch.from_numpy(np.array(seq)).float().unsqueeze(0)
                if torch.cuda.is_available(): seq_tensor = seq_tensor.to("cuda")
                
                # 1. Forward Pass (Logits for Exploration)
                logits = self.brain(seq_tensor, return_logits=True)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                
                # 2. Stochastic Sampler
                dist = torch.distributions.Categorical(logits=logits)
                action_idx = dist.sample()
                log_prob = dist.log_prob(action_idx)
                
                action = action_idx.item()
                if action in [1, 2]: # Only optimize for trades
                    price = float(df.iloc[idx]['close'])
                    exit_px = float(df.iloc[idx+6]['close']) # 30 min outcome
                    raw_pnl = (exit_px - price) / price if action == 1 else (price - exit_px) / price
                    reward = raw_pnl - 0.0008 # Fee friction
                    
                    batch_log_probs.append(log_prob)
                    batch_rewards.append(reward)
                    batch_entropies.append(entropy)
            
            # 3. Deep Update (Policy Gradient + Entropy Bonus)
            if len(batch_log_probs) < 4: continue
            
            rewards = torch.tensor(batch_rewards, dtype=torch.float32)
            if torch.cuda.is_available(): rewards = rewards.to("cuda")
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            loss = 0
            for lp, r, ent in zip(batch_log_probs, rewards, batch_entropies):
                # Loss = -LogProb * Reward - Beta * Entropy
                loss -= (lp * r) + (5.0 * ent) # 5x Entropy Bonus to Shatter Lock
                
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
            self.optimizer.step()
            
            if (gen + 1) % 10 == 0:
                console.print(f"  Gen {gen+1}/{generations} | Neural ROI: {rewards.mean():+.4f} | Entropy: {sum(batch_entropies)/len(batch_entropies):.1f}")

        torch.save(self.brain.state_dict(), "models/transformer_deep_oracle_v1.pth")
        console.print("\n[bold green]MISSION COMPLETE:[/bold green] Deep Oracle v1 persisted.")

if __name__ == "__main__":
    import asyncio
    top_4 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    evolver = DeepTransformerEvolver(top_4)
    asyncio.run(evolver.evolve(generations=100)) # First 10% of the mission
