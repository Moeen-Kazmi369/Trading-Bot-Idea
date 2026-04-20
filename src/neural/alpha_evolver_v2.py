import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import random
from src.neural.imitation_learner import PolicyNetwork
from rich.console import Console

console = Console()

class AlphaEvolverV2:
    def __init__(self, symbols, model_path="models/seed_brain_v1.pth"):
        self.symbols = symbols
        self.brain = PolicyNetwork(input_dim=5, output_dim=4)
        if torch.cuda.is_available(): self.brain.to("cuda")
        
        if os.path.exists(model_path):
            self.brain.load_state_dict(torch.load(model_path))
            console.print(f"[bold green]EVOLVER:[/bold green] Seeding evolution with {model_path}")
            
        self.optimizer = optim.Adam(self.brain.parameters(), lr=1e-5) # Tempered LR for stability
        self.datafeeds = {sym: pd.read_csv(f"data/raw/{sym}_5m.csv") for sym in symbols}
        
    async def run_generation(self, generation_id):
        self.brain.train()
        batch_log_probs = []
        batch_rewards = []
        
        TOTAL_TRADES_PER_GEN = 128
        FEE = 0.0008
        
        for _ in range(TOTAL_TRADES_PER_GEN):
            sym = random.choice(self.symbols)
            df = self.datafeeds[sym]
            start_idx = random.randint(50, len(df) - 100)
            
            window = df.iloc[start_idx-5:start_idx]
            state = [
                (window['close'].iloc[-1] / window['close'].iloc[0]) - 1,
                (window['volume'].iloc[-1] / window['volume'].mean()) - 1,
                (window['high'].max() - window['low'].min()) / window['close'].mean(),
                0, 0
            ]
            
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
            if torch.cuda.is_available(): state_tensor = state_tensor.to("cuda")
            
            # Neural Softener to prevent Categorical NaN
            probs = self.brain(state_tensor) + 1e-8 
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            
            action = action_idx.item()
            exit_idx = start_idx + 3
            entry_px = float(df.iloc[start_idx]['close'])
            exit_px = float(df.iloc[exit_idx]['close'])
            
            if action in [1, 2]:
                raw_pnl = (exit_px - entry_px) / entry_px if action == 1 else (entry_px - exit_px) / entry_px
                net_pnl = raw_pnl - FEE
                
                # Reward Shaping: Optimize for Alpha
                reward = net_pnl
                if net_pnl < 0:
                    reward = -net_pnl * 0.5 # Subsidize exploration of alternatives
                
                batch_log_probs.append(log_prob)
                batch_rewards.append(reward)
        
        if not batch_log_probs: return 0
        
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        if torch.cuda.is_available(): batch_rewards = batch_rewards.to("cuda")
        
        if len(batch_rewards) > 1:
            batch_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-6)
            
        loss = 0
        for log_p, r in zip(batch_log_probs, batch_rewards):
            loss -= log_p * r
            
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0) # Gradient Clipping
        self.optimizer.step()
        
        return batch_rewards.mean().item()

    async def evolve(self, generations=10):
        console.print(f"[bold cyan]PHASE 5 v2.1:[/bold cyan] Starting Stabilized Evolution...")
        for g in range(generations):
            avg_reward = await self.run_generation(g)
            console.print(f"  Gen {g+1}/{generations} | Neural Delta: {avg_reward:+.4f}")
            
        torch.save(self.brain.state_dict(), "models/evolved_brain_v2.pth")
        console.print("\n[bold green]SPRINT COMPLETE:[/bold green] Brain v2 saved.")

if __name__ == "__main__":
    import asyncio
    top_8 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "LINKUSDT", "AVAXUSDT"]
    evolver = AlphaEvolverV2(top_8)
    asyncio.run(evolver.evolve())
