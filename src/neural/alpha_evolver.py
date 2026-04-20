import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from src.neural.trading_env import ProductionTradingEnv
from src.neural.imitation_learner import PolicyNetwork
from rich.console import Console
from rich.progress import Progress

console = Console()

class AlphaEvolver:
    """
    Project Chronos - Alpha Evolution Engine
    Uses Reinforcement Learning (REINFORCE) to evolve the seed brain.
    """
    
    def __init__(self, model_path="models/seed_brain_v1.pth"):
        self.df = pd.read_csv("data/raw/BTCUSDT_5m.csv")
        self.env = ProductionTradingEnv(self.df)
        
        # Load the Seed Brain
        self.model = PolicyNetwork(input_dim=5, output_dim=4)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            console.print("[bold green]EVOLVER:[/bold green] Seed Brain v1 loaded for evolution.")
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4) # Low learning rate for stability

    def evolve(self, episodes=50):
        console.print(f"[bold yellow]EVOLVING:[/bold yellow] Starting {episodes} Evolution Generations...")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Evolving Species...", total=episodes)
            
            for ep in range(episodes):
                state = self.env.reset()
                log_probs = []
                rewards = []
                
                done = False
                while not done:
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                    probs = self.model(state_tensor)
                    
                    # Sample action from probability distribution
                    m = torch.distributions.Categorical(probs)
                    action = m.sample()
                    
                    log_probs.append(m.log_prob(action))
                    
                    # Step environment
                    next_state, reward, done, _ = self.env.step(action.item())
                    
                    rewards.append(reward)
                    state = next_state
                    
                    if len(rewards) > 5000: break # Limiting one gen to 5k bars for speed
                
                # Update Policy (The 'Neural' Learning)
                self.update_policy(log_probs, rewards)
                
                progress.update(task, advance=1)
                
                if (ep + 1) % 10 == 0:
                    console.print(f"  Gen {ep+1} | Cumulative Reward: {sum(rewards):.2f}")

        # Save the Evolved Brain
        torch.save(self.model.state_dict(), "models/evolved_brain_v1.pth")
        console.print("\n[bold green]EVOLUTION COMPLETE:[/bold green] Evolved Brain v1 saved.")

    def update_policy(self, log_probs, rewards):
        """
        Policy Gradient Update Logic
        """
        discounted_rewards = []
        R = 0
        gamma = 0.99
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
            
        discounted_rewards = torch.tensor(discounted_rewards)
        # Normalize rewards for stable training
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * reward)
            
        self.optimizer.zero_grad()
        loss = torch.stack(loss).sum()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    evolver = AlphaEvolver()
    evolver.evolve(episodes=50) # Starting with 50 generations to see optimization
