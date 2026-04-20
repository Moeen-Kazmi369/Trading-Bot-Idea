import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from src.neural.trading_env import ProductionTradingEnv
from src.strategies.chronos_mtf_trap_hunter import ChronosMTFTrapHunter
from rich.console import Console

console = Console()

class PolicyNetwork(nn.Module):
    """
    The Brain (Actor)
    A simple but powerful Feed-Forward network to start the imitation.
    """
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.net(x)

def generate_expert_trajectories(symbol="BTCUSDT"):
    """
    Uses the 45% ROI Trap Hunter to generate 'Correct Answers' for the AI.
    """
    # Use the same logic as our profitable MTF Trap Hunter
    strategy = ChronosMTFTrapHunter(symbol)
    df = strategy.load_aligned_data()
    
    trajectories = []
    
    # Simulate the strategy and record 'State -> Action' pairs
    # Action Space: 0=Stay, 1=Long, 2=Short, 3=Exit
    # Current MTF Hunter is Short-only for Traps.
    
    position = 0
    for i in range(50, len(df)-1):
        row = df.iloc[i]
        action = 0 # Default: Stay
        
        if position == 0:
            if row['htf_trap_active'] == 1 and row['close'] < row['htf_trigger_px']:
                action = 2 # GO SHORT (The Expert Move)
                position = -1
        elif position == -1:
            # We approximate the strategy's exit (simplified for imitation)
            if row['close'] > row['open']: # Just an example exit condition for the AI to learn
                 action = 3
                 position = 0
        
        # Build the 'State' (same as Environment)
        window = df.iloc[i-5:i]
        state = [
            (window['close'].iloc[-1] / window['close'].iloc[0]) - 1,
            (window['volume'].iloc[-1] / window['volume'].mean()) - 1,
            (window['high'].max() - window['low'].min()) / window['close'].mean(),
            position,
            0 # P&L
        ]
        
        trajectories.append((state, action))
        
    return trajectories

def train_imitation_model():
    console.print("[bold cyan]PHASE 2B:[/bold cyan] Generating Expert trajectories from Trap Hunter...")
    expert_data = generate_expert_trajectories()
    
    # Convert to Tensors
    states = torch.tensor([x[0] for x in expert_data], dtype=torch.float32)
    actions = torch.tensor([x[1] for x in expert_data], dtype=torch.long)
    
    # Initialize Brain
    model = PolicyNetwork(input_dim=5, output_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    console.print(f"[bold yellow]TRAINING:[/bold yellow] Teaching AI to 'Imitate' the Trap Hunter ({len(expert_data)} samples)...")
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(states)
        loss = criterion(output, actions)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            console.print(f"  Epoch {epoch+1}/100 | Brain Divergence (Loss): {loss.item():.4f}")
            
    # Save the 'Seed Brain'
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/seed_brain_v1.pth")
    console.print("\n[bold green]IMITATION COMPLETE:[/bold green] Seed Brain v1 saved to models/")

if __name__ == "__main__":
    train_imitation_model()
