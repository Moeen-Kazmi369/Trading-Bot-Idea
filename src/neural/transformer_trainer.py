import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
from src.neural.transformer_brain import HybridSentinel, prepare_15d_manifold
from rich.console import Console

console = Console()

def train_production_v17():
    console.print(f"[bold cyan]MASTER FORGE:[/bold cyan] Initiating 15D Hybrid Locked Seeding...")
    
    # 1. Load Data
    df = pd.read_csv("data/raw/BTCUSDT_5m.csv")
    df = prepare_15d_manifold(df)
    
    target_cols = [
        'vel', 'vol_rel', 'u_wick', 'l_wick', 'z_score', 'mtf_proxy', 'hi_rel', 'lo_rel',
        'buy_wall_prox', 'sell_wall_prox', 'rsi', 'macd_hist', 'atr', 'bb_up_dist', 'bb_lo_dist'
    ]
    feature_matrix = df[target_cols].values
    
    # 2. Expert Labeling (Volatility-Locked)
    states = []
    labels = []
    seq_len = 50
    
    for i in range(seq_len + 1, len(df)-10):
        row = df.iloc[i]
        label = 0
        
        if row['atr'] > 0.0020: 
            if row['bb_lo_dist'] < 0.003 and row['close'] > row['open']:
                label = 1 # LONG
            elif row['bb_up_dist'] < 0.003 and row['close'] < row['open']:
                label = 2 # SHORT
                
        states.append(feature_matrix[i-seq_len:i])
        labels.append(label)
        
    X = torch.tensor(np.array(states), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.long)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    # 3. Training
    model = HybridSentinel(feature_dim=15, seq_len=50, hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(15):
        epoch_loss = 0
        for b_X, b_y in loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_X)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        console.print(f"  Epoch {epoch+1}/15 | Manifold Loss: {epoch_loss/len(loader):.4f}")
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/transformer_hybrid_v17.pth")
    console.print("[bold green]HYBRID ORACLE READY.[/bold green]")

if __name__ == "__main__":
    train_production_v17()
