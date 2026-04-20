import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import re
from src.neural.transformer_brain import HybridSentinel, prepare_15d_manifold
from rich.console import Console

console = Console()

def get_latest_snapshot():
    snap_dir = "models/snapshots"
    if not os.path.exists(snap_dir): return None, 0
    
    files = [f for f in os.listdir(snap_dir) if f.endswith(".pth")]
    if not files: return None, 0
    
    # Extract epoch numbers using regex
    epochs = []
    for f in files:
        match = re.search(r'epoch_(\d+)', f)
        if match:
            epochs.append((int(match.group(1)), f))
            
    if not epochs: return None, 0
    
    latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
    return os.path.join(snap_dir, latest_file), latest_epoch

def run_deep_forge(epochs=100, lr=1e-5):
    console.print(f"[bold cyan]DEEP FORGE RECOVERY:[/bold cyan] Scanning for snapshots...")
    
    # 1. Load Data
    df = pd.read_csv("data/raw/BTCUSDT_5m.csv")
    df = prepare_15d_manifold(df)
    
    target_cols = [
        'vel', 'vol_rel', 'u_wick', 'l_wick', 'z_score', 'mtf_proxy', 'hi_rel', 'lo_rel',
        'buy_wall_prox', 'sell_wall_prox', 'rsi', 'macd_hist', 'atr', 'bb_up_dist', 'bb_lo_dist'
    ]
    feature_matrix = df[target_cols].values
    
    # 2. Expert Seeding
    states = []
    labels = []
    seq_len = 50
    for i in range(seq_len + 1, len(df)-10):
        row = df.iloc[i]
        label = 0
        if row['atr'] > 0.0020:
            if row['bb_lo_dist'] < 0.003 and row['close'] > row['open']:
                label = 1
            elif row['bb_up_dist'] < 0.003 and row['close'] < row['open']:
                label = 2
        states.append(feature_matrix[i-seq_len:i])
        labels.append(label)
    
    X = torch.tensor(np.array(states), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=512, shuffle=True)
    
    # 3. Model & Registry Recovery
    model = HybridSentinel(feature_dim=15, seq_len=50, hidden_dim=128)
    snap_path, start_epoch = get_latest_snapshot()
    
    if snap_path:
        model.load_state_dict(torch.load(snap_path))
        console.print(f"[bold green]RECOVERED:[/bold green] Resuming from Epoch {start_epoch} snapshot.")
    elif os.path.exists("models/transformer_hybrid_v17.pth"):
        model.load_state_dict(torch.load("models/transformer_hybrid_v17.pth"))
        console.print("[bold yellow]REVERSION:[/bold yellow] No snapshots found. Starting from v1.7.1 weights.")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        for b_X, b_y in loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_X)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        console.print(f"  Epoch {epoch+1}/{epochs} | Manifold Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            snap_path = f"models/snapshots/chronos_v18_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), snap_path)
            console.print(f"  [bold blue]SNAPSHOT:[/bold blue] Saved to {snap_path}")
            
    torch.save(model.state_dict(), "models/transformer_deep_forge_v18.pth")
    console.print("\n[bold green]DEEP FORGE COMPLETE.[/bold green]")

if __name__ == "__main__":
    run_deep_forge()
