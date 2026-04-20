import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
from src.neural.transformer_brain import UniversalOracleV2
from src.neural.universal_manifold import prepare_30d_universal_manifold
from rich.console import Console

console = Console()

def run_cloud_dry_run():
    console.print(f"[bold cyan]CLOUD PRE-FLIGHT:[/bold cyan] Initiating 10-Epoch Sanity Check...")
    
    # 1. Load a Small 1-Month Slice (~8,500 bars)
    df_raw = pd.read_csv("data/raw/BTCUSDT_5m.csv").iloc[:8500]
    df, cols = prepare_30d_universal_manifold(df_raw, timeframe_mins=5)
    
    # Verify Dimensions
    console.print(f"  [bold blue]SHAPE CHECK:[/bold blue] Manifold Width: {len(cols)}")
    
    feat_vals = df[cols].values
    X, C, Y = [], [], []
    for i in range(50, len(df)-10):
        X.append(feat_vals[i-50:i])
        C.append([df.iloc[i]['c_timeframe'], df.iloc[i]['c_fee']])
        # Dummy Profit Label
        roi = (df.iloc[i+6]['close'] - df.iloc[i]['close']) / df.iloc[i]['close']
        Y.append(1 if roi > 0.0016 else (2 if roi < -0.0016 else 0))
        
    loader = DataLoader(TensorDataset(torch.tensor(np.array(X)).float(), 
                                      torch.tensor(np.array(C)).float(), 
                                      torch.tensor(np.array(Y)).long()), batch_size=128)
    
    # 2. Model & Seeding
    model = UniversalOracleV2(feature_dim=30, hidden_dim=256)
    model.collaborator_seeding()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 3. 10-Epoch Burn
    for epoch in range(10):
        epoch_loss = 0
        for bX, bC, bY in loader:
            bX, bC, bY = bX.to(device), bC.to(device), bY.to(device)
            optimizer.zero_grad(); out = model(bX, bC); loss = criterion(out, bY)
            loss.backward(); optimizer.step(); epoch_loss += loss.item()
        console.print(f"  Epoch {epoch+1}/10 | Dry Run Loss: {epoch_loss/len(loader):.4f}")
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cloud_siege_dummy_v1.pth")
    console.print("\n[bold green]SANITY CHECK PASSED.[/bold green] Tensors aligned. Cloud deployment authorized.")

if __name__ == "__main__":
    run_cloud_dry_run()
