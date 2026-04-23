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

def prepare_split_data(sym_train="BTCUSDT", sym_val="ETHUSDT", tf=60):
    """
    Implements a Cross-Symbol Temporal Split.
    Training on Symbol A, Validating on Symbol B (Out-of-Sample Time).
    """
    # 1. Training Set (80% of BTC)
    df_train_raw = pd.read_csv(f"data/raw/{sym_train}_5m.csv").iloc[:40000]
    df_train, cols = prepare_30d_universal_manifold(df_train_raw, timeframe_mins=tf)
    
    # 2. Validation Set (Last 20% of ETH)
    df_val_raw = pd.read_csv(f"data/raw/{sym_val}_5m.csv").iloc[40000:]
    df_val, _ = prepare_30d_universal_manifold(df_val_raw, timeframe_mins=tf)
    
    def get_tensors(df, feature_cols):
        X, C, Y = [], [], []
        feat_vals = df[feature_cols].values
        for i in range(50, len(df)-10, 50):
            X.append(feat_vals[i-50:i])
            C.append([df.iloc[i]['c_timeframe'], df.iloc[i]['c_fee']])
            # Simple Profit Expert Label
            roi = (df.iloc[i+6]['close'] - df.iloc[i]['close']) / df.iloc[i]['close']
            label = 0
            if roi > 0.0016: label = 1
            elif roi < -0.0016: label = 2
            Y.append(label)
        return torch.tensor(np.array(X)).float(), torch.tensor(np.array(C)).float(), torch.tensor(np.array(Y)).long()

    train_X, train_C, train_Y = get_tensors(df_train, cols)
    val_X, val_C, val_Y = get_tensors(df_val, cols)
    
    return train_X, train_C, train_Y, val_X, val_C, val_Y

def train_with_guard(epochs=1000):
    console.print(f"[bold cyan]COLAB MASTERY FORGE:[/bold cyan] Initiating 1,000-Epoch Deep Siege...")
    
    # 1. Load Guarded Data
    t_X, t_C, t_Y, v_X, v_C, v_Y = prepare_split_data()
    train_loader = DataLoader(TensorDataset(t_X, t_C, t_Y), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(v_X, v_C, v_Y), batch_size=512)
    
    # 2. Model Init
    model = UniversalOracleV2(feature_dim=30, hidden_dim=256)
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Ultra Slow-Burn
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 3. Training with Overfit Monitoring
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bX, bC, bY in train_loader:
            bX, bC, bY = bX.to(device), bC.to(device), bY.to(device)
            optimizer.zero_grad(); out = model(bX, bC); loss = criterion(out, bY)
            loss.backward(); optimizer.step(); train_loss += loss.item()
            
        # Validation Scan (The Jury)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bX, bC, bY in val_loader:
                bX, bC, bY = bX.to(device), bC.to(device), bY.to(device)
                out = model(bX, bC); loss = criterion(out, bY); val_loss += loss.item()
        
        t_avg = train_loss / len(train_loader)
        v_avg = val_loss / len(val_loader)
        
        console.print(f"  Epoch {epoch+1}/{epochs} | Train Loss: [bold blue]{t_avg:.4f}[/bold blue] | Val Loss: [bold yellow]{v_avg:.4f}[/bold yellow]")
        
        # Overfit Warning
        if v_avg > t_avg * 1.2:
            console.print("  [bold red]OVERFIT WARNING:[/bold red] Validation divergence detected. Memorization active.")

    torch.save(model.state_dict(), "models/universal_oracle_guarded_v2.pth")
    console.print("\n[bold green]CLOUD SIEGE GUARD COMPLETE.[/bold green]")

if __name__ == "__main__":
    train_with_guard()
