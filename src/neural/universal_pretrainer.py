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

def run_pretraining(epochs=20):
    console.print(f"[bold cyan]PRE-TRAINING:[/bold cyan] Initiating Universal Alphabet Cycle (30D Prediction)...")
    
    # 1. Load Diversified Datasets (Multi-Resolution)
    # We use 5m and 1h to teach the 'Zoom' invariant properties
    df_5m = pd.read_csv("data/raw/BTCUSDT_5m.csv")
    df_5m, cols = prepare_30d_universal_manifold(df_5m, timeframe_mins=5)
    
    df_1h = pd.read_csv("data/raw/BTCUSDT_5m.csv").iloc[::12].copy() # Resampled 1h
    df_1h, _ = prepare_30d_universal_manifold(df_1h, timeframe_mins=60)
    
    # Combined State Manifold
    feature_matrix_5m = df_5m[cols].values
    feature_matrix_1h = df_1h[cols].values
    
    states = []
    targets = [] # The next 30D state
    conds = []
    
    seq_len = 50
    # Sampling 5m behavior
    for i in range(seq_len, len(df_5m)-1, 50): # Strided sampling for speed
        states.append(feature_matrix_5m[i-seq_len:i])
        targets.append(feature_matrix_5m[i+1]) # Target = Next Bar
        conds.append([df_5m.iloc[i]['c_timeframe'], df_5m.iloc[i]['c_fee']])
        
    # Sampling 1h behavior
    for i in range(seq_len, len(df_1h)-1, 20):
        states.append(feature_matrix_1h[i-seq_len:i])
        targets.append(feature_matrix_1h[i+1])
        conds.append([df_1h.iloc[i]['c_timeframe'], df_1h.iloc[i]['c_fee']])

    X = torch.tensor(np.array(states), dtype=torch.float32)
    T = torch.tensor(np.array(targets), dtype=torch.float32)
    C = torch.tensor(np.array(conds), dtype=torch.float32)
    
    dataset = TensorDataset(X, C, T)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 2. Model & Heuristic Seeding
    model = UniversalOracleV2(feature_dim=30, hidden_dim=256)
    model.heuristic_seed()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss() # Minimizing prediction error of the next state
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    console.print(f"[bold yellow]ALPHABET FORGE:[/bold yellow] Training on {len(X)} Sequential Transitions...")
    for epoch in range(epochs):
        epoch_loss = 0
        for b_X, b_C, b_T in loader:
            b_X, b_C, b_T = b_X.to(device), b_C.to(device), b_T.to(device)
            optimizer.zero_grad()
            
            # Forward in 'State' Mode (Predicting the next bar)
            pred_state = model(b_X, b_C, mode="state")
            loss = criterion(pred_state, b_T)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        console.print(f"  Epoch {epoch+1}/{epochs} | Reading Comprehension (MSE): {epoch_loss/len(loader):.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/universal_alphabet_v2.pth")
    console.print("\n[bold green]ALPHABET SOILDIFIED.[/bold green] Oracle is now fluent in 30D market physics.")

if __name__ == "__main__":
    run_pretraining()
