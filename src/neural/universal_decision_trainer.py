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

def prepare_split_data():
    """
    Universal Mastery Forge: 6 Coins x 3 Resolutions (18x Data Scaling).
    TURBO-OPTIMIZED for rapid tensor assembly.
    """
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
    timeframes = [5, 15, 60]
    
    all_train_X, all_train_C, all_train_Y = [], [], []
    all_val_X, all_val_C, all_val_Y = [], [], []
    
    for sym in symbols:
        for tf in timeframes:
            path = f"data/raw/{sym}_5m.csv"
            if not os.path.exists(path): continue
            
            console.print(f"  [bold blue]TURBO-FORGING:[/bold blue] {sym} at {tf}m Resolution...")
            df_raw = pd.read_csv(path)
            
            if tf > 5:
                df_raw = df_raw.iloc[::(tf//5)].copy()
                
            split_idx = int(len(df_raw) * 0.8)
            df_train, cols = prepare_30d_universal_manifold(df_raw.iloc[:split_idx], timeframe_mins=tf)
            df_val, _ = prepare_30d_universal_manifold(df_raw.iloc[split_idx:], timeframe_mins=tf)
            
            def get_tensors_turbo(df, feature_cols):
                feat_vals = df[feature_cols].values
                closes = df['close'].values
                c_tf = df['c_timeframe'].values
                c_fe = df['c_fee'].values
                
                step = 25 if tf == 5 else 5
                idx = np.arange(50, len(df)-10, step)
                
                if len(idx) == 0: return None, None, None

                # Vectorized Slice Assembly
                X = np.stack([feat_vals[i-50:i] for i in idx])
                C = np.column_stack([c_tf[idx], c_fe[idx]])
                
                roi = (closes[idx+6] - closes[idx]) / closes[idx]
                Y = np.zeros(len(idx), dtype=int)
                Y[roi > 0.0016] = 1
                Y[roi < -0.0016] = 2
                
                return X, C, Y

            tx, tc, ty = get_tensors_turbo(df_train, cols)
            vx, vc, vy = get_tensors_turbo(df_val, cols)
            
            if tx is not None:
                all_train_X.append(tx); all_train_C.append(tc); all_train_Y.append(ty)
            if vx is not None:
                all_val_X.append(vx); all_val_C.append(vc); all_val_Y.append(vy)
    
    # Accelerated Tensor Conversion
    return torch.from_numpy(np.concatenate(all_train_X)).float(), \
           torch.from_numpy(np.concatenate(all_train_C)).float(), \
           torch.from_numpy(np.concatenate(all_train_Y)).long(), \
           torch.from_numpy(np.concatenate(all_val_X)).float(), \
           torch.from_numpy(np.concatenate(all_val_C)).float(), \
           torch.from_numpy(np.concatenate(all_val_Y)).long()

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
        
        # --- AUTO-SNAPSHOT every 50 epochs ---
        if (epoch + 1) % 50 == 0:
            snap_path = f"models/universal_oracle_snap_{epoch+1}.pth"
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), snap_path)
            console.print(f"  [bold cyan]SNAPSHOT:[/bold cyan] Progress saved to {snap_path}")

    torch.save(model.state_dict(), "models/universal_oracle_guarded_v2.pth")
    console.print("\n[bold green]CLOUD SIEGE GUARD COMPLETE.[/bold green]")

if __name__ == "__main__":
    train_with_guard()
