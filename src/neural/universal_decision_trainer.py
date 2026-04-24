import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import sys
import shutil
from src.neural.transformer_brain import UniversalOracleV2
from src.neural.universal_manifold import prepare_30d_universal_manifold
from rich.console import Console

# force_terminal=True ensures rich streams live even inside Colab subprocesses
console = Console(force_terminal=True, highlight=False)

# Google Drive backup directory (survives runtime resets)
DRIVE_SNAP_DIR = "/content/drive/MyDrive/ChronosData/snapshots"
LOCAL_SNAP_DIR = "models"

SEQ_LEN = 50
FEATURE_DIM = 30


def _extract_windows_fast(df: pd.DataFrame, feature_cols: list, tf: int):
    """
    TURBO WINDOW EXTRACTOR
    ──────────────────────
    Replaces the slow `df.iloc[i]` hot-loop with pure NumPy stride tricks.
    Speed improvement: ~40–80x vs the original loop.

    Returns
    -------
    X : np.ndarray  (N, SEQ_LEN, FEATURE_DIM)
    C : np.ndarray  (N, 2)
    Y : np.ndarray  (N,)  int64
    """
    feat_vals  = df[feature_cols].values.astype(np.float32)   # (T, 30)
    close_vals = df['close'].values.astype(np.float64)
    c_tf_vals  = df['c_timeframe'].values.astype(np.float32)
    c_fee_vals = df['c_fee'].values.astype(np.float32)

    step  = 25 if tf == 5 else 5
    T     = len(df)
    idxs  = np.arange(SEQ_LEN, T - 10, step)          # anchor indices

    if len(idxs) == 0:
        empty_X = np.zeros((0, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        empty_C = np.zeros((0, 2),                    dtype=np.float32)
        empty_Y = np.zeros((0,),                      dtype=np.int64)
        return empty_X, empty_C, empty_Y

    # ── Build X via stride trick (zero-copy view, then contiguous copy once) ──
    # shape (T, 30)  → sliding windows of length SEQ_LEN
    from numpy.lib.stride_tricks import as_strided
    s0, s1 = feat_vals.strides
    full_windows = as_strided(
        feat_vals,
        shape=(T - SEQ_LEN + 1, SEQ_LEN, FEATURE_DIM),
        strides=(s0, s0, s1)
    )                                                   # (T-SEQ+1, 50, 30)
    X = full_windows[idxs - SEQ_LEN].copy()            # (N, 50, 30)  ← ONE copy

    # ── Conditioning ──
    C = np.stack([c_tf_vals[idxs], c_fee_vals[idxs]], axis=1)   # (N, 2)

    # ── Labels (vectorised, no Python loop) ──
    future_idx = np.clip(idxs + 6, 0, T - 1)
    roi = (close_vals[future_idx] - close_vals[idxs]) / (close_vals[idxs] + 1e-10)
    Y = np.where(roi > 0.0016, 1, np.where(roi < -0.0016, 2, 0)).astype(np.int64)

    return X, C, Y


def prepare_split_data():
    """
    Universal Mastery Forge: 6 Coins x 3 Resolutions (18x Data Scaling).
    TURBO MODE: NumPy stride tricks + per-symbol CSV caching.
    """
    symbols    = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
    timeframes = [5, 15, 60]

    train_X_parts, train_C_parts, train_Y_parts = [], [], []
    val_X_parts,   val_C_parts,   val_Y_parts   = [], [], []

    for sym in symbols:
        path = f"data/raw/{sym}_5m.csv"
        if not os.path.exists(path):
            console.print(f"  [bold red]MISSING:[/bold red] {path} — Skipping {sym}")
            continue

        # ── Cache the CSV read ONCE per symbol ──
        df_base = pd.read_csv(path)

        for tf in timeframes:
            console.print(f"  [bold blue]FORGING:[/bold blue] {sym} at {tf}m Resolution...")

            df_raw = df_base.iloc[::(tf // 5)].copy() if tf > 5 else df_base.copy()
            df_raw = df_raw.reset_index(drop=True)

            split_idx = int(len(df_raw) * 0.8)
            df_train, cols = prepare_30d_universal_manifold(df_raw.iloc[:split_idx].reset_index(drop=True), timeframe_mins=tf)
            df_val,   _    = prepare_30d_universal_manifold(df_raw.iloc[split_idx:].reset_index(drop=True), timeframe_mins=tf)

            # Pad cols to exactly FEATURE_DIM
            while len(cols) < FEATURE_DIM:
                cols.append('rsi')

            tX, tC, tY = _extract_windows_fast(df_train, cols, tf)
            vX, vC, vY = _extract_windows_fast(df_val,   cols, tf)

            if len(tX): train_X_parts.append(tX); train_C_parts.append(tC); train_Y_parts.append(tY)
            if len(vX): val_X_parts.append(vX);   val_C_parts.append(vC);   val_Y_parts.append(vY)

    console.print(f"  [bold green]TENSOR FORGE:[/bold green] Stacking {sum(len(x) for x in train_X_parts):,} train + {sum(len(x) for x in val_X_parts):,} val windows...")

    # ── np.concatenate is MUCH faster than np.array(list_of_arrays) ──
    t_X = torch.from_numpy(np.concatenate(train_X_parts, axis=0))
    t_C = torch.from_numpy(np.concatenate(train_C_parts, axis=0))
    t_Y = torch.from_numpy(np.concatenate(train_Y_parts, axis=0))
    v_X = torch.from_numpy(np.concatenate(val_X_parts,   axis=0))
    v_C = torch.from_numpy(np.concatenate(val_C_parts,   axis=0))
    v_Y = torch.from_numpy(np.concatenate(val_Y_parts,   axis=0))

    console.print(f"  [bold green]FORGE COMPLETE:[/bold green] Train={len(t_X):,} | Val={len(v_X):,} | Shape={tuple(t_X.shape)}")
    return t_X, t_C, t_Y, v_X, v_C, v_Y


def _save_snapshot(model, epoch, best_val_loss):
    """Dual-save: local models/ AND Google Drive (survives runtime resets)."""
    os.makedirs(LOCAL_SNAP_DIR, exist_ok=True)
    local_path = f"{LOCAL_SNAP_DIR}/universal_oracle_snap_{epoch}.pth"
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "best_val_loss": best_val_loss}, local_path)

    # Mirror to Drive if available
    if os.path.exists("/content/drive"):
        os.makedirs(DRIVE_SNAP_DIR, exist_ok=True)
        drive_path = f"{DRIVE_SNAP_DIR}/universal_oracle_snap_{epoch}.pth"
        shutil.copy(local_path, drive_path)
        print(f"  SNAPSHOT: Epoch {epoch} saved → Drive ({drive_path})", flush=True)
    else:
        print(f"  SNAPSHOT: Epoch {epoch} saved → Local only", flush=True)


def _find_latest_snapshot():
    """
    Scans Drive first (persistent), then local. Returns (path, epoch) or (None, 0).
    """
    best_epoch = 0
    best_path  = None

    for snap_dir in [DRIVE_SNAP_DIR, LOCAL_SNAP_DIR]:
        if not os.path.exists(snap_dir):
            continue
        for fname in os.listdir(snap_dir):
            if fname.startswith("universal_oracle_snap_") and fname.endswith(".pth"):
                try:
                    ep = int(fname.replace("universal_oracle_snap_", "").replace(".pth", ""))
                    if ep > best_epoch:
                        best_epoch = ep
                        best_path  = os.path.join(snap_dir, fname)
                except ValueError:
                    pass

    return best_path, best_epoch


def train_with_guard(epochs=1000):
    console.print(f"[bold cyan]COLAB MASTERY FORGE:[/bold cyan] Initiating 1,000-Epoch Deep Siege...")

    # 1. Turbo Data Load
    t_X, t_C, t_Y, v_X, v_C, v_Y = prepare_split_data()
    train_loader = DataLoader(TensorDataset(t_X, t_C, t_Y), batch_size=512, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(v_X, v_C, v_Y), batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    # 2. Model Init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"  [bold green]DEVICE:[/bold green] {str(device).upper()}")

    model = UniversalOracleV2(feature_dim=FEATURE_DIM, hidden_dim=256)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)   # Ultra Slow-Burn
    criterion = nn.CrossEntropyLoss()

    os.makedirs(LOCAL_SNAP_DIR, exist_ok=True)

    # 3. AUTO-RESUME: Find and load the latest snapshot from Drive
    start_epoch   = 0
    best_val_loss = float('inf')
    snap_path, snap_epoch = _find_latest_snapshot()

    if snap_path:
        checkpoint = torch.load(snap_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch   = checkpoint["epoch"]          # resume AFTER this epoch
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        print(f"  RESUMED: Loaded Epoch {start_epoch} snapshot from {snap_path}", flush=True)
        print(f"  Best Val Loss so far: {best_val_loss:.4f}", flush=True)
    else:
        print("  FRESH START: No snapshot found — training from Epoch 1", flush=True)

    model.to(device)

    # 4. Training Loop with Overfit Guard + Auto-Snapshot
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for bX, bC, bY in train_loader:
            bX, bC, bY = bX.to(device, non_blocking=True), bC.to(device, non_blocking=True), bY.to(device, non_blocking=True)
            optimizer.zero_grad()
            out  = model(bX, bC)
            loss = criterion(out, bY)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Scan (The Jury)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bX, bC, bY in val_loader:
                bX, bC, bY = bX.to(device, non_blocking=True), bC.to(device, non_blocking=True), bY.to(device, non_blocking=True)
                out  = model(bX, bC)
                loss = criterion(out, bY)
                val_loss += loss.item()

        t_avg = train_loss / len(train_loader)
        v_avg = val_loss   / len(val_loader)

        # Divergence warning
        gap = v_avg - t_avg
        gap_tag = f"  [bold red]⚠ OVERFIT GAP={gap:.3f}[/bold red]" if gap > 0.08 else ""
        # Plain print with flush=True — guaranteed live heartbeat in any environment
        print(f"  Epoch {epoch+1}/{epochs} | Train: {t_avg:.4f} | Val: {v_avg:.4f}{gap_tag}", flush=True)

        # Best model tracker
        if v_avg < best_val_loss:
            best_val_loss = v_avg
            best_path = f"{LOCAL_SNAP_DIR}/universal_oracle_best.pth"
            torch.save({"epoch": epoch+1, "model_state": model.state_dict(), "best_val_loss": best_val_loss}, best_path)
            if os.path.exists("/content/drive"):
                os.makedirs(DRIVE_SNAP_DIR, exist_ok=True)
                shutil.copy(best_path, f"{DRIVE_SNAP_DIR}/universal_oracle_best.pth")

        # Auto-Snapshot every 50 epochs → saved to Drive
        if (epoch + 1) % 50 == 0:
            _save_snapshot(model, epoch+1, best_val_loss)

    final_local = f"{LOCAL_SNAP_DIR}/universal_oracle_guarded_v2.pth"
    torch.save({"epoch": epochs, "model_state": model.state_dict(), "best_val_loss": best_val_loss}, final_local)
    if os.path.exists("/content/drive"):
        os.makedirs(DRIVE_SNAP_DIR, exist_ok=True)
        shutil.copy(final_local, f"{DRIVE_SNAP_DIR}/universal_oracle_guarded_v2.pth")
    console.print(f"\n[bold green]CLOUD SIEGE GUARD COMPLETE.[/bold green] Best Val Loss: [bold yellow]{best_val_loss:.4f}[/bold yellow]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Universal Mastery Forge Trainer")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train (default: 1000)")
    args = parser.parse_args()
    
    train_with_guard(epochs=args.epochs)
