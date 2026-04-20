"""
Demand Zone Discovery - R&D Sprint
====================================
Tests 4 mathematical methods for finding Demand Zones on historical BTC data.
Each method is self-validated by checking if price BOUNCED at the zone.

Methods:
  M1: Volume Profile High Volume Node (HVN)
  M2: Fractal Swing Low Cluster
  M3: Price Imbalance / Fair Value Gap (Demand Vacuum)
  M4: Volatility Compression (Bollinger Width Squeeze)

Data Horizons (per your assumption):
  1m  → last 4 hours  = 240 bars
  5m  → last 1 day    = 288 bars
  15m → last 3 days   = 288 bars
"""

import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

# ─────────────────────────────────────────────
# DATA HORIZON CONFIG (Your assumption, validated)
# ─────────────────────────────────────────────
HORIZON_BARS = {
    "1m":  240,   # 4 hours
    "5m":  288,   # 1 day
    "15m": 288,   # 3 days
}

SYMBOL = "BTCUSDT"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_df(symbol, timeframe):
    path = f"data/raw/{symbol}_{timeframe}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["body"] = abs(df["close"] - df["open"])
    df["avg_body"] = df["body"].rolling(50).mean()
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def validate_bounce(df, zone_idx, zone_high, zone_low, lookahead=20):
    """
    After zone is identified at zone_idx, check if price:
      1. Touches the zone in the next `lookahead` candles
      2. Then closes higher (demand confirmed)
    Returns: (touched: bool, bounced: bool)
    """
    future = df.iloc[zone_idx + 1 : zone_idx + 1 + lookahead]
    if future.empty:
        return False, False
    touched = (future["low"] <= zone_high) & (future["close"] >= zone_low)
    if not touched.any():
        return False, False
    first_touch_idx = touched.idxmax()
    after_touch = df.iloc[first_touch_idx + 1 : first_touch_idx + 6]
    if after_touch.empty:
        return True, False
    bounced = after_touch["close"].max() > df.iloc[first_touch_idx]["high"]
    return True, bool(bounced)


# ─────────────────────────────────────────────
# METHOD 1: Volume Profile HVN
# Demand Zone = price bucket with the highest volume concentration
# ─────────────────────────────────────────────
def method_m1_volume_hvn(df, horizon):
    results = []
    for i in range(horizon, len(df) - 25, horizon // 2):
        window = df.iloc[i - horizon : i]
        # Build volume profile with 50 price buckets
        price_range = window["high"].max() - window["low"].min()
        if price_range == 0: continue
        bucket_size = price_range / 50
        window = window.copy()
        window["bucket"] = ((window["close"] - window["low"].min()) / bucket_size).astype(int).clip(0, 49)
        vol_profile = window.groupby("bucket")["volume"].sum()
        # HVN = bucket with max volume
        hvn_bucket = vol_profile.idxmax()
        zone_low  = window["low"].min() + hvn_bucket * bucket_size
        zone_high = zone_low + bucket_size
        touched, bounced = validate_bounce(df, i, zone_high, zone_low)
        results.append({
            "method": "M1 Vol Profile HVN",
            "index": i,
            "timestamp": df.iloc[i]["timestamp"],
            "zone_high": round(zone_high, 2),
            "zone_low": round(zone_low, 2),
            "touched": touched,
            "bounced": bounced,
            "valid": bounced
        })
    return results


# ─────────────────────────────────────────────
# METHOD 2: Fractal Swing Low Cluster
# Fractal Low = candle N is the lowest low of N±3 neighbours
# Demand Zone = 3 or more fractal lows within 0.5% price band
# ─────────────────────────────────────────────
def method_m2_fractal_cluster(df, horizon):
    results = []
    neighbour = 3
    # Find all fractal lows in entire df
    fractal_lows = []
    for i in range(neighbour, len(df) - neighbour):
        window_lows = df.iloc[i - neighbour : i + neighbour + 1]["low"]
        if df.iloc[i]["low"] == window_lows.min():
            fractal_lows.append((i, df.iloc[i]["low"], df.iloc[i]["timestamp"]))
    
    # Group fractals into clusters within 0.5% band
    for j in range(1, len(fractal_lows)):
        idx_j, price_j, ts_j = fractal_lows[j]
        idx_prev, price_prev, _ = fractal_lows[j - 1]
        if abs(price_j - price_prev) / price_prev < 0.005:  # 0.5% band
            zone_low  = min(price_j, price_prev) * 0.9995
            zone_high = max(price_j, price_prev) * 1.0005
            touched, bounced = validate_bounce(df, idx_j, zone_high, zone_low)
            results.append({
                "method": "M2 Fractal Cluster",
                "index": idx_j,
                "timestamp": ts_j,
                "zone_high": round(zone_high, 2),
                "zone_low": round(zone_low, 2),
                "touched": touched,
                "bounced": bounced,
                "valid": bounced
            })
    return results


# ─────────────────────────────────────────────
# METHOD 3: Price Imbalance / Fair Value Gap
# Gap = high of candle[i] < low of candle[i+2]
# This gap is a "demand vacuum" — price tends to fill it
# ─────────────────────────────────────────────
def method_m3_fvg(df, horizon):
    results = []
    for i in range(1, len(df) - 3):
        c1, c3 = df.iloc[i], df.iloc[i + 2]
        # Bullish FVG (Demand): c1 high < c3 low (gap where demand lives)
        if c1["high"] < c3["low"]:
            zone_low  = c1["high"]
            zone_high = c3["low"]
            gap_pct = (zone_high - zone_low) / zone_low * 100
            if gap_pct < 0.05: continue  # Ignore micro-noise gaps
            touched, bounced = validate_bounce(df, i + 2, zone_high, zone_low)
            results.append({
                "method": "M3 FVG Gap",
                "index": i + 2,
                "timestamp": df.iloc[i + 2]["timestamp"],
                "zone_high": round(zone_high, 2),
                "zone_low": round(zone_low, 2),
                "touched": touched,
                "bounced": bounced,
                "valid": bounced
            })
    return results


# ─────────────────────────────────────────────
# METHOD 4: Volatility Compression (BB Squeeze)
# A "coiled spring" — when the market goes very quiet,
# a big move always follows. The compression ZONE = demand.
# ─────────────────────────────────────────────
def method_m4_bb_squeeze(df, horizon):
    results = []
    period = 20
    df = df.copy()
    df["bb_std"] = df["close"].rolling(period).std()
    df["bb_mean"] = df["close"].rolling(period).mean()
    df["bb_width"] = df["bb_std"] / df["bb_mean"]  # Normalized width
    df["bb_width_min"] = df["bb_width"].rolling(horizon).min()
    
    for i in range(horizon + period, len(df) - 5):
        row = df.iloc[i]
        # Squeeze: current width is at its minimum for the horizon
        if row["bb_width"] <= row["bb_width_min"] * 1.05:
            zone_low  = row["close"] - row["bb_std"]
            zone_high = row["close"] + row["bb_std"]
            touched, bounced = validate_bounce(df, i, zone_high, zone_low)
            results.append({
                "method": "M4 BB Squeeze",
                "index": i,
                "timestamp": row["timestamp"],
                "zone_high": round(zone_high, 2),
                "zone_low": round(zone_low, 2),
                "touched": touched,
                "bounced": bounced,
                "valid": bounced
            })
    return results


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run_rnd():
    timeframes = ["1m", "5m", "15m"]
    grand_results = []

    for tf in timeframes:
        df = load_df(SYMBOL, tf)
        if df is None:
            console.print(f"[red]MISSING DATA: {SYMBOL}_{tf}.csv[/red]")
            continue
        
        horizon = HORIZON_BARS[tf]
        console.print(f"\n[bold cyan]Testing {tf} | {len(df)} bars | Horizon: {horizon}[/bold cyan]")
        
        m1 = method_m1_volume_hvn(df, horizon)
        m2 = method_m2_fractal_cluster(df, horizon)
        m3 = method_m3_fvg(df, horizon)
        m4 = method_m4_bb_squeeze(df, horizon)
        
        for method_results in [m1, m2, m3, m4]:
            total = len(method_results)
            valid = sum(1 for r in method_results if r["valid"])
            touched = sum(1 for r in method_results if r["touched"])
            win_rate = (valid / touched * 100) if touched > 0 else 0
            
            method_name = method_results[0]["method"] if method_results else "N/A"
            grand_results.append({
                "timeframe": tf,
                "method": method_name,
                "total_zones": total,
                "price_touched_zone": touched,
                "confirmed_bounces": valid,
                "bounce_rate_%": round(win_rate, 1),
                "samples": method_results
            })

    return grand_results


def print_summary(grand_results):
    table = Table(title="DEMAND ZONE R&D SUMMARY REPORT (BTC)")
    table.add_column("Timeframe", style="cyan")
    table.add_column("Method", style="yellow")
    table.add_column("Zones Found", style="white")
    table.add_column("Price Touched", style="magenta")
    table.add_column("Bounced (Valid)", style="green")
    table.add_column("Bounce Rate %", style="bold white")
    table.add_column("Verdict", style="bold")
    
    for r in grand_results:
        rate = r["bounce_rate_%"]
        if rate >= 60:
            verdict = "[+] STRONG"
        elif rate >= 40:
            verdict = "[~] MODERATE"
        else:
            verdict = "[-] WEAK"
        table.add_row(
            r["timeframe"], r["method"],
            str(r["total_zones"]), str(r["price_touched_zone"]),
            str(r["confirmed_bounces"]), f"{rate}%", verdict
        )
    
    console.print(table)
    
    # Print 2 Manual Verification Samples from the BEST performing method per timeframe
    console.print("\n[bold yellow]── MANUAL VERIFICATION SAMPLES (Check these on your chart) ──[/bold yellow]")
    by_tf = {}
    for r in grand_results:
        tf = r["timeframe"]
        if tf not in by_tf or r["bounce_rate_%"] > by_tf[tf]["bounce_rate_%"]:
            by_tf[tf] = r

    for tf, best in by_tf.items():
        valid_samples = [s for s in best["samples"] if s["valid"]][:2]
        if not valid_samples:
            continue
        console.print(f"\n[bold green]{tf} Best Method: {best['method']} ({best['bounce_rate_%']}% Bounce Rate)[/bold green]")
        v_table = Table()
        v_table.add_column("Timestamp (UTC)", style="magenta")
        v_table.add_column("Zone High", style="green")
        v_table.add_column("Zone Low", style="red")
        v_table.add_column("Tip: Look For...", style="yellow")
        for s in valid_samples:
            v_table.add_row(
                str(s["timestamp"]),
                str(s["zone_high"]),
                str(s["zone_low"]),
                "Price entering zone → Green bounce candle"
            )
        console.print(v_table)


if __name__ == "__main__":
    console.print("[bold]Starting Demand Zone R&D Sprint...[/bold]")
    results = run_rnd()
    print_summary(results)
