"""
Latest Demand Zones - M2 Fractal Cluster (The Winner)
Extracts the most recent VALID demand zones for manual chart verification.
"""
import pandas as pd
import numpy as np

HORIZON_BARS = {"1m": 240, "5m": 288, "15m": 288}
SYMBOL = "BTCUSDT"

def load_df(symbol, timeframe):
    path = f"data/raw/{symbol}_{timeframe}.csv"
    df = pd.read_csv(path)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

def validate_bounce(df, zone_idx, zone_high, zone_low, lookahead=20):
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

def get_latest_zones(symbol, timeframe, top_n=3):
    df = load_df(symbol, timeframe)
    neighbour = 3
    fractal_lows = []
    for i in range(neighbour, len(df) - neighbour):
        window_lows = df.iloc[i - neighbour : i + neighbour + 1]["low"]
        if df.iloc[i]["low"] == window_lows.min():
            fractal_lows.append((i, df.iloc[i]["low"], df.iloc[i]["timestamp"]))

    # Group into clusters
    all_zones = []
    for j in range(1, len(fractal_lows)):
        idx_j, price_j, ts_j = fractal_lows[j]
        idx_prev, price_prev, _ = fractal_lows[j - 1]
        if abs(price_j - price_prev) / price_prev < 0.005:
            zone_low  = min(price_j, price_prev) * 0.9995
            zone_high = max(price_j, price_prev) * 1.0005
            touched, bounced = validate_bounce(df, idx_j, zone_high, zone_low)
            all_zones.append({
                "timeframe": timeframe,
                "timestamp_utc": ts_j,
                "zone_high": round(zone_high, 2),
                "zone_low": round(zone_low, 2),
                "touched": touched,
                "bounced": bounced,
                "valid": bounced
            })

    # Sort by timestamp descending, return last valid ones
    valid = [z for z in all_zones if z["valid"]]
    valid_sorted = sorted(valid, key=lambda x: x["timestamp_utc"], reverse=True)
    return valid_sorted[:top_n]

print("=" * 65)
print("  LATEST VALID DEMAND ZONES - M2 Fractal Cluster")
print("  Manual Verification Guide: Open chart in UTC timezone")
print("=" * 65)

for tf in ["1m", "5m", "15m"]:
    zones = get_latest_zones(SYMBOL, tf, top_n=3)
    print(f"\n--- {tf} Timeframe (last {len(zones)} confirmed zones) ---")
    for i, z in enumerate(zones, 1):
        local_ts = z["timestamp_utc"].tz_convert("Asia/Karachi")
        print(f"  Zone #{i}")
        print(f"    Timestamp (UTC)   : {z['timestamp_utc']}")
        print(f"    Timestamp (PKT)   : {local_ts}  <-- use this on your chart")
        print(f"    Demand Zone       : ${z['zone_low']:,.2f}  to  ${z['zone_high']:,.2f}")
        print(f"    Validated (Bounce): YES")
        print(f"    What to look for  : Price dipped into this zone then closed GREEN above it")
        print()
