import pandas as pd

HORIZON = {"1m": 240, "5m": 288, "15m": 288}
BAR_MINUTES = {"1m": 1, "5m": 5, "15m": 15}

files = {
    "1m":  "data/raw/BTCUSDT_1m.csv",
    "5m":  "data/raw/BTCUSDT_5m.csv",
    "15m": "data/raw/BTCUSDT_15m.csv",
}

for tf, path in files.items():
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    total_bars = len(df)
    start = df["timestamp"].min()
    end   = df["timestamp"].max()
    span_days = (end - start).days

    horizon_bars = HORIZON[tf]
    horizon_minutes = horizon_bars * BAR_MINUTES[tf]
    horizon_hours = horizon_minutes / 60

    print(f"Timeframe : {tf}")
    print(f"  File          : {path}")
    print(f"  Total Bars    : {total_bars:,}")
    print(f"  Starts From   : {start}")
    print(f"  Ends At       : {end}")
    print(f"  Total Span    : {span_days} days")
    print(f"  Horizon Used  : {horizon_bars} bars = {horizon_hours:.0f} hours of context per scan window")
    print()
