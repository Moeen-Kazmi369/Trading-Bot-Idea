import pandas as pd
from src.strategies.order_block import OrderBlockDetector

df = pd.read_csv("data/raw/BTCUSDT_1m.csv")
detector = OrderBlockDetector(df)
obs = detector.find_bullish_order_blocks() + detector.find_bearish_order_blocks()

avg_vol = sum(o['vol_ratio'] for o in obs) / len(obs)
avg_body = sum(o['body_ratio'] for o in obs) / len(obs)
fvg_count = sum(1 for o in obs if o['has_fvg'])

print(f"Total OBs: {len(obs)}")
print(f"Avg Volume Ratio: {avg_vol:.2f}")
print(f"Avg Body Ratio: {avg_body:.2f}")
print(f"FVG Count: {fvg_count} ({fvg_count/len(obs)*100:.2f}%)")
