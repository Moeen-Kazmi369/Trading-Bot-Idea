import os
import pandas as pd
import random
from src.strategies.order_block import OrderBlockDetector
from src.backtester.engine import BacktestEngine
from rich.console import Console

console = Console()

def get_samples():
    data_path = "data/raw/"
    # Focus on BTC, ETH, SOL for easy verification
    sample_files = ["BTCUSDT_5m.csv", "ETHUSDT_5m.csv", "SOLUSDT_5m.csv", "LTCUSDT_1m.csv"]
    
    all_executed_trades = []

    for file in sample_files:
        if not os.path.exists(os.path.join(data_path, file)): continue
        
        df = pd.read_csv(os.path.join(data_path, file))
        symbol = file.split("_")[0]
        tf = file.split("_")[1].replace(".csv", "")
        
        detector = OrderBlockDetector(df)
        bull_obs = detector.find_bullish_order_blocks()
        bear_obs = detector.find_bearish_order_blocks()
        
        engine = BacktestEngine(df, bull_obs, bear_obs)
        # We need to capture the trades with timestamps
        # I'll modify the engine temporarily or just run it and capture the list
        # Since I wrote the engine.py, I'll just look for the logic
        
        # Actually, I'll just re-implement the simulation here briefly to get the TIMESTAMPS
        for ob in bull_obs:
            # Simple simulation to find A FEW that were executed
            start_idx = ob['index'] + 1
            z_hh, z_ll = ob['zone_hh'], ob['zone_ll']
            entry = z_hh + (0.03 * ob['height'])
            
            # Check next 50 candles for entry
            for i in range(start_idx, min(start_idx + 50, len(df))):
                if df.iloc[i]['low'] <= entry and df.iloc[i]['low'] > z_ll:
                    # Trade entered!
                    all_executed_trades.append({
                        'coin': symbol,
                        'tf': tf,
                        'time': df.iloc[ob['index']]['timestamp'], # Pattern Time
                        'type': 'BULLISH',
                        'entry': f"{entry:.4f}",
                        'ob_index': ob['index']
                    })
                    break
        
        for ob in bear_obs:
            start_idx = ob['index'] + 1
            z_hh, z_ll = ob['zone_hh'], ob['zone_ll']
            entry = z_ll - (0.03 * ob['height'])
            for i in range(start_idx, min(start_idx + 50, len(df))):
                if df.iloc[i]['high'] >= entry and df.iloc[i]['high'] < z_hh:
                    all_executed_trades.append({
                        'coin': symbol,
                        'tf': tf,
                        'time': df.iloc[ob['index']]['timestamp'],
                        'type': 'BEARISH',
                        'entry': f"{entry:.4f}",
                        'ob_index': ob['index']
                    })
                    break

    # Pick 5 random ones
    random.shuffle(all_executed_trades)
    samples = all_executed_trades[:5]
    
    for i, s in enumerate(samples):
        console.print(f"\n[bold cyan]SAMPLE TRADE #{i+1}[/bold cyan]")
        console.print(f"Coin: {s['coin']} ({s['tf']})")
        console.print(f"Type: {s['type']}")
        console.print(f"Pattern Detected at: {s['time']}")
        console.print(f"Entry Price: {s['entry']}")
        console.print(f"Check your chart for the Green-Red-Green pattern right at this timestamp.")

if __name__ == "__main__":
    get_samples()
