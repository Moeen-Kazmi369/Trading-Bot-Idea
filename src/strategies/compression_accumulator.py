import pandas as pd
import numpy as np

class CompressionAccumulator:
    def __init__(self, df):
        self.df = df.copy()
        
    def find_signals(self):
        signals = []
        # Need at least 10 candles for SMA
        for i in range(10, len(self.df) - 2):
            # 1. Volume Ratio (T-1 Vol / Avg Vol last 10)
            avg_vol = self.df.iloc[i-10:i]['volume'].mean()
            vol_ratio = self.df.iloc[i]['volume'] / avg_vol if avg_vol > 0 else 0
            
            # 2. Range Ratio (T-1 Range / Avg Range last 10)
            current_range = self.df.iloc[i]['high'] - self.df.iloc[i]['low']
            ranges = (self.df.iloc[i-10:i]['high'] - self.df.iloc[i-10:i]['low'])
            avg_range = ranges.mean()
            range_ratio = current_range / avg_range if avg_range > 0 else 0
            
            # 3. Body Ratio (T-1 Body / T-1 Range)
            current_body = abs(self.df.iloc[i]['close'] - self.df.iloc[i]['open'])
            body_ratio = current_body / current_range if current_range > 0 else 0
            
            # 4. Anchor (Flatness last 4 candles)
            four_min_move = abs(self.df.iloc[i]['close'] - self.df.iloc[i-4]['open']) / self.df.iloc[i-4]['open']
            
            # CHECK GEMINI'S SECRET SIGNATURE
            if vol_ratio > 1.3 and range_ratio < 0.85 and body_ratio < 0.4 and four_min_move < 0.001:
                # Prediction: >0.4% move in next 120s (2 candles)
                # Bias: Use 10-period SMA
                sma_10 = self.df.iloc[i-10:i]['close'].mean()
                direction = "LONG" if self.df.iloc[i]['close'] > sma_10 else "SHORT"
                
                signals.append({
                    'index': i,
                    'timestamp': self.df.iloc[i]['timestamp'],
                    'type': direction,
                    'price': self.df.iloc[i]['close'],
                    'vol_ratio': vol_ratio,
                    'range_ratio': range_ratio
                })
        return signals

if __name__ == "__main__":
    df = pd.read_csv("data/raw/BTCUSDT_1m.csv")
    strategy = CompressionAccumulator(df)
    signals = strategy.find_signals()
    print(f"Discovered {len(signals)} Compression Accumulator Signals!")
    if signals:
        print(f"Sample Signal: {signals[0]}")
