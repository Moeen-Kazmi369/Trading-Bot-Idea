import pandas as pd
from src.strategies.order_block import OrderBlockDetector
from src.strategies.compression_accumulator import CompressionAccumulator
from src.backtester.engine import BacktestEngine
from rich.console import Console

console = Console()

class HybridEngine(BacktestEngine):
    def run(self, accumulator_signals):
        # Create a set of indices where accumulator signals fired
        sig_indices = {s['index'] for s in accumulator_signals}
        
        # Only keep Order Blocks that had a Gemini signal nearby (within 10 candles)
        def is_hybrid(ob_idx):
            for i in range(ob_idx - 10, ob_idx + 1):
                if i in sig_indices:
                    return True
            return False

        filtered_bulls = [ob for ob in self.bull_obs if is_hybrid(ob['index'])]
        filtered_bears = [ob for ob in self.bear_obs if is_hybrid(ob['index'])]
        
        console.print(f"[bold cyan]Hybrid Strategy: Filtered {len(self.bull_obs)+len(self.bear_obs)} OBs down to {len(filtered_bulls)+len(filtered_bears)} verified setups.[/bold cyan]")
        
        self.bull_obs = filtered_bulls
        self.bear_obs = filtered_bears
        
        return super().run()

if __name__ == "__main__":
    df = pd.read_csv("data/raw/BTCUSDT_1m.csv")
    
    # 1. Get OBs
    detector = OrderBlockDetector(df)
    bull_obs = detector.find_bullish_order_blocks()
    bear_obs = detector.find_bearish_order_blocks()
    
    # 2. Get Accumulator Signals
    accumulator = CompressionAccumulator(df)
    signals = accumulator.find_signals()
    
    # 3. Run Hybrid
    engine = HybridEngine(df, bull_obs, bear_obs)
    engine.run(signals)
