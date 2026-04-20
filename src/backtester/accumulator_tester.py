import pandas as pd
from src.strategies.compression_accumulator import CompressionAccumulator
from rich.console import Console
from rich.table import Table

console = Console()

class AccumulatorBacktester:
    def __init__(self, df, signals):
        self.df = df
        self.signals = signals
        self.trades = []

    def run(self):
        for sig in self.signals:
            start_idx = sig['index'] + 1
            entry_price = sig['price']
            sig_type = sig['type']
            
            # Simple Exit Logic: 
            # Exit after 2 candles OR hit 0.4% profit
            # Stop Loss at 0.2%
            target_pct = 0.004
            sl_pct = 0.002
            
            for i in range(start_idx, min(start_idx + 3, len(self.df))):
                row = self.df.iloc[i]
                
                if sig_type == "LONG":
                    # Check SL
                    if row['low'] <= entry_price * (1 - sl_pct):
                        self.trades.append({'result': 'LOSS', 'pnl': -1})
                        break
                    # Check TP
                    if row['high'] >= entry_price * (1 + target_pct):
                        self.trades.append({'result': 'WIN', 'pnl': 2})
                        break
                else: # SHORT
                    if row['high'] >= entry_price * (1 + sl_pct):
                        self.trades.append({'result': 'LOSS', 'pnl': -1})
                        break
                    if row['low'] <= entry_price * (1 - target_pct):
                        self.trades.append({'result': 'WIN', 'pnl': 2})
                        break
                
                # Time exit after 2 candles
                if i == start_idx + 2:
                    current_pnl = 0
                    if sig_type == "LONG":
                        current_pnl = (row['close'] - entry_price) / entry_price
                    else:
                        current_pnl = (entry_price - row['close']) / entry_price
                    
                    if current_pnl > 0:
                        self.trades.append({'result': 'WIN', 'pnl': 1})
                    else:
                        self.trades.append({'result': 'LOSS', 'pnl': -1})

        return self._generate_report()

    def _generate_report(self):
        if not self.trades:
            return "No trades taken."
            
        tdf = pd.DataFrame(self.trades)
        wins = len(tdf[tdf['result'] == 'WIN'])
        total = len(tdf)
        win_rate = (wins/total)*100
        pnl = tdf['pnl'].sum()
        
        table = Table(title="STRATEGY v2: Compression Accumulator (Gemini Discovered)")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Total Trades", str(total))
        table.add_row("Win Rate", f"{win_rate:.2f}%")
        table.add_row("Total PnL", f"{pnl:.2f}")
        console.print(table)

if __name__ == "__main__":
    df = pd.read_csv("data/raw/BTCUSDT_1m.csv")
    strategy = CompressionAccumulator(df)
    signals = strategy.find_signals()
    
    tester = AccumulatorBacktester(df, signals)
    tester.run()
