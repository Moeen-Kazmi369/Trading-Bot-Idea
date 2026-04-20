import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

class ChronosAnomalyFilter:
    """
    Chronos Anomaly Filter v1 - Final Strategy
    Logic: Spark Detection + Trap Exclusion.
    The bot only trades when a move is discovered WITHOUT 'Trap Signatures'.
    """
    
    def __init__(self, symbol="BTCUSDT", timeframe="5m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_path = f"data/raw/{symbol}_{timeframe}.csv"
        
    def load_data(self):
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def calculate_indicators(self, df):
        df = df.copy()
        # --- 1. THE SPARK (Momentum Signal) ---
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['upper_band'] = rolling_mean + (1.5 * rolling_std)
        df['spark_signal'] = np.where(df['close'] > df['upper_band'], 1, 0)
        
        # --- 2. THE TRAP DETECTOR (Shadow Absorption Signal) ---
        df['body'] = abs(df['close'] - df['open'])
        df['lower_wick'] = np.where(df['close'] > df['open'], 
                                     df['open'] - df['low'], 
                                     df['close'] - df['low'])
        df['vol_sma'] = df['volume'].rolling(window=14).mean()
        df['wick_to_body'] = df['lower_wick'] / (df['body'] + 1e-9)
        df['vol_surge'] = df['volume'] / (df['vol_sma'] + 1e-9)
        
        # We classify a 'Trap' if wick/vol signals were high in the last 5 bars
        df['is_trap'] = np.where((df['wick_to_body'] > 1.2) & (df['vol_surge'] > 1.2), 1, 0)
        df['trap_score'] = df['is_trap'].rolling(window=5).max()
        
        return df

    def backtest(self):
        df = self.load_data()
        df = self.calculate_indicators(df)
        
        balance = 1000.0
        position = 0
        entry_price = 0
        trades = []
        
        # Simulation parameters
        tp_target = 0.02 # 2% Take Profit
        sl_target = 0.01 # 1% Stop Loss
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            
            # --- EXIT LOGIC ---
            if position == 1:
                # Check TP/SL
                if row['high'] >= entry_price * (1 + tp_target):
                    profit = balance * tp_target
                    balance += profit
                    trades.append({'time': row['timestamp'], 'result': 'WIN', 'profit': profit})
                    position = 0
                elif row['low'] <= entry_price * (1 - sl_target):
                    loss = balance * sl_target
                    balance -= loss
                    trades.append({'time': row['timestamp'], 'result': 'LOSS', 'profit': -loss})
                    position = 0
            
            # --- ENTRY LOGIC (THE ANOMALY FILTER) ---
            elif position == 0:
                # Rule: High Spark AND LOW Trap Score
                if row['spark_signal'] == 1 and row['trap_score'] == 0:
                    position = 1
                    entry_price = row['close']
                    
        return balance, trades

if __name__ == "__main__":
    strategy = ChronosAnomalyFilter()
    final_balance, trades = strategy.backtest()
    
    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']
    
    table = Table(title="CHRONOS ANOMALY FILTER - BACKTEST REPORT")
    table.add_column("Metric", style="magenta")
    table.add_column("Result", style="bold white")
    table.add_row("Starting Capital", "$1,000.00")
    table.add_row("Ending Capital", f"${final_balance:,.2f}")
    table.add_row("Total Trades", str(len(trades)))
    table.add_row("Win Rate", f"{(len(wins)/len(trades)*100 if trades else 0):.2f}%")
    table.add_row("Profit Factor", f"{(abs(sum([t['profit'] for t in wins])) / abs(sum([t['profit'] for t in losses])) if losses else 0):.2f}")
    
    console.print(table)
