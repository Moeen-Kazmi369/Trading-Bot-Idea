import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

class ChronosTrapHunter:
    """
    Chronos Trap Hunter v1
    Logic: Inverse Strategy. 
    1. Identify 'Shadow Absorption' (Long wicks + High Volume).
    2. Wait for price to BREAK BELOW the low of that absorption candle.
    3. SHORT the failure (Trading the retail stop-losses).
    """
    
    def __init__(self, symbol="BTCUSDT", timeframe="5m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_path = f"data/raw/{symbol}_{timeframe}.csv"
        
    def load_data(self):
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def calculate_logic(self, df):
        df = df.copy()
        
        # --- 1. DETECT THE TRAP CANDLE ---
        df['body'] = abs(df['close'] - df['open'])
        df['lower_wick'] = np.where(df['close'] > df['open'], 
                                     df['open'] - df['low'], 
                                     df['close'] - df['low'])
        df['vol_sma'] = df['volume'].rolling(window=20).mean()
        
        # Trap Signal: Massive wick + Heavy Volume
        # This is where retail thinks "Institutional Buying!"
        df['is_potential_trap'] = np.where(
            (df['lower_wick'] > df['body'] * 1.5) & 
            (df['volume'] > df['vol_sma'] * 1.5), 1, 0
        )
        
        # Store the low of the trap candle as our 'Trigger Line'
        df['trap_trigger_price'] = np.where(df['is_potential_trap'] == 1, df['low'], np.nan)
        df['trap_trigger_price'] = df['trap_trigger_price'].ffill().shift(1)
        
        # A trap is active for 5 bars
        df['trap_age'] = df['is_potential_trap'].rolling(window=5).sum()
        
        return df

    def backtest(self):
        df = self.load_data()
        df = self.calculate_logic(df)
        
        balance = 1000.0
        position = 0 # 0=None, -1=Short
        entry_price = 0
        trades = []
        
        # Simulation parameters
        tp_target = 0.015 # 1.5% profit on the crash
        sl_target = 0.007 # 0.7% stop loss (tight)
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            
            # --- EXIT LOGIC ---
            if position == -1:
                # Win (Price went DOWN for a Short)
                if row['low'] <= entry_price * (1 - tp_target):
                    profit = balance * tp_target
                    balance += profit
                    trades.append({'time': row['timestamp'], 'result': 'WIN', 'profit': profit})
                    position = 0
                # Loss (Price went UP for a Short)
                elif row['high'] >= entry_price * (1 + sl_target):
                    loss = balance * sl_target
                    balance -= loss
                    trades.append({'time': row['timestamp'], 'result': 'LOSS', 'profit': -loss})
                    position = 0
            
            # --- ENTRY LOGIC (THE TRAP HUNTER) ---
            elif position == 0:
                # Rule: A trap candle occurred recently AND price just broke BELOW its low
                if row['trap_age'] > 0 and row['close'] < row['trap_trigger_price']:
                    position = -1 # Go SHORT
                    entry_price = row['close']
                    
        return balance, trades

if __name__ == "__main__":
    strategy = ChronosTrapHunter()
    final_balance, trades = strategy.backtest()
    
    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']
    
    table = Table(title="CHRONOS TRAP HUNTER - BACKTEST REPORT")
    table.add_column("Metric", style="magenta")
    table.add_column("Result", style="bold white")
    table.add_row("Starting Capital", "$1,000.00")
    table.add_row("Ending Capital", f"${final_balance:,.2f}")
    table.add_row("Total Trades", str(len(trades)))
    table.add_row("Win Rate", f"{(len(wins)/len(trades)*100 if trades else 0):.2f}%")
    table.add_row("Profit Factor", f"{(abs(sum([t['profit'] for t in wins])) / abs(sum([t['profit'] for t in losses])) if losses else 0):.2f}")
    
    console.print(table)
