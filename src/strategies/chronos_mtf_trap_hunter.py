import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

class ChronosMTFTrapHunter:
    """
    Chronos MTF Trap Hunter v1
    Goal: Trade the liquidation of 15m traps on the 5m timeframe.
    """
    
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.tf_5m = "5m"
        self.tf_15m = "15m"
        self.path_5m = f"data/raw/{symbol}_5m.csv"
        self.path_15m = f"data/raw/{symbol}_15m.csv"
        
    def load_aligned_data(self):
        df_5m = pd.read_csv(self.path_5m)
        df_15m = pd.read_csv(self.path_15m)
        
        df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
        df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
        
        # --- 1. CALCULATE 15m TRAP SIGNALS ---
        df_15m['body'] = abs(df_15m['close'] - df_15m['open'])
        df_15m['lower_wick'] = np.where(df_15m['close'] > df_15m['open'], 
                                         df_15m['open'] - df_15m['low'], 
                                         df_15m['close'] - df_15m['low'])
        df_15m['vol_sma'] = df_15m['volume'].rolling(window=20).mean()
        
        # 15m Trap: Clearer and stronger than 5m
        df_15m['htf_trap'] = np.where(
            (df_15m['lower_wick'] > df_15m['body'] * 1.5) & 
            (df_15m['volume'] > df_15m['vol_sma'] * 1.5), 1, 0
        )
        df_15m['htf_trigger_px'] = np.where(df_15m['htf_trap'] == 1, df_15m['low'], np.nan)
        
        # --- 2. MERGE 15m SIGNALS INTO 5m ---
        # We use ffill so the '15m Trap' state persists for the next three 5m candles
        df_15m_signals = df_15m[['timestamp', 'htf_trap', 'htf_trigger_px']]
        df = pd.merge_asof(
            df_5m.sort_values('timestamp'), 
            df_15m_signals.sort_values('timestamp'), 
            on='timestamp', 
            direction='backward'
        )
        
        # A 15m trap is 'Hunting' for the next 45 minutes (9 bars of 5m)
        df['htf_trap_active'] = df['htf_trap'].rolling(window=9).max()
        df['htf_trigger_px'] = df['htf_trigger_px'].ffill()
        
        return df

    def backtest(self):
        df = self.load_aligned_data()
        
        balance = 1000.0
        position = 0
        entry_price = 0
        trades = []
        
        # Simulation parameters - We tighten SL and expand TP for MTF moves
        tp_target = 0.025 # 2.5% Target (Liquidations are fast)
        sl_target = 0.008 # 0.8% Stop Loss
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            
            # --- EXIT LOGIC ---
            if position == -1:
                if row['low'] <= entry_price * (1 - tp_target):
                    profit = balance * tp_target
                    balance += profit
                    trades.append({'time': row['timestamp'], 'result': 'WIN', 'profit': profit})
                    position = 0
                elif row['high'] >= entry_price * (1 + sl_target):
                    loss = balance * sl_target
                    balance -= loss
                    trades.append({'time': row['timestamp'], 'result': 'LOSS', 'profit': -loss})
                    position = 0
            
            # --- ENTRY LOGIC (MTF TRAP HUNTER) ---
            elif position == 0:
                # Rule: Is there a 15m Trap Active AND did 5m price just break BELOW that 15m low?
                if row['htf_trap_active'] == 1 and row['close'] < row['htf_trigger_px']:
                    position = -1
                    entry_price = row['close']
                    
        return balance, trades

if __name__ == "__main__":
    strategy = ChronosMTFTrapHunter()
    final_balance, trades = strategy.backtest()
    
    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']
    
    table = Table(title="CHRONOS MTF TRAP HUNTER (15m -> 5m)")
    table.add_column("Metric", style="magenta")
    table.add_column("Result", style="bold white")
    table.add_row("Starting Capital", "$1,000.00")
    table.add_row("Ending Capital", f"${final_balance:,.2f}")
    table.add_row("Total Trades", str(len(trades)))
    table.add_row("Win Rate", f"{(len(wins)/len(trades)*100 if trades else 0):.2f}%")
    table.add_row("Profit Factor", f"{(abs(sum([t['profit'] for t in wins])) / abs(sum([t['profit'] for t in losses])) if losses else 0):.2f}")
    
    console.print(table)
