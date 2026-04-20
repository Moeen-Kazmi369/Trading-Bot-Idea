import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

class BacktestEngine:
    def __init__(self, df, bull_obs, bear_obs):
        self.df = df
        self.bull_obs = bull_obs
        self.bear_obs = bear_obs
        self.trades = []
        
        # Pre-convert to numpy for 100x speed
        self.lows = df['low'].values
        self.highs = df['high'].values
        self.closes = df['close'].values

    def run(self, apply_refinements=False):
        # Filter OBs based on refinements
        active_bulls = self.bull_obs
        active_bears = self.bear_obs
        
        if apply_refinements:
            active_bulls = [ob for ob in self.bull_obs if ob['vol_ratio'] >= 1.1 and ob['body_ratio'] >= 0.6]
            active_bears = [ob for ob in self.bear_obs if ob['vol_ratio'] >= 1.1 and ob['body_ratio'] >= 0.6]

        # Process Bullish
        for ob in active_bulls:
            self._simulate_trade(ob, 'BULLISH')
            
        # Process Bearish
        for ob in active_bears:
            self._simulate_trade(ob, 'BEARISH')
            
        return self._generate_report()

    def _simulate_trade(self, ob, ob_type):
        start_idx = ob['index'] + 1
        zone_hh, zone_ll, height = ob['zone_hh'], ob['zone_ll'], ob['height']
        
        # Calculated Levels
        entry, sl, tp = 0, 0, 0
        if ob_type == 'BULLISH':
            entry = zone_hh + (0.03 * height)
            sl = entry - (1.33 * height)
            tp = entry + (2.0 * (entry - sl))
        else:
            entry = zone_ll - (0.03 * height)
            sl = entry + (1.33 * height)
            tp = entry - (2.0 * (sl - entry))

        state = "WAITING"
        # Slice arrays to avoid O(N*M) - iterate only after OB index
        l_slice = self.lows[start_idx:]
        h_slice = self.highs[start_idx:]
        
        for i in range(len(l_slice)):
            curr_low = l_slice[i]
            curr_high = h_slice[i]
            
            if state == "WAITING":
                if ob_type == 'BULLISH':
                    if curr_low <= zone_ll: break # Expired
                    if curr_low <= entry: state = "IN"
                else:
                    if curr_high >= zone_hh: break # Expired
                    if curr_high >= entry: state = "IN"
            
            elif state == "IN":
                if ob_type == 'BULLISH':
                    if curr_low <= sl:
                        self.trades.append({'result': 'LOSS', 'pnl': -1})
                        return
                    if curr_high >= tp:
                        self.trades.append({'result': 'WIN', 'pnl': 2})
                        return
                else:
                    if curr_high >= sl:
                        self.trades.append({'result': 'LOSS', 'pnl': -1})
                        return
                    if curr_low <= tp:
                        self.trades.append({'result': 'WIN', 'pnl': 2})
                        return

    def _generate_report(self):
        if not self.trades: return "No trades taken."
        tdf = pd.DataFrame(self.trades)
        wins, total = len(tdf[tdf['result'] == 'WIN']), len(tdf)
        win_rate = (wins / total) * 100
        total_pnl = tdf['pnl'].sum()
        
        days = len(self.df) // 1440
        table = Table(title=f"Report - {days} Days")
        table.add_column("Metric"); table.add_column("Value")
        table.add_row("Total Trades", str(total))
        table.add_row("Win Rate", f"{win_rate:.2f}%")
        table.add_row("Total PnL", f"{total_pnl:.2f}")
        console.print(table)
        return tdf
