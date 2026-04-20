import pandas as pd
import numpy as np
import os
from src.neural.dimensional_expansion import prepare_15d_manifold
from rich.console import Console

console = Console()

class ForensicHarvester:
    """
    Project Chronos - Genetic Harvester
    Forces trades to map indicator states to win rates.
    """
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.cols_of_interest = [
            'vel', 'vol_rel', 'u_wick', 'l_wick', 'z_score', 'mtf_proxy', 'hi_rel', 'lo_rel',
            'buy_wall_prox', 'sell_wall_prox', 'rsi', 'macd_hist', 'atr', 'bb_up_dist', 'bb_lo_dist'
        ]

    def run_harvest(self, months=2):
        console.print(f"[bold cyan]HARVESTER:[/bold cyan] Initiating {months}-month data mine on {self.symbol}...")
        
        df = pd.read_csv(f"data/raw/{self.symbol}_5m.csv")
        df = prepare_15d_manifold(df)
        
        # Taking last 2 months (~17,280 bars)
        df_target = df.tail(17280).copy()
        
        harvest_data = []
        
        # Genetic Harvest: Force trade every 1 hour (12 bars) to get diverse states
        for i in range(0, len(df_target) - 30, 12):
            # 1. Capture State (15D)
            state = df_target.iloc[i][self.cols_of_interest].to_dict()
            
            # 2. Simulate Random Directional Choice (Force Alpha)
            # We'll log outcomes for BOTH Long and Short for every point
            price = df_target.iloc[i]['close']
            exit_px = df_target.iloc[i + 6]['close'] # 30 min window
            
            # LONG Outcome
            roi_long = (exit_px - price) / price
            state_long = state.copy()
            state_long['direction'] = 'long'
            state_long['roi'] = roi_long
            state_long['win'] = 1 if roi_long > 0.0008 else 0 # Net of fee
            harvest_data.append(state_long)
            
            # SHORT Outcome
            roi_short = (price - exit_px) / price
            state_short = state.copy()
            state_short['direction'] = 'short'
            state_short['roi'] = roi_short
            state_short['win'] = 1 if roi_short > 0.0008 else 0
            harvest_data.append(state_short)
            
        harvest_df = pd.DataFrame(harvest_data)
        os.makedirs("research", exist_ok=True)
        harvest_df.to_csv("research/forensic_harvest.csv", index=False)
        console.print(f"[bold green]HARVEST COMPLETE:[/bold green] {len(harvest_df)} states logged to research/forensic_harvest.csv")

if __name__ == "__main__":
    harvester = ForensicHarvester()
    harvester.run_harvest()
