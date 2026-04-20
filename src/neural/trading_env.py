import pandas as pd
import numpy as np
import os
from rich.console import Console

console = Console()

class ProductionTradingEnv:
    """
    FRONT-END NEURAL ENVIRONMENT (2026 Standard)
    This is the 'Gym' where the AI learns to trade.
    Implements: Differential Sharpe Ratio, Friction, and Dense Rewards.
    """
    
    def __init__(self, df, fee_pct=0.0004, initial_balance=1000.0):
        self.df = df
        self.fee_pct = fee_pct
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        self.step_idx = 50 # Allow for lookback
        self.balance = self.initial_balance
        self.position = 0 # 0=None, 1=Long, -1=Short
        self.entry_price = 0
        self.inventory = 0
        self.max_drawdown = 0
        self.peak_balance = self.initial_balance
        
        # DSR Statistics (Differential Sharpe Ratio)
        self.dsr_stats = {'mean': 0.0, 'var': 1.0}
        self.done = False
        
        return self._get_observation()

    def _get_observation(self):
        """
        Builds the raw Market DNA Tensors (The Input to the Brain)
        """
        if self.step_idx >= len(self.df) - 1:
            self.done = True
            return np.zeros(10) # End of episode
            
        # Snapshot of the last 5 bars (Volume + Price Action)
        window = self.df.iloc[self.step_idx-5 : self.step_idx]
        
        # Feature Engineering: Raw Tensors
        obs = [
            (window['close'].iloc[-1] / window['close'].iloc[0]) - 1, # Price Momentum
            (window['volume'].iloc[-1] / window['volume'].mean()) - 1, # Volume Intensity
            (window['high'].max() - window['low'].min()) / window['close'].mean(), # Range/Volatility
            self.position, # Current Position State
            (self.df.iloc[self.step_idx]['close'] / self.entry_price - 1) if self.position != 0 else 0 # Unrealized P&L
        ]
        return np.array(obs, dtype=np.float32)

    def calculate_reward(self, pnl, vol_ema):
        """
        IMPLEMETATION OF THE PRODUCTION-GRADE REWARD (DSR)
        """
        # 1. Net PNL with Friction (Fees)
        net_pnl = pnl - (abs(pnl) * self.fee_pct)
        
        # 2. Volatility Scaling
        norm_pnl = net_pnl / (vol_ema + 1e-6)
        
        # 3. Differential Sharpe Ratio (Incremental improvement to strategy)
        eta = 0.01 
        new_mean = (1 - eta) * self.dsr_stats['mean'] + eta * net_pnl
        new_var = (1 - eta) * self.dsr_stats['var'] + eta * (net_pnl**2)
        
        reward = (new_var * (net_pnl - self.dsr_stats['mean']) - 
                  0.5 * self.dsr_stats['mean'] * (net_pnl**2 - new_var)) / (new_var**1.5 + 1e-6)
        
        # Update internal stats
        self.dsr_stats = {'mean': new_mean, 'var': new_var}
        
        # 4. Inventory/Drawdown Circuit Breakers
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown > 0.02:
            reward -= 5.0 # Heavy negative impulse for hitting 2% DD limit
            
        return reward

    def step(self, action):
        """
        Action Space: 0=Stay, 1=Long, 2=Short, 3=Exit
        """
        current_price = self.df.iloc[self.step_idx]['close']
        vol_ema = self.df['high'].rolling(window=14).std().iloc[self.step_idx] / current_price
        
        step_pnl = 0
        
        # Execute Action
        if action == 1 and self.position == 0: # ENTER LONG
            self.position = 1
            self.entry_price = current_price
            step_pnl = -self.fee_pct # Entry Cost
        elif action == 2 and self.position == 0: # ENTER SHORT
            self.position = -1
            self.entry_price = current_price
            step_pnl = -self.fee_pct # Entry Cost
        elif action == 3 and self.position != 0: # EXIT
            if self.position == 1:
                step_pnl = (current_price - self.entry_price) / self.entry_price - self.fee_pct
            else:
                step_pnl = (self.entry_price - current_price) / self.entry_price - self.fee_pct
            self.position = 0
        elif self.position != 0: # HOLD (Dense Reward)
            if self.position == 1:
                step_pnl = (current_price - self.df.iloc[self.step_idx-1]['close']) / self.df.iloc[self.step_idx-1]['close']
            else:
                step_pnl = (self.df.iloc[self.step_idx-1]['close'] - current_price) / self.df.iloc[self.step_idx-1]['close']

        # Update Account
        self.balance *= (1 + step_pnl)
        if self.balance > self.peak_balance: self.peak_balance = self.balance
        
        # Calculate Reward
        reward = self.calculate_reward(step_pnl, vol_ema)
        
        self.step_idx += 1
        return self._get_observation(), reward, self.done, {}

if __name__ == "__main__":
    console.print("[bold green]NEURAL TRADING ENVIRONMENT INITIALIZED.[/bold green]")
    console.print("Ready for Phase 2: Imitation Learning.")
