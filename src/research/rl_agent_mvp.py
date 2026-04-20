import pandas as pd
import numpy as np
import random
import os
from rich.console import Console
from rich.table import Table

console = Console()

class TradingEnv:
    def __init__(self, df, transaction_cost=0.0005):
        self.df = df
        self.transaction_cost = transaction_cost
        self.reset()
        
    def reset(self):
        self.current_step = 5 # Start index to allow lookback
        self.position = 0 # 0 = Neutral, 1 = Long
        self.entry_price = 0
        self.done = False
        self.balance = 1000.0 # Starting capital
        return self._get_state()
        
    def _get_state(self):
        # We simplify the infinite market into a finite "State Space" for the AI
        if self.current_step >= len(self.df) - 1:
            self.done = True
            return (0, 0, 0)
            
        lookback = self.df.iloc[self.current_step - 5 : self.current_step]
        
        # Feature 1: Price momentum (0: Dump, 1: Flat, 2: Pump)
        pct_change = (lookback['close'].iloc[-1] - lookback['close'].iloc[0]) / lookback['close'].iloc[0]
        if pct_change < -0.002: momentum = 0
        elif pct_change > 0.002: momentum = 2
        else: momentum = 1
            
        # Feature 2: Volatility squeeze (0: Low Vol, 1: High Vol)
        price_range = (lookback['high'].max() - lookback['low'].min()) / lookback['close'].mean()
        volatility = 0 if price_range < 0.005 else 1
        
        return (momentum, volatility, self.position)
        
    def step(self, action):
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0
        
        if action == 1 and self.position == 0: # Buy
            self.position = 1
            self.entry_price = current_price
            reward = -self.transaction_cost # Small penalty for trading (fees)
            
        elif action == 2 and self.position == 1: # Sell
            self.position = 0
            profit_pct = (current_price - self.entry_price) / self.entry_price
            reward = profit_pct - self.transaction_cost
            self.balance = self.balance * (1 + reward)
            
        elif action == 0 and self.position == 1: # Hold while Long
            # Unrealized reward to guide the agent
            profit_pct = (current_price - self.entry_price) / self.entry_price
            reward = profit_pct * 0.1 # Small breadcrumb reward
            
        self.current_step += 1
        return self._get_state(), reward * 100, self.done # Scale reward up for Q-learning
        
class QAgent:
    def __init__(self):
        # Q-Table stores the expected profit for every Action in every State
        # States: 3 momentums * 2 volatilities * 2 positions = 12 states
        # Actions: 3
        self.q_table = np.zeros((3, 2, 2, 3)) 
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_decay = 0.99
        
    def act(self, state, exploit_only=False):
        m, v, p = state
        if not exploit_only and random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2]) # Explore randomly
        return np.argmax(self.q_table[m, v, p]) # Exploit learned strategy
        
    def learn(self, state, action, reward, next_state):
        m, v, p = state
        nm, nv, np_pos = next_state
        
        # The core Q-Learning equation (Bellman equation)
        old_value = self.q_table[m, v, p, action]
        future_optimal = np.max(self.q_table[nm, nv, np_pos])
        
        # Give the agent a new brain connection
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * future_optimal)
        self.q_table[m, v, p, action] = new_value

def run_simulation():
    # Load 1 month of BTC 5m data for the AI to train on
    path = "data/raw/BTCUSDT_5m.csv"
    if not os.path.exists(path):
        console.print("[red]Data not found.[/red]")
        return
        
    df = pd.read_csv(path)
    # Take the last 30 days to keep it fast
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'] >= df['timestamp'].max() - pd.Timedelta(days=30)].reset_index(drop=True)
    
    env = TradingEnv(df)
    agent = QAgent()
    
    epochs = 10 # Let the AI trade history 10 times to "Learn"
    
    console.print("\n[bold cyan]🧠 SPAWNING REINFORCEMENT LEARNING AGENT (Q-Learning)[/bold cyan]")
    console.print(f"Training on {len(df)} candles...")
    
    # --- TRAINING PHASE ---
    for e in range(epochs):
        state = env.reset()
        total_reward = 0
        while not env.done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        agent.epsilon *= agent.epsilon_decay # Slowly transition from exploring to exploiting
        console.print(f"  Epoch {e+1}/{epochs} | Score (Rewards): {total_reward:.2f} | Epsilon (Randomness): {agent.epsilon:.2f}")

    # --- TESTING PHASE (Exploit Only) ---
    console.print("\n[bold green]🚀 TRAINING COMPLETE. RUNNING BLIND TEST (No Exploration)[/bold green]")
    state = env.reset()
    trades_won = 0
    trades_lost = 0
    starting_balance = env.balance
    
    while not env.done:
        action = agent.act(state, exploit_only=True)
        next_state, reward, done = env.step(action)
        
        # Track purely completed trades
        if action == 2 and state[2] == 1: 
            if reward > 0: trades_won += 1
            else: trades_lost += 1
            
        state = next_state

    final_balance = env.balance
    roi = ((final_balance - starting_balance) / starting_balance) * 100
    
    table = Table(title="AI Agent Performance Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Result", style="bold white")
    table.add_row("Starting Capital", f"${starting_balance:,.2f}")
    table.add_row("Ending Capital", f"${final_balance:,.2f}")
    table.add_row("Total Return (ROI)", f"{'[green]' if roi > 0 else '[red]'}{roi:.2f}%")
    table.add_row("Winning Trades", str(trades_won))
    table.add_row("Losing Trades", str(trades_lost))
    
    console.print(table)
    console.print("[dim italic]Note: This is an MVP. The agent created its own strategy from scratch with zero human rules.[/dim italic]\n")

if __name__ == "__main__":
    run_simulation()
