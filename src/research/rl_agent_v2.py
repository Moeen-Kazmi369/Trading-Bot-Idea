"""
Reinforcement Learning Agent v2 - Q-Learning
Fixes:
  1. 1000 training epochs (was 10)
  2. Inactivity penalty forces the agent to actually trade
  3. Richer 5-feature state space for better market vision
  4. Proper epsilon decay to reach near-zero randomness by end of training
"""
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
        self.tc = transaction_cost
        self.reset()

    def reset(self):
        self.step_idx = 20
        self.position = 0
        self.entry_price = 0
        self.flat_streak = 0   # Counts how many steps agent stays flat
        self.balance = 1000.0
        self.done = False
        return self._state()

    def _state(self):
        if self.step_idx >= len(self.df) - 1:
            self.done = True
            return (0, 0, 0, 0, 0)

        w = self.df.iloc[self.step_idx - 20 : self.step_idx]
        c = self.df.iloc[self.step_idx]

        # Feature 1: Short momentum (last 5 bars)
        short_ret = (c['close'] - w.iloc[-5]['close']) / w.iloc[-5]['close']
        mom_s = 0 if short_ret < -0.002 else (2 if short_ret > 0.002 else 1)

        # Feature 2: Long momentum (last 20 bars)
        long_ret = (c['close'] - w.iloc[0]['close']) / w.iloc[0]['close']
        mom_l = 0 if long_ret < -0.005 else (2 if long_ret > 0.005 else 1)

        # Feature 3: Volatility (are we in a squeeze?)
        price_range = (w['high'].max() - w['low'].min()) / w['close'].mean()
        vol = 0 if price_range < 0.004 else 1

        # Feature 4: Volume surge (is big money moving in?)
        vol_surge = 0 if c['volume'] < w['volume'].mean() else 1

        # Feature 5: Position
        pos = self.position

        return (mom_s, mom_l, vol, vol_surge, pos)

    def step(self, action):
        price = self.df.iloc[self.step_idx]['close']
        reward = 0

        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = price
            self.flat_streak = 0
            reward = -self.tc * 100  # Small cost for entering

        elif action == 2 and self.position == 1:  # Sell / Close
            self.position = 0
            profit = (price - self.entry_price) / self.entry_price
            reward = profit * 100 - self.tc * 100
            self.balance *= (1 + profit - self.tc)
            self.flat_streak = 0

        else:  # Hold
            self.flat_streak += 1
            if self.position == 0 and self.flat_streak > 30:
                reward = -0.05  # Inactivity penalty — forces the agent to seek trades

        self.step_idx += 1
        return self._state(), reward, self.done


class QAgent:
    def __init__(self):
        # State dimensions: 3 x 3 x 2 x 2 x 2 = 72 states, 3 actions
        self.q = np.zeros((3, 3, 2, 2, 2, 3))
        self.lr = 0.15
        self.gamma = 0.95
        self.epsilon = 1.0

    def act(self, state, exploit=False):
        if not exploit and random.random() < self.epsilon:
            return random.randint(0, 2)
        return int(np.argmax(self.q[state]))

    def learn(self, s, a, r, ns):
        current = self.q[s][a]
        best_next = np.max(self.q[ns])
        self.q[s][a] = (1 - self.lr) * current + self.lr * (r + self.gamma * best_next)


def run():
    path = "data/raw/BTCUSDT_5m.csv"
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Split: 80% train, 20% test (no future peeking)
    cutoff = int(len(df) * 0.8)
    train_df = df.iloc[:cutoff].reset_index(drop=True)
    test_df  = df.iloc[cutoff:].reset_index(drop=True)

    agent = QAgent()
    epochs = 1000
    decay = 0.99 ** (1 / epochs * 10) # Reaches ~0.05 by end

    console.print(f"\n[bold cyan]AGENT SPAWNED - Training on {len(train_df)} bars ({epochs} epochs)...[/bold cyan]")

    for e in range(epochs):
        env = TradingEnv(train_df)
        state = env.reset()
        while not env.done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
        agent.epsilon = max(0.02, agent.epsilon * 0.993)

        if (e + 1) % 100 == 0:
            console.print(f"  Epoch {e+1}/{epochs} | Epsilon: {agent.epsilon:.3f}")

    # -- BLIND TEST on held-out data the agent NEVER saw --
    console.print(f"\n[bold green]TESTING on unseen {len(test_df)} bars...[/bold green]")
    env = TradingEnv(test_df)
    state = env.reset()
    wins, losses = 0, 0
    start_bal = env.balance

    while not env.done:
        prev_state = state
        action = agent.act(state, exploit=True)
        state, reward, done = env.step(action)
        if action == 2 and prev_state[4] == 1:  # Closed a trade
            if reward > 0: wins += 1
            else: losses += 1

    final_bal = env.balance
    total_trades = wins + losses
    roi = (final_bal - start_bal) / start_bal * 100
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    table = Table(title="RL Agent v2 — Blind Test Report (Unseen Data)")
    table.add_column("Metric", style="cyan")
    table.add_column("Result", style="bold white")
    table.add_row("Starting Capital",   f"${start_bal:,.2f}")
    table.add_row("Ending Capital",     f"${final_bal:,.2f}")
    table.add_row("ROI",                f"[green]{roi:.2f}%[/green]" if roi > 0 else f"[red]{roi:.2f}%[/red]")
    table.add_row("Total Trades Made",  str(total_trades))
    table.add_row("Wins / Losses",      f"{wins} / {losses}")
    table.add_row("Win Rate",           f"{win_rate:.1f}%")
    console.print(table)


if __name__ == "__main__":
    run()
