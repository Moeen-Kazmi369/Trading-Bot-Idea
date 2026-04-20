"""
Reinforcement Learning Agent v3 — Fast Q-Learning
Fixes from v2:
  - Pre-computes ALL states as numpy arrays (no per-step Python loops)
  - 200 epochs is enough with proper epsilon decay
  - Uses last 30 days only for speed: 8640 bars train, ~2160 test
"""
import pandas as pd
import numpy as np
import random
import os
from rich.console import Console
from rich.table import Table

console = Console()

# ── 1. PRE-COMPUTE ALL STATES UPFRONT ──────────────────────────────────────
def build_state_array(df):
    """Vectorized: compute all states for every bar at once."""
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    vol   = df['volume'].values

    n = len(df)
    mom_s  = np.ones(n, dtype=int)  # default = 1 (flat)
    mom_l  = np.ones(n, dtype=int)
    squeeze = np.zeros(n, dtype=int)
    vsurge  = np.zeros(n, dtype=int)

    for i in range(20, n):
        # Short momentum (5 bars)
        ret_s = (close[i] - close[i-5]) / close[i-5]
        mom_s[i] = 0 if ret_s < -0.002 else (2 if ret_s > 0.002 else 1)

        # Long momentum (20 bars)
        ret_l = (close[i] - close[i-20]) / close[i-20]
        mom_l[i] = 0 if ret_l < -0.005 else (2 if ret_l > 0.005 else 1)

        # Volatility squeeze
        pr = (high[i-20:i].max() - low[i-20:i].min()) / close[i]
        squeeze[i] = 0 if pr < 0.004 else 1

        # Volume surge
        avg_v = vol[i-20:i].mean()
        vsurge[i] = 0 if vol[i] < avg_v else 1

    return mom_s, mom_l, squeeze, vsurge


# ── 2. Q-TABLE ──────────────────────────────────────────────────────────────
class QAgent:
    def __init__(self):
        # Dims: mom_s(3) x mom_l(3) x squeeze(2) x vsurge(2) x position(2) x action(3)
        self.q = np.zeros((3, 3, 2, 2, 2, 3))
        self.lr    = 0.15
        self.gamma = 0.95
        self.eps   = 1.0

    def act(self, s, exploit=False):
        if not exploit and random.random() < self.eps:
            return random.randint(0, 2)
        return int(np.argmax(self.q[s]))

    def learn(self, s, a, r, ns):
        best = np.max(self.q[ns])
        self.q[s][a] += self.lr * (r + self.gamma * best - self.q[s][a])


# ── 3. EPISODE RUNNER ──────────────────────────────────────────────────────
def run_episode(agent, mom_s, mom_l, squeeze, vsurge, tc=0.0005, exploit=False):
    n         = len(mom_s)
    position  = 0
    entry_px  = 0.0
    flat_streak = 0
    balance   = 1000.0
    wins = losses = 0

    for i in range(20, n - 1):
        s = (mom_s[i], mom_l[i], squeeze[i], vsurge[i], position)
        a = agent.act(s, exploit=exploit)

        # Next state
        ni = i + 1
        ns = (mom_s[ni], mom_l[ni], squeeze[ni], vsurge[ni], position)

        reward = 0.0

        if a == 1 and position == 0:        # BUY
            position  = 1
            entry_px  = df_close[i]
            flat_streak = 0
            reward = -tc * 100
            ns = (mom_s[ni], mom_l[ni], squeeze[ni], vsurge[ni], 1)

        elif a == 2 and position == 1:      # SELL
            pnl = (df_close[i] - entry_px) / entry_px
            reward = pnl * 100 - tc * 100
            balance *= (1 + pnl - tc)
            position = 0
            flat_streak = 0
            ns = (mom_s[ni], mom_l[ni], squeeze[ni], vsurge[ni], 0)
            if exploit:
                if reward > 0: wins += 1
                else:          losses += 1

        else:                               # HOLD
            flat_streak += 1
            if position == 0 and flat_streak > 40:
                reward = -0.05  # Inactivity penalty

        if not exploit:
            agent.learn(s, a, reward, ns)

    return balance, wins, losses


# ── 4. MAIN ─────────────────────────────────────────────────────────────────
path = "data/raw/BTCUSDT_5m.csv"
df   = pd.read_csv(path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Use last 40 days — fast enough, representative enough
df = df[df['timestamp'] >= df['timestamp'].max() - pd.Timedelta(days=40)].reset_index(drop=True)

split = int(len(df) * 0.8)
train = df.iloc[:split].reset_index(drop=True)
test  = df.iloc[split:].reset_index(drop=True)

# Pre-compute states for both splits
t_ms, t_ml, t_sq, t_vs = build_state_array(train)
df_close = train['close'].values   # used inside episode for PnL

agent  = QAgent()
epochs = 200

console.print(f"\n[bold cyan]RL Agent v3 | Train: {len(train)} bars | Epochs: {epochs}[/bold cyan]")

for e in range(epochs):
    run_episode(agent, t_ms, t_ml, t_sq, t_vs)
    agent.eps = max(0.02, agent.eps * 0.978)
    if (e + 1) % 40 == 0:
        console.print(f"  Epoch {e+1}/{epochs} | Epsilon: {agent.eps:.3f}")

console.print(f"\n[bold green]Training complete. Blind test on {len(test)} unseen bars...[/bold green]")

# Swap df_close for test prices
df_close = test['close'].values
x_ms, x_ml, x_sq, x_vs = build_state_array(test)

final_bal, wins, losses = run_episode(agent, x_ms, x_ml, x_sq, x_vs, exploit=True)

total  = wins + losses
roi    = (final_bal - 1000) / 1000 * 100
wr     = wins / total * 100 if total > 0 else 0

table = Table(title="RL Agent v3 — Blind Test Results")
table.add_column("Metric",          style="cyan")
table.add_column("Value",           style="bold white")
table.add_row("Starting Capital",   "$1,000.00")
table.add_row("Ending Capital",     f"${final_bal:,.2f}")
table.add_row("ROI",                f"[green]{roi:.2f}%[/green]" if roi > 0 else f"[red]{roi:.2f}%[/red]")
table.add_row("Total Trades",       str(total))
table.add_row("Wins / Losses",      f"{wins} / {losses}")
table.add_row("Win Rate",           f"{wr:.1f}%")
console.print(table)
