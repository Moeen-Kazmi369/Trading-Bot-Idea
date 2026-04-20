"""
RL Agent v3 — 1m Timeframe Run
Same Q-Learning agent, adapted for 1m speed:
  - Lookback: 10 bars (instead of 20)
  - Momentum thresholds tighter (1m moves are smaller)
  - 200 epochs on last 30 days of 1m BTC data
"""
import pandas as pd
import numpy as np
import random
from rich.console import Console
from rich.table import Table

console = Console()

TIMEFRAME = "1m"
DAYS      = 30
EPOCHS    = 200
LOOKBACK  = 10

def build_state_array(df):
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    vol   = df['volume'].values
    n     = len(df)

    mom_s   = np.ones(n, dtype=int)
    mom_l   = np.ones(n, dtype=int)
    squeeze = np.zeros(n, dtype=int)
    vsurge  = np.zeros(n, dtype=int)

    for i in range(LOOKBACK, n):
        # Short momentum (5 bars)
        ret_s = (close[i] - close[i-5]) / close[i-5]
        mom_s[i] = 0 if ret_s < -0.001 else (2 if ret_s > 0.001 else 1)

        # Long momentum (LOOKBACK bars)
        ret_l = (close[i] - close[i-LOOKBACK]) / close[i-LOOKBACK]
        mom_l[i] = 0 if ret_l < -0.002 else (2 if ret_l > 0.002 else 1)

        # Volatility
        pr = (high[i-LOOKBACK:i].max() - low[i-LOOKBACK:i].min()) / close[i]
        squeeze[i] = 0 if pr < 0.002 else 1

        # Volume surge
        avg_v = vol[i-LOOKBACK:i].mean()
        vsurge[i] = 0 if vol[i] < avg_v else 1

    return mom_s, mom_l, squeeze, vsurge


class QAgent:
    def __init__(self):
        self.q   = np.zeros((3, 3, 2, 2, 2, 3))
        self.lr  = 0.15
        self.gamma = 0.95
        self.eps = 1.0

    def act(self, s, exploit=False):
        if not exploit and random.random() < self.eps:
            return random.randint(0, 2)
        return int(np.argmax(self.q[s]))

    def learn(self, s, a, r, ns):
        best = np.max(self.q[ns])
        self.q[s][a] += self.lr * (r + self.gamma * best - self.q[s][a])


def run_episode(agent, mom_s, mom_l, squeeze, vsurge, close_px, tc=0.0005, exploit=False):
    n = len(mom_s)
    position = 0
    entry_px = 0.0
    flat_streak = 0
    balance = 1000.0
    wins = losses = 0

    for i in range(LOOKBACK, n - 1):
        s  = (mom_s[i], mom_l[i], squeeze[i], vsurge[i], position)
        a  = agent.act(s, exploit=exploit)
        ni = i + 1

        reward = 0.0
        new_pos = position

        if a == 1 and position == 0:       # BUY
            new_pos  = 1
            entry_px = close_px[i]
            flat_streak = 0
            reward = -tc * 100

        elif a == 2 and position == 1:     # SELL
            pnl = (close_px[i] - entry_px) / entry_px
            reward = pnl * 100 - tc * 100
            balance *= (1 + pnl - tc)
            new_pos = 0
            flat_streak = 0
            if exploit:
                if reward > 0: wins += 1
                else:          losses += 1

        else:                              # HOLD
            flat_streak += 1
            if position == 0 and flat_streak > 60:
                reward = -0.05

        ns = (mom_s[ni], mom_l[ni], squeeze[ni], vsurge[ni], new_pos)

        if not exploit:
            agent.learn(s, a, reward, ns)

        position = new_pos

    return balance, wins, losses


# ── LOAD DATA ───────────────────────────────────────────────
path = f"data/raw/BTCUSDT_{TIMEFRAME}.csv"
df = pd.read_csv(path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[df['timestamp'] >= df['timestamp'].max() - pd.Timedelta(days=DAYS)].reset_index(drop=True)

split   = int(len(df) * 0.8)
train   = df.iloc[:split].reset_index(drop=True)
test    = df.iloc[split:].reset_index(drop=True)

t_ms, t_ml, t_sq, t_vs = build_state_array(train)
x_ms, x_ml, x_sq, x_vs = build_state_array(test)

agent = QAgent()

console.print(f"\n[bold cyan]RL Agent | {TIMEFRAME} | Train: {len(train)} bars | Test: {len(test)} bars | Epochs: {EPOCHS}[/bold cyan]")

for e in range(EPOCHS):
    run_episode(agent, t_ms, t_ml, t_sq, t_vs, train['close'].values)
    agent.eps = max(0.02, agent.eps * 0.978)
    if (e + 1) % 40 == 0:
        console.print(f"  Epoch {e+1}/{EPOCHS} | Epsilon: {agent.eps:.3f}")

console.print(f"\n[bold green]Blind test on {len(test)} unseen 1m bars...[/bold green]")

final_bal, wins, losses = run_episode(
    agent, x_ms, x_ml, x_sq, x_vs,
    test['close'].values, exploit=True
)

total = wins + losses
roi   = (final_bal - 1000) / 1000 * 100
wr    = wins / total * 100 if total > 0 else 0

table = Table(title=f"RL Agent — {TIMEFRAME} Blind Test (Last {DAYS} Days)")
table.add_column("Metric",         style="cyan")
table.add_column("Value",          style="bold white")
table.add_row("Starting Capital",  "$1,000.00")
table.add_row("Ending Capital",    f"${final_bal:,.2f}")
table.add_row("ROI",               f"[green]{roi:.2f}%[/green]" if roi > 0 else f"[red]{roi:.2f}%[/red]")
table.add_row("Total Trades",      str(total))
table.add_row("Wins / Losses",     f"{wins} / {losses}")
table.add_row("Win Rate",          f"{wr:.1f}%")
console.print(table)
