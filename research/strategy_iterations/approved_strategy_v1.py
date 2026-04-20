import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

def get_candle_type(open_, high, low, close):
    body = abs(close - open_)
    range_ = high - low
    if range_ == 0: return "Doji"
    body_pct = body / range_
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    if body_pct < 0.15: return "Doji"
    if body_pct < 0.4 and lower_wick > 1.3 * body and upper_wick < 0.3 * range_: return "Hammer"
    if body_pct < 0.45 and upper_wick > 0.3 * range_ and lower_wick > 0.3 * range_: return "Spinning Top"
    return "Normal"

def run_approved_strategy(symbol, timeframe):
    data_path = f"data/raw/{symbol}_{timeframe}.csv"
    if not os.path.exists(data_path): return []
    
    df = pd.read_csv(data_path)
    # Convert to numeric to avoid errors
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col])
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Momentum Engine: rolling(50) mean body
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(window=50).mean()
    
    # Tracking for Breakouts
    df['res_20'] = df['high'].shift(1).rolling(window=20).max()
    df['sup_20'] = df['low'].shift(1).rolling(window=20).min()
    
    setups = []
    
    for i in range(50, len(df)):
        c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        momentum_avg = df.iloc[i]['avg_body']
        
        # 1. BULLISH GRG Rules
        if c1['close'] > c1['open'] and c2['close'] < c2['open'] and c3['close'] > c3['open']:
            # LL Staircase: C1 Low < C2 Low < C3 Low
            if c1['low'] < c2['low'] and c2['low'] < c3['low']:
                # Momentum Engine: 3 candles combined > 4.5x Avg
                combined_body = c1['body'] + c2['body'] + c3['body']
                if combined_body > 4.5 * momentum_avg:
                    # Breakout: C3 High > Last 20 Highs
                    if c3['high'] > df.iloc[i]['res_20']:
                        candle_kind = get_candle_type(c2['open'], c2['high'], c2['low'], c2['close'])
                        setups.append({
                            'time': c2['timestamp'],
                            'type': 'BULLISH (Demand)',
                            'zone': f"{c2['high']} - {c2['low']}",
                            'plus_point': candle_kind,
                            'price': c3['close']
                        })

        # 2. BEARISH RGR Rules
        if c1['close'] < c1['open'] and c2['close'] > c2['open'] and c3['close'] < c3['open']:
            # HH Staircase: C1 High > C2 High > C3 High
            if c1['high'] > c2['high'] and c2['high'] > c3['high']:
                # Momentum Engine: 3 candles combined > 4.5x Avg
                combined_body = c1['body'] + c2['body'] + c3['body']
                if combined_body > 4.5 * momentum_avg:
                    # Breakout: C3 Low < Last 20 Lows
                    if c3['low'] < df.iloc[i]['sup_20']:
                        candle_kind = get_candle_type(c2['open'], c2['high'], c2['low'], c2['close'])
                        setups.append({
                            'time': c2['timestamp'],
                            'type': 'BEARISH (Supply)',
                            'zone': f"{c2['high']} - {c2['low']}",
                            'plus_point': candle_kind,
                            'price': c3['close']
                        })
                        
    return setups

if __name__ == "__main__":
    results = run_approved_strategy("BTCUSDT", "5m")
    
    table = Table(title="APPROVED STRATEGY EXECUTION (BTC 5m)")
    table.add_column("OB Timestamp (UTC)", style="magenta")
    table.add_column("Type", style="cyan")
    table.add_column("OB Zone (H-L)", style="yellow")
    table.add_column("Plus Point", style="green")
    table.add_column("Exit/Entry Price", style="bold white")
    
    for s in results[-10:]:
        table.add_row(str(s['time']), s['type'], s['zone'], s['plus_point'], f"{s['price']:.2f}")
        
    console.print(table)
    console.print(f"\n[bold cyan]TOTAL APPROVED SETUPS FOUND:[/bold cyan] {len(results)}")
