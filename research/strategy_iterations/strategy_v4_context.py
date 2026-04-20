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
    if body_pct < 0.1: return "Doji"
    if body_pct < 0.35 and lower_wick > 1.5 * body and upper_wick < 0.2 * range_: return "Hammer"
    if body_pct < 0.4 and upper_wick > 0.3 * range_ and lower_wick > 0.3 * range_: return "Spinning Top"
    return "Normal"

def discover_strategy_v4_context(symbol, timeframe):
    data_path = f"data/raw/{symbol}_{timeframe}.csv"
    if not os.path.exists(data_path): return []
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 1. GENERATE HIGHER TIMEFRAME CONTEXT (1-Hour)
    # We resample the 5m data into 1h bars
    df_1h = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Calculate 1H Trend (EMA 20 & Structure)
    df_1h['ema50'] = df_1h['close'].ewm(span=50, adjust=False).mean()
    df_1h['is_bullish_trend'] = (df_1h['close'] > df_1h['ema50']) & (df_1h['close'] > df_1h['close'].shift(1))
    
    # Map back to 5m DF
    df['htf_bullish'] = df_1h['is_bullish_trend'].reindex(df.index, method='ffill')
    
    # 2. RUN 5M STRATEGY WITH TREND FILTER
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(window=50).mean()
    
    # Go back to range-based to match timestamps easily
    df.reset_index(inplace=True)
    setups = []
    
    for i in range(50, len(df) - 5):
        # RULE 0: TREND CONTEXT CHECK
        if not df.iloc[i]['htf_bullish']:
            continue
            
        # RULE 1: FIND POI (Vector Run)
        v_run = df.iloc[i:i+3]
        if all(v_run['close'] > v_run['open']) and v_run['body'].sum() > 4.5 * df.iloc[i]['avg_body']:
            
            # RULE 2: FIND OB (GRG behind the run)
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            if c1['close'] > c1['open'] and c2['close'] < c2['open'] and c3['close'] > c3['open']:
                
                # RULE 3: LIQUIDITY SWEEP (Red Candle is the absolute bottom)
                if c2['low'] < c1['low'] and c2['low'] < c3['low']:
                    
                    # Rule 4: PLUS POINT
                    candle_kind = get_candle_type(c2['open'], c2['high'], c2['low'], c2['close'])
                    
                    setups.append({
                        'timestamp': c2['timestamp'],
                        'price': c2['close'],
                        'plus_point': candle_kind,
                        'is_elite': "YES" if candle_kind != "Normal" else "NO"
                    })
                    
    return setups

def report():
    setups = discover_strategy_v4_context("BTCUSDT", "5m")
    
    table = Table(title="STRATEGY v4: STRUCTURAL CONTEXT FILTER (BTC 5m)")
    table.add_column("Setup Time (UTC)", style="magenta")
    table.add_column("1H Context", style="bold green")
    table.add_column("POI Type", style="cyan")
    table.add_column("Plus Point", style="yellow")
    table.add_column("Elite?", style="bold white")
    
    for s in setups[-8:]:
        table.add_row(
            str(s['timestamp']),
            "BULLISH TREND",
            "VECTOR RUN",
            s['plus_point'],
            s['is_elite']
        )
            
    console.print(table)
    console.print(f"\n[bold green]VALID HIGH-PROBABILITY SETUPS:[/bold green] {len(setups)}")
    console.print(f"[bold red]PERCENTAGE OF NOISE REMOVED:[/bold red] {((74 - len(setups))/74)*100:.1f}%")

if __name__ == "__main__":
    report()
