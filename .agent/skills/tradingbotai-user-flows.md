# User Flows - TradingBotAI

## Mission
To map out the core interactions of the TradingBotAI system, ensuring every step from data to trade is logical and testable.

## Flow 1: Strategy Backtesting (M1)
1.  **Fetcher:** Get historical 1m OHLCV for X pairs.
2.  **Detector:** Identify Order Block patterns (GRG/RGR).
3.  **Simulate:** When a pattern is found, simulate an entry at HH + 3% (Bullish).
4.  **Validate:** Determine if price hits TP (2x SL) or SL (133% offset).
5.  **Report:** Count wins/losses and calculate stats.

## Flow 2: Automated Researcher (M2) 24/7
1.  **Statistical Scan:** Find coins/times where price moved > 2% in < 5 mins.
2.  **Identify Points:** Record these "Interesting Points."
3.  **Analyze (Gemini):** Pass OHLCV snapshots to Gemini to ask: "What happened before this move?"
4.  **Hypothesis:** Gemini generates a new pattern formula.
5.  **Test:** Feed the new formula to M1 to backtest. If win rate > 60%, log as a "Discovered Pattern."

## Flow 3: Testnet Trading (M3) - Phase 3
1.  **Coin Selection:** Filter coins by RSI/Fib/Indicators.
2.  **Stream:** Start Binance WebSocket for filtered coins.
3.  **Active Scan:** Continuously scan real-time 1m candles for active patterns.
4.  **Execution:** Send Testnet Market/Limit Order when setup triggers.
5.  **Outcome:** Log profit/loss based on real Testnet execution.

## Flow 4: Live Trading (M4) - Phase 3
1.  **Criteria:** M3 reports win rate > 60% with profit factor > 1.5.
2.  **Switch:** Change API Keys to Live Future/Spot account.
3.  **Execute:** Same flow as M3 but with real capital.
4.  **Stop-Switch:** Kill strategy if drawdown exceeds 10%.
