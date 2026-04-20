# Technical Risks - TradingBotAI

## Potential Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Overfitting** | High | Use separate "training" and "testing" data sets for every strategy before graduating it. |
| **LLM Hallucinations** | Med | Never trust an AI-generated strategy without a rigorous backtest. Let numbers, not text, decide. |
| **Binance API Limits** | Med | Use local caching for historical data. Batch high-volume requests and respect Rate Limits. |
| **Data Integrity** | Med | Validate OHLCV data for gaps or errors that could skew backtest results (e.g., flash crashes). |
| **Strategy Decay** | High | Run 24/7 monitoring of "active" strategy performance. When win rates drop, flag for re-discovery. |
| **Market Slippage** | Med | Factor in realistic slippage/fees into backtests (especially for 1m scalping). |
| **Look-ahead Bias** | High | Strict codebase rules: Ensure logic only accesses data from *before* the current decision point. |

## Identified Unknowns
- Gemini API performance for complex, numeric OHLCV pattern generation.
- Real-world latency for 1m Binance WebSocket entries.
- High-level coin selection impact on Order Block win rates.
