# MVP Scope Guard - TradingBotAI

## Mission
To prevent scope creep during the build of **Module 1 (Strategy Trainer)** and **Module 2 (Pattern Discovery)**.

## Core Scope (M1/M2)
- **Data (S1):** OHLCV fetcher (REST) + SQLite/Pandas data storage.
- **Strategy (S2):** Order Block Detection (GRG/RGR), LLM-hypothesized patterns.
- **Backtesting (S3):** Trading simulation (Entry/SL/TP only), calculation of basic stats.
- **Refinement (S4):** Automated strategy optimization (M1) and Pattern discovery (M2).

## Out of Scope (M1/M2)
- **Real-Time Data (O1):** WebSocket streaming or live trading. (Save for M3).
- **UI/Web Dashboard (O2):** Next.js or charts. (Save for Phase 3E).
- **Social/News (O3):** Sentiment analysis, news-scraping. (Never).
- **Portfolio (O4):** Multi-account management. (Never).

## Review Checklist
Before adding a new feature, ask:
1.  **Does it directly improve the backtesting engine's accuracy?**
2.  **Does it directly enable the AI to find "Interesting Points" (Layer 1)?**
3.  **Is it required to prove that the strategy is statistically sound?**

If "No" to all three, defer it to **V3 (M3/M4)**.
