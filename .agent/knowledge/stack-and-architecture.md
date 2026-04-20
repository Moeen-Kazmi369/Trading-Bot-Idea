# Stack and Architecture - TradingBotAI

## Suggested Stack
- **Language:** Python 3.12+ (standard for data/AI)
- **AI Brain:** Google Gemini API (via `google-genai` SDK) - User has Google AI Studio Developer Plan.
- **Binance API:** `python-binance` or `ccxt` for candle data (Rest) and execution (WS).
- **Data Analysis:** `pandas`, `numpy`, `scipy` for OHLCV processing and statistical analysis.
- **Database:** SQLite (initially for MVP), PostgreSQL (scaling).
- **Scheduling:** `APScheduler` for 24/7 background loops.
- **TUI (Reporting):** `Rich` for beautiful terminal output and JSON reports.
- **Frontend (Future):** Next.js with WebSockets.

## Architecture
The system is built as a **4-Module Pipeline** for modularity and scalability.

### M1: Strategy Trainer (Backtester & Refiner)
Takes a pattern (like Order Blocks) and backtests it against massive historical data. It learns what makes a specific instance of a pattern "valid" or "invalid" to improve accuracy.

### M2: Pattern Discovery (The Researcher)
The heart of the system. It runs 24/7 on historical data seeking "Interesting Points" where big moves occurred. It then uses Gemini AI to design formulas or patterns that predict these points, then feeds them to M1.

### M3: Real-Time Tester (The Sandbox)
Executes production-ready strategies on the Binance Testnet. This is paper trading using live WebSocket data.

### M4: Live Trader (The Executor)
Graduates proven M3 strategies to live futures/spot trading.

## Discovery Pipeline (3-Layer Discovery)
1.  **Layer 1:** Find "Interesting Points" (historical facts).
2.  **Layer 2:** Find "Pattern/Formula" (prediction logic).
3.  **Layer 3:** Find "Trading Strategy" (entry/SL/TP optimization).

## Decisions Made
- **Price Action Only:** Use OHLCV data as the "truth," as everything else is noise or already baked into the price.
- **Human-in-the-Loop Seeds:** Start with the developer's 3-year market experience (Order Blocks) to prime the AI.
- **Testnet-First:** No live execution without proven statistical success.
