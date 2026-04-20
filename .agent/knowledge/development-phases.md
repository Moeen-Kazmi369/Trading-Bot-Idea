# Development Phases - TradingBotAI

## MVP: Module 1 — Strategy Trainer
> Goal: Proof of concept with Order Blocks.

- **Phase 1A: Setup & Data Fetching**
    - Connect to Binance.
    - Fetch 1m historical OHLCV.
    - SQLite storage.
- **Phase 1B: Order Block Detector**
    - Implement GRG/RGR logic correctly.
    - Identify candidate Order Blocks in historical data.
- **Phase 1C: Backtesting Engine**
    - Simulate trades with entry/SL/TP rules.
    - Calculate win rate, profit factor, max drawdown.
- **Phase 1D: Self-Refinement v1**
    - Identify common features of "winning" Order Blocks vs. "losing" ones.
    - Adjust parameters and re-test automatically.

## V2: Module 2 — Pattern Discovery
> Goal: Autonomous researcher using Gemini.

- **Phase 2A: Statistical Analyzer**
    - Find "Interesting Points" (high-volume moves, sharp reversals).
- **Phase 2B: LLM Researcher**
    - Integrate Gemini API.
    - Generate hypothesis formulas/patterns based on data.
- **Phase 2C: Experiment Runner**
    - Automatically backtest and validate LLM-generated patterns.
- **Phase 2D: Pattern Registry**
    - Save, version, and rank discovered patterns.
- **Phase 2E: 24/7 researcher loop.**

## V3: Module 3 & 4 — Prototyping & Live
> Goal: Move to the market with live data.

- **Phase 3A: Coin Selection Filters** (RSI, Fibonacci, indicators).
- **Phase 3B: Real-Time Streamer** (Binance WebSocket).
- **Phase 3C: Testnet Trading Executor.**
- **Phase 3D: Live Strategy Graduation.**
- **Phase 3E: Web Dashboard** (Next.js).
