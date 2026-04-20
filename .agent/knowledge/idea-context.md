# IDEA CONTEXT
============
Idea Name:          TradingBotAI
Problem:            Retail traders lose because they follow publicly known, exploited 
                    strategies. Existing bots are dumb if/else systems. No system 
                    discovers, validates, and adapts to hidden market controller 
                    patterns — or evolves when those patterns change.

Target Users:       You — 3-year experienced trader with deep market understanding.
                    Personal weapon first, potential product later.

Core Solution:      AI-powered autonomous trading system that uses your market 
                    knowledge as seed intelligence, trains/refines strategies via 
                    backtesting, discovers entirely new patterns through a 3-layer 
                    pipeline, and self-improves 24/7. Starts on Binance Testnet, 
                    graduates to live when proven.

Must-Have Features:
  - Historical 1m candle data fetcher from Binance (all futures pairs)
  - Order block strategy (GRG/RGR) as seed strategy
  - Backtesting engine with trade simulation
  - Self-learning loop (test → analyze → refine → retest)
  - 3-layer discovery pipeline (Interesting Points → Pattern → Strategy)
  - Gemini AI integration for hypothesis generation & experiment design
  - Performance reporting (win rate, PnL, drawdown, profit factor)
  - Pattern registry for discovered strategies
  - Coin selection filter layer (RSI, Fibonacci, indicators)

Nice-to-Have Features:
  - Web dashboard (Next.js) for stats, real-time updates, controls
  - Spot trading support
  - Multi-timeframe analysis
  - Telegram/Discord notifications
  - Portfolio management

Out of Scope:
  - News/sentiment-based trading
  - Social media signal following
  - Copy trading from other traders
  - Fixed if/else bots that never learn

Suggested Stack:
  - Language:       Python 3.12+
  - AI Brain:       Google Gemini API (via google-genai SDK)
  - Binance:        python-binance / ccxt
  - Data:           pandas, numpy, scipy
  - Database:       SQLite (MVP) → PostgreSQL (scale)
  - Scheduling:     APScheduler
  - Logging:        Rich (console) + JSON reports
  - Frontend:       Next.js + WebSocket (later)

Architecture:
  4-module system:
  - M1: Strategy Trainer (backtesting + self-refinement)
  - M2: Pattern Discovery (AI-powered 3-layer pipeline, 24/7)
  - M3: Real-Time Tester (Binance Testnet)
  - M4: Live Trader (real money, futures → spot)

  3-Layer Discovery Pipeline:
  - Layer 1: Find Interesting Points (big moves in data)
  - Layer 2: Find Pattern/Formula (what predicts those points)
  - Layer 3: Find Trading Strategy (optimal entry/SL/TP)

  Coin Selection Pre-Filter:
  - Technical indicators (RSI, Fibonacci, etc.) filter which coins 
    are worth scanning for patterns

Development Phases:
  - MVP:  Module 1 — Strategy Trainer
          Data fetcher + Order block detector + Backtesting engine +
          Performance reports + Self-refinement v1
  - V2:   Module 2 — Pattern Discovery
          Statistical analyzer + Gemini AI brain + Experiment runner +
          3-layer pipeline + Pattern registry + 24/7 loop
  - V3:   Module 3+4 — Real Trading
          Coin selection filters + Real-time streaming + Testnet +
          Live trading + Web dashboard

Technical Risks:
  - Overfitting to historical data (mitigate: train/test splits)
  - LLM hypothesis quality (mitigate: validate with backtesting)
  - Data volume (mitigate: efficient storage, batch processing)
  - Binance rate limits (mitigate: caching, backoff)
  - API costs (mitigate: optimize prompts, budget caps)
  - Look-ahead bias (mitigate: strict temporal separation)

Seed Intelligence (Your Knowledge):
  - Market controller vs retailer dynamics
  - Order Block Strategy (Bullish GRG / Bearish RGR)
  - 1m timeframe focus for scalping
  - Entry at HH + 3% of candle height
  - SL at 133% of candle height from entry
  - TP at 2x SL distance
  - Order blocks expire when price touches/crosses them
  - Indicators for coin selection, NOT for trading decisions
  - Patterns change when leaked — system must continuously adapt

Notes:
  - User has Google AI Studio Developer Plan (Gemini API)
  - User will provide Binance API keys when needed
  - Testnet first, always. Live only when stats are proven.
  - System should be modular — M1's strategy trainer is reusable
    for any strategy M2 discovers
