# Git Rules - TradingBotAI

## Branch Naming
- `main` / `master`: Only production-ready or fully tested code.
- `mvp/module-1`: Active development for Module 1.
- `v2/module-2`: Active development for Module 2.
- `feature/[name]`: New features (e.g., `feature/binance-websocket`).
- `fix/[name]`: Bug fixes (e.g., `fix/ohlcv-data-gaps`).
- `research/[name]`: Experimental pattern research branches.

## Commit Conventions
- Use descriptive, imperative commits: `Add Binance data fetcher` or `Implement GRG order block detection`.
- Follow "Atomic Commits": One logical change per commit.
- Never commit API keys or sensitive `.env` files. (Use `.gitignore`).

## Remote Action Rules
- Push to GitHub/repository frequently to avoid logical work loss.
- Review recent backtest results before merging features that impact trade execution logic.
- Ensure all logic is documented, especially OHLCV processing and SL/TP calculation.

## Tagging
- `v0.1.0-mvp`: First working backtest version.
- `v0.2.0-discovery`: First Gemini-AI pattern discovery integration.
- `v1.0.0-testnet-ready`: Ready for paper trading on Binance.
