# Decision Log - TradingBotAI

## Mission
To document key technical and product decisions, ensuring the reasoning is captured for long-term consistency.

## Decisions Log

| Date | ID | Decision | Reasoning | Status |
|---|---|---|---|---|
| 2026-03-20 | D01 | **Use Python/Pandas.** | Industry standard for trading data, backtesting, and AI/ML. | ✅ Active |
| 2026-03-20 | D02 | **Google Gemini API.** | User has Google AI Studio Developer Plan; Gemini is top-tier for data/reasoning. | ✅ Active |
| 2026-03-20 | D03 | **Price Action Only.** | Developer's core belief: Price action contains all necessary truths (no noise). | ✅ Active |
| 2026-03-20 | D04 | **4-Module Pipeline.** | Modular development allows M1/M2 to be perfected before M3/M4 trading. | ✅ Active |
| 2026-03-20 | D05 | **Binance REST (MVP).** | REST is simpler for fetching historical candles than WebSockets (for M1/M2). | ✅ Active |
| 2026-03-20 | D06 | **3-Layer Discovery.** | Clearly defines the automated researcher loop (M2). | ✅ Active |

## Decision Recording Process
When making a major decision:
1.  **State the problem.**
2.  **State the chosen solution.**
3.  **Explain the trade-offs (Pros/Cons).**
4.  **Log it here.**
