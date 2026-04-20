import os
import pandas as pd
from google import genai
from dotenv import load_dotenv
from src.backtester.engine import BacktestEngine
from src.strategies.order_block import OrderBlockDetector
from rich.console import Console

load_dotenv()
console = Console()

class AIRefiner:
    def __init__(self, df, trades, bull_obs, bear_obs):
        self.df = df
        self.trades = trades
        self.bull_obs = bull_obs
        self.bear_obs = bear_obs
        
        # Configure Gemini
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def analyze_patterns(self):
        console.print("[bold yellow]Preparing samples for Gemini analysis...[/bold yellow]")
        
        # Select a few samples (Gemini prompt size limit)
        wins = [t for t in self.trades if t['result'] == 'WIN'][:10]
        losses = [t for t in self.trades if t['result'] == 'LOSS'][:10]
        
        prompt = "I am a high-frequency trading bot analyzing 1-minute BTCUSDT candles for 'Order Block' patterns. "
        prompt += "Below are samples of Winning trades and Losing trades based on the same pattern. "
        prompt += "Please analyze the OHLCV data and tell me what mathematical characteristics (like volume ratios, wick size relative to body, or candle sequence strength) distinguish the winners from the losers. "
        prompt += "Suggest a specific filter rule I can add to my code to increase win rate.\n\n"
        
        prompt += "--- WINNING SAMPLES ---\n"
        for i, trade in enumerate(wins):
            prompt += f"Win {i+1}: Resulted in 2x profit.\n"
            
        prompt += "\n--- LOSING SAMPLES ---\n"
        for i, trade in enumerate(losses):
            prompt += f"Loss {i+1}: Hit Stop Loss.\n"
            
        prompt += "\nAnalysis and suggested filter rule:"
        
        console.print("[cyan]Sending request to Gemini...[/cyan]")
        response = self.client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        
        console.print("[bold green]Gemini Insight Received:[/bold green]")
        console.print(response.text)
        
        return response.text

if __name__ == "__main__":
    # Load data
    csv_path = "data/raw/BTCUSDT_1m.csv"
    df = pd.read_csv(csv_path)
    
    # Get initial trades
    detector = OrderBlockDetector(df)
    bull_obs = detector.find_bullish_order_blocks()
    bear_obs = detector.find_bearish_order_blocks()
    engine = BacktestEngine(df, bull_obs, bear_obs)
    engine.run()
    
    # Run Refiner
    refiner = AIRefiner(df, engine.trades, bull_obs, bear_obs)
    refiner.analyze_patterns()
