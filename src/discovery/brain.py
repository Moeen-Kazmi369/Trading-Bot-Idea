import os
import pandas as pd
from google import genai
from dotenv import load_dotenv
from src.discovery.scanner import InterestScanner
from rich.console import Console

load_dotenv()
console = Console()

class DiscoveryBrain:
    def __init__(self, points):
        self.points = points
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def crack_pattern(self):
        console.print(f"[bold yellow]Analyzing {len(self.points)} explosive points with Gemini 3 Flash...[/bold yellow]")
        
        # Prepare data summary for prompt
        data_summary = ""
        for i, p in enumerate(self.points[:30]): # First 30 points to stay within reasonable prompt size
            data_summary += f"Explosion {i+1} (Move: {p['move_pct']:.2f}%):\n"
            for j, c in enumerate(p['context_ohlcv']):
                vol_change = "N/A" # Simple placeholder for prompt logic
                data_summary += f"  T-{10-j}: O:{c['open']}, H:{c['high']}, L:{c['low']}, C:{c['close']}, V:{c['volume']}\n"
            data_summary += "---\n"

        prompt = f"""
I am searching for the mathematical signature that precedes explosive price moves (>0.4% in 1 minute) in BTCUSDT.
Below is the OHLCV data for the 10 candles IMMEDIATELY PRECEDING {len(self.points[:30])} such explosions.

TASK:
1. Identify any recurring mathematical sequences or ratios in the Open, High, Low, Close, and Volume data.
2. Specifically look for 'Compression' (decreasing range) or 'Volume Exhaustion/Spiking' before the move.
3. Propose a NEW trading pattern formula based on these insights that I can program.

DATA:
{data_summary}

Structure your answer to provide a programmatic logic (e.g., 'If Candle[i-1].Volume > X and Avg(Body[i-3:i-1]) < Y').
"""
        
        console.print("[cyan]Consulting the Gemini Brain...[/cyan]")
        response = self.client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        
        console.print("[bold green]Discovery Insight Received:[/bold green]")
        console.print(response.text)
        return response.text

if __name__ == "__main__":
    df = pd.read_csv("data/raw/BTCUSDT_1m.csv")
    scanner = InterestScanner(df)
    points = scanner.find_big_moves(threshold_pct=0.4)
    
    brain = DiscoveryBrain(points)
    brain.crack_pattern()
