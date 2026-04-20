import os
import random
import pandas as pd
from dotenv import load_dotenv
from google import genai
from rich.console import Console

load_dotenv()
console = Console()

class GeminiPatternCracker:
    """
    Project Chronos - Frontier Pattern Cracker
    Uses Gemini 1.5 Pro to identify the 'Invisible Divergence' between 
    Real Launches and Market Controller Traps.
    """
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.dataset_dir = "research/chronos_dataset"
        
    def load_random_samples(self, count=5):
        pos_files = [f for f in os.listdir(self.dataset_dir) if f.startswith("launchpad_")]
        neg_files = [f for f in os.listdir(self.dataset_dir) if f.startswith("trap_")]
        
        selected_pos = random.sample(pos_files, count)
        selected_neg = random.sample(neg_files, count)
        
        return selected_pos, selected_neg

    def prepare_data_text(self, pos_files, neg_files):
        text = "FRONTIER QUANTITATIVE RESEARCH DATASET v2 (WITH WICK DATA)\n"
        text += "=========================================================\n\n"
        
        def format_file(filename, label):
            df = pd.read_csv(os.path.join(self.dataset_dir, filename))
            # Keep raw data for wick analysis
            summary = df[['open', 'high', 'low', 'close', 'volume']].tail(50).to_string(index=False)
            return f"SAMPLE ID: {filename} | CLASSIFICATION: {label}\n{summary}\n\n"

        text += "--- SUCCESSFUL TREND LAUNCHPADS (POSITIVE) ---\n"
        for f in pos_files:
            text += format_file(f, "SUCCESSFUL TREND")
            
        text += "--- MARKET CONTROLLER TRAPS (NEGATIVE) ---\n"
        for f in neg_files:
            text += format_file(f, "FAILED TRAP")
            
        return text

    def crack_pattern(self):
        console.print("[bold cyan]CHRONOS:[/bold cyan] Selecting mystery samples for research...")
        pos, neg = self.load_random_samples(5)
        data_text = self.prepare_data_text(pos, neg)
        
        prompt = f"""
        YOU ARE A FRONTIER QUANTITATIVE RESEARCHER. 
        PREVIOUS FORMULA HAD TOO LOW SENSITIVITY (Recall < 5%).
        
        UPGRADE MISSION:
        You must find a MORE SENSITIVE and DISTINCT mathematical signature for Launchpads.
        
        NEW FOCUS AREAS:
        1. WICK ANALYSIS: Detect "Shadow Absorption." Are long lower wicks on high volume (rejection) present in Launchpads?
        2. VSA (Volume Spread Analysis): Is the spread (High-Low) shrinking while Volume is rising?
        3. MOMENTUM SHIFT: Detect the 'Initial Spark' where the last 3 candles break the local volatility band.
        
        OUTPUT REQUIREMENT:
        1. A brief explanation of the "Shadow Absorption" you discovered.
        2. A Python function `calculate_anomaly_score(df)` that returns a score (0 to 100).
        3. IMPORTANT: Ensure the score is calibrated so that Launchpads hit > 70 regularly.
        
        DATASET:
        {data_text}
        """
        
        console.print("[bold yellow]RESEARCHING v2:[/bold yellow] Sending Wick Data to Gemini Brain...")
        
        response = self.client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )
        
        console.print("\n[bold green]FRONTIER DISCOVERY COMPLETE:[/bold green]\n")
        console.print(response.text)
        
        # Save the hypothesis
        with open("research/latest_hypothesis.txt", "w") as f:
            f.write(response.text)

if __name__ == "__main__":
    cracker = GeminiPatternCracker()
    cracker.crack_pattern()
