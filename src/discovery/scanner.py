import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

class InterestScanner:
    def __init__(self, df):
        self.df = df
        
    def find_big_moves(self, threshold_pct=0.5):
        """Finds points where price moved significantly in 1-5 minutes."""
        interesting_points = []
        
        for i in range(10, len(self.df) - 10):
            # Check for 1-minute explosive moves
            move = ((self.df.iloc[i]['close'] - self.df.iloc[i]['open']) / self.df.iloc[i]['open']) * 100
            
            if abs(move) >= threshold_pct:
                interesting_points.append({
                    'index': i,
                    'timestamp': self.df.iloc[i]['timestamp'],
                    'move_pct': move,
                    'context_ohlcv': self.df.iloc[i-10:i].to_dict('records') # Previous 10 candles
                })
                
        return interesting_points

if __name__ == "__main__":
    df = pd.read_csv("data/raw/BTCUSDT_1m.csv")
    scanner = InterestScanner(df)
    points = scanner.find_big_moves(threshold_pct=0.8) # 0.8% in 1 min is massive for BTC
    
    console.print(f"[bold green]Found {len(points)} highly interesting explosive points![/bold green]")
    
    if points:
        table = Table(title="Sample Interesting Points (Explosive Moves)")
        table.add_column("Timestamp")
        table.add_column("Move %")
        
        for p in points[:10]:
            color = "green" if p['move_pct'] > 0 else "red"
            table.add_row(str(p['timestamp']), f"[{color}]{p['move_pct']:.2f}%[/{color}]")
            
        console.print(table)
