import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.table import Table

console = Console()

def crack_alpha():
    console.print(f"[bold cyan]ALPHA CRACKER:[/bold cyan] Analyzing Forensic Harvest (2,876 Snapshots)...")
    
    df = pd.read_csv("research/forensic_harvest.csv")
    
    # Preprocessing
    # We strip 'roi' and 'win' from features
    features = [c for c in df.columns if c not in ['roi', 'win', 'direction']]
    X = df[features]
    y = df['win']
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)
    
    # 1. Feature Importance Dashboard
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    table = Table(title="PHASE 11: FEATURE IMPORTANCE DASHBOARD")
    table.add_column("Rank", style="cyan")
    table.add_column("Indicator", style="green")
    table.add_column("Impact Score", style="yellow")
    
    for r in range(min(10, len(features))):
        table.add_row(str(r+1), features[indices[r]], f"{importances[indices[r]]:.4f}")
    
    console.print(table)
    
    # 2. Rule Extraction (High-Probability States)
    # We'll look for simple correlations in the top 3 features
    top_f = features[indices[0]]
    mid_f = features[indices[1]]
    
    console.print(f"\n[bold cyan]FORENSIC RULES DISCOVERED:[/bold cyan]")
    
    # Example: Check top feature percentiles
    for q in [0.1, 0.25, 0.75, 0.9]:
        val = df[top_f].quantile(q)
        # Check win rate when top_f is beyond this quantile
        subset = df[df[top_f] > val] if q > 0.5 else df[df[top_f] < val]
        wr = subset['win'].mean() * 100
        direction = ">" if q > 0.5 else "<"
        console.print(f" Rule: If {top_f} {direction} {val:.4f} -> Observed Win-Rate: [bold]{wr:.2f}%[/bold]")

if __name__ == "__main__":
    crack_alpha()
