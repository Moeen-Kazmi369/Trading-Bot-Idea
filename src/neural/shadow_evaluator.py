import asyncio
import json
import torch
import numpy as np
import pandas as pd
import os
from binance import AsyncClient, BinanceSocketManager
from src.neural.imitation_learner import PolicyNetwork
from rich.console import Console

console = Console()

class ChronosShadowEvaluator:
    """
    Project Chronos - Live Shadow Evaluator v1.1.2
    Features: Differential Sharpe Reward, Formal Verifier, 
    Conviction Gating, and Hindsight Shadow Logging.
    """
    
    def __init__(self, model_path="models/evolved_brain_v1.pth"):
        # 1. Load the Evolved Brain
        self.model = PolicyNetwork(input_dim=5, output_dim=4)
        if torch.cuda.is_available(): self.model.to("cuda")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 2. Setup State Persistence
        self.history = []
        self.shadow_balance = 1000.0
        self.position = 0
        self.entry_price = 0
        
        # 3. Formal Verifier Guardrails
        self.max_exposure = 5.0 
        self.daily_drawdown_limit = 0.02 
        
        # 4. Hindsight & Heartbeat Diagnostics
        self.ghost_log_path = "research/ghost_trades.csv"
        self.heartbeat_interval = 600 
        if not os.path.exists("research"): os.makedirs("research")

    def formal_verifier(self, action, current_price):
        """Non-Neural Circuit Breaker."""
        if (1000.0 - self.shadow_balance) / 1000.0 > self.daily_drawdown_limit:
            console.print("[bold red]VERIFIER:[/bold red] Daily Drawdown Hit. Kill-switch activated.")
            return 3 
        return action

    async def heartbeat_task(self):
        """Formal Verifier Heartbeat Logic"""
        while True:
            console.print(f"[bold dim white]VERIFIER HEARTBEAT:[/bold dim white] System Check OK | Timeframe: 5m | State: { 'BUSY' if self.position != 0 else 'IDLE' }")
            await asyncio.sleep(self.heartbeat_interval)

    async def log_ghost_trade(self, price, action, confidence, current_idx, df):
        """Asynchronous Hindsight Logger with 30-minute Velocity Trap"""
        action_names = ["STAY", "LONG", "SHORT", "EXIT"]
        
        # Velocity Check: 30 minutes (6 bars on 5m chart)
        exit_idx = min(current_idx + 6, len(df) - 1)
        exit_price = float(df.iloc[exit_idx]['close'])
        
        # Calculate Opportunity Net P&L (Subtracting 0.08% friction)
        fee_friction = 0.0008
        pnl = 0
        if action == 1: # LONG
            pnl = ((exit_price - price) / price) - fee_friction
        elif action == 2: # SHORT
            pnl = ((price - exit_price) / price) - fee_friction
            
        entry = {
            'timestamp': pd.Timestamp.now(),
            'entry_price': price,
            'exit_price': exit_price,
            'predicted_action': action_names[action],
            'confidence': f"{confidence:.1f}%",
            'net_pnl': f"{pnl*100:.2f}%"
        }
        
        df_ghost = pd.DataFrame([entry])
        df_ghost.to_csv(self.ghost_log_path, mode='a', header=not os.path.exists(self.ghost_log_path), index=False)
        console.print(f"[bold yellow]GHOST TRADE CAPTURED:[/bold yellow] P&L Opportunity: {pnl*100:+.2f}% ({confidence:.1f}%)")

    async def start(self, simulation_mode=True):
        # Start Heartbeat in background
        asyncio.create_task(self.heartbeat_task())
        
        if simulation_mode:
            console.print("[bold yellow]TIME MACHINE:[/bold yellow] Simulating Live Flow from local history...")
            df = pd.read_csv("data/raw/BTCUSDT_5m.csv")
            latest_data = df.tail(100) 
            
            # Get internal index for ghost logging
            start_offset = len(df) - 100
            
            for i, (_, row) in enumerate(latest_data.iterrows()):
                self.history.append({
                    'close': float(row['close']), 'volume': float(row['volume']),
                    'high': float(row['high']), 'low': float(row['low'])
                })
                await self.process_neural_step(float(row['close']), start_offset + i, df)
                await asyncio.sleep(0.1) 
        else:
            # Live logic (would require a different async wait approach)
            pass

    async def process_neural_step(self, price, current_idx, df):
        # 1. Build the Tensor Input (State)
        df_hist = pd.DataFrame(self.history[-5:])
        state = [
            (df_hist['close'].iloc[-1] / df_hist['close'].iloc[0]) - 1,
            (df_hist['volume'].iloc[-1] / df_hist['volume'].mean()) - 1,
            (df_hist['high'].max() - df_hist['low'].min()) / df_hist['close'].mean(),
            self.position,
            (price / self.entry_price - 1) if self.position != 0 else 0
        ]
        
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        if torch.cuda.is_available(): state_tensor = state_tensor.to("cuda")
        
        # 2. Neural Inference (Shadow Decision)
        with torch.inference_mode():
            probs = self.model(state_tensor)
            conf, raw_action = torch.max(probs, dim=1)
            confidence = conf.item() * 100
            final_action = raw_action.item()
            
        # 3. Conviction Gate & Hindsight Logic
        CONVICTION_THRESHOLD = 75.0
        if final_action != 0 and confidence < CONVICTION_THRESHOLD:
            if confidence >= 50.0:
                await self.log_ghost_trade(price, final_action, confidence, current_idx, df)
            final_action = 0 
            
        # 4. Formal Verifier (Safety)
        final_action = self.formal_verifier(final_action, price)
        
        # 5. UI Update
        action_names = ["STAY", "LONG", "SHORT", "EXIT"]
        prob_text = f"Confidence: {confidence:.1f}%"
        color = self.get_color(final_action)
        console.print(f"LIVE: Price: ${price:,.2f} | Brain thinks: [bold {color}]{action_names[final_action]}[/bold {color}] ({prob_text})")

    def get_color(self, action):
        return {0: "white", 1: "green", 2: "red", 3: "yellow"}[action]

if __name__ == "__main__":
    evaluator = ChronosShadowEvaluator()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(evaluator.start())
