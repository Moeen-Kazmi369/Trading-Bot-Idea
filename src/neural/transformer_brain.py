import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console

console = Console()

class UniversalOracleV2(nn.Module):
    """
    Chronos-Universal v2.2.8 (Momentum-Locked Sentinel)
    Implements 12/26 EMA Trigger Awareness and 200 EMA Macro Bias.
    """
    def __init__(self, feature_dim=30, conditioning_dim=2, seq_len=50, hidden_dim=256):
        super(UniversalOracleV2, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # 1. Feature Masking (Collaborator Trigger Priorities)
        self.feature_mask = nn.Parameter(torch.ones(feature_dim))
        
        # 2. Sequential Foundations
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.cond_proj = nn.Linear(conditioning_dim, hidden_dim)
        
        # 3. Foundation Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, 
            dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.decider = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 4))

    def forward(self, x_seq, x_cond):
        x_seq = x_seq * self.feature_mask
        x = self.feature_proj(x_seq) + self.pos_embedding
        c = self.cond_proj(x_cond).unsqueeze(1)
        x = x + c 
        x = self.transformer(x)
        
        latent = x[:, -1, :] # The Global Context
        logits = self.decider(latent)
        return F.softmax(logits, dim=-1)

    def collaborator_seeding(self):
        """
        Finalizing the Phase 13.5 'Prior Weights'.
        Index 0: EMA 200 (Macro Support)
        Index 1: EMA 12/26 Delta (Momentum Trigger)
        Index 5 & 6: BB Pierce Upper/Lower (Liquidity Pulse)
        """
        with torch.no_grad():
            self.feature_mask[0] = 0.5   # Macro: Significant but not overwhelming
            self.feature_mask[1] = 2.0   # Trigger: High Priority Momentum
            self.feature_mask[5] = 1.5   # Pulse: Exhaustion Alert
            
            # Decider Seeding
            # If EMA 12/26 Delta is Positive -> Lean Long
            self.decider[-1].weight[1, 1] = 1.2 
            # If BB Pierce Upper -> Lean Exit/Short
            self.decider[-1].weight[3, 5] = 1.5
            
            console.print("[bold green]COLLABORATOR SEEDING:[/bold green] 12/26 Triggers and 200 EMA support initialized.")
