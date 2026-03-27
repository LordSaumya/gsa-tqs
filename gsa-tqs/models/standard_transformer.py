import torch
import torch.nn as nn
from typing import Tuple, Union, Dict, Any, Optional

from modules.output_layer import InvariantPoolAndOutput


class StandardTransformer(nn.Module):
    """
    Standard (non-equivariant) transformer for ablation study comparison.
    Matches the architecture of EquivariantTransformer but uses standard PyTorch components.
    """
    
    def __init__(
        self,
        num_sites: int = 8,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        output_mode: str = "polar",
        phase_init_zero: bool = True,
        dropout: float = 0.1,
        d_ff: Optional[int] = None,
    ):
        super().__init__()
        
        self.num_sites = num_sites
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_mode = output_mode
        
        # Input embedding layer (same as EquivariantTransformer)
        self.state_embedding = nn.Linear(1, d_model)
        
        # Standard multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Feed-forward layers after each attention layer
        d_ff = d_ff or 4 * d_model
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalisation
        self.layer_norms_attn = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.layer_norms_ff = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Output layer (same as EquivariantTransformer)
        self.output_layer = InvariantPoolAndOutput(
            d_model=d_model,
            output_mode=output_mode,
            phase_init_zero=phase_init_zero,
        )
    
    def forward(
        self,
        X: torch.Tensor,
        state_indices: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass through the standard transformer.
        
        Args:
            X: Input tensor of shape [B, n] or [B, n, 1]
            state_indices: Optional state indices (for compatibility, not used)
        
        Returns:
            Output tensor (amplitude and phase in polar mode)
        """
        # Ensure input is [B, n, 1]
        if X.dim() == 2:
            X = X.unsqueeze(-1).float()
        
        # Embed input: [B, n, 1] -> [B, n, d_model]
        X = self.state_embedding(X)
        
        # Standard transformer layers with residual connections
        for attn_layer, ff_layer, ln_attn, ln_ff in zip(
            self.attention_layers,
            self.feed_forward_layers,
            self.layer_norms_attn,
            self.layer_norms_ff,
        ):
            # Self-attention with pre-normalization (pre-LN transformer)
            X_norm = ln_attn(X)
            attn_output, _ = attn_layer(X_norm, X_norm, X_norm)
            X = X + attn_output  # Residual connection
            
            # Feed-forward with pre-normalization
            X_norm = ln_ff(X)
            ff_output = ff_layer(X_norm)
            X = X + ff_output  # Residual connection
        
        # Pooling + output
        output = self.output_layer(X.unsqueeze(2))  # Add group dimension for compatibility
        
        return output
