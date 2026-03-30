import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional

from modules.output_layer import InvariantPoolAndOutput


class MultiheadAttentionWithRPE(nn.Module):
    """
    Multi-head attention with learnable per-head Relative Positional Encoding (RPE).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_sites: int,
        dropout: float = 0.0,
        bias: bool = True,
        relative_bias_init_std: float = 0.02,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_sites = num_sites
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Learnable per-head relative position bias embeddings
        self.relative_pos_bias = nn.Embedding(num_sites, num_heads)
        nn.init.normal_(self.relative_pos_bias.weight, mean=0.0, std=relative_bias_init_std)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rel_pos_indices: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with learnable per-head relative positional biases.
        
        Args:
            query, key, value: [B, N, D]
            rel_pos_indices: [N, N] relative position indices (optional, computed if not provided)
        
        Returns:
            output: [B, N, D]
            attn_weights: [B, num_heads, N, N]
        """
        B, N, D = query.shape
        
        # Project Q, K, V
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        
        # Reshape to [B, num_heads, N, head_dim]
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: [B, num_heads, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Add learnable per-head relative position bias
        if rel_pos_indices is None:
            # Compute relative position indices: [N, N]
            positions = torch.arange(N, device=query.device)
            rel_pos_indices = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()  # [N, N]
        
        # Get position biases for all heads: [num_heads, N, N]
        rel_pos_bias = self.relative_pos_bias(rel_pos_indices)  # [N, N, num_heads]
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1)  # [num_heads, N, N]
        
        # Add to scores: [B, num_heads, N, N] + [1, num_heads, N, N]
        scores = scores + rel_pos_bias.unsqueeze(0)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values: [B, num_heads, N, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back: [B, num_heads, N, head_dim] -> [B, N, D]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, N, D)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights


class StandardTransformer(nn.Module):
    """
    Standard (non-equivariant) transformer for ablation study comparison.
    Matches the architecture of EquivariantTransformer but uses standard PyTorch components
    with learnable absolute positional encodings.
    """
    
    def __init__(
        self,
        num_sites: int = 8,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        output_mode: str = "polar",
        phase_init_zero: bool = True,
        d_ff: Optional[int] = None,
        pe_mode: int = 0, # Positional encoding mode: 0 = None, 1 = Learnable absolute, 2 = RPE
    ):
        super().__init__()
        
        self.num_sites = num_sites
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_mode = output_mode
        self.pe_mode = pe_mode
        
        # Input embedding layer (same as EquivariantTransformer)
        self.state_embedding = nn.Linear(1, d_model)
        
        # Learnable absolute positional encoding (only for pe_mode 1)
        if pe_mode == 1:
            self.positional_encoding = nn.Parameter(torch.randn(1, num_sites, d_model))
        else:
            self.register_parameter('positional_encoding', None)
        
        # Feedforward dimension
        d_ff = d_ff or (4 * d_model)
        
        total_layers = num_layers + 1
        
        # Attention layers based on pe_mode
        if pe_mode == 2:
            # Use relative positional encoding
            self.attention_layers = nn.ModuleList([
                MultiheadAttentionWithRPE(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    num_sites=num_sites,
                )
                for _ in range(total_layers)
            ])
        else:
            # Use standard multi-head attention (for modes 0 and 1)
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    batch_first=True,
                )
                for _ in range(total_layers)
            ])
        
        # Layer normalisation
        self.layer_norms_attn = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(total_layers)
        ])

        # Feedforward networks after attention
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(total_layers)
        ])
        
        # Layer norms for feedforward
        self.layer_norms_ff = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(total_layers)
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
        """
        # Ensure input is [B, n, 1]
        if X.dim() == 2:
            X = X.unsqueeze(-1).float()
        
        # Embed input: [B, n, 1] -> [B, n, d_model]
        X = self.state_embedding(X)
        
        # Add learnable absolute positional encoding (only for pe_mode 1)
        if self.pe_mode == 1:
            X = X + self.positional_encoding

        # Standard transformer layers with residual connections
        for attn_layer, ln_attn, ff_layer, ln_ff in zip(
            self.attention_layers,
            self.layer_norms_attn,
            self.ff_layers,
            self.layer_norms_ff,
        ):
            # Self-attention with pre-normalisation (pre-LN transformer)
            X_norm = ln_attn(X)
            
            # Apply attention based on pe_mode
            if self.pe_mode == 2:
                # RPE attention
                attn_output, _ = attn_layer(X_norm, X_norm, X_norm)
            else:
                # Standard attention (modes 0 and 1)
                attn_output, _ = attn_layer(X_norm, X_norm, X_norm)
            
            X = X + attn_output  # Residual connection
            
            # Feedforward sublayer
            X_norm = ln_ff(X)
            X_ff = ff_layer(X_norm)
            X = X + X_ff  # Residual connection
        
        # Pooling + output
        output = self.output_layer(X.unsqueeze(2))  # Use same output layer as EquivariantTransformer for comparison
        
        return output
