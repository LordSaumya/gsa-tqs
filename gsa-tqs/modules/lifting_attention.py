import torch
import torch.nn as nn
import math
from typing import Optional


class LiftingAttention(nn.Module):
    """Lifting self-attention layer (from site representation to lifted group representation)."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        group_size: int,
        num_sites: int,
        d_hidden: Optional[int] = None,
        relative_bias_init_std: float = 0.02,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_sites = num_sites
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(self.num_heads * self.d_v, self.d_hidden)
        
        # Layer normalisation
        self.layernorm = nn.LayerNorm(d_model)
        
        # Relative positional bias embedding
        num_distances = num_sites
        self.rho_lift = nn.Embedding(num_distances, num_heads)
        nn.init.normal_(self.rho_lift.weight, mean=0.0, std=relative_bias_init_std)
        
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        X: torch.Tensor,
        spatial_diff: torch.Tensor,
        group_action_space: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: [B, n, d_model] -> [B, n, |H|, d_hidden]"""
        batch_size, n, d_model = X.shape
        
        X = self.layernorm(X)
        
        # Project and reshape
        Q = self.W_q(X).reshape(batch_size, n, self.num_heads, self.d_k)
        K = self.W_k(X).reshape(batch_size, n, self.num_heads, self.d_k)
        V = self.W_v(X).reshape(batch_size, n, self.num_heads, self.d_v)
        
        Q = Q.permute(0, 2, 1, 3)  # [B, M, n, d_k]
        K = K.permute(0, 2, 3, 1)  # [B, M, d_k, n]
        logits = torch.matmul(Q, K) / self.scale  # [B, M, n, n]
        logits = logits.permute(0, 2, 3, 1)  # [B, n, n, M]
        
        # Get relative biases
        lifted_dist = group_action_space[:, spatial_diff]  # [|H|, n, n]
        R = self.rho_lift(lifted_dist)  # [|H|, n, n, M]
        
        # Broadcast and add
        A = logits.unsqueeze(1) + R.unsqueeze(0)
        
        # Softmax over key dimension
        A = torch.softmax(A, dim=3)
                
        # Aggregate with V
        # A: [B, |H|, n, n, M] -> [B, |H|, M, n, n]
        # V: [B, n, M, d_v] -> [B, M, n, d_v] -> [B, 1, M, n, d_v] for broadcast
        A = A.permute(0, 1, 4, 2, 3)  # [B, |H|, M, n, n]
        V = V.permute(0, 2, 1, 3)  # [B, M, n, d_v]
        V_expanded = V.unsqueeze(1)  # [B, 1, M, n, d_v] for broadcasting
        O = torch.matmul(A, V_expanded)  # [B, |H|, M, n, d_v]
        
        # Reshape: [B, |H|, M, n, d_v] -> [B, n, |H|, M*d_v]
        O = O.permute(0, 3, 1, 2, 4)  # [B, n, |H|, M, d_v]
        O = O.reshape(batch_size, n, self.group_size, -1)  # [B, n, |H|, M*d_v]
        
        # Project
        O = self.W_o(O)  # [B, n, |H|, d_hidden]
        
        return O
