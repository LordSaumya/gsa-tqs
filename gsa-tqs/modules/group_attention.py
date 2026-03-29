import torch
import torch.nn as nn
import math


class GroupAttention(nn.Module):
    """Group self-attention layer (equivariant attention within group manifold)."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        group_size: int,
        num_sites: int,
        relative_bias_init_std: float = 0.02,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_sites = num_sites
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Exact embedding allocation: |S| distances * |H| point group operations
        num_embeddings = num_sites * group_size
        self.rho_group = nn.Embedding(num_embeddings, num_heads)
        nn.init.normal_(self.rho_group.weight, mean=0.0, std=relative_bias_init_std)
        
        self.layernorm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        X: torch.Tensor,
        spatial_diff: torch.Tensor,
        group_mult: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n, h, d_model = X.shape
        residual = X
        M = self.num_heads
        
        X = self.layernorm(X)
        
        Q = self.W_q(X).reshape(batch_size, n, h, M, self.d_k)
        K = self.W_k(X).reshape(batch_size, n, h, M, self.d_k)
        V = self.W_v(X).reshape(batch_size, n, h, M, self.d_v)
        
        Q = Q.permute(0, 3, 1, 2, 4)  # [B, M, n_q, h_q, d_k]
        K = K.permute(0, 3, 1, 2, 4)  # [B, M, n_k, h_k, d_k]
        V = V.permute(0, 3, 1, 2, 4)  # [B, M, n_k, h_k, d_v]

        Q_grouped = Q.reshape(batch_size * M, n * h, self.d_k)
        K_grouped = K.reshape(batch_size * M, n * h, self.d_k)

        # Logits: [B*M, n_q*h_q, n_k*h_k]
        logits = torch.matmul(Q_grouped, K_grouped.transpose(-2, -1)) / self.scale
        
        # Reshape to explicit dimensional topology
        logits = logits.reshape(batch_size, M, n, h, n, h)
        # Permute to: [B, n_q, n_k, h_q, h_k, M]
        logits = logits.permute(0, 2, 4, 3, 5, 1)

        # Construct 4D metric index
        idx = (
            spatial_diff.unsqueeze(2).unsqueeze(3) * self.group_size
            + group_mult.unsqueeze(0).unsqueeze(0)
        )
        
        R = self.rho_group(idx) # [n, n, h, h, M]
        A = logits + R.unsqueeze(0) # [B, n_q, n_k, h_q, h_k, M]

        # Permute A to [B, n_q, h_q, M, n_k, h_k]
        A_permuted = A.permute(0, 1, 3, 5, 2, 4)
        
        # Flatten the (n_k, h_k) dimensions into a single target distribution
        A_flat = A_permuted.reshape(batch_size, n, h, M, n * h)
        
        Attn_flat = torch.softmax(A_flat, dim=-1)

        Attn_grouped = Attn_flat.permute(0, 3, 1, 2, 4).reshape(batch_size * M, n * h, n * h)
        
        V_grouped = V.reshape(batch_size * M, n * h, self.d_v)
        
        O_grouped = torch.matmul(Attn_grouped, V_grouped)
        
        # Reshape back to explicit topology: [B, M, n_q, h_q, d_v]
        O = O_grouped.reshape(batch_size, M, n, h, self.d_v)
        
        # Permute to standard multi-head output: [B, n_q, h_q, M, d_v]
        O = O.permute(0, 2, 3, 1, 4)
        
        # Merge heads: [B, n_q, h_q, d_model]
        O = O.reshape(batch_size, n, h, d_model)

        O = self.W_o(O)

        return O + residual
