import torch
import torch.nn as nn
from typing import Tuple, Union, Dict, Any, Optional
import os

from lattice_buffers import LatticeBufferRegistry
from modules.lifting_attention import LiftingAttention
from modules.group_attention import GroupAttention
from modules.output_layer import InvariantPoolAndOutput


class EquivariantTransformer(nn.Module):
    """
    Full group equivariant transformer for encoding quantum states.
    """
    
    def __init__(
        self,
        lattice_type: str = "1d_dihedral",
        num_sites: int = 8,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        output_mode: str = "polar",
        phase_init_zero: bool = True,
        d_ff: Optional[int] = None,
        **lattice_kwargs,
    ):
        """
        Initialise EquivariantTransformer.
        """
        super().__init__()
        
        self.lattice_type = lattice_type
        self.num_sites = num_sites
        self.d_model = d_model
        self.num_layers = num_layers
        self.output_mode = output_mode
        self.state_embedding = nn.Linear(1, d_model)
        
        # Feedforward dimension
        d_ff = d_ff or (4 * d_model)
        
        # Lattice buffer registry
        self.lattice_registry = LatticeBufferRegistry(
            lattice_type=lattice_type,
            **{"n": num_sites, **lattice_kwargs}
        )
        config = self.lattice_registry.get_lattice_config()
        self.group_size = config["group_size"]
        self.register_buffer("spatial_diff", config["spatial_diff"])
        self.register_buffer("group_action_space", config["group_action_space"])
        self.register_buffer("group_mult", config["group_mult"])
        
        # Lifting layer
        self.lifting = LiftingAttention(
            d_model=d_model,
            num_heads=num_heads,
            group_size=self.group_size,
            num_sites=num_sites,
            d_hidden=d_model,
        )
        
        # Layer norm and feedforward after lifting
        self.layer_norm_lifting = nn.LayerNorm(d_model)
        self.ff_lifting = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )
        
        # Equivariant attention layers
        self.attention_layers = nn.ModuleList([
            GroupAttention(
                d_model=d_model,
                num_heads=num_heads,
                group_size=self.group_size,
                num_sites=num_sites,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms and feedforward networks for attention layers
        self.layer_norms_attn = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(num_layers)
        ])
        
        # Output layer
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
        Forward pass through the full model.
        """
        # Ensure input is [B, n, d_model]
        if X.dim() == 2:
            X = X.unsqueeze(-1).float()
            X = self.state_embedding(X) # [B, n, d_model]
        
        # Lifting: non-equivariant -> equivariant
        # [B, n, d_model] -> [B, n, |H|, d_model]
        X_eq = self.lifting(X, self.spatial_diff, self.group_action_space)
        
        # Feedforward after lifting
        X_norm = self.layer_norm_lifting(X_eq)
        X_ff = self.ff_lifting(X_norm)
        X_eq = X_eq + X_ff  # Residual connection
        
        # Equivariant attention layers
        X_att = X_eq
        for i, layer in enumerate(self.attention_layers):
            # Attention
            X_att_out = layer(X_att, self.spatial_diff, self.group_mult)
            X_att = X_att + X_att_out  # Residual connection
            
            # Feedforward
            X_norm = self.layer_norms_attn[i](X_att)
            X_ff = self.ff_layers[i](X_norm)
            X_att = X_att + X_ff  # Residual connection
        
        # Pooling + output
        output = self.output_layer(X_att)
        
        return output
    
    def verify_initialisation(self) -> Dict[str, Any]:
        """
        Verify critical initialisation constraints.
        """
        return {
            "phase_zeros": self.output_layer.verify_phase_init(),
            "buffers_persistent": not any(
                p.requires_grad for p in self.lattice_registry.parameters()
            ),
            "lattice_config": self.lattice_registry.get_lattice_config(),
        }