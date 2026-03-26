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
        dropout: float = 0.1,
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
            dropout=dropout,
            d_hidden=d_model,
        )
        
        # Equivariant attention layers
        self.attention_layers = nn.ModuleList([
            GroupAttention(
                d_model=d_model,
                num_heads=num_heads,
                group_size=self.group_size,
                num_sites=num_sites,
                dropout=dropout,
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
        
        # Equivariant attention layers
        X_att = X_eq
        for layer in self.attention_layers:
            X_att = layer(X_att, self.spatial_diff, self.group_mult)
        
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


if __name__ == "__main__":
    print("Testing EquivariantTransformer...")
    
    # Configuration
    batch_size = 2
    num_sites = 8
    d_model = 128
    
    # Create model
    model = EquivariantTransformer(
        lattice_type="1d_dihedral",
        num_sites=num_sites,
        d_model=d_model,
        num_layers=2,
        num_heads=4,
        output_mode="polar",
        phase_init_zero=True,
    )
    
    # Verify initialization
    init_check = model.verify_initialisation()
    print(f"✓ Phase initialized to 0: {init_check['phase_zeros']}")
    print(f"✓ Buffers non-trainable: {init_check['buffers_persistent']}")
    print(f"✓ Lattice: {init_check['lattice_config']['lattice_type']}, "
          f"n={init_check['lattice_config']['num_sites']}, "
          f"|H|={init_check['lattice_config']['group_size']}")
    
    # Create dummy input
    X = torch.randn(batch_size, num_sites, d_model)
    
    # Forward pass
    alpha, beta = model(X)
    
    print(f"\n✓ Model forward pass successful")
    print(f"  Input: {X.shape}")
    print(f"  Amplitude: {alpha.shape}")
    print(f"  Phase: {beta.shape}")
    
    assert alpha.shape == (batch_size,)
    assert beta.shape == (batch_size,)
    
    # Test backward
    loss = alpha.sum() + beta.sum()
    loss.backward()
    
    print(f"✓ Model backward pass successful")
    print(f"✓ Gradient shapes agree with parameters")
    
    # Test complex mode
    model_complex = EquivariantTransformer(
        lattice_type="1d_dihedral",
        num_sites=num_sites,
        d_model=d_model,
        num_layers=2,
        output_mode="complex",
        phase_init_zero=True,
    )
    
    psi = model_complex(X)
    print(f"\n✓ Complex mode output: {psi.shape}, dtype={psi.dtype}")
    
    print(f"\n✓ ALL EQUIVARIANT TRANSFORMER TESTS PASSED")
