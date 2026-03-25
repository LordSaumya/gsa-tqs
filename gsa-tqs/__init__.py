"""
Group-Equivariant Self-Attention - Tensor Quantum States (GSA-TQS)
A PyTorch implementation of group equivariant transformers for quantum state encoding.
"""

from lattice_utils import (
    diff_tensor_1d,
    diff_tensor_2d,
    group_action_space_1d_dihedral,
    group_mult_table_1d_dihedral,
    make_lattice_config,
)
from tensor_utils import (
    safe_einsum,
    broadcast_and_add_bias,
    flatten_joint_dims,
    unflatten_from,
    safe_softmax,
    reshape_for_heads,
    reshape_for_heads_and_group,
)
from constants import (
    LatticeConfig,
    AttentionConfig,
    ModelConfig,
    validate_model_config,
)
from lattice_buffers import LatticeBufferRegistry

__version__ = "0.1.0"
__all__ = [
    # Lattice utilities
    "diff_tensor_1d",
    "diff_tensor_2d",
    "group_action_space_1d_dihedral",
    "group_mult_table_1d_dihedral",
    "make_lattice_config",
    # Tensor utilities
    "safe_einsum",
    "broadcast_and_add_bias",
    "flatten_joint_dims",
    "unflatten_from",
    "safe_softmax",
    "reshape_for_heads",
    "reshape_for_heads_and_group",
    # Constants
    "LatticeConfig",
    "AttentionConfig",
    "ModelConfig",
    "validate_model_config",
    # Buffers
    "LatticeBufferRegistry",
]
