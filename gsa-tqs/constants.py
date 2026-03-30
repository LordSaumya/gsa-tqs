from typing import TypedDict, Optional
import torch


class LatticeConfig(TypedDict):
    """Configuration for lattice geometry and symmetry group."""
    spatial_diff: torch.Tensor  # [num_sites, num_sites] or [..., d] for multiD
    group_action_space: torch.Tensor  # [group_size, num_sites]
    group_mult: torch.Tensor  # [group_size, group_size] Cayley table
    num_sites: int
    group_size: int
    lattice_type: str  # "1d_dihedral", "2d_square", etc.


class AttentionConfig(TypedDict, total=False):
    """Configuration for attention modules."""
    num_heads: int
    d_model: int
    d_k: int  # Key/query dimension per head (typically d_model / num_heads)
    d_v: int  # Value dimension per head (typically d_model / num_heads)
    d_hidden: int  # Intermediate hidden dimension (same as d_model typically)
    attention_type: str  # "lifting" or "group"


class ModelConfig(TypedDict, total=False):
    """Full model configuration combining lattice and attention specs."""
    lattice: LatticeConfig
    attention: AttentionConfig
    num_layers: int  # Number of group attention blocks (excluding lifting)
    use_layernorm: bool
    use_residuals: bool
    output_mode: str  # "polar" (alpha + phase) or "complex"
    phase_init_zero: bool  # Whether to initialise phase head to 0.0
    relative_bias_init_std: float  # Std dev for relative bias embeddings


# ===== Hyperparameter Defaults =====

DEFAULT_D_K_FACTOR: int = 64
"""Default d_k dimension. Typically d_k = d_model / num_heads."""

DEFAULT_ATTENTION_CONFIG: AttentionConfig = {
    "num_heads": 4,
    "d_model": 128,
    "d_k": 128 // 4,  # = 32
    "d_v": 128 // 4,  # = 32
    "d_hidden": 128,
    "attention_type": "lifting",
}

DEFAULT_MODEL_CONFIG: ModelConfig = {
    "num_layers": 2,
    "use_layernorm": True,
    "use_residuals": True,
    "output_mode": "polar",
    "phase_init_zero": True,
    "relative_bias_init_std": 0.02,
}

PHASE_INIT_VALUE: float = 0.0
RELATIVE_BIAS_INIT_STD: float = 0.02
AMPLITUDE_INIT_STD: float = 0.02

DEFAULT_HAMILTONIAN_DTYPE: torch.dtype = torch.float64
"""Default dtype for Hamiltonians. Using float64 for real-valued Hamiltonians (X and Z operators only)."""

def validate_model_config(config: ModelConfig) -> None:
    """
    Validate model configuration for consistency and reasonableness.
    """
    att = config.get("attention", {})
    d_model = att.get("d_model", 128)
    num_heads = att.get("num_heads", 4)
    d_k = att.get("d_k", d_model // num_heads)
    d_v = att.get("d_v", d_model // num_heads)
    
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
    
    if d_k * num_heads != d_model:
        raise ValueError(
            f"d_k ({d_k}) * num_heads ({num_heads}) = {d_k * num_heads}, "
            f"but d_model = {d_model}"
        )
    
    if config.get("num_layers", 2) < 1:
        raise ValueError("num_layers must be >= 1")
    
    if config.get("output_mode") != "polar" and config.get("phase_init_zero"):
        raise ValueError("phase_init_zero=True only makes sense for output_mode='polar'")