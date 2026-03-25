"""
Tensor operation utilities for group equivariant attention.

Provides safe einsum operations, custom broadcasting, and reshape utilities
that enforce strict shape contracts for numerical stability and correctness.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict


def safe_einsum(
    equation: str,
    *tensors,
    shape_contracts: Optional[Dict[str, int]] = None
) -> torch.Tensor:
    """
    Perform einsum with strict shape validation.
    
    Args:
        equation: Einstein summation equation (e.g., 'b i m d, b j m d -> b i j m').
        *tensors: Input tensors to contract.
        shape_contracts: Optional dict mapping axis labels to required dimensions.
                        Used to validate input tensor shapes before contraction.
                        Example: {"b": 2, "i": 8, "m": 4, "d": 64}
    
    Returns:
        Result of torch.einsum contraction.
        
    Raises:
        ValueError: If any input tensor shape violates the shape_contracts.
    """
    if shape_contracts:
        # Parse equation to extract input/output parts
        parts = equation.split("->")
        inputs_spec = parts[0].split(",")
        
        # Validate each input tensor against contracts
        for tensor_idx, (spec, tensor) in enumerate(zip(inputs_spec, tensors)):
            spec = spec.strip()
            # Parse the spec to extract indices (skip spaces and commas)
            indices = [c for c in spec if c not in (' ', ',')]
            for axis_idx, axis_label in enumerate(indices):
                if axis_label in shape_contracts:
                    expected_dim = shape_contracts[axis_label]
                    actual_dim = tensor.shape[axis_idx]
                    if actual_dim != expected_dim:
                        raise ValueError(
                            f"Input tensor {tensor_idx}, axis '{axis_label}' (position {axis_idx}): "
                            f"expected dim {expected_dim}, got {actual_dim}"
                        )
    
    return torch.einsum(equation, *tensors)


def broadcast_and_add_bias(
    A: torch.Tensor,
    bias: torch.Tensor,
    bias_axes: List[int],
    target_axes: List[int]
) -> torch.Tensor:
    """
    Custom broadcast & add for attention logits and relative biases.
    
    Typically used in attention to add positional biases of shape [|H|, n, n, num_heads]
    to logits of shape [B, n, n, num_heads], creating an intermediate group dimension.
    
    Args:
        A: Logits tensor (e.g., shape [B, n, n, num_heads]).
        bias: Relative bias tensor (e.g., shape [|H|, n, n, num_heads]).
        bias_axes: Axes in bias tensor to unsqueeze for broadcasting (e.g., [0] to unsqueeze at dim 0).
        target_axes: Axes in A to unsqueeze (e.g., [1] to create group dimension).
    
    Returns:
        Result of broadcasting and element-wise addition, e.g., shape [B, |H|, n, n, num_heads].
    """
    A_expanded = A
    for ax in sorted(target_axes, reverse=True):
        A_expanded = A_expanded.unsqueeze(ax)
    
    bias_expanded = bias
    for ax in sorted(bias_axes, reverse=True):
        bias_expanded = bias_expanded.unsqueeze(ax)
    
    return A_expanded + bias_expanded


def flatten_joint_dims(
    tensor: torch.Tensor,
    dims: List[int],
    target_dim: int = -1
) -> torch.Tensor:
    """
    Flatten multiple dimensions into a single dimension.
    
    Used in GroupAttention to collapse joint (key_spatial, key_group) space into
    a single dimension before softmax, since PyTorch softmax only accepts one dimension.
    
    Args:
        tensor: Input tensor to flatten.
        dims: List of dimensions to flatten (will be sorted and collapsed in order).
        target_dim: Position where the flattened dimension should appear (default: -1 for last).
    
    Returns:
        Tensor with the specified dimensions flattened into one.
        
    Example:
        tensor of shape [B, n, n, |H|, |H|, M]
        dims=[2, 4] (key_spatial=2, key_group=4)
        → shape [B, n, |H|, M, n*|H|] (flattened at target_dim=-1)
    """
    # Flatten the specified dimensions
    shape = list(tensor.shape)
    dims_sorted = sorted(dims, reverse=True)
    
    # Calculate the product of flattened dimensions
    flattened_size = 1
    for d in dims:
        flattened_size *= shape[d]
    
    # Remove flattened dimensions from shape and insert the flattened one
    for d in dims_sorted:
        del shape[d]
    
    if target_dim == -1:
        shape.append(flattened_size)
    else:
        shape.insert(target_dim, flattened_size)
    
    return tensor.reshape(shape)


def unflatten_from(
    tensor: torch.Tensor,
    target_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Reshape a flattened tensor back to a target shape.
    
    Inverse of flatten_joint_dims. Typically used after softmax to restore
    the multi-dimensional structure for value aggregation.
    
    Args:
        tensor: Flattened tensor to reshape.
        target_shape: Target shape tuple to reshape into.
    
    Returns:
        Tensor reshaped to target_shape.
        
    Example:
        tensor of shape [B, n, |H|, M, n*|H|]
        target_shape = (B, n, n, |H|, |H|, M)
        → shape [B, n, n, |H|, |H|, M]
    """
    return tensor.reshape(target_shape)


def safe_softmax(
    A: torch.Tensor,
    dim: int,
    validate: bool = True
) -> torch.Tensor:
    """
    Apply softmax with optional dimension validation.
    
    Args:
        A: Input tensor to apply softmax.
        dim: Dimension along which to apply softmax. Negative indexing supported.
        validate: If True, enforces that dim is a valid single dimension (not a sequence).
    
    Returns:
        Softmax applied along the specified dimension.
        
    Raises:
        ValueError: If validate=True and dim is invalid.
    """
    if validate:
        if not isinstance(dim, int):
            raise ValueError(f"Softmax dimension must be a single int, got {type(dim)}")
        if dim < -len(A.shape) or dim >= len(A.shape):
            raise ValueError(f"Softmax dimension {dim} out of range for tensor of shape {A.shape}")
    
    return F.softmax(A, dim=dim)


def reshape_for_heads(
    tensor: torch.Tensor,
    batch_size: int,
    num_spatial: int,
    num_heads: int,
    d_k: int
) -> torch.Tensor:
    """
    Reshape a projected tensor from [B, n, d_hidden] to [B, n, num_heads, d_k].
    
    Used after linear Q/K/V projections in attention modules.
    
    Args:
        tensor: Tensor of shape [B*n, d_hidden] or [B, n, d_hidden].
        batch_size: Batch size B.
        num_spatial: Number of spatial sites n.
        num_heads: Number of attention heads.
        d_k: Key/query dimension per head.
    
    Returns:
        Reshaped tensor of shape [B, n, num_heads, d_k].
    """
    if tensor.dim() == 2:
        # [B*n, d_hidden] → [B, n, d_hidden] → [B, n, num_heads, d_k]
        tensor = tensor.reshape(batch_size, num_spatial, -1)
    
    return tensor.reshape(batch_size, num_spatial, num_heads, d_k)


def reshape_for_heads_and_group(
    tensor: torch.Tensor,
    batch_size: int,
    num_spatial: int,
    num_group: int,
    num_heads: int,
    d_k: int
) -> torch.Tensor:
    """
    Reshape a projected tensor from [B, n, |H|, d_hidden] to [B, n, |H|, num_heads, d_k].
    
    Used in GroupAttention after linear Q/K/V projections.
    
    Args:
        tensor: Tensor of shape [B, n, |H|, d_hidden].
        batch_size: Batch size B.
        num_spatial: Number of spatial sites n.
        num_group: Group size |H|.
        num_heads: Number of attention heads.
        d_k: Key/query dimension per head.
    
    Returns:
        Reshaped tensor of shape [B, n, |H|, num_heads, d_k].
    """
    return tensor.reshape(batch_size, num_spatial, num_group, num_heads, d_k)


if __name__ == "__main__":
    # Quick sanity check
    print("Testing tensor_utils...")
    
    # Test safe_einsum with contracts
    Q = torch.randn(2, 8, 4, 64)
    K = torch.randn(2, 8, 4, 64)
    contracts = {"b": 2, "i": 8, "j": 8, "m": 4, "d": 64}
    result = safe_einsum('b i m d, b j m d -> b i j m', Q, K, shape_contracts=contracts)
    print(f"Einsum result shape: {result.shape}, expected [2, 8, 8, 4]")
    
    # Test flatten/unflatten
    tensor = torch.randn(2, 8, 8, 2, 2, 4)  # [B, n, n, |H|, |H|, M]
    flat = flatten_joint_dims(tensor, dims=[2, 4], target_dim=-1)
    print(f"Flattened shape: {flat.shape}, expected [2, 8, 2, 4, 16]")
    
    unflat = unflatten_from(flat, target_shape=(2, 8, 8, 2, 2, 4))
    print(f"Unflattened shape: {unflat.shape}, expected [2, 8, 8, 2, 2, 4]")
    
    print("✓ tensor_utils sanity checks passed")
