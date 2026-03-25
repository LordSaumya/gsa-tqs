import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict


def safe_einsum(
    equation: str,
    *tensors,
    shape_contracts: Optional[Dict[str, int]] = None
) -> torch.Tensor:
    """
    Perform einsum with shape validation.
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
    """
    return tensor.reshape(target_shape)


def safe_softmax(
    A: torch.Tensor,
    dim: int,
    validate: bool = True
) -> torch.Tensor:
    """
    Apply softmax with optional dimension validation.
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
    """
    if tensor.dim() == 2:
        # [B*n, d_hidden] -> [B, n, d_hidden] -> [B, n, num_heads, d_k]
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
    """
    return tensor.reshape(batch_size, num_spatial, num_group, num_heads, d_k)
