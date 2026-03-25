import torch
from typing import Dict, Tuple


def diff_tensor_1d(n: int) -> torch.Tensor:
    """
    Generate 1D periodic spatial difference matrix. Element [i, j] = (j - i) mod n.
    """
    spatial_diff = torch.zeros((n, n), dtype=torch.long)
    for i in range(n):
        for j in range(n):
            spatial_diff[i, j] = (j - i) % n
    return spatial_diff


def diff_tensor_2d(x_num: int, y_num: int) -> torch.Tensor:
    """
    Generate 2D periodic spatial difference matrix for a rectangular lattice. Element [i, j] = (dx, dy) where dx = (j_x - i_x) mod x_num, dy = (j_y - i_y) mod y_num.
    """
    n_sites = x_num * y_num
    spatial_diff_2d = torch.zeros((n_sites, n_sites, 2), dtype=torch.long)
    
    for i in range(n_sites):
        i_x, i_y = i % x_num, i // x_num
        for j in range(n_sites):
            j_x, j_y = j % x_num, j // x_num
            dx = (j_x - i_x) % x_num
            dy = (j_y - i_y) % y_num
            spatial_diff_2d[i, j] = torch.tensor([dx, dy], dtype=torch.long)
    
    return spatial_diff_2d


def group_action_space_1d_dihedral(n: int) -> torch.Tensor:
    """
    Generate group action on space for 1D Dihedral group D_n.
    """
    group_action_space = torch.zeros((2, n), dtype=torch.long)
    for d in range(n):
        group_action_space[0, d] = d  # Identity: d -> d
        group_action_space[1, d] = (-d) % n  # Reflection: d -> -d mod n
    return group_action_space


def group_mult_table_1d_dihedral() -> torch.Tensor:
    """
    Generate group multiplication (Cayley) table for 1D Dihedral group D_n.
    """
    group_mult = torch.tensor([
        [0, 1],
        [1, 0]
    ], dtype=torch.long)
    return group_mult


def make_lattice_config(
    lattice_type: str,
    n: int = None,
    x_num: int = None,
    y_num: int = None
) -> Dict:
    """
    Factory function to generate a complete lattice configuration.
    """
    if lattice_type == "1d_dihedral":
        if n is None:
            raise ValueError("For 1d_dihedral, n (number of sites) must be provided.")
        return {
            "spatial_diff": diff_tensor_1d(n),
            "group_action_space": group_action_space_1d_dihedral(n),
            "group_mult": group_mult_table_1d_dihedral(),
            "num_sites": n,
            "group_size": 2,
            "lattice_type": "1d_dihedral",
        }
    elif lattice_type == "2d_square":
        if x_num is None or y_num is None:
            raise ValueError("For 2d_square, x_num and y_num must be provided.")
        raise NotImplementedError("2D square lattice support deferred to Phase 2.")
    else:
        raise ValueError(f"Unsupported lattice_type: {lattice_type}")
