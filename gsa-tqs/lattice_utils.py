import torch
from typing import Dict, Tuple


def diff_tensor_1d(n: int) -> torch.Tensor:
    """
    Generate 1D periodic spatial difference matrix.
    
    For a 1D periodic chain with n sites, compute (j - i) mod n for all pairs (i, j).
    
    Args:
        n: Number of sites in the chain.
        
    Returns:
        Tensor of shape [n, n] with dtype torch.long. Element [i, j] = (j - i) % n.
    """
    spatial_diff = torch.zeros((n, n), dtype=torch.long)
    for i in range(n):
        for j in range(n):
            spatial_diff[i, j] = (j - i) % n
    return spatial_diff


def diff_tensor_2d(x_num: int, y_num: int) -> torch.Tensor:
    """
    Generate 2D periodic spatial difference matrix for a rectangular lattice.
    
    For a 2D lattice with x_num x y_num sites, compute the 2D difference vector
    (dx, dy) mod (x_num, y_num) for all pairs of lattice sites.
    
    Site indexing: site at (x, y) has linear index = y * x_num + x.
    
    Args:
        x_num: Number of sites along x-direction.
        y_num: Number of sites along y-direction.
        
    Returns:
        Tensor of shape [x_num*y_num, x_num*y_num, 2] with dtype torch.long.
        Element [i, j] = [dx, dy] where dx = (j_x - i_x) % x_num, dy = (j_y - i_y) % y_num.
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
    
    The Dihedral group D_n has |D_n| = 2 elements: Identity (h=0) and Reflection (h=1).
    For each group element h and spatial difference d, compute h^{-1} * d.
    
    - Identity (h=0): d -> d
    - Reflection (h=1): d -> -d mod n
    
    Args:
        n: Number of sites (and group order factor). Group size is 2.
        
    Returns:
        Tensor of shape [2, n] with dtype torch.long.
        Element [h, d] = h^{-1} * d (the transformed spatial difference).
    """
    group_action_space = torch.zeros((2, n), dtype=torch.long)
    for d in range(n):
        group_action_space[0, d] = d  # Identity: d -> d
        group_action_space[1, d] = (-d) % n  # Reflection: d -> -d mod n
    return group_action_space


def group_mult_table_1d_dihedral() -> torch.Tensor:
    """
    Generate group multiplication (Cayley) table for 1D Dihedral group D_n.
    
    The Dihedral group D_n has 2 elements: {e, P} (Identity and Reflection).
    Compute h_bar^{-1} * h_tilde for all pairs (h_bar, h_tilde).
    
    Cayley table:
    | h_bar \\ h_tilde |  0  |  1  |
    |       0          |  0  |  1  |
    |       1          |  1  |  0  |
    
    Returns:
        Tensor of shape [2, 2] with dtype torch.long representing the Cayley table.
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
    
    Args:
        lattice_type: Type of lattice. Supported: "1d_dihedral" (future: "2d_square").
        n: Number of sites (for 1D lattices).
        x_num, y_num: Lattice dimensions (for 2D lattices).
        
    Returns:
        Dictionary with keys:
        - "spatial_diff": Spatial difference tensor
        - "group_action_space": Group action on space
        - "group_mult": Group multiplication table
        - "num_sites": Total number of sites
        - "group_size": Size of the group (|H|)
        - "lattice_type": Type string for reference
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
