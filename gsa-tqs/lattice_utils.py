import torch
from typing import Dict

def diff_tensor_1d(n: int) -> torch.Tensor:
    """
    Generate 1D periodic spatial difference matrix. Element [i, j] = (j - i) mod n.
    """
    spatial_diff = torch.zeros((n, n), dtype=torch.long)
    for i in range(n):
        for j in range(n):
            spatial_diff[i, j] = (j - i) % n
    return spatial_diff


def diff_tensor_2d(n: int) -> torch.Tensor:
    """
    Generate 2D periodic spatial difference matrix for a square lattice.
    Element [i, j] is a single integer index representing (dx, dy).
    """
    n_sites = n * n
    spatial_diff_2d = torch.zeros((n_sites, n_sites), dtype=torch.long)
    
    for i in range(n_sites):
        i_x, i_y = i % n, i // n
        for j in range(n_sites):
            j_x, j_y = j % n, j // n
            dx = (j_x - i_x) % n
            dy = (j_y - i_y) % n
            spatial_diff_2d[i, j] = dy * n + dx # Single integer encoding of (dx, dy)
            
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


def define_d4_transformations(n: int) -> list:
    """
    Define the 8 transformations of the D4 (dihedral) group acting on 2D relative
    distance vectors (dx, dy) on a square lattice with periodic boundary conditions.
    """
    def identity(dx, dy):
        return dx, dy
    
    def rotate_90(dx, dy):
        # (dx, dy) -> (-dy mod n, dx)
        return (-dy) % n, dx
    
    def rotate_180(dx, dy):
        # (dx, dy) -> (-dx mod n, -dy mod n)
        return (-dx) % n, (-dy) % n
    
    def rotate_270(dx, dy):
        # (dx, dy) -> (dy, -dx mod n)
        return dy, (-dx) % n
    
    def reflect_vertical(dx, dy):
        # Reflect across vertical axis: (dx, dy) -> (-dx mod n, dy)
        return (-dx) % n, dy
    
    def reflect_horizontal(dx, dy):
        # Reflect across horizontal axis: (dx, dy) -> (dx, -dy mod n)
        return dx, (-dy) % n
    
    def reflect_diag1(dx, dy):
        # Reflect across main diagonal: (dx, dy) -> (dy, dx)
        return dy, dx
    
    def reflect_diag2(dx, dy):
        # Reflect across anti-diagonal: (dx, dy) -> (-dy mod n, -dx mod n)
        return (-dy) % n, (-dx) % n
    
    return [identity, rotate_90, rotate_180, rotate_270, reflect_vertical, 
            reflect_horizontal, reflect_diag1, reflect_diag2]


def group_action_space_2d_square(n: int) -> torch.Tensor:
    """
    Generate group action on space for 2D square lattice with D4 symmetry.
    """
    n_sites = n * n
    transformations = define_d4_transformations(n)
    
    # Define inverses: each transformation paired with its inverse index
    inverses = [0, 3, 2, 1, 4, 5, 6, 7]  # r^-1 = r3, r2^-1 = r2, r3^-1 = r, etc.
    
    group_action_space = torch.zeros((8, n_sites), dtype=torch.long)
    
    for h in range(8):
        h_inv = inverses[h]
        h_inv_transform = transformations[h_inv]
        
        for d in range(n_sites):
            # Convert flattened index d to (dx, dy)
            dx = d % n
            dy = d // n
            
            # Apply h^-1 transformation
            dx_new, dy_new = h_inv_transform(dx, dy)
            
            # Flatten back to index
            d_new = dy_new * n + dx_new
            group_action_space[h, d] = d_new
    
    return group_action_space


def group_mult_table_2d_square() -> torch.Tensor:
    """
    Generate group multiplication (Cayley) table for D4 (dihedral group of square).
    """
    # Use a larger lattice to avoid ambiguities in element identification
    n = 8
    
    transformations = define_d4_transformations(n)
    inverses = [0, 3, 2, 1, 4, 5, 6, 7]  # Inverse indices for each element
    
    # Identify group element by applying it to (1, 0) and (0, 1)
    def get_transformation_signature(g_idx):
        transform = transformations[g_idx]
        r1 = transform(1, 0)
        r2 = transform(0, 1)
        r1 = (r1[0] % n, r1[1] % n)
        r2 = (r2[0] % n, r2[1] % n)

        return (r1, r2)
    
    # Build signature lookup table
    signatures = {}
    for g_idx in range(8):
        sig = get_transformation_signature(g_idx)
        signatures[sig] = g_idx
    
    # Compute Cayley table
    group_mult = torch.zeros((8, 8), dtype=torch.long)
    
    for h in range(8):
        h_inv_idx = inverses[h]
        h_inv_transform = transformations[h_inv_idx]
        
        for h_tilde in range(8):
            h_tilde_transform = transformations[h_tilde]
            
            test1 = h_tilde_transform(1, 0)
            test1 = h_inv_transform(test1[0], test1[1])
            test1 = (test1[0] % n, test1[1] % n)
            
            test2 = h_tilde_transform(0, 1)
            test2 = h_inv_transform(test2[0], test2[1])
            test2 = (test2[0] % n, test2[1] % n)
            
            sig = (test1, test2)
            result_group_idx = signatures[sig]
            group_mult[h, h_tilde] = result_group_idx
    
    return group_mult


def make_lattice_config(
    lattice_type: str,
    n: int = None,
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
        if n is None:
            raise ValueError("For 2d_square, n (lattice size) must be provided.")
        return {
            "spatial_diff": diff_tensor_2d(n),
            "group_action_space": group_action_space_2d_square(n),
            "group_mult": group_mult_table_2d_square(),
            "num_sites": n * n,
            "group_size": 8,
            "lattice_type": "2d_square",
            "n": n,
        }
    else:
        raise ValueError(f"Unsupported lattice_type: {lattice_type}")
