import pennylane as qml
import torch
import numpy as np

def build_pennylane_tfim(
    n_sites: int,
    J: float,
    Omega: float,
    pbc: bool = True,
    lattice_type: str = "1d_dihedral",
) -> qml.Hamiltonian:
    """
    Construct a Transverse Field Ising Model (TFIM) Hamiltonian using PennyLane.
    """
    obs = []
    coeffs = []
    
    if lattice_type == "1d_dihedral":
        # 1D TFIM: -J * sum z_i z_{i+1} - Omega * sum x_i
        
        # Interaction terms: -J * Z_i Z_{i+1}
        for i in range(n_sites):
            if i == n_sites - 1 and not pbc:
                continue
            j = (i + 1) % n_sites
            obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
            coeffs.append(-J)
        
        # Transverse field terms: -Omega * X_i
        for i in range(n_sites):
            obs.append(qml.PauliX(i))
            coeffs.append(-Omega)
    
    elif lattice_type == "2d_square":
        # 2D square lattice TFIM on an n*n grid

        n = n_sites  # n_sites is the side length for 2D
        total_sites = n * n
        
        # Horizontal interaction terms: -J * Z_{x,y} Z_{x+1,y}
        for y in range(n):
            for x in range(n):
                i = y * n + x
                j = y * n + ((x + 1) % n)  # Periodic boundary in x
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                coeffs.append(-J)
        
        # Vertical interaction terms: -J * Z_{x,y} Z_{x,y+1}
        for y in range(n):
            for x in range(n):
                i = y * n + x
                j = ((y + 1) % n) * n + x  # Periodic boundary in y
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                coeffs.append(-J)
        
        # Transverse field terms: -Omega * X_i
        for i in range(total_sites):
            obs.append(qml.PauliX(i))
            coeffs.append(-Omega)
    
    else:
        raise ValueError(f"Unsupported lattice_type: {lattice_type}")
    
    return qml.Hamiltonian(coeffs, obs)


def hamiltonian_to_torch_sparse(
    H_qml: qml.Hamiltonian,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Convert a PennyLane Hamiltonian to a PyTorch sparse CSR tensor.
    For real-valued Hamiltonians (X and Z operators), explicitly takes the real part.
    """
    # Get dense matrix representation from PennyLane Hamiltonian
    from scipy.sparse import csr_matrix
    H_dense = qml.matrix(H_qml)
    # Use real part
    H_dense = np.real(H_dense)
    H_scipy = csr_matrix(H_dense)
    
    # Extract CSR format components
    data = torch.tensor(H_scipy.data, dtype=torch.float64, device=device)
    indices = torch.tensor(H_scipy.indices, dtype=torch.long, device=device)
    indptr = torch.tensor(H_scipy.indptr, dtype=torch.long, device=device)
    
    # Construct PyTorch sparse CSR tensor
    H_torch = torch.sparse_csr_tensor(
        crow_indices=indptr,
        col_indices=indices,
        values=data,
        size=H_scipy.shape,
        device=device,
    )
    
    return H_torch


def build_hamiltonian_dense(
    H_qml: qml.Hamiltonian,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Convert a PennyLane Hamiltonian to a dense PyTorch tensor.
    For real-valued Hamiltonians (X and Z operators), explicitly takes the real part.
    """
    H_dense = qml.matrix(H_qml)
    # Use real part
    H_dense = np.real(H_dense)
    H_torch = torch.tensor(H_dense, dtype=torch.float64, device=device)
    return H_torch