import pennylane as qml
import torch
import numpy as np
from typing import Tuple, Optional

def build_pennylane_tfim(
    n_sites: int,
    J: float,
    Omega: float,
    pbc: bool = True,
) -> qml.Hamiltonian:
    """
    Construct a 1D Transverse Field Ising Model (TFIM) Hamiltonian using PennyLane.
    """
    obs = []
    coeffs = []
    
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
    
    return qml.Hamiltonian(coeffs, obs)


def hamiltonian_to_torch_sparse(
    H_qml: qml.Hamiltonian,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Convert a PennyLane Hamiltonian to a PyTorch sparse CSR tensor.
    """
    # Get dense matrix representation from PennyLane Hamiltonian
    from scipy.sparse import csr_matrix
    H_dense = qml.matrix(H_qml)
    H_scipy = csr_matrix(H_dense)
    
    # Extract CSR format components
    data = torch.tensor(H_scipy.data, dtype=torch.complex64, device=device)
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
    """
    H_dense = qml.matrix(H_qml)
    H_torch = torch.tensor(H_dense, dtype=torch.complex64, device=device)
    return H_torch