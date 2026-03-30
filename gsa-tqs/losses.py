import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, Dict, Any


class BaseVMCLoss(nn.Module, ABC):
    """
    Abstract base class for Variational Monte Carlo loss functions.
    """
    
    @abstractmethod
    def forward(
        self,
        model: nn.Module,
        H_context: Union[torch.Tensor, Dict[str, Any]],
        J_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the energy and loss for variational optimisation.
        """
        pass


class ExactRayleighQuotientLoss(BaseVMCLoss):
    """
    Exact Variational Monte Carlo loss for small quantum systems (n <= 20).
    """
    
    def __init__(
        self,
        n_sites: int,
        device: str = "cpu",
    ):
        """
        Initialise the exact Rayleigh quotient loss.
        """
        super().__init__()
        
        self.n_sites = n_sites
        self.device = device
        self.basis_size = 2 ** n_sites
        
        ints = torch.arange(self.basis_size, device=device).unsqueeze(1)
        bits = (ints >> torch.arange(n_sites - 1, -1, -1, device=device)) & 1
        # basis: shape [2^n, n_sites]
        self.register_buffer("basis", (bits * 2 - 1).float())
    
    def forward(
        self,
        model: nn.Module,
        H: torch.Tensor,
        J_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the exact Rayleigh quotient.
        """
        # Expand J_params to match batch size
        J_batch = J_params.expand(self.basis.size(0), -1) # [basis_size, num_params]
        
        # Forward pass: evaluate model on all basis states
        alpha, phase = model(self.basis, state_indices=J_batch)
        
        Psi = torch.polar(torch.exp(alpha), phase)
        Psi = Psi.to(dtype=torch.complex128) if H.dtype == torch.float64 else Psi
        
        Psi_conj = torch.conj(Psi)
        norm_sq = torch.dot(Psi_conj, Psi) # Normalisation constant
        
        # norm_sq should be real; take real part and clamp to avoid division by zero
        norm_sq_real = norm_sq.real
        norm_sq_real = torch.clamp(norm_sq_real, min=1e-10)
        
        # Energy numerator: <psi | H | psi>
        if H.is_sparse:
            H_Psi = torch.sparse.mm(H, Psi.unsqueeze(1)).squeeze(1)
        else:
            H_Psi = torch.mv(H, Psi)
        
        energy_numerator = torch.dot(Psi_conj, H_Psi)
        
        # Rayleigh quotient: E = <psi | H | psi> / <psi | psi>
        energy_complex = energy_numerator / norm_sq_real
        
        # Extract real part (physical energy must be real by hermiticity of H)
        energy = energy_complex.real
        
        # Minimise energy
        loss = energy
        
        return energy, loss


class ExactRayleighQuotientLossWithGradNorm(BaseVMCLoss):
    """
    Extended exact loss with gradient norm regularisation.
    """
    
    def __init__(
        self,
        n_sites: int,
        device: str = "cpu",
        grad_reg_weight: float = 0.0,
    ):
        """
        Initialise the regularised exact loss.
        
        Args:
            n_sites: Number of lattice sites.
            device: Device to place tensors on ('cpu' or 'cuda').
            grad_reg_weight: Weight for gradient norm regularisation (0 = no regularisation).
        """
        super().__init__()    
        self.n_sites = n_sites
        self.device = device
        self.grad_reg_weight = grad_reg_weight
        self.basis_size = 2 ** n_sites
        
        # Pre-compute the full computational basis
        ints = torch.arange(self.basis_size, device=device).unsqueeze(1)
        bits = (ints >> torch.arange(n_sites - 1, -1, -1, device=device)) & 1
        self.register_buffer("basis", (bits * 2 - 1).float())
    
    def forward(
        self,
        model: nn.Module,
        H_sparse: torch.Tensor,
        J_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the exact Rayleigh quotient with optional gradient regularisation.
        """
        # Expand J_params to match batch size
        J_batch = J_params.expand(self.basis.size(0), -1)
        
        # Forward pass
        alpha, phase = model(self.basis, state_indices=J_batch)
        Psi = torch.polar(torch.exp(alpha.squeeze()), phase.squeeze())
        Psi = Psi.to(dtype=torch.complex128) if H_sparse.dtype == torch.float64 else Psi
        
        # Normalisation
        Psi_conj = torch.conj(Psi)
        norm_sq = torch.dot(Psi_conj, Psi)
        
        # Energy computation
        H_Psi = torch.mv(H_sparse, Psi)
        energy_numerator = torch.dot(Psi_conj, H_Psi)
        energy_complex = energy_numerator / norm_sq
        energy = energy_complex.real
        
        # Loss with optional regularisation
        loss = energy
        
        if self.grad_reg_weight > 0.0:
            # Compute gradient norm regularisation
            param_grad_norms = torch.tensor(0.0, device=self.device)
            for param in model.parameters():
                if param.requires_grad:
                    param_grad_norms = param_grad_norms + torch.norm(param)
            
            loss = energy + self.grad_reg_weight * param_grad_norms
        
        return energy, loss