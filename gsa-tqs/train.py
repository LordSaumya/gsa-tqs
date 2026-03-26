import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Optional

from models.equivariant_transformer import EquivariantTransformer
from losses import ExactRayleighQuotientLoss, BaseVMCLoss


def initialise_model(
    n_sites: int,
    d_model: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, int]:
    """
    Initialise the equivariant neural network ansatz.
    """
    model = EquivariantTransformer(
        lattice_type="1d_dihedral",
        num_sites=n_sites,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        output_mode="polar",
        phase_init_zero=True,
        dropout=0.1,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    return model, num_params


def train_vmc(
    model: torch.nn.Module,
    optimiser: optim.Optimizer,
    loss_fn: BaseVMCLoss,
    H_context: torch.Tensor,
    J_params: torch.Tensor,
    steps: int = 1000,
    log_interval: int = 10,
    device: str = "cpu",
    gradient_clip_value: float = 1.0,
) -> Dict[str, List[float]]:
    """
    Run the Variational Monte Carlo optimisation loop. Can switch between exact and MCMC evaluation by swapping loss_fn and H_context.
    """
    model.train()
    
    history = {
        "energy": [],
        "loss": [],
        "steps": [],
        "grad_norm": [],
    }
    
    for step in range(steps):
        optimiser.zero_grad()
        
        # Forward & backward
        energy, loss = loss_fn(model, H_context, J_params)
        loss.backward()
        
        # Compute gradient norm
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += torch.norm(param.grad).item() ** 2
        total_grad_norm = (total_grad_norm) ** 0.5
        
        # Gradient clipping
        if gradient_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
        
        # Update
        optimiser.step()
    
    return history
