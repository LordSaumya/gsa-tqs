import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import wandb
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from torch.amp import autocast, GradScaler

from hamiltonian import build_pennylane_tfim, hamiltonian_to_torch_sparse
from models.equivariant_transformer import EquivariantTransformer
from models.standard_transformer import StandardTransformer
from losses import ExactRayleighQuotientLoss, BaseVMCLoss


def compute_ground_state_energy_from_sparse(H_sparse: torch.Tensor) -> float:
    """
    Compute the ground state energy directly from sparse Hamiltonian matrix.
    """
    # Convert PyTorch sparse CSR tensor to scipy sparse matrix
    crow_indices = H_sparse.crow_indices().cpu().numpy()
    col_indices = H_sparse.col_indices().cpu().numpy()
    values = H_sparse.values().cpu().numpy()
    
    H_scipy = csr_matrix((values, col_indices, crow_indices), shape=H_sparse.shape)
    
    # Compute only the ground state (lowest eigenvalue) using sparse eigenvalue solver
    eigenvalues, _ = eigsh(H_scipy, k=1, which='SA')
    
    return float(eigenvalues[0])


def initialise_model(
    n_sites: int,
    d_model: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    device: str = "cpu",
    lattice_type: str = "1d_dihedral",
    **lattice_kwargs,
) -> Tuple[torch.nn.Module, int]:
    """
    Initialise the equivariant neural network ansatz.
    """
    model = EquivariantTransformer(
        lattice_type=lattice_type,
        num_sites=n_sites,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        output_mode="polar",
        phase_init_zero=True,
        **lattice_kwargs,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    return model, num_params


def initialise_ablation_models(
    n_sites: int,
    d_model: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    device: str = "cpu",
    lattice_type: str = "1d_dihedral",
    **lattice_kwargs,
) -> Tuple[torch.nn.Module, torch.nn.Module, int, int]:
    """
    Initialise models for ablation study.
    """

    # Standard model (same hyperparameters, uses actual number of sites)
    if lattice_type == "2d_square":
        actual_n_sites = n_sites * n_sites
    else:
        actual_n_sites = n_sites

    # Equivariant model
    eq_model = EquivariantTransformer(
        lattice_type=lattice_type,
        num_sites=actual_n_sites,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        output_mode="polar",
        phase_init_zero=True,
        **lattice_kwargs,
    ).to(device)
    
    std_model = StandardTransformer(
        num_sites=actual_n_sites,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        output_mode="polar",
        phase_init_zero=True,
        pe_mode=1 # Absolute positional encoding
    ).to(device)
    
    eq_num_params = sum(p.numel() for p in eq_model.parameters())
    std_num_params = sum(p.numel() for p in std_model.parameters())
    
    return eq_model, std_model, eq_num_params, std_num_params


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
    ground_state_energy: float = None,
) -> Dict[str, List[float]]:
    """
    Run VMC optimisation loop. Can switch between exact and MCMC evaluation by swapping loss_fn and H_context.
    """
    model.train()
    
    history = {
        "energy": [],
        "loss": [],
        "steps": [],
        "grad_norm": [],
    }
    
    scaler = GradScaler('cuda')
    
    for step in range(steps):
        optimiser.zero_grad()
        
        with autocast('cuda'):
            energy, loss = loss_fn(model, H_context, J_params)
        
        scaler.scale(loss).backward()
        
        if gradient_clip_value > 0:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value).item()
        else:
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += torch.norm(param.grad).item() ** 2
            total_grad_norm = (total_grad_norm) ** 0.5
        
        scaler.step(optimiser)
        scaler.update()
        
        # Log metrics
        energy_val = energy.item() if isinstance(energy, torch.Tensor) else energy
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
        
        history["energy"].append(energy_val)
        history["loss"].append(loss_val)
        history["grad_norm"].append(total_grad_norm)
        history["steps"].append(step)
        
        # Log to wandb
        if wandb.run is not None:
            log_dict = {
                "energy": energy_val,
                "loss": loss_val,
                "grad_norm": total_grad_norm,
            }
            
            # Add ground state energy error if available
            if ground_state_energy is not None:
                energy_error = energy_val - ground_state_energy
                log_dict["energy_error_vs_gs"] = energy_error
            
            wandb.log(log_dict)
        
        # Console logging
        if log_interval > 0 and (step + 1) % log_interval == 0:
            print(f"Step {step + 1}/{steps} | Energy: {energy_val:.6f} | Loss: {loss_val:.6f} | Grad norm: {total_grad_norm:.6f}")
    
    return history


def train_vmc_ablation(
    eq_model: torch.nn.Module,
    std_model: torch.nn.Module,
    eq_optimiser: optim.Optimizer,
    std_optimiser: optim.Optimizer,
    loss_fn: BaseVMCLoss,
    H_context: torch.Tensor,
    J_params: torch.Tensor,
    steps: int = 1000,
    log_interval: int = 10,
    device: str = "cpu",
    gradient_clip_value: float = 1.0,
    ground_state_energy: float = None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Run VMC optimisation for both equivariant and standard transformers simultaneously.
    """
    eq_model.train()
    std_model.train()
    
    eq_history = {
        "energy": [],
        "loss": [],
        "steps": [],
        "grad_norm": [],
    }
    
    std_history = {
        "energy": [],
        "loss": [],
        "steps": [],
        "grad_norm": [],
    }
    
    eq_scaler = GradScaler('cuda')
    std_scaler = GradScaler('cuda')
    
    for step in range(steps):
        eq_optimiser.zero_grad()
        
        with autocast('cuda'):
            eq_energy, eq_loss = loss_fn(eq_model, H_context, J_params)
        
        eq_scaler.scale(eq_loss).backward()
        
        if gradient_clip_value > 0:
            eq_grad_norm = torch.nn.utils.clip_grad_norm_(eq_model.parameters(), gradient_clip_value).item()
        else:
            eq_grad_norm = 0.0
            for param in eq_model.parameters():
                if param.grad is not None:
                    eq_grad_norm += torch.norm(param.grad).item() ** 2
            eq_grad_norm = (eq_grad_norm) ** 0.5
        
        eq_scaler.step(eq_optimiser)
        eq_scaler.update()
        
        eq_energy_val = eq_energy.item() if isinstance(eq_energy, torch.Tensor) else eq_energy
        eq_loss_val = eq_loss.item() if isinstance(eq_loss, torch.Tensor) else eq_loss
        
        eq_history["energy"].append(eq_energy_val)
        eq_history["loss"].append(eq_loss_val)
        eq_history["grad_norm"].append(eq_grad_norm)
        eq_history["steps"].append(step)
        
        std_optimiser.zero_grad()
        
        with autocast('cuda'):
            std_energy, std_loss = loss_fn(std_model, H_context, J_params)
        
        std_scaler.scale(std_loss).backward()
        
        if gradient_clip_value > 0:
            std_grad_norm = torch.nn.utils.clip_grad_norm_(std_model.parameters(), gradient_clip_value).item()
        else:
            std_grad_norm = 0.0
            for param in std_model.parameters():
                if param.grad is not None:
                    std_grad_norm += torch.norm(param.grad).item() ** 2
            std_grad_norm = (std_grad_norm) ** 0.5
        
        std_scaler.step(std_optimiser)
        std_scaler.update()
        
        std_energy_val = std_energy.item() if isinstance(std_energy, torch.Tensor) else std_energy
        std_loss_val = std_loss.item() if isinstance(std_loss, torch.Tensor) else std_loss
        
        std_history["energy"].append(std_energy_val)
        std_history["loss"].append(std_loss_val)
        std_history["grad_norm"].append(std_grad_norm)
        std_history["steps"].append(step)
        
        if wandb.run is not None:
            log_dict = {
                "eq_energy": eq_energy_val,
                "eq_loss": eq_loss_val,
                "eq_grad_norm": eq_grad_norm,
                "std_energy": std_energy_val,
                "std_loss": std_loss_val,
                "std_grad_norm": std_grad_norm,
                "diff_energy_eq_std": eq_energy_val - std_energy_val,
            }
            
            # Add ground state energy error if available
            if ground_state_energy is not None:
                log_dict["eq_energy_error_vs_gs"] = eq_energy_val - ground_state_energy
                log_dict["std_energy_error_vs_gs"] = std_energy_val - ground_state_energy
            
            wandb.log(log_dict)
        
        # Console logging
        if log_interval > 0 and (step + 1) % log_interval == 0:
            print(f"Step {step + 1}/{steps}")
            print(f"  Equivariant  | Energy: {eq_energy_val:.6f} | Grad: {eq_grad_norm:.6f}")
            print(f"  Standard     | Energy: {std_energy_val:.6f} | Grad: {std_grad_norm:.6f}")
    
    return eq_history, std_history

def optimise_vmc(
    n_sites: int,
    J: float,
    Omega: float,
    d_model: int,
    num_layers: int,
    num_heads: int,
    learning_rate: float,
    steps: int,
    device: str = None,
    seed: int = 42,
    use_wandb: bool = True,
    wandb_project: str = "gsa-tqs",
    wandb_entity: str = None,
    ground_state_energy: float = None,
    lattice_type: str = "1d_dihedral",
):
    """
    Initialise model and run VMC optimisation for the given Hamiltonian.
    If ground_state_energy is not provided, it will be calculated from the Hamiltonian.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    
    # Initialise wandb
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                "n_sites": n_sites,
                "J": J,
                "Omega": Omega,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "learning_rate": learning_rate,
                "steps": steps,
                "device": device,
                "seed": seed,
            },
            tags=["vmc", "1d-tfim"],
        )
    
    # Calculate actual number of sites based on lattice type
    if lattice_type == "1d_dihedral":
        actual_n_sites = n_sites
    elif lattice_type == "2d_square":
        actual_n_sites = n_sites * n_sites
    else:
        raise ValueError(f"Unsupported lattice_type: {lattice_type}")
    
    # Build sparse Hamiltonian
    H_qml = build_pennylane_tfim(n_sites, J, Omega, pbc=True, lattice_type=lattice_type)
    H_tensor_sparse = hamiltonian_to_torch_sparse(H_qml, device=device)
    
    # Calculate ground state energy if not provided
    if ground_state_energy is None:
        H_sparse_cpu = hamiltonian_to_torch_sparse(H_qml, device="cpu")
        ground_state_energy = compute_ground_state_energy_from_sparse(H_sparse_cpu)
        print(f"Calculated ground state energy: {ground_state_energy:.10f}")
    else:
        print(f"Using provided ground state energy: {ground_state_energy:.10f}")
    
    # Log ground state energy to wandb
    if use_wandb:
        wandb.config.update({"ground_state_energy": ground_state_energy})
    
    # Initialise model
    model, num_params = initialise_model(
        n_sites=n_sites,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
        lattice_type=lattice_type,
    )
    
    # Setup loss and optimiser
    loss_fn = ExactRayleighQuotientLoss(n_sites=actual_n_sites, device=device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    J_params = torch.tensor([[J, Omega]], device=device, dtype=torch.float32)
    
    # Train
    history = train_vmc(
        model=model,
        optimiser=optimiser,
        loss_fn=loss_fn,
        H_context=H_tensor_sparse,
        J_params=J_params,
        steps=steps,
        log_interval=max(1, steps // 20),
        device=device,
        gradient_clip_value=1.0,
        ground_state_energy=ground_state_energy,
    )
    
    return model, history, num_params, ground_state_energy


def optimise_vmc_ablation(
    n_sites: int,
    J: float,
    Omega: float,
    d_model: int,
    num_layers: int,
    num_heads: int,
    learning_rate: float,
    steps: int,
    device: str = None,
    seed: int = 42,
    use_wandb: bool = True,
    wandb_project: str = "gsa-tqs",
    wandb_entity: str = None,
    ground_state_energy: float = None,
    lattice_type: str = "1d_dihedral",
    log_interval: int = 250,
):
    """
    Initialise both equivariant and standard models and run VMC optimisation together for ablation study.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    
    # Initialise wandb with ablation tag
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                "n_sites": n_sites,
                "J": J,
                "Omega": Omega,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "learning_rate": learning_rate,
                "steps": steps,
                "device": device,
                "seed": seed,
                "ablation_study": True,
            },
            tags=["vmc", "1d-tfim", "ablation"],
        )
    
    # Calculate actual number of sites based on lattice type
    if lattice_type == "1d_dihedral":
        actual_n_sites = n_sites
    elif lattice_type == "2d_square":
        actual_n_sites = n_sites * n_sites
    else:
        raise ValueError(f"Unsupported lattice_type: {lattice_type}")
    
    # Build sparse Hamiltonian
    H_qml = build_pennylane_tfim(n_sites, J, Omega, pbc=True, lattice_type=lattice_type)
    H_tensor_sparse = hamiltonian_to_torch_sparse(H_qml, device=device)
    
    # Calculate ground state energy if not provided
    if ground_state_energy is None:
        H_sparse_cpu = hamiltonian_to_torch_sparse(H_qml, device="cpu")
        ground_state_energy = compute_ground_state_energy_from_sparse(H_sparse_cpu)
        print(f"Calculated ground state energy: {ground_state_energy:<15.10f}")
    else:
        print(f"Using provided ground state energy: {ground_state_energy:<15.10f}")
    
    # Log ground state energy to wandb
    if use_wandb:
        wandb.config.update({"ground_state_energy": ground_state_energy})
    
    # Initialise both models
    eq_model, std_model, eq_num_params, std_num_params = initialise_ablation_models(
        n_sites=n_sites,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
        lattice_type=lattice_type,
    )
    
    print(f"Equivariant model: {eq_num_params} parameters")
    print(f"Standard model: {std_num_params} parameters")
    
    # Setup loss and optimisers
    loss_fn = ExactRayleighQuotientLoss(n_sites=actual_n_sites, device=device)
    eq_optimiser = optim.Adam(eq_model.parameters(), lr=learning_rate)
    std_optimiser = optim.Adam(std_model.parameters(), lr=learning_rate)
    J_params = torch.tensor([[J, Omega]], device=device, dtype=torch.float32)

    if log_interval == 0:
        log_interval = 2 * steps  # Disable logging if set to 0
    
    # Train both models simultaneously
    eq_history, std_history = train_vmc_ablation(
        eq_model=eq_model,
        std_model=std_model,
        eq_optimiser=eq_optimiser,
        std_optimiser=std_optimiser,
        loss_fn=loss_fn,
        H_context=H_tensor_sparse,
        J_params=J_params,
        steps=steps,
        log_interval=log_interval,
        device=device,
        gradient_clip_value=1.0,
        ground_state_energy=ground_state_energy,
    )
    
    return eq_model, std_model, eq_history, std_history, eq_num_params, std_num_params, ground_state_energy
def evaluate_model(
    model: torch.nn.Module,
    n_sites: int,
    J: float,
    Omega: float,
    device: str = "cpu",
    return_ground_state: bool = False,
    lattice_type: str = "1d_dihedral",
) -> Tuple[float, Optional[float]]:
    """
    Evaluate a trained model on the TFIM Hamiltonian for a given system size.
    """
    model.eval()  # Set to evaluation mode
    
    # Calculate actual number of sites
    if lattice_type == "1d_dihedral":
        actual_n_sites = n_sites
    elif lattice_type == "2d_square":
        actual_n_sites = n_sites * n_sites
    else:
        raise ValueError(f"Unsupported lattice_type: {lattice_type}")
    
    with torch.no_grad():
        # Build sparse Hamiltonian for system size
        H_qml = build_pennylane_tfim(n_sites, J, Omega, pbc=True, lattice_type=lattice_type)
        H_tensor_sparse = hamiltonian_to_torch_sparse(H_qml, device=device)
        
        # Create loss function
        loss_fn = ExactRayleighQuotientLoss(n_sites=actual_n_sites, device=device)
        J_params = torch.tensor([[J, Omega]], device=device, dtype=torch.float32)
        
        # Compute model energy
        energy, _ = loss_fn(model, H_tensor_sparse, J_params)
        energy_val = energy.item()
        
        # Compute ground state energy
        ground_state_energy = None
        if return_ground_state:
            H_sparse_cpu = hamiltonian_to_torch_sparse(H_qml, device="cpu")
            ground_state_energy = compute_ground_state_energy_from_sparse(H_sparse_cpu)
    
    return energy_val, ground_state_energy
