import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import pennylane as qml
import wandb

from hamiltonian import build_pennylane_tfim, build_hamiltonian_dense
from models.equivariant_transformer import EquivariantTransformer
from models.standard_transformer import StandardTransformer
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


def initialise_ablation_models(
    n_sites: int,
    d_model: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, torch.nn.Module, int, int]:
    """
    Initialise both equivariant and standard transformer models for ablation study.
    
    Returns:
        Tuple of (equivariant_model, standard_model, eq_num_params, std_num_params)
    """
    # Equivariant model
    eq_model = EquivariantTransformer(
        lattice_type="1d_dihedral",
        num_sites=n_sites,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        output_mode="polar",
        phase_init_zero=True,
        dropout=0.1,
    ).to(device)
    
    # Standard model (same hyperparameters)
    std_model = StandardTransformer(
        num_sites=n_sites,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        output_mode="polar",
        phase_init_zero=True,
        dropout=0.1,
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
    
    Args:
        eq_model: Equivariant transformer model
        std_model: Standard transformer model
        eq_optimiser: Optimizer for equivariant model
        std_optimiser: Optimizer for standard model
        loss_fn: Loss function
        H_context: Hamiltonian tensor
        J_params: Hamiltonian parameters
        steps: Number of training steps
        log_interval: Logging interval
        device: Device to train on
        gradient_clip_value: Gradient clipping value
        ground_state_energy: Ground state energy for error calculation
    
    Returns:
        Tuple of (eq_history, std_history)
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
    
    for step in range(steps):
        eq_optimiser.zero_grad()
        
        # Forward & backward
        eq_energy, eq_loss = loss_fn(eq_model, H_context, J_params)
        eq_loss.backward()
        
        # Compute gradient norm
        eq_grad_norm = 0.0
        for param in eq_model.parameters():
            if param.grad is not None:
                eq_grad_norm += torch.norm(param.grad).item() ** 2
        eq_grad_norm = (eq_grad_norm) ** 0.5
        
        # Gradient clipping
        if gradient_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(eq_model.parameters(), gradient_clip_value)
        
        # Update
        eq_optimiser.step()
        
        # Extract values
        eq_energy_val = eq_energy.item() if isinstance(eq_energy, torch.Tensor) else eq_energy
        eq_loss_val = eq_loss.item() if isinstance(eq_loss, torch.Tensor) else eq_loss
        
        eq_history["energy"].append(eq_energy_val)
        eq_history["loss"].append(eq_loss_val)
        eq_history["grad_norm"].append(eq_grad_norm)
        eq_history["steps"].append(step)
        
        std_optimiser.zero_grad()
        
        # Forward & backward
        std_energy, std_loss = loss_fn(std_model, H_context, J_params)
        std_loss.backward()
        
        # Compute gradient norm
        std_grad_norm = 0.0
        for param in std_model.parameters():
            if param.grad is not None:
                std_grad_norm += torch.norm(param.grad).item() ** 2
        std_grad_norm = (std_grad_norm) ** 0.5
        
        # Gradient clipping
        if gradient_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(std_model.parameters(), gradient_clip_value)
        
        # Update
        std_optimiser.step()
        
        # Extract values
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
            print(f"  Equivariant  | Energy: {eq_energy_val:.6f} | Loss: {eq_loss_val:.6f} | Grad: {eq_grad_norm:.6f}")
            print(f"  Standard     | Energy: {std_energy_val:.6f} | Loss: {std_loss_val:.6f} | Grad: {std_grad_norm:.6f}")
    
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
    
    # Build Hamiltonian
    H_qml = build_pennylane_tfim(n_sites, J, Omega, pbc=True)
    H_tensor = build_hamiltonian_dense(H_qml, device=device)
    
    # Calculate ground state energy if not provided
    if ground_state_energy is None:
        # Move to CPU for eigenvalue calculation if needed
        H_cpu = H_tensor.cpu() if H_tensor.device.type == 'cuda' else H_tensor
        eigenvalues = torch.linalg.eigvalsh(H_cpu)
        ground_state_energy = eigenvalues[0].item()
        print(f"Calculated ground state energy: {ground_state_energy:.6f}")
    else:
        print(f"Using provided ground state energy: {ground_state_energy:.6f}")
    
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
    )
    
    # Setup loss and optimiser
    loss_fn = ExactRayleighQuotientLoss(n_sites=n_sites, device=device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    J_params = torch.tensor([[J, Omega]], device=device, dtype=torch.float32)
    
    # Train
    history = train_vmc(
        model=model,
        optimiser=optimiser,
        loss_fn=loss_fn,
        H_context=H_tensor,
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
):
    """
    Initialise both equivariant and standard models and run VMC optimisation together for ablation study.
    
    Args:
        n_sites: Number of sites
        J: Ising coupling strength
        Omega: Transverse field strength
        d_model: Model embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        learning_rate: Learning rate
        steps: Number of training steps
        device: Device to train on
        seed: Random seed
        use_wandb: Whether to use W&B logging
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        ground_state_energy: Ground state energy (computed if not provided)
    
    Returns:
        Tuple of (eq_model, std_model, eq_history, std_history, eq_num_params, std_num_params, ground_state_energy)
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
    
    # Build Hamiltonian
    H_qml = build_pennylane_tfim(n_sites, J, Omega, pbc=True)
    H_tensor = build_hamiltonian_dense(H_qml, device=device)
    
    # Calculate ground state energy if not provided
    if ground_state_energy is None:
        H_cpu = H_tensor.cpu() if H_tensor.device.type == 'cuda' else H_tensor
        eigenvalues = torch.linalg.eigvalsh(H_cpu)
        ground_state_energy = eigenvalues[0].item()
        print(f"Calculated ground state energy: {ground_state_energy:.6f}")
    else:
        print(f"Using provided ground state energy: {ground_state_energy:.6f}")
    
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
    )
    
    print(f"Equivariant model: {eq_num_params} parameters")
    print(f"Standard model: {std_num_params} parameters")
    
    # Setup loss and optimisers
    loss_fn = ExactRayleighQuotientLoss(n_sites=n_sites, device=device)
    eq_optimiser = optim.Adam(eq_model.parameters(), lr=learning_rate)
    std_optimiser = optim.Adam(std_model.parameters(), lr=learning_rate)
    J_params = torch.tensor([[J, Omega]], device=device, dtype=torch.float32)
    
    # Train both models simultaneously
    eq_history, std_history = train_vmc_ablation(
        eq_model=eq_model,
        std_model=std_model,
        eq_optimiser=eq_optimiser,
        std_optimiser=std_optimiser,
        loss_fn=loss_fn,
        H_context=H_tensor,
        J_params=J_params,
        steps=steps,
        log_interval=max(1, steps // 20),
        device=device,
        gradient_clip_value=1.0,
        ground_state_energy=ground_state_energy,
    )
    
    return eq_model, std_model, eq_history, std_history, eq_num_params, std_num_params, ground_state_energy
