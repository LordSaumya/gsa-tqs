"""
Main entry point: GSA-TQS VMC optimisation.

Single function for initialisation and training, with results saving.
"""

import torch
import torch.optim as optim
import json
from pathlib import Path

from hamiltonian import build_pennylane_tfim, build_hamiltonian_dense
from losses import ExactRayleighQuotientLoss
from train import initialise_model, train_vmc


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
):
    """
    Initialise model and run VMC optimisation for the given Hamiltonian.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    
    # Build Hamiltonian
    H_qml = build_pennylane_tfim(n_sites, J, Omega, pbc=True)
    H_tensor = build_hamiltonian_dense(H_qml, device=device)
    
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
    )
    
    return model, history, num_params


if __name__ == "__main__":
    # Physical system
    n_sites = 10
    J, Omega = 1.0, 1.0
    
    # Model architecture
    d_model = 128
    num_layers = 2
    num_heads = 4
    
    # Training
    learning_rate = 1e-4
    steps = 500
    
    print(f"\n{'='*70}")
    print("GSA-TQS: Group Equivariant Variational Monte Carlo")
    print(f"{'='*70}\n")
    
    # Run optimisation
    model, history, num_params = optimise_vmc(
        n_sites=n_sites,
        J=J,
        Omega=Omega,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        learning_rate=learning_rate,
        steps=steps,
    )
    
    # Save results
    results_dir = "results"
    Path(results_dir).mkdir(exist_ok=True)
    
    energy_improvement = history['energy'][0] - history['energy'][-1]
    print(f"\nResults: E_initial={history['energy'][0]:.6f}, E_final={history['energy'][-1]:.6f}, ΔE={energy_improvement:.6f}\n")
    
    # Save training history
    history_path = Path(results_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "energy": history["energy"],
            "loss": history["loss"],
            "grad_norm": history["grad_norm"],
        }, f, indent=2)
    print(f"Saved history to {history_path}")
    
    # Save model weights
    model_path = Path(results_dir) / "model_weights.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    # Save config
    config_path = Path(results_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "physical": {"n_sites": n_sites, "J": J, "Omega": Omega},
            "architecture": {"d_model": d_model, "num_layers": num_layers, "num_heads": num_heads},
            "training": {"learning_rate": learning_rate, "steps": steps},
            "results": {
                "initial_energy": history["energy"][0],
                "final_energy": history["energy"][-1],
                "num_parameters": num_params,
            }
        }, f, indent=2)
    print(f"Saved config to {config_path}")