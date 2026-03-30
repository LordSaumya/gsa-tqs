import torch
import wandb
import os
from train import optimise_vmc, optimise_vmc_ablation, evaluate_model, transfer_weights_topological
from models.equivariant_transformer import EquivariantTransformer
from models.standard_transformer import StandardTransformer


if __name__ == "__main__":
    # 1D Physical system
    n_sites_1d = 11
    J, Omega = 1.0, 1.0
    lattice_type = "1d_dihedral"
    
    # Model architecture hyperparameters (possible values)
    d_model = 4
    num_layers = 2
    num_heads = 2
    seed = [1, 2, 3]

    print("Architecture hyperparameters:")
    print(f"  d_model: {d_model}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads}")
    print(f"  seed: {seed}")

    # Training
    learning_rate = 1e-4
    steps = 5000

    results = optimise_vmc_ablation(
        n_sites_1d, J, Omega, d_model, num_layers, num_heads, learning_rate, steps, use_wandb=False, lattice_type=lattice_type
    )
    
    # Finish wandb run
    wandb.finish()