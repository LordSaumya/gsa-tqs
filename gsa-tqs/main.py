import wandb
from train import optimise_vmc, optimise_vmc_ablation


if __name__ == "__main__":
    # Physical system
    n_sites = 10
    J, Omega = 3.0, 5.0
    
    # Model architecture
    d_model = 256
    num_layers = 3
    num_heads = 8
    
    # Training
    learning_rate = 1e-4
    steps = 5000
    
    results = optimise_vmc_ablation(
        n_sites, J, Omega, d_model, num_layers, num_heads, learning_rate, steps
    )

    # Finish wandb run
    wandb.finish()