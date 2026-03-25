"""
Output Layer: Invariant Pooling and Polar Output for group equivariant networks.

Extracts an invariant representation from the equivariant features and projects
to amplitude and phase outputs (or complex wavefunction).

Module 4 from the architectural specification.

CRITICAL INITIALIZATION:
The phase projection head (W_beta, b_beta) MUST be initialized to 0.0.
This forces the initial phase to be purely real, preventing phase chaos
before the network begins learning.
"""

import torch
import torch.nn as nn
from typing import Tuple, Union


class InvariantPoolAndOutput(nn.Module):
    """
    Invariant pooling and output projection layer.
    
    Extracts an invariant representation by summing over spatial and group dimensions,
    then projects to amplitude and phase (or complex) outputs.
    
    Key Mathematical Properties:
    - Invariance: Output is invariant under group actions on input
    - Initialization: Phase projection head initialized to 0.0 for stability
    - Polar representation: Outputs (amplitude, phase) or complex tensor
    
    Attributes:
        d_model: Input feature dimension
        output_mode: "polar" (amplitude + phase) or "complex"
        phase_init_zero: Whether to initialize phase head to 0.0 (CRITICAL)
    """
    
    def __init__(
        self,
        d_model: int,
        output_mode: str = "polar",
        phase_init_zero: bool = True,
        use_complex: bool = False,
    ):
        """
        Initialize InvariantPoolAndOutput module.
        
        Args:
            d_model: Input feature dimension [B, n, |H|, d_model].
            output_mode: "polar" or "complex".
            phase_init_zero: If True, initialize phase projection to 0.0 (CRITICAL).
            use_complex: If True, output complex tensor instead of separate amplitude/phase.
        """
        super().__init__()
        
        self.d_model = d_model
        self.output_mode = output_mode
        self.phase_init_zero = phase_init_zero
        self.use_complex = use_complex
        
        if output_mode not in ("polar", "complex"):
            raise ValueError(f"Unrecognized output_mode: {output_mode}")
        
        # Amplitude projection head
        self.W_alpha = nn.Linear(d_model, 1)
        self.b_alpha = nn.Parameter(torch.zeros(1))
        
        # Phase projection head
        self.W_beta = nn.Linear(d_model, 1)
        self.b_beta = nn.Parameter(torch.zeros(1))
        
        # CRITICAL INITIALIZATION
        if phase_init_zero:
            self._initialize_phase_to_zero()
        else:
            # Standard initialization for both heads
            nn.init.xavier_uniform_(self.W_alpha.weight)
            nn.init.xavier_uniform_(self.W_beta.weight)
    
    def _initialize_phase_to_zero(self) -> None:
        """
        Initialize phase projection head to exactly 0.0.
        
        This forces Phi(s) = 0 for all configurations at initialization,
        making the starting wavefunction purely real and positive.
        The optimizer can then smoothly introduce complex phases as
        dictated by the loss landscape.
        
        Mathematical Justification:
        Without this, the extensive summation in the pooling layer induces
        severe instability at epoch 0, particularly in the phase.
        """
        # Zero-initialize phase weights and bias (INCLUDING Linear layer bias)
        nn.init.zeros_(self.W_beta.weight)
        if self.W_beta.bias is not None:
            nn.init.zeros_(self.W_beta.bias)
        self.b_beta.data.zero_()
        
        # Standard initialization for amplitude
        nn.init.xavier_uniform_(self.W_alpha.weight)
    
    def forward(
        self,
        X: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass: extract invariant features and project to output.
        
        Args:
            X: Input tensor of shape [B, n, |H|, d_model].
        
        Returns:
            If output_mode="polar":
                Tuple[amplitude, phase] where each is [B, 1]
            If output_mode="complex":
                Complex tensor [B] = amplitude * exp(i*phase)
        
        Mathematical Flow:
        1. Unweighted invariant trace: h_pool = sum over spatial & group dims
           Shape: [B, d_model]
        2. Amplitude projection: alpha = W_alpha(h_pool) + b_alpha
        3. Phase projection: beta = W_beta(h_pool) + b_beta
        4. If complex mode: output = alpha * exp(i*beta)
        """
        batch_size = X.shape[0]
        
        # 1. Compute unweighted invariant trace
        # X: [B, n, |H|, d_model]
        # Sum over spatial (dim 1) and group (dim 2) dimensions
        h_pool = X.sum(dim=(1, 2))  # [B, d_model]
        
        # 2. Amplitude projection
        alpha = self.W_alpha(h_pool) + self.b_alpha  # [B, 1]
        alpha = alpha.squeeze(-1)  # [B]
        
        # 3. Phase projection
        beta = self.W_beta(h_pool) + self.b_beta  # [B, 1]
        beta = beta.squeeze(-1)  # [B]
        
        if self.output_mode == "polar":
            # Return as polar coordinates (amplitude, phase)
            return alpha, beta
        elif self.output_mode == "complex":
            # Convert to complex: r * exp(i*phi)
            real_part = alpha * torch.cos(beta)
            imag_part = alpha * torch.sin(beta)
            return torch.complex(real_part, imag_part)
    
    def verify_phase_init(self) -> bool:
        """
        Verify that phase projection head is initialized to 0.0.
        
        Returns:
            True if phase head is exactly 0.0, False otherwise.
        """
        tolerance = 1e-10
        phase_weights_zero = torch.allclose(
            self.W_beta.weight, torch.zeros_like(self.W_beta.weight),
            atol=tolerance
        )
        phase_bias_zero = torch.allclose(
            self.b_beta, torch.zeros_like(self.b_beta),
            atol=tolerance
        )
        # Also check the Linear layer's bias
        linear_bias_zero = True
        if self.W_beta.bias is not None:
            linear_bias_zero = torch.allclose(
                self.W_beta.bias, torch.zeros_like(self.W_beta.bias),
                atol=tolerance
            )
        return phase_weights_zero and phase_bias_zero and linear_bias_zero


if __name__ == "__main__":
    print("Testing InvariantPoolAndOutput...")
    
    # Configuration
    batch_size = 2
    n_sites = 8
    group_size = 2
    d_model = 128
    
    # Create module
    module = InvariantPoolAndOutput(
        d_model=d_model,
        output_mode="polar",
        phase_init_zero=True,
    )
    
    # Verify phase initialization
    is_phase_zero = module.verify_phase_init()
    print(f"✓ Phase initialization verified: {is_phase_zero}")
    assert is_phase_zero, "Phase head should be initialized to 0.0"
    
    # Create dummy input
    X = torch.randn(batch_size, n_sites, group_size, d_model)
    
    # Forward pass
    alpha, beta = module(X)
    
    print(f"✓ OutputLayer forward pass successful")
    print(f"  Input shape: {X.shape}")
    print(f"  Amplitude shape: {alpha.shape}")
    print(f"  Phase shape: {beta.shape}")
    
    assert alpha.shape == (batch_size,), f"Amplitude shape mismatch: {alpha.shape}"
    assert beta.shape == (batch_size,), f"Phase shape mismatch: {beta.shape}"
    
    # Check phase values are close to 0 for first batch
    # (they should be close since W_beta and b_beta are 0)
    initial_phase_magnitude = torch.abs(beta).mean()
    print(f"  Initial phase magnitude: {initial_phase_magnitude.item():.6f}")
    assert initial_phase_magnitude < 0.01, "Phase should be close to 0 at init"
    
    # Test complex mode
    print(f"\n✓ Testing complex output mode...")
    module_complex = InvariantPoolAndOutput(
        d_model=d_model,
        output_mode="complex",
        phase_init_zero=True,
    )
    
    psi = module_complex(X)
    print(f"  Wavefunction (complex) shape: {psi.shape}")
    assert psi.shape == (batch_size,), f"Complex wavefunction shape mismatch: {psi.shape}"
    assert psi.dtype == torch.complex64 or psi.dtype == torch.complex128, \
        f"Wavefunction should be complex, got {psi.dtype}"
    
    # Test backward pass
    loss = alpha.sum() + beta.sum()
    loss.backward()
    print(f"\n✓ Backward pass successful")
    
    print(f"\n✓ ALL OUTPUT LAYER TESTS PASSED")
