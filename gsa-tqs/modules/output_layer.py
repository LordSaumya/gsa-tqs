import torch
import torch.nn as nn
from typing import Tuple, Union


class InvariantPoolAndOutput(nn.Module):
    """
    Invariant pooling and output projection layer.
    """
    
    def __init__(
        self,
        d_model: int,
        output_mode: str = "polar",
        phase_init_zero: bool = True,
        use_complex: bool = False,
    ):
        """
        Initialise InvariantPoolAndOutput module.
        """
        super().__init__()
        
        self.d_model = d_model
        self.output_mode = output_mode
        self.phase_init_zero = phase_init_zero
        self.use_complex = use_complex
        
        if output_mode not in ("polar", "complex"):
            raise ValueError(f"Unrecognised output_mode: {output_mode}")
        
        # Amplitude projection head
        self.W_alpha = nn.Linear(d_model, 1)
        self.b_alpha = nn.Parameter(torch.zeros(1))
        
        # Phase projection head
        self.W_beta = nn.Linear(d_model, 1)
        self.b_beta = nn.Parameter(torch.zeros(1))
        
        if phase_init_zero:
            self._initialise_phase_to_zero()
        else:
            # Standard initialisation for both heads
            nn.init.xavier_uniform_(self.W_alpha.weight)
            nn.init.xavier_uniform_(self.W_beta.weight)
    
    def _initialise_phase_to_zero(self) -> None:
        """
        Initialise phase projection head to 0.0.
        """
        # Zero-initialise phase weights and bias
        nn.init.zeros_(self.W_beta.weight)
        if self.W_beta.bias is not None:
            nn.init.zeros_(self.W_beta.bias)
        self.b_beta.data.zero_()
        
        # Standard initialisation for amplitude
        nn.init.xavier_uniform_(self.W_alpha.weight)
    
    def forward(
        self,
        X: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass: extract invariant features and project to output.
        """        
        # Sum over sites and group elements
        h_pool = X.sum(dim=(1, 2))  # [B, d_model]
        
        # Amplitude projection
        alpha = self.W_alpha(h_pool) + self.b_alpha  # [B, 1]
        alpha = alpha.squeeze(-1)  # [B]
        
        # Phase projection
        beta = self.W_beta(h_pool) + self.b_beta  # [B, 1]
        beta = beta.squeeze(-1)  # [B]
        
        if self.output_mode == "polar":
            # Return as polar coordinates (log amplitude, phase)
            return alpha, beta
        elif self.output_mode == "complex":
            amplitude = torch.exp(alpha)
            # Return exp()
            return torch.polar(amplitude, beta)

    def verify_phase_init(self) -> bool:
        """
        Verify that phase projection head is initialised to 0.0.
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
        linear_bias_zero = True
        if self.W_beta.bias is not None:
            linear_bias_zero = torch.allclose(
                self.W_beta.bias, torch.zeros_like(self.W_beta.bias),
                atol=tolerance
            )
        return phase_weights_zero and phase_bias_zero and linear_bias_zero