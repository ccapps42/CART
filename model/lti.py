import math
import torch
import torch.nn as nn
from .config import CARTConfig


class LTIInjection(nn.Module):
    """
    LTI-stable recurrent injection (Parcae, Prairie et al. 2026).

    Replaces the standard residual connection:
        h = h_input + transformer_out
    with:
        h = A * h_input + transformer_out

    A is a learnable diagonal matrix parameterized as sigmoid(a_param),
    which guarantees every diagonal entry is in (0, 1) and therefore
    the spectral radius rho(A) < 1 by construction. This prevents
    residual explosion at high loop counts (R=6, R=8).

    Initialization: a_param = sigmoid_inverse(lti_init_value) so that
    A starts at lti_init_value (default 0.9) — stable but close to
    standard residual behavior (A=1). The model learns to tighten or
    relax A during training.

    The learned A values are a paper result: if A settles near 0.9
    uniformly, the forgetting mechanism is inert. If A is smaller at
    high R than low R, the model has learned to discard stale loop
    states more aggressively when it has more iterations available.
    Log rho(A) periodically during training.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        v = config.lti_init_value
        # sigmoid_inverse(v) = log(v / (1 - v))
        init_val = math.log(v / (1.0 - v))
        self.a_param = nn.Parameter(torch.full((config.d_model,), init_val))

    def forward(
        self,
        h_input: torch.Tensor,        # [B, T, d_model] — from hyper-connection
        transformer_out: torch.Tensor, # [B, T, d_model] — from CoreBlock
    ) -> torch.Tensor:
        A = torch.sigmoid(self.a_param)  # [d_model], all values in (0, 1)
        return A * h_input + transformer_out

    def spectral_radius(self) -> float:
        """Max diagonal value of A. Log this periodically during training."""
        with torch.no_grad():
            return torch.sigmoid(self.a_param).max().item()
