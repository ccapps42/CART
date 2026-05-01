import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import CARTConfig


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network.
    output = down(silu(gate(x)) * up(x))
    No bias terms anywhere.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        d = config.d_model
        h = config.ffn_intermediate
        self.gate = nn.Linear(d, h, bias=False)
        self.up   = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
