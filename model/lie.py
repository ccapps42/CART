import math
import torch
import torch.nn as nn
from .config import CARTConfig


class LoopIndexEmbedding(nn.Module):
    """
    Sinusoidal loop-index embedding (LIE).

    Injects a signal encoding the current loop iteration r into h_input
    before each CoreBlock pass. This gives the shared-weight block
    positional awareness in the loop dimension — it can learn that
    iteration 1 and iteration 6 are different computational contexts
    warranting different behavior.

    Without LIE, the CoreBlock processes every iteration identically
    except for the changing value of h_t. With LIE, the block can
    learn a genuine computational schedule across loop depth (e.g.,
    coarse pattern matching in early loops, fine-grained refinement
    in late loops).

    Implementation:
        - Pre-compute sinusoidal encodings for loop indices 0..max_loops-1
        - At each loop r, look up pe[r] and project to d_model
        - Add the projected signal to h_input (broadcast across [B, T])

    lie_dim = 32 (fixed, not swept). Parameter cost: 32 * d_model.
    At d=768 that is 24,576 parameters — negligible.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        self.lie_dim = config.lie_dim
        self.proj = nn.Linear(config.lie_dim, config.d_model, bias=False)
        # Pre-compute sinusoidal table for up to 16 loop indices
        pe = self._build_sinusoidal(max_loops=16, lie_dim=config.lie_dim)
        self.register_buffer("pe", pe)  # [16, lie_dim], not a parameter

    def _build_sinusoidal(self, max_loops: int, lie_dim: int) -> torch.Tensor:
        pe = torch.zeros(max_loops, lie_dim)
        pos = torch.arange(max_loops).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, lie_dim, 2).float() * -(math.log(10000.0) / lie_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, h: torch.Tensor, r: int) -> torch.Tensor:
        # h: [B, T, d_model]
        # r: current loop index, 0-based
        signal = self.proj(self.pe[r])  # [d_model]
        return h + signal               # broadcast: [B, T, d_model] + [d_model]
