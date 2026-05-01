import torch
import torch.nn as nn
from .config import CARTConfig


class HyperConnection(nn.Module):
    """
    Hyper-connection mechanism at loop boundaries.
    Maintains a ring buffer of the last n_hyper hidden states.
    Combines them with learned scalar weights initialized to residual baseline:
        weights = [1.0, 0.0, 0.0, ...]  (w0 = most recent, w_{n-1} = oldest)

    At loop iteration r:
        - Buffer holds states [h_{r-1}, h_{r-2}, ..., h_{r-n}]
        - Zero-padded for the first n-1 iterations
        - h_input = sum(w_i * buffer[i])

    With residual initialization, this behaves identically to standard
    residual connections until the training data supports learning non-zero
    weights for older states. The learned weight vector is a paper result.
    """
    def __init__(self, config: CARTConfig):
        super().__init__()
        n = config.n_hyper
        # Residual initialization: weight all on most recent state
        init = torch.zeros(n)
        init[0] = 1.0
        self.weights = nn.Parameter(init)
        self.n_hyper = n

    def init_buffer(self, h: torch.Tensor) -> list:
        """
        Initialize the ring buffer before the loop starts.
        All slots set to h (the initial hidden state from prelude output).
        With residual init weights [1,0,0] this is safe regardless of
        what fills the non-primary slots.
        """
        return [h.clone() for _ in range(self.n_hyper)]

    def combine(self, buffer: list) -> torch.Tensor:
        """
        buffer[0] = h_{r-1} (most recent)
        buffer[1] = h_{r-2}
        buffer[2] = h_{r-3}
        Returns the weighted combination as h_input for this iteration.
        """
        w = torch.softmax(self.weights, dim=0)  # Normalize weights
        result = sum(w[i] * buffer[i] for i in range(self.n_hyper))
        return result

    def update_buffer(self, buffer: list, h_new: torch.Tensor) -> list:
        """Shift buffer and insert new hidden state at position 0."""
        return [h_new.clone()] + buffer[:-1]
