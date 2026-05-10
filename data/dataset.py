import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class FixedOrderDataset(Dataset):
    """
    Reads a pre-tokenized .bin file in fixed order.
    Every run uses identical token sequences at identical positions.
    No shuffling. This is intentional for sweep comparability.

    Each item is a sequence of seq_len tokens.
    Training: use tinystories_train.bin
    Eval: use data/val/*_val.bin files
    """
    def __init__(self, bin_path, seq_len: int):
        self.data = np.fromfile(str(bin_path), dtype=np.uint16)
        self.seq_len = seq_len
        # -1 because each item returns seq_len input + 1 next token as target
        self.n_seqs = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * self.seq_len
        tokens = torch.from_numpy(
            self.data[start:start + self.seq_len + 1].astype(np.int64)
        )
        return tokens[:-1], tokens[1:]  # input, target
