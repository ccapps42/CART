"""
Compute perplexity on a held-out .bin file.
Called by train_one.py at eval steps.
Also callable standalone for post-training evaluation.

Usage:
    python eval/perplexity.py --checkpoint checkpoints/{id}/step_3000.pt
                              --data data/val/wikipedia_val.bin
                              --seq-len 512
"""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import FixedOrderDataset
from model.cart import CART
from model.config import CARTConfig


def compute_perplexity(
    model: CART,
    bin_path,
    seq_len: int,
    device: torch.device,
    max_batches: int = 0,       # 0 = full dataset
    batch_size: int = 1,
) -> float:
    """
    Compute perplexity on a held-out .bin file.

    max_batches=0 evaluates the full file (post-sweep quality eval).
    max_batches=50 is the fast in-sweep approximation (see train_one.py).
    """
    if not Path(bin_path).exists():
        return float("nan")

    model.eval()
    dataset = FixedOrderDataset(bin_path, seq_len)
    n = len(dataset) if max_batches == 0 else min(max_batches, len(dataset))
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        i = 0
        while i < n:
            batch_end = min(i + batch_size, n)
            xs, ys = [], []
            for j in range(i, batch_end):
                x, y = dataset[j]
                xs.append(x)
                ys.append(y)
            x_batch = torch.stack(xs).to(device)
            y_batch = torch.stack(ys).to(device)
            _, loss = model(x_batch, y_batch)
            total_loss += loss.item()
            total_batches += 1
            i = batch_end

    model.train()
    return math.exp(total_loss / total_batches) if total_batches > 0 else float("nan")


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> CART:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = CART(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--data", required=True,
                        help="Path to held-out .bin file")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0=full dataset, N=cap at N batches")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    model = load_model_from_checkpoint(args.checkpoint, device)
    cfg = model.config
    print(f"Config: d={cfg.d_model} R={cfg.n_loops} P={cfg.n_prelude}")

    param_counts = model.count_parameters()
    print(f"Params: {param_counts['total']:,} (effective {param_counts['effective']:,})")

    print(f"\nEvaluating on: {args.data}")
    ppl = compute_perplexity(
        model,
        args.data,
        seq_len=args.seq_len,
        device=device,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
    )
    print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
