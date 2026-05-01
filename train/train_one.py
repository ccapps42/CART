"""
Trains exactly one config. Reads config from results.db, writes results,
marks status complete or failed. Never loops over configs.

Usage:
    python train/train_one.py --config-id <id> --db results.db

    # Testing only:
    python train/train_one.py --config-id <id> --db results.db --max-steps 50
"""
import argparse
import math
import sqlite3
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import CARTConfig
from model.cart import CART
from data.dataset import FixedOrderDataset
from train.lr_schedule import get_lr

# ---------------------------------------------------------------------------
# Fixed hyperparameters — identical across all sweep runs
# ---------------------------------------------------------------------------
SEQ_LEN        = 512
BATCH_SIZE     = 4
GRAD_ACCUM     = 8        # effective batch = 4 * 8 * 512 = 16,384 tokens/step
TOTAL_STEPS    = 1500     # Stage 1
WARMUP_STEPS   = 100
PEAK_LR        = 3e-4
MIN_LR         = 3e-5
WEIGHT_DECAY   = 0.1
GRAD_CLIP      = 1.0
EVAL_STEPS     = (500, 1500)
LOG_INTERVAL   = 50       # Tier 1 log every 50 steps
MAX_EVAL_BATCHES = 50

# ---------------------------------------------------------------------------
# Default paths (relative to project root, may be overridden via CLI)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_DEFAULT_TRAIN_BIN = _ROOT / "data" / "tinystories_train.bin"
_DEFAULT_VAL_DIR   = _ROOT / "data" / "val"
CKPT_DIR  = _ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def load_config_row(conn, config_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM configs WHERE config_id = ?", (config_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"config_id '{config_id}' not found in DB")
    return row


def mark_running(conn, config_id: str):
    conn.execute(
        "UPDATE configs SET status='running', started_at=datetime('now') WHERE config_id=?",
        (config_id,),
    )
    conn.commit()


def mark_complete(conn, config_id: str):
    conn.execute(
        "UPDATE configs SET status='complete', completed_at=datetime('now') WHERE config_id=?",
        (config_id,),
    )
    conn.commit()


def mark_failed(conn, config_id: str, error_msg: str):
    conn.execute(
        """UPDATE configs SET status='failed', completed_at=datetime('now'),
           error_msg=?, retry_count=retry_count+1
           WHERE config_id=?""",
        (error_msg[:2000], config_id),
    )
    conn.commit()


def write_train_log(conn, config_id, step, loss, grad_norm, lr,
                    n_tokens_seen, wall_sec, spectral_radius):
    conn.execute(
        """INSERT INTO train_log
           (config_id, step, train_loss, grad_norm, lr,
            n_tokens_seen, lti_spectral_radius, wall_sec, recorded_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
        (config_id, step, loss, grad_norm, lr,
         n_tokens_seen, spectral_radius, wall_sec),
    )
    conn.commit()


def write_results(conn, config_id, step, ppl_tiny, ppl_wiki, ppl_edu,
                  peak_vram_gb, tokens_per_sec, spectral_radius, wall_sec):
    conn.execute(
        """INSERT INTO results
           (config_id, step, eval_ppl_tiny, eval_ppl_wiki, eval_ppl_edu,
            peak_vram_gb, tokens_per_sec, lti_spectral_radius,
            wall_sec, recorded_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
        (config_id, step, ppl_tiny, ppl_wiki, ppl_edu,
         peak_vram_gb, tokens_per_sec, spectral_radius, wall_sec),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def eval_perplexity(model, val_bin_path, seq_len, device,
                    max_batches=MAX_EVAL_BATCHES) -> float:
    if not Path(val_bin_path).exists():
        return float("nan")
    model.eval()
    dataset = FixedOrderDataset(val_bin_path, seq_len)
    total_loss = 0.0
    n = min(max_batches, len(dataset))
    with torch.no_grad():
        for i in range(n):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
    model.train()
    return math.exp(total_loss / n) if n > 0 else float("nan")


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, config_id, scaler=None):
    d = CKPT_DIR / config_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"step_{step}.pt"
    payload = {
        "step": step,
        "config": model.config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, str(path))
    return path


# ---------------------------------------------------------------------------
# Rolling tokens/sec
# ---------------------------------------------------------------------------

def rolling_tps(step_times: deque) -> float:
    if len(step_times) < 2:
        return 0.0
    elapsed = step_times[-1][0] - step_times[0][0]
    tokens = sum(t[1] for t in step_times)
    return tokens / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--db", default="results.db")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override total training steps (for testing only)")
    parser.add_argument("--train-bin", default=None,
                        help="Override training .bin path (for testing only)")
    parser.add_argument("--val-dir", default=None,
                        help="Override val/ directory (for testing only)")
    args = parser.parse_args()

    # Resolve paths
    train_bin = Path(args.train_bin) if args.train_bin else _DEFAULT_TRAIN_BIN
    val_dir   = Path(args.val_dir)   if args.val_dir   else _DEFAULT_VAL_DIR
    val_files = {
        "tiny":    val_dir / "tinystories_val.bin",
        "wiki":    val_dir / "wikipedia_val.bin",
        "fineweb": val_dir / "fineweb_edu_val.bin",
    }

    conn = open_db(args.db)
    row = load_config_row(conn, args.config_id)
    config_id = args.config_id

    total_steps = args.max_steps if args.max_steps else TOTAL_STEPS
    # For test runs, fire Tier 2 eval at the final step
    eval_steps = set(EVAL_STEPS) if not args.max_steps else {total_steps}

    try:
        mark_running(conn, config_id)

        # --- Build config and model ---
        cfg = CARTConfig(
            d_model=row["d_model"],
            n_loops=row["n_loops"],
            n_prelude=row["n_prelude"],
        )
        cfg.validate()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Config: d={cfg.d_model} R={cfg.n_loops} P={cfg.n_prelude}  "
              f"device={device}")

        torch.manual_seed(row["seed"])
        model = CART(cfg).to(device)
        if cfg.d_model >= 768:
            model.enable_gradient_checkpointing()

        param_counts = model.count_parameters()
        print(f"Params: total={param_counts['total']:,}  "
              f"effective={param_counts['effective']:,}  "
              f"leverage={param_counts['leverage']:.2f}x")

        # --- Optimizer ---
        try:
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(
                model.parameters(),
                lr=PEAK_LR,
                betas=(0.9, 0.95),
                weight_decay=WEIGHT_DECAY,
            )
            print("Optimizer: AdamW8bit (bitsandbytes)")
        except ImportError:
            print("WARNING: bitsandbytes not available, falling back to torch AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=PEAK_LR,
                betas=(0.9, 0.95),
                weight_decay=WEIGHT_DECAY,
            )

        # --- Data ---
        if not train_bin.exists():
            raise FileNotFoundError(
                f"Training data not found: {train_bin}\n"
                "Run: python data/tokenize.py --output-dir data/"
            )
        train_ds = FixedOrderDataset(train_bin, SEQ_LEN)
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,   # Fixed order — required for sweep comparability
            num_workers=0,   # Windows: no multiprocessing in DataLoader
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        train_iter = iter(train_loader)

        def next_batch():
            nonlocal train_iter
            try:
                return next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                return next(train_iter)

        # --- Mixed precision ---
        use_amp = (device.type == "cuda")
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp else torch.autocast(device_type="cpu", enabled=False)
        )

        # --- Training loop ---
        train_start = time.perf_counter()
        step_times: deque = deque(maxlen=100)
        n_tokens_seen = 0
        last_loss = float("nan")
        last_grad_norm = float("nan")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for step in range(1, total_steps + 1):
            # Set learning rate
            lr = get_lr(step, WARMUP_STEPS, total_steps, PEAK_LR, MIN_LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation
            optimizer.zero_grad()
            accum_loss = 0.0

            for _ in range(GRAD_ACCUM):
                x, y = next_batch()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with amp_ctx:
                    _, loss = model(x, y)
                    loss = loss / GRAD_ACCUM
                loss.backward()
                accum_loss += loss.item()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            batch_tokens = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN
            n_tokens_seen += batch_tokens
            step_times.append((time.perf_counter(), batch_tokens))
            last_loss = accum_loss
            last_grad_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)

            # --- Tier 1: lightweight log every LOG_INTERVAL steps ---
            if step % LOG_INTERVAL == 0:
                wall_sec = time.perf_counter() - train_start
                sr = model.lti.spectral_radius()
                write_train_log(
                    conn, config_id, step, last_loss, last_grad_norm, lr,
                    n_tokens_seen, wall_sec, sr,
                )
                tps = rolling_tps(step_times)
                print(f"step {step:4d}/{total_steps}  loss={last_loss:.4f}  "
                      f"grad={last_grad_norm:.3f}  lr={lr:.2e}  "
                      f"tps={tps:.0f}  rho={sr:.4f}")
                if sr > 0.99:
                    print(f"WARNING: spectral radius {sr:.4f} at step {step} "
                          f"— possible instability")

            # --- Tier 2: full eval at checkpoints ---
            if step in eval_steps:
                print(f"\n[Tier 2 eval @ step {step}]")
                ppl_tiny = eval_perplexity(model, val_files["tiny"],    SEQ_LEN, device)
                ppl_wiki = eval_perplexity(model, val_files["wiki"],    SEQ_LEN, device)
                ppl_edu  = eval_perplexity(model, val_files["fineweb"], SEQ_LEN, device)
                peak_vram = (torch.cuda.max_memory_allocated() / 1e9
                             if device.type == "cuda" else 0.0)
                tps  = rolling_tps(step_times)
                wall = time.perf_counter() - train_start
                write_results(
                    conn, config_id, step,
                    ppl_tiny, ppl_wiki, ppl_edu,
                    peak_vram, tps, model.lti.spectral_radius(), wall,
                )
                print(f"  ppl_tiny={ppl_tiny:.2f}  ppl_wiki={ppl_wiki:.2f}  "
                      f"ppl_edu={ppl_edu:.2f}  vram={peak_vram:.2f}GB  tps={tps:.0f}")
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()

        # --- Save final checkpoint ---
        ckpt_path = save_checkpoint(model, optimizer, total_steps, config_id)
        print(f"\nCheckpoint saved: {ckpt_path}")

        mark_complete(conn, config_id)
        wall_total = time.perf_counter() - train_start
        print(f"Done. Total time: {wall_total/60:.1f} min  "
              f"({wall_total/total_steps:.2f} sec/step)")

    except Exception:
        tb = traceback.format_exc()
        print(f"FAILED:\n{tb}", file=sys.stderr)
        mark_failed(conn, config_id, tb)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
