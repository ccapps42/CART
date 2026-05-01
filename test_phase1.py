"""
Phase 1 verification: forward pass, parameter counts, spectral radius, LIE distinctness.
"""
import sys
import torch

sys.path.insert(0, "K:/projects/Model_Paper_1")

from model.config import CARTConfig
from model.cart import CART

def main():
    print("=== Phase 1 Verification ===\n")

    config = CARTConfig(d_model=256, n_loops=2, n_prelude=2)
    config.validate()
    print(f"Config: d_model={config.d_model}, n_loops={config.n_loops}, "
          f"n_prelude={config.n_prelude}, n_heads={config.n_heads}, "
          f"d_kv_latent={config.d_kv_latent}, ffn_intermediate={config.ffn_intermediate}")

    model = CART(config)
    model.eval()

    # --- Forward pass ---
    B, T = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    targets   = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        logits, loss = model(input_ids, targets)

    assert logits.shape == (B, T, config.vocab_size), \
        f"Expected logits shape {(B, T, config.vocab_size)}, got {logits.shape}"
    assert loss is not None and loss.item() > 0, \
        f"Expected positive loss, got {loss.item()}"
    print(f"\nForward pass OK")
    print(f"  logits shape : {tuple(logits.shape)}")
    print(f"  loss         : {loss.item():.4f}  (expected ~{torch.log(torch.tensor(config.vocab_size)).item():.2f})")

    # --- Parameter counts ---
    params = model.count_parameters()
    print(f"\nParameter counts:")
    for k, v in params.items():
        print(f"  {k:15s}: {v:,.0f}")

    # --- Spectral radius ---
    rho = model.lti.spectral_radius()
    assert 0.88 <= rho <= 0.92, f"Expected spectral radius ~0.9, got {rho}"
    print(f"\nLTI spectral radius: {rho:.6f}  (expected ~0.9)")

    # --- LIE distinctness ---
    h = torch.randn(1, 8, config.d_model)
    with torch.no_grad():
        h0 = model.lie(h, 0)
        h1 = model.lie(h, 1)
    diff = (h0 - h1).abs().max().item()
    assert diff > 1e-4, f"LIE r=0 and r=1 should produce different outputs, max diff={diff}"
    print(f"\nLIE distinctness OK: max |h(r=0) - h(r=1)| = {diff:.6f}")

    # --- Gradient flow ---
    model.train()
    logits, loss = model(input_ids, targets)
    loss.backward()
    assert model.hyper.weights.grad is not None, "hyper.weights.grad is None — gradient not flowing"
    print(f"\nGradient flow OK: hyper.weights.grad = {model.hyper.weights.grad}")

    print("\n=== Phase 1 PASSED ===")

if __name__ == "__main__":
    main()
