"""Compare from-scratch implementations against library reference implementations."""

import numpy as np


def _to_numpy(x):
    """Convert tensors, arrays, lists and scalars to a float64 NumPy array."""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def check_close(name, ours, reference, atol=1e-6, rtol=1e-5):
    """Assert that `ours` matches `reference` element-wise and print the result.

    Uses the same criterion as `np.allclose`: |ours - reference| <= atol + rtol * |reference|.
    Raises AssertionError on mismatch, so a broken implementation stops the notebook
    instead of silently producing a wrong plot.
    """
    a, b = _to_numpy(ours), _to_numpy(reference)
    if a.shape != b.shape:
        # Allow harmless shape differences such as (n, 1) vs (n,)
        if a.size == b.size:
            a, b = a.reshape(-1), b.reshape(-1)
        else:
            raise AssertionError(f"✗ {name}: shape mismatch {a.shape} vs {b.shape}")

    max_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(
            f"✗ {name}: max |diff| = {max_diff:.3e} exceeds tolerance (atol={atol}, rtol={rtol})"
        )
    print(f"✓ {name}: matches (max |diff| = {max_diff:.2e})")


def check_agreement(name, ours, reference, min_agreement=0.95):
    """Assert that two label vectors agree on at least `min_agreement` of samples.

    Useful for iterative methods (gradient descent, sampling) that should reach the
    same decisions as the library without matching its parameters exactly.
    """
    a, b = _to_numpy(ours).reshape(-1), _to_numpy(reference).reshape(-1)
    if a.shape != b.shape:
        raise AssertionError(f"✗ {name}: shape mismatch {a.shape} vs {b.shape}")

    agreement = float(np.mean(a == b))
    if agreement < min_agreement:
        raise AssertionError(
            f"✗ {name}: only {agreement:.1%} agreement (expected ≥ {min_agreement:.0%})"
        )
    print(f"✓ {name}: {agreement:.1%} agreement")
