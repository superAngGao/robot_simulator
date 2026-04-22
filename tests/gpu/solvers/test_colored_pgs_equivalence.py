"""Kernel-level equivalence test: batched_colored_pgs_all_iters vs per-color loop.

Verifies that the fused kernel produces bit-identical lambdas to running
batched_colored_pgs_step in the original iter×color Python loop, using a
synthetic contact fixture with known multi-point manifold structure.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import warp as wp

    HAS_WARP = True
except ImportError:
    HAS_WARP = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_WARP, reason="Warp or CUDA not available"),
]

# Fixture constants
N = 2  # environments
NC = 6  # contacts: 3 body pairs × 2 points each (simulates box-ground manifold)
CONDIM = 3  # normal + 2 friction
MAX_ROWS = NC * CONDIM  # 18
MAX_ITER = 20
MU = 0.5


def _build_arrays(device):
    """Build synthetic arrays that exercise multi-point contact on same body pair.

    Contact layout (per env):
      c=0,1  → bodies (0,1)  — simulates 2-point manifold on body pair 0-1
      c=2,3  → bodies (1,2)  — simulates 2-point manifold on body pair 1-2
      c=4,5  → bodies (0,2)  — simulates 2-point manifold on body pair 0-2

    After greedy coloring:
      c=0 → color 0  (body 0,1 free)
      c=1 → color 1  (body 0,1 already used by c=0)
      c=2 → color 0  (body 1,2 not yet used at color 0 by this pair)
        ... actual colors depend on greedy scan order, but the key property is
        that contacts sharing a body get different colors.
    """
    rng = np.random.default_rng(42)

    # Diagonal-dominant W → guaranteed PGS convergence
    W_np = np.zeros((N, MAX_ROWS, MAX_ROWS), dtype=np.float32)
    for env in range(N):
        A = rng.standard_normal((MAX_ROWS, MAX_ROWS)).astype(np.float32)
        W_np[env] = A @ A.T + np.eye(MAX_ROWS, dtype=np.float32) * MAX_ROWS

    W_diag_np = np.array(
        [[W_np[env, i, i] for i in range(MAX_ROWS)] for env in range(N)],
        dtype=np.float32,
    )

    v_free_np = rng.standard_normal((N, MAX_ROWS)).astype(np.float32) * 0.1

    contact_active_np = np.ones((N, NC), dtype=np.int32)

    # Body indices — contacts share bodies to force multi-color assignment
    contact_bi_np = np.array([[0, 0, 1, 1, 0, 0]] * N, dtype=np.int32)
    contact_bj_np = np.array([[1, 1, 2, 2, 2, 2]] * N, dtype=np.int32)

    W = wp.array(W_np, dtype=wp.float32, device=device)
    W_diag = wp.array(W_diag_np, dtype=wp.float32, device=device)
    v_free = wp.array(v_free_np, dtype=wp.float32, device=device)
    contact_active = wp.array(contact_active_np, dtype=wp.int32, device=device)
    contact_bi = wp.array(contact_bi_np, dtype=wp.int32, device=device)
    contact_bj = wp.array(contact_bj_np, dtype=wp.int32, device=device)

    return W, W_diag, v_free, contact_active, contact_bi, contact_bj


class TestFusedKernelEquivalence:
    """Fused kernel must produce identical lambdas to the original per-color loop."""

    def test_lambda_parity_synthetic_manifold(self):
        """lambdas from fused kernel == original iter×color loop, atol=1e-5."""
        from physics.backends.warp.colored_pgs_kernels import (
            batched_colored_pgs_all_iters,
            batched_colored_pgs_step,
            batched_greedy_coloring,
        )

        device = "cuda"
        W, W_diag, v_free, contact_active, contact_bi, contact_bj = _build_arrays(device)

        contact_color = wp.zeros((N, NC), dtype=wp.int32, device=device)
        n_colors = wp.zeros(N, dtype=wp.int32, device=device)
        nb = 3  # bodies

        wp.launch(
            batched_greedy_coloring,
            dim=N,
            device=device,
            inputs=[contact_active, contact_bi, contact_bj, NC, nb],
            outputs=[contact_color, n_colors],
        )

        # --- Reference: original per-color loop ---
        lambdas_ref = wp.zeros((N, MAX_ROWS), dtype=wp.float32, device=device)
        n_colors_host = n_colors.numpy()
        max_colors = int(n_colors_host.max())

        for _ in range(MAX_ITER):
            for color in range(max_colors):
                wp.launch(
                    batched_colored_pgs_step,
                    dim=N,
                    device=device,
                    inputs=[
                        W,
                        W_diag,
                        v_free,
                        lambdas_ref,
                        contact_active,
                        contact_color,
                        MU,
                        color,
                        NC,
                        MAX_ROWS,
                    ],
                )

        # --- Test: fused kernel ---
        lambdas_fused = wp.zeros((N, MAX_ROWS), dtype=wp.float32, device=device)
        wp.launch(
            batched_colored_pgs_all_iters,
            dim=N,
            device=device,
            inputs=[
                W,
                W_diag,
                v_free,
                lambdas_fused,
                contact_active,
                contact_color,
                n_colors,
                MU,
                MAX_ITER,
                NC,
                MAX_ROWS,
            ],
        )

        ref = lambdas_ref.numpy()
        fused = lambdas_fused.numpy()

        assert np.allclose(ref, fused, atol=1e-5), (
            f"Max lambda diff: {np.abs(ref - fused).max():.2e}\n"
            f"ref[:6]:   {ref[0, :6]}\n"
            f"fused[:6]: {fused[0, :6]}"
        )

    def test_lambda_parity_zero_contacts(self):
        """Fused kernel is a no-op when no contacts are active."""
        from physics.backends.warp.colored_pgs_kernels import (
            batched_colored_pgs_all_iters,
            batched_greedy_coloring,
        )

        device = "cuda"
        W, W_diag, v_free, _, contact_bi, contact_bj = _build_arrays(device)

        contact_active = wp.zeros((N, NC), dtype=wp.int32, device=device)  # all inactive
        contact_color = wp.zeros((N, NC), dtype=wp.int32, device=device)
        n_colors = wp.zeros(N, dtype=wp.int32, device=device)

        wp.launch(
            batched_greedy_coloring,
            dim=N,
            device=device,
            inputs=[contact_active, contact_bi, contact_bj, NC, 3],
            outputs=[contact_color, n_colors],
        )

        lambdas = wp.zeros((N, MAX_ROWS), dtype=wp.float32, device=device)
        wp.launch(
            batched_colored_pgs_all_iters,
            dim=N,
            device=device,
            inputs=[
                W,
                W_diag,
                v_free,
                lambdas,
                contact_active,
                contact_color,
                n_colors,
                MU,
                MAX_ITER,
                NC,
                MAX_ROWS,
            ],
        )

        assert np.allclose(lambdas.numpy(), 0.0), "Inactive contacts must leave lambdas at zero"

    def test_normal_lambdas_nonnegative(self):
        """Normal impulse rows must satisfy λ_n ≥ 0 (non-penetration cone)."""
        from physics.backends.warp.colored_pgs_kernels import (
            batched_colored_pgs_all_iters,
            batched_greedy_coloring,
        )

        device = "cuda"
        W, W_diag, v_free, contact_active, contact_bi, contact_bj = _build_arrays(device)
        contact_color = wp.zeros((N, NC), dtype=wp.int32, device=device)
        n_colors = wp.zeros(N, dtype=wp.int32, device=device)

        wp.launch(
            batched_greedy_coloring,
            dim=N,
            device=device,
            inputs=[contact_active, contact_bi, contact_bj, NC, 3],
            outputs=[contact_color, n_colors],
        )

        lambdas = wp.zeros((N, MAX_ROWS), dtype=wp.float32, device=device)
        wp.launch(
            batched_colored_pgs_all_iters,
            dim=N,
            device=device,
            inputs=[
                W,
                W_diag,
                v_free,
                lambdas,
                contact_active,
                contact_color,
                n_colors,
                MU,
                MAX_ITER,
                NC,
                MAX_ROWS,
            ],
        )

        lam = lambdas.numpy()
        normal_rows = lam[:, ::CONDIM]  # rows 0, 3, 6, ... are normal rows
        assert (normal_rows >= -1e-7).all(), f"Normal lambdas must be ≥ 0, got min={normal_rows.min():.2e}"
