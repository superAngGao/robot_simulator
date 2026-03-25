"""
Jacobi PGS contact solver — GPU-friendly parallel variant.

Identical to Gauss-Seidel PGS except that each iteration reads
from the *previous* iteration's lambda (not the partially-updated
current iteration).  This removes all data dependencies between
constraint rows, making every row updatable in parallel.

Trade-off: ~2x more iterations to converge (information propagates
one contact per iteration instead of immediately).  On GPU with
thousands of threads this is more than offset by parallelism.

A relaxation factor omega in (0, 1] dampens updates to improve
stability (omega=1.0 is standard Jacobi, 0.5-0.8 is typical).

References:
  Erleben (2007) — Velocity-based shock propagation (Jacobi variant)
  MuJoCo MJX — uses Jacobi PGS on GPU with ~100 iterations
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..spatial import Vec3, Vec6
from .pgs_solver import (
    ContactConstraint,
    _build_contact_frame,
    _compute_angular_jacobian_row,
    _compute_linear_jacobian_row,
    _row_to_contact,
)


class JacobiPGSContactSolver:
    """Jacobi (parallel) PGS solver — same interface as PGSContactSolver.

    Args:
        max_iter : Maximum iterations (typically 2x of serial PGS).
        tolerance: Convergence tolerance on impulse change.
        omega    : Relaxation factor in (0, 1]. Lower = more stable, slower.
        erp      : Error Reduction Parameter (Baumgarte).
        cfm      : Constraint Force Mixing (regularization).
    """

    def __init__(
        self,
        max_iter: int = 60,
        tolerance: float = 1e-6,
        omega: float = 0.7,
        erp: float = 0.2,
        cfm: float = 1e-6,
    ) -> None:
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.omega = omega
        self.erp = erp
        self.cfm = cfm
        self._warm_cache: dict[tuple[int, int], list[tuple[Vec3, NDArray]]] = {}

    def solve(
        self,
        contacts: list[ContactConstraint],
        body_v: list[Vec6],
        body_X_world: list,
        inv_mass: list[float],
        inv_inertia: list[NDArray],
        dt: float,
    ) -> list[Vec6]:
        """Solve contact constraints and return spatial impulses per body."""
        nc = len(contacts)
        num_bodies = len(body_v)
        if nc == 0:
            self._warm_cache.clear()
            return [np.zeros(6) for _ in range(num_bodies)]

        # Build contact frames
        for c in contacts:
            c.tangent1, c.tangent2 = _build_contact_frame(c.normal)

        # ── Row offsets ──
        row_offsets = []
        offset = 0
        for c in contacts:
            row_offsets.append(offset)
            offset += c.condim
        n_rows = offset

        # ── Build Jacobian ──
        J_body_i = np.zeros((n_rows, 6))
        J_body_j = np.zeros((n_rows, 6))

        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            directions_linear = [c.normal]
            if c.condim >= 3:
                directions_linear.extend([c.tangent1, c.tangent2])
            for d_idx, direction in enumerate(directions_linear):
                row = base + d_idx
                if c.body_i >= 0:
                    J_body_i[row] = _compute_linear_jacobian_row(
                        direction,
                        c.point,
                        body_X_world[c.body_i].r,
                        body_X_world[c.body_i].R,
                    )
                if c.body_j >= 0:
                    J_body_j[row] = -_compute_linear_jacobian_row(
                        direction,
                        c.point,
                        body_X_world[c.body_j].r,
                        body_X_world[c.body_j].R,
                    )
            if c.condim >= 4:
                row = base + 3
                if c.body_i >= 0:
                    J_body_i[row] = _compute_angular_jacobian_row(c.normal, body_X_world[c.body_i].R)
                if c.body_j >= 0:
                    J_body_j[row] = -_compute_angular_jacobian_row(c.normal, body_X_world[c.body_j].R)
            if c.condim >= 6:
                for t_idx, tang in enumerate([c.tangent1, c.tangent2]):
                    row = base + 4 + t_idx
                    if c.body_i >= 0:
                        J_body_i[row] = _compute_angular_jacobian_row(tang, body_X_world[c.body_i].R)
                    if c.body_j >= 0:
                        J_body_j[row] = -_compute_angular_jacobian_row(tang, body_X_world[c.body_j].R)

        # ── Delassus W ──
        W = np.zeros((n_rows, n_rows))
        body_contact_rows: dict[int, list[int]] = {}
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            for bi in [c.body_i, c.body_j]:
                if bi < 0:
                    continue
                if bi not in body_contact_rows:
                    body_contact_rows[bi] = []
                body_contact_rows[bi].extend(range(base, base + c.condim))

        for bi, rows in body_contact_rows.items():
            m_inv = inv_mass[bi]
            I_inv = inv_inertia[bi]
            J_rows = np.zeros((len(rows), 6))
            for ri, r in enumerate(rows):
                ci = _row_to_contact(r, row_offsets, nc)
                c = contacts[ci]
                J_rows[ri] = J_body_i[r] if c.body_i == bi else J_body_j[r]
            for ri1, r1 in enumerate(rows):
                j1_lin = J_rows[ri1, :3]
                j1_ang = J_rows[ri1, 3:]
                Minv_j1 = np.concatenate([m_inv * j1_lin, I_inv @ j1_ang])
                for ri2, r2 in enumerate(rows):
                    W[r1, r2] += J_rows[ri2] @ Minv_j1

        for i in range(n_rows):
            W[i, i] += self.cfm

        # ── Free velocity ──
        v_free = np.zeros(n_rows)
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            v_contact = np.zeros(3)
            omega_contact = np.zeros(3)
            for bi, sign in [(c.body_i, 1.0), (c.body_j, -1.0)]:
                if bi < 0:
                    continue
                X = body_X_world[bi]
                v_b = body_v[bi]
                v_lin_w = X.R @ v_b[:3]
                omega_w = X.R @ v_b[3:]
                r = c.point - X.r
                v_contact += sign * (v_lin_w + np.cross(omega_w, r))
                omega_contact += sign * omega_w
            v_free[base] = np.dot(v_contact, c.normal)
            if c.condim >= 3:
                v_free[base + 1] = np.dot(v_contact, c.tangent1)
                v_free[base + 2] = np.dot(v_contact, c.tangent2)
            if c.condim >= 4:
                v_free[base + 3] = np.dot(omega_contact, c.normal)
            if c.condim >= 6:
                v_free[base + 4] = np.dot(omega_contact, c.tangent1)
                v_free[base + 5] = np.dot(omega_contact, c.tangent2)

        # ── Bias ──
        bias = np.zeros(n_rows)
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            erp = c.erp if c.erp is not None else self.erp
            baumgarte = -erp / dt * c.depth
            restitution_bias = 0.0
            if c.restitution > 0.0 and v_free[base] < -0.01:
                restitution_bias = c.restitution * v_free[base]
            bias[base] = baumgarte + restitution_bias

        # ── Warm starting ──
        lambdas = np.zeros(n_rows)
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            key = (min(c.body_i, c.body_j), max(c.body_i, c.body_j))
            if key in self._warm_cache:
                if c.body_i >= 0:
                    local_pt = body_X_world[c.body_i].R.T @ (c.point - body_X_world[c.body_i].r)
                else:
                    local_pt = c.point
                best_dist = 0.02
                best_lam = None
                for cached_local, cached_lam in self._warm_cache[key]:
                    dist = np.linalg.norm(local_pt - cached_local)
                    if dist < best_dist:
                        best_dist = dist
                        best_lam = cached_lam
                if best_lam is not None:
                    n_copy = min(len(best_lam), c.condim)
                    lambdas[base : base + n_copy] = best_lam[:n_copy]

        # ── Jacobi PGS iterations (all rows read from OLD, write to NEW) ──
        omega = self.omega
        W_diag = np.diag(W).copy()

        for iteration in range(self.max_iter):
            lambdas_old = lambdas.copy()  # snapshot for reads
            max_delta = 0.0

            # Precompute W @ lambdas_old (shared across all rows)
            Wl = W @ lambdas_old

            for ci, c in enumerate(contacts):
                base = row_offsets[ci]

                # Normal (row 0): lambda_n >= 0
                residual = v_free[base] + Wl[base] + bias[base]
                delta = -residual / W_diag[base]
                raw = lambdas_old[base] + omega * delta
                new_n = max(0.0, raw)
                lambdas[base] = new_n
                max_delta = max(max_delta, abs(new_n - lambdas_old[base]))

                if c.condim >= 3:
                    limit_t = c.mu * new_n
                    for row_off in (1, 2):
                        idx = base + row_off
                        res = v_free[idx] + Wl[idx] + bias[idx]
                        d = -res / W_diag[idx]
                        raw_val = lambdas_old[idx] + omega * d
                        new_val = np.clip(raw_val, -limit_t, limit_t)
                        lambdas[idx] = new_val
                        max_delta = max(max_delta, abs(new_val - lambdas_old[idx]))

                if c.condim >= 4:
                    idx = base + 3
                    limit_s = c.mu_spin * new_n
                    res = v_free[idx] + Wl[idx] + bias[idx]
                    d = -res / W_diag[idx]
                    raw_val = lambdas_old[idx] + omega * d
                    new_val = np.clip(raw_val, -limit_s, limit_s)
                    lambdas[idx] = new_val
                    max_delta = max(max_delta, abs(new_val - lambdas_old[idx]))

                if c.condim >= 6:
                    limit_r = c.mu_roll * new_n
                    for row_off in (4, 5):
                        idx = base + row_off
                        res = v_free[idx] + Wl[idx] + bias[idx]
                        d = -res / W_diag[idx]
                        raw_val = lambdas_old[idx] + omega * d
                        new_val = np.clip(raw_val, -limit_r, limit_r)
                        lambdas[idx] = new_val
                        max_delta = max(max_delta, abs(new_val - lambdas_old[idx]))

            if max_delta < self.tolerance:
                break

        # ── Update warm cache ──
        self._warm_cache.clear()
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            key = (min(c.body_i, c.body_j), max(c.body_i, c.body_j))
            lam = lambdas[base : base + c.condim].copy()
            if c.body_i >= 0:
                local_pt = body_X_world[c.body_i].R.T @ (c.point - body_X_world[c.body_i].r)
            else:
                local_pt = c.point.copy()
            if key not in self._warm_cache:
                self._warm_cache[key] = []
            self._warm_cache[key].append((local_pt.copy(), lam))

        # ── Impulses to body forces ──
        body_impulses = [np.zeros(6) for _ in range(num_bodies)]
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            J_world_linear = lambdas[base] * c.normal
            if c.condim >= 3:
                J_world_linear += lambdas[base + 1] * c.tangent1
                J_world_linear += lambdas[base + 2] * c.tangent2
            J_world_angular = np.zeros(3)
            if c.condim >= 4:
                J_world_angular += lambdas[base + 3] * c.normal
            if c.condim >= 6:
                J_world_angular += lambdas[base + 4] * c.tangent1
                J_world_angular += lambdas[base + 5] * c.tangent2
            for bi, sign in [(c.body_i, 1.0), (c.body_j, -1.0)]:
                if bi < 0:
                    continue
                X = body_X_world[bi]
                f_world = sign * J_world_linear
                r = c.point - X.r
                torque_world = np.cross(r, f_world) + sign * J_world_angular
                body_impulses[bi][:3] += X.R.T @ f_world
                body_impulses[bi][3:] += X.R.T @ torque_world

        return body_impulses
