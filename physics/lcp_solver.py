"""
LCP contact solver using Projected Gauss-Seidel (PGS).

Solves the contact Linear Complementarity Problem:
  0 <= lambda_n  perp  v_n >= 0       (Signorini: no penetration)
  |lambda_t| <= mu * lambda_n          (Coulomb friction cone)

The velocity-level formulation:
  v_contact = v_free + W @ lambda
  where:
    v_free  = contact velocity without constraint forces
    W       = J @ M^{-1} @ J^T   (full Delassus operator)
    lambda  = constraint impulses [normal; tangent_x; tangent_y] per contact

PGS iterates over the full W matrix (not diagonal approximation):
  for each constraint row i:
    residual_i = v_free_i + sum_j(W_ij * lambda_j) + bias_i
    delta = -residual_i / W_ii
    lambda_i += delta
    lambda_i = project(lambda_i)  # Signorini / friction cone

Warm starting: reuse lambda from previous step as initial guess,
matched by contact ID (body pair + proximity).

References:
  Catto (2005) — Iterative Dynamics with Temporal Coherence (GDC)
  Erleben (2007) — Velocity-based shock propagation
  Bullet: btSequentialImpulseConstraintSolver (full row updates)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .gjk_epa import ContactManifold
from .spatial import Vec3, Vec6


@dataclass
class ContactConstraint:
    """One contact point ready for LCP solving."""

    body_i: int  # body index (-1 for ground)
    body_j: int  # body index (-1 for ground)
    point: Vec3  # world position
    normal: Vec3  # from j to i (unit)
    tangent1: Vec3  # friction direction 1
    tangent2: Vec3  # friction direction 2
    depth: float  # penetration (positive)
    mu: float  # friction coefficient
    restitution: float = 0.0  # coefficient of restitution [0,1]


def _build_contact_frame(normal: Vec3) -> tuple[Vec3, Vec3]:
    """Build orthonormal tangent vectors for a contact normal."""
    n = normal / np.linalg.norm(normal)
    if abs(n[0]) < 0.9:
        t1 = np.cross(n, np.array([1, 0, 0]))
    else:
        t1 = np.cross(n, np.array([0, 1, 0]))
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


def _compute_contact_jacobian_row(
    direction: Vec3,
    point: Vec3,
    body_origin: Vec3,
    body_R: NDArray,
) -> Vec6:
    """Compute one row of the contact Jacobian for one body.

    Maps body spatial velocity [lin; ang] (body frame) to
    contact velocity along `direction` (world frame).

    Returns J_row (6,) such that v_contact_dir = J_row @ v_body.
    """
    # v_point_world = R @ v_lin_body + (R @ omega_body) × r_world
    # dot(d, v_point) = d^T R v_lin + d^T (R omega × r)
    #                 = d^T R v_lin + (r × d)^T R omega
    #                 = (R^T d)^T v_lin + (R^T (r × d))^T omega
    r_world = point - body_origin
    rxd = np.cross(r_world, direction)
    J_lin = body_R.T @ direction  # (3,) in body frame
    J_ang = body_R.T @ rxd  # (3,) in body frame
    return np.concatenate([J_lin, J_ang])


class PGSContactSolver:
    """Projected Gauss-Seidel LCP solver with full Delassus matrix.

    Args:
        max_iter : Maximum PGS iterations.
        tolerance: Convergence tolerance on impulse change.
        erp      : Error Reduction Parameter (Baumgarte stabilization).
        cfm      : Constraint Force Mixing (regularization).
    """

    def __init__(
        self,
        max_iter: int = 30,
        tolerance: float = 1e-6,
        erp: float = 0.2,
        cfm: float = 1e-6,
    ) -> None:
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.erp = erp
        self.cfm = cfm
        # Warm starting cache: (body_i, body_j) → list of (point, lambda_3)
        self._warm_cache: dict[tuple[int, int], list[tuple[Vec3, NDArray]]] = {}

    def solve(
        self,
        contacts: list[ContactConstraint],
        body_v: list[Vec6],  # spatial velocities in body frame
        body_X_world: list,  # SpatialTransform per body
        inv_mass: list[float],  # 1/mass per body (0 for ground)
        inv_inertia: list[NDArray],  # 3x3 inverse inertia per body (zeros for ground)
        dt: float,
    ) -> list[Vec6]:
        """Solve contact constraints and return spatial impulses per body.

        Returns: list of Vec6 impulses in body frame.
        """
        nc = len(contacts)
        num_bodies = len(body_v)
        if nc == 0:
            self._warm_cache.clear()
            return [np.zeros(6) for _ in range(num_bodies)]

        # Build contact frames
        for c in contacts:
            c.tangent1, c.tangent2 = _build_contact_frame(c.normal)

        n_rows = nc * 3  # 3 constraint rows per contact (normal + 2 tangent)

        # ── Build full Jacobian J (n_rows × 6*num_bodies) ──
        # Stored as sparse: for each row, we store J_i and J_j (6-vectors for body_i and body_j)
        J_body_i = np.zeros((n_rows, 6))  # Jacobian for body_i
        J_body_j = np.zeros((n_rows, 6))  # Jacobian for body_j

        for ci, c in enumerate(contacts):
            directions = [c.normal, c.tangent1, c.tangent2]
            for d_idx, direction in enumerate(directions):
                row = ci * 3 + d_idx
                if c.body_i >= 0:
                    J_body_i[row] = _compute_contact_jacobian_row(
                        direction, c.point, body_X_world[c.body_i].r,
                        body_X_world[c.body_i].R,
                    )
                if c.body_j >= 0:
                    J_body_j[row] = -_compute_contact_jacobian_row(
                        direction, c.point, body_X_world[c.body_j].r,
                        body_X_world[c.body_j].R,
                    )

        # ── Build full Delassus operator W = J M⁻¹ Jᵀ (n_rows × n_rows) ──
        # W[r1, r2] = J_i[r1]^T @ M_i^{-1} @ J_i[r2] + J_j[r1]^T @ M_j^{-1} @ J_j[r2]
        # where M^{-1} = diag(1/m * I3, I_inv) for each body
        W = np.zeros((n_rows, n_rows))

        for ci, c in enumerate(contacts):
            for bi, J_b in [(c.body_i, J_body_i), (c.body_j, J_body_j)]:
                if bi < 0:
                    continue
                # M_inv for body bi: [inv_mass * I3, 0; 0, inv_inertia]
                m_inv = inv_mass[bi]
                I_inv = inv_inertia[bi]

                for r1 in range(ci * 3, ci * 3 + 3):
                    j1_lin = J_b[r1, :3]
                    j1_ang = J_b[r1, 3:]
                    # M_inv @ J[r1] = [m_inv * j1_lin; I_inv @ j1_ang]
                    Minv_j1_lin = m_inv * j1_lin
                    Minv_j1_ang = I_inv @ j1_ang

                    for r2 in range(n_rows):
                        j2 = J_b[r2]
                        W[r1, r2] += j2[:3] @ Minv_j1_lin + j2[3:] @ Minv_j1_ang

        # Add CFM to diagonal (regularization)
        for i in range(n_rows):
            W[i, i] += self.cfm

        # ── Compute free velocity at contacts ──
        v_free = np.zeros(n_rows)
        for ci, c in enumerate(contacts):
            v_contact = np.zeros(3)
            for bi, sign in [(c.body_i, 1.0), (c.body_j, -1.0)]:
                if bi < 0:
                    continue
                X = body_X_world[bi]
                v_b = body_v[bi]
                v_lin_w = X.R @ v_b[:3]
                omega_w = X.R @ v_b[3:]
                r = c.point - X.r
                v_point = v_lin_w + np.cross(omega_w, r)
                v_contact += sign * v_point

            v_free[ci * 3 + 0] = np.dot(v_contact, c.normal)
            v_free[ci * 3 + 1] = np.dot(v_contact, c.tangent1)
            v_free[ci * 3 + 2] = np.dot(v_contact, c.tangent2)

        # Baumgarte bias + restitution
        # PGS solves: v_free + W*lambda + bias >= 0, lambda >= 0
        # Baumgarte: bias = -erp/dt * depth (negative, pushes velocity positive)
        # Restitution: target v_n = -e * v_incoming → bias -= e * v_free_n
        #   (v_free_n < 0 for incoming, so -e*v_free_n > 0, making bias more negative
        #    which drives lambda larger)
        bias = np.zeros(n_rows)
        for ci, c in enumerate(contacts):
            baumgarte = -self.erp / dt * c.depth
            restitution_bias = 0.0
            if c.restitution > 0.0 and v_free[ci * 3] < -0.01:
                # Newton restitution: v_after = -e * v_before
                # bias -= e * v_free (v_free < 0, so this makes bias more negative)
                restitution_bias = c.restitution * v_free[ci * 3]  # negative
            bias[ci * 3] = baumgarte + restitution_bias

        # ── Warm starting: match by body-local coordinates (Bullet approach) ──
        lambdas = np.zeros(n_rows)
        for ci, c in enumerate(contacts):
            key = (min(c.body_i, c.body_j), max(c.body_i, c.body_j))
            if key in self._warm_cache:
                # Compute contact point in body_i local frame (or world for ground)
                if c.body_i >= 0:
                    local_pt = body_X_world[c.body_i].R.T @ (c.point - body_X_world[c.body_i].r)
                else:
                    local_pt = c.point  # ground: world coords are stable
                best_dist = 0.02  # match threshold in local coords [m]
                best_lam = None
                for cached_local, cached_lam in self._warm_cache[key]:
                    dist = np.linalg.norm(local_pt - cached_local)
                    if dist < best_dist:
                        best_dist = dist
                        best_lam = cached_lam
                if best_lam is not None:
                    lambdas[ci * 3: ci * 3 + 3] = best_lam

        # ── PGS iterations (full row updates) ──
        for iteration in range(self.max_iter):
            max_delta = 0.0

            for ci, c in enumerate(contacts):
                # Normal
                idx_n = ci * 3
                old_n = lambdas[idx_n]
                # residual = v_free + W[row, :] @ lambda + bias
                residual_n = v_free[idx_n] + W[idx_n] @ lambdas + bias[idx_n]
                delta_n = -residual_n / W[idx_n, idx_n]
                new_n = max(0.0, old_n + delta_n)
                lambdas[idx_n] = new_n
                max_delta = max(max_delta, abs(new_n - old_n))

                # Tangent 1
                idx_t1 = ci * 3 + 1
                old_t1 = lambdas[idx_t1]
                residual_t1 = v_free[idx_t1] + W[idx_t1] @ lambdas + bias[idx_t1]
                delta_t1 = -residual_t1 / W[idx_t1, idx_t1]
                max_f = c.mu * lambdas[idx_n]
                new_t1 = np.clip(old_t1 + delta_t1, -max_f, max_f)
                lambdas[idx_t1] = new_t1
                max_delta = max(max_delta, abs(new_t1 - old_t1))

                # Tangent 2
                idx_t2 = ci * 3 + 2
                old_t2 = lambdas[idx_t2]
                residual_t2 = v_free[idx_t2] + W[idx_t2] @ lambdas + bias[idx_t2]
                delta_t2 = -residual_t2 / W[idx_t2, idx_t2]
                new_t2 = np.clip(old_t2 + delta_t2, -max_f, max_f)
                lambdas[idx_t2] = new_t2
                max_delta = max(max_delta, abs(new_t2 - old_t2))

            if max_delta < self.tolerance:
                break

        # ── Update warm start cache (store in body-local coordinates) ──
        self._warm_cache.clear()
        for ci, c in enumerate(contacts):
            key = (min(c.body_i, c.body_j), max(c.body_i, c.body_j))
            lam = lambdas[ci * 3: ci * 3 + 3].copy()
            if c.body_i >= 0:
                local_pt = body_X_world[c.body_i].R.T @ (c.point - body_X_world[c.body_i].r)
            else:
                local_pt = c.point.copy()
            if key not in self._warm_cache:
                self._warm_cache[key] = []
            self._warm_cache[key].append((local_pt.copy(), lam))

        # ── Convert impulses to spatial forces per body ──
        body_impulses = [np.zeros(6) for _ in range(num_bodies)]

        for ci, c in enumerate(contacts):
            J_world = (
                lambdas[ci * 3] * c.normal
                + lambdas[ci * 3 + 1] * c.tangent1
                + lambdas[ci * 3 + 2] * c.tangent2
            )

            for bi, sign in [(c.body_i, 1.0), (c.body_j, -1.0)]:
                if bi < 0:
                    continue
                X = body_X_world[bi]
                f_world = sign * J_world
                r = c.point - X.r
                torque_world = np.cross(r, f_world)
                body_impulses[bi][:3] += X.R.T @ f_world
                body_impulses[bi][3:] += X.R.T @ torque_world

        return body_impulses
