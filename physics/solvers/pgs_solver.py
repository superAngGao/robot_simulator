"""
LCP contact solver using Projected Gauss-Seidel (PGS).

Solves the contact Linear Complementarity Problem with variable contact
dimensions (MuJoCo condim semantics):

  condim=1: normal only (frictionless)
  condim=3: normal + 2 tangent (Coulomb sliding friction)        [default]
  condim=4: normal + 2 tangent + 1 spin (torsional friction)
  condim=6: normal + 2 tangent + 1 spin + 2 rolling friction

Constraint rows per contact:
  - Normal:   v_n >= 0, lambda_n >= 0                (Signorini)
  - Tangent:  |lambda_t| <= mu * lambda_n            (Coulomb cone)
  - Spin:     |lambda_s| <= mu_spin * lambda_n
  - Rolling:  |lambda_r| <= mu_roll * lambda_n

Sliding Jacobian (normal/tangent): maps spatial velocity to linear
contact velocity along a direction d:
  v_d = d^T (v_lin_world + omega_world x r)

Angular Jacobian (spin/rolling): maps spatial velocity to angular
contact velocity along a direction d:
  omega_d = d^T omega_world

References:
  Catto (2005) — Iterative Dynamics with Temporal Coherence (GDC)
  Erleben (2007) — Velocity-based shock propagation
  MuJoCo docs — Contact parameters (condim, friction triple)
  Bullet: btSequentialImpulseConstraintSolver (full row updates)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..spatial import Vec3, Vec6

CONDIM_VALID = (1, 3, 4, 6)


@dataclass
class ContactConstraint:
    """One contact point ready for LCP solving.

    Attributes:
        condim: Contact dimension (1, 3, 4, or 6).
            1 = normal only (frictionless)
            3 = normal + 2 tangent (default)
            4 = normal + 2 tangent + torsional spin
            6 = normal + 2 tangent + torsional spin + 2 rolling
        mu:       Sliding (tangent) friction coefficient.
        mu_spin:  Torsional friction coefficient (condim >= 4).
        mu_roll:  Rolling friction coefficient (condim == 6).
    """

    body_i: int  # body index (-1 for ground)
    body_j: int  # body index (-1 for ground)
    point: Vec3  # world position
    normal: Vec3  # from j to i (unit)
    tangent1: Vec3  # friction direction 1 (set by solver)
    tangent2: Vec3  # friction direction 2 (set by solver)
    depth: float  # penetration (positive)
    mu: float  # sliding friction coefficient
    condim: int = 3
    mu_spin: float = 0.0  # torsional friction
    mu_roll: float = 0.0  # rolling friction
    restitution: float = 0.0  # coefficient of restitution [0,1]
    erp: float | None = None  # per-contact ERP override (None = use solver default)
    slop: float | None = None  # per-contact slop override (None = use solver default)


def _build_contact_frame(normal: Vec3) -> tuple[Vec3, Vec3]:
    """Build orthonormal tangent vectors for a contact normal."""
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        # Degenerate: fallback to z-up
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = normal / norm
    if abs(n[0]) < 0.9:
        t1 = np.cross(n, np.array([1, 0, 0]))
    else:
        t1 = np.cross(n, np.array([0, 1, 0]))
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


def _compute_linear_jacobian_row(
    direction: Vec3,
    point: Vec3,
    body_origin: Vec3,
    body_R: NDArray,
) -> Vec6:
    """Jacobian row mapping body spatial velocity to linear contact velocity.

    v_contact_dir = direction^T (v_lin_world + omega_world x r)

    Returns J_row (6,) in body frame: v_contact_dir = J_row @ v_body.
    """
    r_world = point - body_origin
    rxd = np.cross(r_world, direction)
    J_lin = body_R.T @ direction
    J_ang = body_R.T @ rxd
    return np.concatenate([J_lin, J_ang])


def _compute_angular_jacobian_row(
    direction: Vec3,
    body_R: NDArray,
) -> Vec6:
    """Jacobian row mapping body spatial velocity to angular contact velocity.

    omega_contact_dir = direction^T (R @ omega_body)

    Used for spin (direction=normal) and rolling (direction=tangent).
    No moment arm contribution — pure angular.

    Returns J_row (6,) in body frame.
    """
    J_lin = np.zeros(3)
    J_ang = body_R.T @ direction
    return np.concatenate([J_lin, J_ang])


class PGSContactSolver:
    """Projected Gauss-Seidel LCP solver with variable condim support.

    Supports condim 1/3/4/6 per contact. Constraint rows per contact
    vary: 1, 3, 4, or 6 rows respectively.

    Args:
        max_iter : Maximum PGS iterations.
        tolerance: Convergence tolerance on impulse change.
        erp      : Error Reduction Parameter (Baumgarte stabilization).
        cfm      : Constraint Force Mixing (regularization on normal rows).
        solimp   : (d_0, d_width, width, midpoint, power) impedance params.
                    Controls per-row R regularization on friction rows:
                    R_i = (1-d)/d * |W_ii|.  Prevents float32 noise from
                    producing spurious friction forces (Q25 fix).
        friction_warmstart: If False, zero friction lambdas each frame
                    instead of warmstarting (Bullet-style, prevents
                    cross-frame noise accumulation).
        max_depenetration_vel: Upper bound on the Baumgarte velocity bias
                    magnitude [m/s] (PhysX ``maxDepenetrationVelocity`` style).
                    Because position correction is folded into the velocity
                    solve, the bias is also the resulting post-solve velocity,
                    so the default is conservative (1 m/s). Prevents deep
                    initial penetration from ejecting the body.
    """

    def __init__(
        self,
        max_iter: int = 30,
        tolerance: float = 1e-6,
        erp: float = 0.2,
        cfm: float = 1e-6,
        slop: float = 0.0,
        solimp: tuple[float, ...] = (0.95, 0.99, 0.001, 0.5, 2.0),
        friction_warmstart: bool = False,
        max_depenetration_vel: float = 1.0,
    ) -> None:
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.erp = erp
        self.cfm = cfm
        self.slop = slop
        self.solimp = solimp
        self.friction_warmstart = friction_warmstart
        self.max_depenetration_vel = max_depenetration_vel
        self._warm_cache: dict[tuple[int, int], list[tuple[Vec3, NDArray]]] = {}

    def _impedance(self, depth: float) -> float:
        """Compute impedance d(r) from solimp parameters.

        Piecewise power-law sigmoid — same as ADMMQPSolver._impedance().
        """
        d_0, d_width, width, midpoint, power = self.solimp
        if width < 1e-10:
            return d_0
        x = min(1.0, max(0.0, abs(depth) / width))
        if x <= 0:
            return d_0
        if x >= 1:
            return d_width
        if power <= 1:
            y = x
        elif x <= midpoint:
            a = 1.0 / max(midpoint ** (power - 1), 1e-10)
            y = a * x**power
        else:
            b_coeff = 1.0 / max((1.0 - midpoint) ** (power - 1), 1e-10)
            y = 1.0 - b_coeff * (1.0 - x) ** power
        return d_0 + y * (d_width - d_0)

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

        # ── Row offset mapping (variable rows per contact) ──
        row_offsets = []  # row_offsets[ci] = start row for contact ci
        offset = 0
        for c in contacts:
            row_offsets.append(offset)
            offset += c.condim
        n_rows = offset

        # ── Build Jacobian rows ──
        # For each row, store which direction and whether it's linear or angular
        J_body_i = np.zeros((n_rows, 6))
        J_body_j = np.zeros((n_rows, 6))

        for ci, c in enumerate(contacts):
            base = row_offsets[ci]

            # Row 0: normal (linear)
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

            # Row 3: spin about normal (angular) — condim >= 4
            if c.condim >= 4:
                row = base + 3
                if c.body_i >= 0:
                    J_body_i[row] = _compute_angular_jacobian_row(
                        c.normal,
                        body_X_world[c.body_i].R,
                    )
                if c.body_j >= 0:
                    J_body_j[row] = -_compute_angular_jacobian_row(
                        c.normal,
                        body_X_world[c.body_j].R,
                    )

            # Rows 4-5: rolling about tangent1, tangent2 (angular) — condim == 6
            if c.condim >= 6:
                for t_idx, tang in enumerate([c.tangent1, c.tangent2]):
                    row = base + 4 + t_idx
                    if c.body_i >= 0:
                        J_body_i[row] = _compute_angular_jacobian_row(
                            tang,
                            body_X_world[c.body_i].R,
                        )
                    if c.body_j >= 0:
                        J_body_j[row] = -_compute_angular_jacobian_row(
                            tang,
                            body_X_world[c.body_j].R,
                        )

        # ── Build full Delassus W = J M⁻¹ Jᵀ ──
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
                # Find which contact this row belongs to
                ci = _row_to_contact(r, row_offsets, nc)
                c = contacts[ci]
                if c.body_i == bi:
                    J_rows[ri] = J_body_i[r]
                else:
                    J_rows[ri] = J_body_j[r]

            for ri1, r1 in enumerate(rows):
                j1_lin = J_rows[ri1, :3]
                j1_ang = J_rows[ri1, 3:]
                Minv_j1 = np.concatenate([m_inv * j1_lin, I_inv @ j1_ang])
                for ri2, r2 in enumerate(rows):
                    W[r1, r2] += J_rows[ri2] @ Minv_j1

        # Per-row regularization: normal rows get uniform cfm,
        # friction rows get self-adaptive R = (1-d)/d * |W_ii| (Q25 fix).
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            # Normal row: uniform cfm
            W[base, base] += self.cfm
            if c.condim >= 3:
                d = self._impedance(c.depth)
                ratio = (1.0 - d) / max(d, 1e-10)
                for j in range(1, c.condim):
                    W[base + j, base + j] += ratio * abs(W[base + j, base + j])

        # ── Free velocity ──
        v_free = np.zeros(n_rows)
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            # Relative point velocity (world frame)
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

            # Linear rows
            v_free[base] = np.dot(v_contact, c.normal)
            if c.condim >= 3:
                v_free[base + 1] = np.dot(v_contact, c.tangent1)
                v_free[base + 2] = np.dot(v_contact, c.tangent2)
            # Spin row
            if c.condim >= 4:
                v_free[base + 3] = np.dot(omega_contact, c.normal)
            # Rolling rows
            if c.condim >= 6:
                v_free[base + 4] = np.dot(omega_contact, c.tangent1)
                v_free[base + 5] = np.dot(omega_contact, c.tangent2)

        # ── Baumgarte bias + restitution (normal row only) ──
        bias = np.zeros(n_rows)
        v_max = self.max_depenetration_vel
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            erp = c.erp if c.erp is not None else self.erp
            slop = c.slop if c.slop is not None else self.slop
            # Position correction: v_ref = depth / τ (MuJoCo QP style).
            # erp is reinterpreted as 1/τ when > 1, or as classic erp/dt when ≤ 1.
            penetration = max(c.depth - slop, 0.0)
            if erp > 1.0:
                baumgarte = -erp * penetration  # 1/τ * (depth - slop)
            else:
                baumgarte = -erp / dt * penetration  # legacy Baumgarte
            # Clamp depenetration velocity magnitude (Bullet/PhysX style).
            # Prevents deep-penetration ejection by bounding the recovery impulse.
            if baumgarte < -v_max:
                baumgarte = -v_max
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
                    # Normal row always warmstarted
                    lambdas[base] = best_lam[0]
                    # Friction rows: only warmstart if enabled (Q25 fix)
                    if self.friction_warmstart:
                        n_copy = min(len(best_lam), c.condim)
                        lambdas[base + 1 : base + n_copy] = best_lam[1:n_copy]

        # ── PGS iterations ──
        for iteration in range(self.max_iter):
            max_delta = 0.0

            for ci, c in enumerate(contacts):
                base = row_offsets[ci]

                # Normal (row 0): lambda_n >= 0
                old_n = lambdas[base]
                residual = v_free[base] + W[base] @ lambdas + bias[base]
                w_diag = W[base, base]
                if w_diag < 1e-12:
                    continue  # degenerate constraint, skip
                delta = -residual / w_diag
                new_n = max(0.0, old_n + delta)
                lambdas[base] = new_n
                max_delta = max(max_delta, abs(new_n - old_n))

                if c.condim >= 3:
                    # Tangent 1 (row 1): |lambda_t1| <= mu * lambda_n
                    _pgs_box_row(
                        base + 1, c.mu * new_n, lambdas, v_free, W, bias, max_delta_ref := [max_delta]
                    )
                    max_delta = max_delta_ref[0]

                    # Tangent 2 (row 2)
                    _pgs_box_row(base + 2, c.mu * new_n, lambdas, v_free, W, bias, max_delta_ref)
                    max_delta = max_delta_ref[0]

                if c.condim >= 4:
                    # Spin (row 3): |lambda_s| <= mu_spin * lambda_n
                    _pgs_box_row(base + 3, c.mu_spin * new_n, lambdas, v_free, W, bias, max_delta_ref)
                    max_delta = max_delta_ref[0]

                if c.condim >= 6:
                    # Rolling 1 (row 4): |lambda_r1| <= mu_roll * lambda_n
                    _pgs_box_row(base + 4, c.mu_roll * new_n, lambdas, v_free, W, bias, max_delta_ref)
                    max_delta = max_delta_ref[0]

                    # Rolling 2 (row 5)
                    _pgs_box_row(base + 5, c.mu_roll * new_n, lambdas, v_free, W, bias, max_delta_ref)
                    max_delta = max_delta_ref[0]

            if max_delta < self.tolerance:
                break

        # ── Update warm start cache ──
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

        # ── Convert impulses to spatial forces per body ──
        body_impulses = [np.zeros(6) for _ in range(num_bodies)]

        for ci, c in enumerate(contacts):
            base = row_offsets[ci]

            # Linear impulse (normal + tangent)
            J_world_linear = lambdas[base] * c.normal
            if c.condim >= 3:
                J_world_linear += lambdas[base + 1] * c.tangent1
                J_world_linear += lambdas[base + 2] * c.tangent2

            # Angular impulse (spin + rolling)
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


def _pgs_box_row(
    idx: int,
    limit: float,
    lambdas: NDArray,
    v_free: NDArray,
    W: NDArray,
    bias: NDArray,
    max_delta_ref: list[float],
) -> None:
    """One PGS update for a box-constrained row: |lambda| <= limit."""
    old = lambdas[idx]
    w_diag = W[idx, idx]
    if w_diag < 1e-12:
        return  # degenerate
    residual = v_free[idx] + W[idx] @ lambdas + bias[idx]
    delta = -residual / w_diag
    new = np.clip(old + delta, -limit, limit)
    lambdas[idx] = new
    max_delta_ref[0] = max(max_delta_ref[0], abs(new - old))


def _row_to_contact(row: int, row_offsets: list[int], nc: int) -> int:
    """Find the contact index that owns a given row."""
    for ci in range(nc - 1, -1, -1):
        if row >= row_offsets[ci]:
            return ci
    return 0
