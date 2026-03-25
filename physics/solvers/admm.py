"""
ADMM contact solver — GPU-friendly, implicit-contact-compatible.

Alternating Direction Method of Multipliers splits the contact QP into:
  Step 1: linear system solve (parallelizable, pre-factorable)
  Step 2: cone projection (per-contact, embarrassingly parallel)
  Step 3: dual variable update (element-wise)

The linear system (M + rho * J^T J) is symmetric positive-definite and
constant across iterations (J doesn't change within a step), so it can
be pre-factored once (Cholesky) and reused for all ADMM iterations.

This solver naturally implements implicit contact integration: the
velocity update in Step 1 already accounts for the contact constraints.

Optional compliant contact mode (contact_stiffness/contact_damping):
  Replaces Baumgarte ERP with spring-damper penetration response,
  eliminating the dt^{-1} amplification that causes divergence.
  Equivalent to MuJoCo solref/solimp compliance model.

Optional adaptive rho (adaptive_rho=True):
  Boyd et al. (2011) scheme: adjusts rho based on primal/dual residual
  ratio, automatically tuning convergence for different contact stiffness.

References:
  Boyd et al. (2011) — Distributed Optimization and Statistical Learning
    via the Alternating Direction Method of Multipliers
  Todorov (2014) — Convex and analytically-invertible dynamics (MuJoCo)
  Macklin et al. (2019) — Non-smooth Newton Methods for ADMM contact
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
)


def _project_cone(s: NDArray, contact: ContactConstraint) -> NDArray:
    """Project s onto the friction cone K for one contact.

    K depends on condim:
      condim=1: s_n >= 0
      condim=3: s_n >= 0, |s_t| <= mu * s_n
      condim=4: + |s_spin| <= mu_spin * s_n
      condim=6: + |s_roll| <= mu_roll * s_n

    Args:
        s: constraint-space vector for this contact, shape (condim,).
        contact: ContactConstraint with condim, mu, mu_spin, mu_roll.

    Returns:
        Projected vector, same shape.
    """
    p = s.copy()
    cd = contact.condim

    # Normal: clamp to >= 0
    p[0] = max(0.0, p[0])

    if cd >= 3:
        # Tangent: project to friction disk |p_t| <= mu * p_n
        limit = contact.mu * p[0]
        t_norm = np.sqrt(p[1] ** 2 + p[2] ** 2)
        if t_norm > limit and t_norm > 1e-12:
            scale = limit / t_norm
            p[1] *= scale
            p[2] *= scale

    if cd >= 4:
        # Spin: box constraint |p_s| <= mu_spin * p_n
        limit_s = contact.mu_spin * p[0]
        p[3] = np.clip(p[3], -limit_s, limit_s)

    if cd >= 6:
        # Rolling: box constraint per axis
        limit_r = contact.mu_roll * p[0]
        p[4] = np.clip(p[4], -limit_r, limit_r)
        p[5] = np.clip(p[5], -limit_r, limit_r)

    return p


class ADMMContactSolver:
    """ADMM contact solver with condim support.

    Args:
        max_iter          : Maximum ADMM iterations.
        tolerance         : Primal residual convergence tolerance.
        rho               : ADMM penalty parameter (larger = faster but less stable).
        erp               : Error Reduction Parameter (Baumgarte stabilization).
                            Ignored when contact_stiffness is set (compliant mode).
        cfm               : Constraint Force Mixing (regularization).
        contact_stiffness : Compliant contact stiffness [1/s^2]. When set,
                            replaces Baumgarte ERP with spring-damper response.
        contact_damping   : Compliant contact damping [1/s]. Defaults to
                            2*sqrt(stiffness) (critical damping) when stiffness
                            is set and damping is None.
        adaptive_rho      : Enable Boyd et al. (2011) adaptive rho.
        rho_scale         : Primal/dual residual ratio threshold (default 10).
        rho_factor        : Rho multiplier/divisor per adaptation (default 2).
    """

    def __init__(
        self,
        max_iter: int = 50,
        tolerance: float = 1e-6,
        rho: float = 1.0,
        erp: float = 0.2,
        cfm: float = 1e-6,
        contact_stiffness: float | None = None,
        contact_damping: float | None = None,
        adaptive_rho: bool = False,
        rho_scale: float = 10.0,
        rho_factor: float = 2.0,
    ) -> None:
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.rho = rho
        self.erp = erp
        self.cfm = cfm
        self.contact_stiffness = contact_stiffness
        if contact_stiffness is not None and contact_damping is None:
            self.contact_damping = 2.0 * np.sqrt(contact_stiffness)
        else:
            self.contact_damping = contact_damping or 0.0
        self.adaptive_rho = adaptive_rho
        self.rho_scale = rho_scale
        self.rho_factor = rho_factor
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
        """Solve contact constraints via ADMM.

        Same interface as PGSContactSolver.solve().
        """
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
        n_body_dofs = num_bodies * 6  # 6 spatial DOFs per body

        # ── Build dense Jacobian J (n_rows × n_body_dofs) ──
        J = np.zeros((n_rows, n_body_dofs))

        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            directions_linear = [c.normal]
            if c.condim >= 3:
                directions_linear.extend([c.tangent1, c.tangent2])
            for d_idx, direction in enumerate(directions_linear):
                row = base + d_idx
                if c.body_i >= 0:
                    J[row, c.body_i * 6 : c.body_i * 6 + 6] += _compute_linear_jacobian_row(
                        direction,
                        c.point,
                        body_X_world[c.body_i].r,
                        body_X_world[c.body_i].R,
                    )
                if c.body_j >= 0:
                    J[row, c.body_j * 6 : c.body_j * 6 + 6] -= _compute_linear_jacobian_row(
                        direction,
                        c.point,
                        body_X_world[c.body_j].r,
                        body_X_world[c.body_j].R,
                    )
            if c.condim >= 4:
                row = base + 3
                if c.body_i >= 0:
                    J[row, c.body_i * 6 : c.body_i * 6 + 6] += _compute_angular_jacobian_row(
                        c.normal,
                        body_X_world[c.body_i].R,
                    )
                if c.body_j >= 0:
                    J[row, c.body_j * 6 : c.body_j * 6 + 6] -= _compute_angular_jacobian_row(
                        c.normal,
                        body_X_world[c.body_j].R,
                    )
            if c.condim >= 6:
                for t_idx, tang in enumerate([c.tangent1, c.tangent2]):
                    row = base + 4 + t_idx
                    if c.body_i >= 0:
                        J[row, c.body_i * 6 : c.body_i * 6 + 6] += _compute_angular_jacobian_row(
                            tang,
                            body_X_world[c.body_i].R,
                        )
                    if c.body_j >= 0:
                        J[row, c.body_j * 6 : c.body_j * 6 + 6] -= _compute_angular_jacobian_row(
                            tang,
                            body_X_world[c.body_j].R,
                        )

        # ── Build block-diagonal mass matrix M (n_body_dofs × n_body_dofs) ──
        M = np.zeros((n_body_dofs, n_body_dofs))
        for bi in range(num_bodies):
            m = 1.0 / inv_mass[bi] if inv_mass[bi] > 1e-10 else 1e10
            I_rot = (
                np.linalg.inv(inv_inertia[bi]) if np.linalg.det(inv_inertia[bi]) > 1e-20 else np.eye(3) * 1e10
            )
            idx = bi * 6
            M[idx : idx + 3, idx : idx + 3] = m * np.eye(3)
            M[idx + 3 : idx + 6, idx + 3 : idx + 6] = I_rot

        # ── Free velocity v̄ = current body velocities (flattened) ──
        v_bar = np.zeros(n_body_dofs)
        for bi in range(num_bodies):
            v_bar[bi * 6 : bi * 6 + 6] = body_v[bi]

        # ── Bias (constraint space) ──
        bias = np.zeros(n_rows)
        v_free_contact = J @ v_bar

        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            if self.contact_stiffness is not None:
                # Compliant mode: spring-damper bias (normalized by impedance)
                #   F = k * depth - d * v_n   (standard spring-damper)
                #   bias = (k * depth - d * v_n) / (k * dt + d)
                #
                # The -d*v_n term provides BIDIRECTIONAL damping:
                #   v_n < 0 (approaching): bias increases (pushes harder)
                #   v_n > 0 (departing):   bias decreases (reduces overcorrection)
                # This is critical for settling — without it, the solver
                # overcorrects and the body oscillates.
                k = self.contact_stiffness
                d = self.contact_damping
                impedance = k * dt + d
                v_n = v_free_contact[base]  # + = departing, - = approaching
                if impedance > 1e-12:
                    bias[base] = (k * c.depth - d * v_n) / impedance
                else:
                    bias[base] = 0.0
            else:
                # Hard mode: Baumgarte ERP
                erp = c.erp if c.erp is not None else self.erp
                baumgarte = erp / dt * c.depth
                bias[base] = baumgarte
            # Restitution (both modes)
            if c.restitution > 0.0 and v_free_contact[base] < -0.01:
                bias[base] += -c.restitution * v_free_contact[base]

        # ── Pre-factor A = M + rho * J^T J ──
        rho = self.rho
        JtJ = J.T @ J
        A = M + rho * JtJ + self.cfm * np.eye(n_body_dofs)
        A_chol = np.linalg.cholesky(A)

        # ── Initialize ADMM variables ──
        s = np.zeros(n_rows)  # slack (constraint-space)
        u = np.zeros(n_rows)  # dual (scaled)

        # Warm starting: initialize s from cache
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
                    s[base : base + n_copy] = best_lam[:n_copy]

        # ── ADMM iterations ──
        Mv_bar = M @ v_bar

        for iteration in range(self.max_iter):
            # Step 1: v⁺ = A⁻¹ (M v̄ + rho * Jᵀ (s - u + bias_scaled))
            rhs = Mv_bar + rho * J.T @ (s - u + bias / rho)
            y = np.linalg.solve(A_chol, rhs)
            v_new = np.linalg.solve(A_chol.T, y)

            # Step 2: s = proj_K(J v⁺ + u)
            Jv = J @ v_new
            z = Jv + u
            s_old = s.copy()
            s_new = np.zeros(n_rows)
            for ci, c in enumerate(contacts):
                base = row_offsets[ci]
                s_new[base : base + c.condim] = _project_cone(z[base : base + c.condim], c)

            # Step 3: u = u + Jv - s
            u = u + Jv - s_new

            # Convergence check
            primal_residual = np.linalg.norm(Jv - s_new)
            dual_residual = np.linalg.norm(rho * J.T @ (s_new - s_old))
            s = s_new

            # Adaptive rho (Boyd et al. 2011)
            if self.adaptive_rho and iteration < self.max_iter - 1:
                need_refactor = False
                if primal_residual > self.rho_scale * max(dual_residual, 1e-20):
                    rho *= self.rho_factor
                    need_refactor = True
                elif dual_residual > self.rho_scale * max(primal_residual, 1e-20):
                    rho /= self.rho_factor
                    need_refactor = True
                if need_refactor:
                    A = M + rho * JtJ + self.cfm * np.eye(n_body_dofs)
                    A_chol = np.linalg.cholesky(A)

            if primal_residual < self.tolerance:
                break

        # ── Extract impulses: lambda = rho * (s - Jv - u) or directly from dual ──
        # The contact impulse is lambda = rho * (s - (Jv + u - s)) = rho * u
        # More precisely: lambda = rho * (Jv⁺ + u - s) ... but we use the
        # change in velocity to compute impulses: impulse = M (v⁺ - v̄)
        dv = v_new - v_bar

        # ── Update warm cache ──
        # Store the constraint-space solution (s) for warm starting
        self._warm_cache.clear()
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            key = (min(c.body_i, c.body_j), max(c.body_i, c.body_j))
            lam = s[base : base + c.condim].copy()
            if c.body_i >= 0:
                local_pt = body_X_world[c.body_i].R.T @ (c.point - body_X_world[c.body_i].r)
            else:
                local_pt = c.point.copy()
            if key not in self._warm_cache:
                self._warm_cache[key] = []
            self._warm_cache[key].append((local_pt.copy(), lam))

        # ── Convert velocity delta to spatial impulses per body ──
        body_impulses = [np.zeros(6) for _ in range(num_bodies)]
        for bi in range(num_bodies):
            idx = bi * 6
            # impulse = M_body @ dv_body
            m = 1.0 / inv_mass[bi] if inv_mass[bi] > 1e-10 else 1e10
            I_rot = (
                np.linalg.inv(inv_inertia[bi]) if np.linalg.det(inv_inertia[bi]) > 1e-20 else np.eye(3) * 1e10
            )
            dv_body = dv[idx : idx + 6]
            body_impulses[bi][:3] = m * dv_body[:3]
            body_impulses[bi][3:] = I_rot @ dv_body[3:]

        return body_impulses
