"""
MuJoCo-style acceleration-level contact QP with R-regularization.

Solves the dual constrained dynamics QP:

    f* = argmin  1/2 f^T (A + R) f  +  f^T (a_u_c - a_ref)
         s.t.    f in Omega  (friction cone)

where:
    A     = J H^{-1} J^T         (constraint-space inverse inertia, joint-space)
    R     = (1-d)/d * diag(A)    (self-adaptive compliance regularization)
    a_u_c = J @ a_u              (unconstrained constraint-space acceleration)
    a_ref = -b*v_c - k*d(r)*r   (spring-damper reference acceleration)

Key differences from our velocity-level ADMM:
    1. Acceleration level — forces, not impulses
    2. Joint-space A via CRBA — captures inertia coupling
    3. R regularization — QP strictly convex, unique stable equilibrium
    4. Self-adaptive R — scales with A_ii, no manual tuning per robot

References:
    Todorov (2014) — Convex and analytically-invertible dynamics
    MuJoCo docs — Computation / Solver / Constraint model
    Boyd et al. (2011) — ADMM
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..spatial import Vec6
from .pgs_solver import ContactConstraint, _build_contact_frame


class MuJoCoStyleSolver:
    """Acceleration-level contact QP with MuJoCo-compatible compliance.

    Args:
        max_iter  : Maximum ADMM iterations.
        tolerance : Primal residual convergence tolerance.
        rho       : ADMM penalty parameter.
        solref    : (timeconst, dampratio) — spring-damper reference.
                    timeconst: time to resolve penetration [s].
                    dampratio: 1.0 = critical damping.
        solimp    : (d_0, d_width, width, midpoint, power) — impedance.
                    d_0: impedance at zero penetration [0..1].
                    d_width: impedance at full penetration [0..1].
                    width: penetration range for transition [m].
    """

    def __init__(
        self,
        max_iter: int = 50,
        tolerance: float = 1e-8,
        rho: float = 1.0,
        solref: tuple[float, float] = (0.02, 1.0),
        solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2),
    ) -> None:
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.rho = rho
        self.solref = solref
        self.solimp = solimp
        # Populated after each solve
        self.last_jacobian: NDArray | None = None
        self.last_forces: NDArray | None = None

    def solve(
        self,
        contacts: list[ContactConstraint],
        tree,
        q: NDArray,
        qdot: NDArray,
        a_u: NDArray,
        dt: float,
    ) -> tuple[NDArray, NDArray]:
        """Solve the contact QP at the acceleration level.

        Args:
            contacts: List of ContactConstraint from collision pipeline.
            tree    : RobotTreeNumpy (for CRBA, contact_jacobian).
            q, qdot : Current generalized state.
            a_u     : Unconstrained generalized acceleration (nv,) from ABA.
            dt      : Time step [s].

        Returns:
            f : (n_rows,) optimal contact forces in constraint space.
            J : (n_rows, nv) joint-space contact Jacobian.
        """
        nc = len(contacts)
        nv = tree.nv
        if nc == 0:
            self.last_jacobian = np.zeros((0, nv))
            self.last_forces = np.zeros(0)
            return np.zeros(0), np.zeros((0, nv))

        # Build contact frames
        for c in contacts:
            c.tangent1, c.tangent2 = _build_contact_frame(c.normal)

        # Row offsets (variable condim per contact)
        row_offsets = []
        offset = 0
        for c in contacts:
            row_offsets.append(offset)
            offset += c.condim
        n_rows = offset

        # ── Build joint-space contact Jacobian J (n_rows x nv) ──
        X_world = tree.forward_kinematics(q)
        J = np.zeros((n_rows, nv))

        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            # Positional Jacobian for this contact point (3 x nv)
            if c.body_i >= 0:
                J_pos_i = tree.contact_jacobian(q, c.body_i, c.point)
            else:
                J_pos_i = np.zeros((3, nv))
            if c.body_j >= 0:
                J_pos_j = tree.contact_jacobian(q, c.body_j, c.point)
            else:
                J_pos_j = np.zeros((3, nv))

            # Relative Jacobian: J_rel maps qdot -> relative contact velocity
            J_pos = J_pos_i - J_pos_j

            # Project onto contact directions
            # Normal (row 0)
            J[base, :] = c.normal @ J_pos
            # Tangent 1, 2 (rows 1, 2)
            if c.condim >= 3:
                J[base + 1, :] = c.tangent1 @ J_pos
                J[base + 2, :] = c.tangent2 @ J_pos
            # Spin (row 3) — angular Jacobian (not implemented yet, zero)
            # Rolling (rows 4-5) — angular Jacobian (not implemented yet, zero)

        # ── Constraint-space inverse inertia A = J H^{-1} J^T ──
        H = tree.crba(q)
        try:
            L = np.linalg.cholesky(H)
            # H^{-1} J^T via Cholesky: solve H X = J^T
            HinvJt = np.linalg.solve(L.T, np.linalg.solve(L, J.T))
        except np.linalg.LinAlgError:
            HinvJt = np.linalg.lstsq(H, J.T, rcond=None)[0]
        A = J @ HinvJt  # (n_rows x n_rows)

        # ── Compliance R = (1-d)/d * diag(A) (self-adaptive) ──
        R_diag = np.zeros(n_rows)
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            d = self._impedance(c.depth)
            ratio = (1.0 - d) / max(d, 1e-10)
            for j in range(c.condim):
                R_diag[base + j] = ratio * abs(A[base + j, base + j])
        R = np.diag(R_diag)

        # ── Reference acceleration a_ref (solref spring-damper) ──
        timeconst, dampratio = self.solref
        # Safety clamp: timeconst >= 2*dt (MuJoCo default)
        timeconst = max(timeconst, 2.0 * dt)
        # b, k use d_width (solimp[1]), not d(0) — per MuJoCo docs
        d_w = self.solimp[1]  # impedance at full penetration
        b = 2.0 / (d_w * timeconst)
        k = 1.0 / (d_w**2 * timeconst**2 * dampratio**2)

        v_c = J @ qdot  # constraint-space velocity
        a_u_c = J @ a_u  # unconstrained constraint-space acceleration

        a_ref = np.zeros(n_rows)
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            d = self._impedance(c.depth)
            # Normal: a_ref = -b*v_n + k*d*depth
            # MuJoCo uses r (signed distance, negative=penetrating): a_ref = -b*v - k*d*r
            # Our depth is positive when penetrating, so r = -depth:
            #   a_ref = -b*v - k*d*(-depth) = -b*v + k*d*depth
            a_ref[base] = -b * v_c[base] + k * d * c.depth
            # Tangent: a_ref = -b*v_t (no position term)
            if c.condim >= 3:
                a_ref[base + 1] = -b * v_c[base + 1]
                a_ref[base + 2] = -b * v_c[base + 2]

        # ── ADMM on dual QP ──
        # min 1/2 f^T (A+R) f + f^T (a_u_c - a_ref)
        # s.t. f in Omega (friction cone)
        AR = A + R
        rhs_const = -(a_u_c - a_ref)  # = a_ref - a_u_c

        # Pre-factor (A + R + rho*I)
        ARrho = AR + self.rho * np.eye(n_rows)
        try:
            ARrho_chol = np.linalg.cholesky(ARrho)
        except np.linalg.LinAlgError:
            ARrho += 1e-6 * np.eye(n_rows)
            ARrho_chol = np.linalg.cholesky(ARrho)

        f = np.zeros(n_rows)
        s = np.zeros(n_rows)
        u = np.zeros(n_rows)

        for iteration in range(self.max_iter):
            # Step 1: f-update — (AR + rho*I) f = rhs_const + rho*(s - u)
            rhs = rhs_const + self.rho * (s - u)
            y = np.linalg.solve(ARrho_chol, rhs)
            f_new = np.linalg.solve(ARrho_chol.T, y)

            # Step 2: s = proj_cone(f + u)
            s_new = self._project_all(f_new + u, contacts, row_offsets)

            # Step 3: dual update
            u = u + f_new - s_new

            # Convergence: both primal AND dual residual must be small
            # (primal alone can be zero on iteration 1 when f is already feasible,
            #  but the solution hasn't converged yet due to rho*I regularization)
            primal_res = np.linalg.norm(f_new - s_new)
            dual_res = np.linalg.norm(self.rho * (s_new - s))
            f, s = f_new, s_new

            if primal_res < self.tolerance and dual_res < self.tolerance:
                break

        self.last_jacobian = J
        self.last_forces = f
        return f, J

    def _impedance(self, depth: float) -> float:
        """Compute impedance d(r) from solimp parameters.

        Piecewise power-law sigmoid from MuJoCo engine_core_constraint.c.
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

    @staticmethod
    def _project_all(
        z: NDArray,
        contacts: list[ContactConstraint],
        row_offsets: list[int],
    ) -> NDArray:
        """Project z onto friction cones for all contacts."""
        s = z.copy()
        for ci, c in enumerate(contacts):
            base = row_offsets[ci]
            # Normal: f_n >= 0
            s[base] = max(0.0, z[base])
            if c.condim >= 3:
                # Tangent: |f_t| <= mu * f_n (disk projection)
                limit = c.mu * s[base]
                t_norm = np.sqrt(z[base + 1] ** 2 + z[base + 2] ** 2)
                if t_norm > limit and t_norm > 1e-12:
                    scale = limit / t_norm
                    s[base + 1] = z[base + 1] * scale
                    s[base + 2] = z[base + 2] * scale
                else:
                    s[base + 1] = z[base + 1]
                    s[base + 2] = z[base + 2]
            if c.condim >= 4:
                limit_s = c.mu_spin * s[base]
                s[base + 3] = np.clip(z[base + 3], -limit_s, limit_s)
            if c.condim >= 6:
                limit_r = c.mu_roll * s[base]
                s[base + 4] = np.clip(z[base + 4], -limit_r, limit_r)
                s[base + 5] = np.clip(z[base + 5], -limit_r, limit_r)
        return s
