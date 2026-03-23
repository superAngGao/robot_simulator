"""
LCP contact solver using Projected Gauss-Seidel (PGS).

Solves the contact Linear Complementarity Problem:
  0 <= lambda_n  perp  v_n >= 0       (Signorini: no penetration)
  |lambda_t| <= mu * lambda_n          (Coulomb friction cone)

The velocity-level formulation:
  v_contact = v_free + W @ lambda
  where:
    v_free  = contact velocity without constraint forces
    W       = J @ M^{-1} @ J^T   (Delassus operator)
    lambda  = constraint impulses [normal; tangent_x; tangent_y]

PGS iterates:
  for each contact i:
    lambda_n_i = project_positive(lambda_n_i - W_nn^{-1} * (v_n_i + ...))
    lambda_t_i = project_friction_cone(lambda_t_i - ..., mu * lambda_n_i)

References:
  Catto (2005) — Iterative Dynamics with Temporal Coherence (GDC)
  Erleben (2007) — Velocity-based shock propagation
  MuJoCo: similar PGS with warm-starting
"""

from __future__ import annotations

from dataclasses import dataclass

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


def _contact_jacobian_body(
    point: Vec3,
    normal: Vec3,
    body_origin: Vec3,
    body_R: NDArray,
) -> NDArray:
    """Compute the contact Jacobian row for one body.

    Maps body spatial velocity (6,) [lin; ang] to contact-frame velocity (scalar).
    J_n = n^T @ [R, -R @ skew(r_local)]  for normal direction.

    Returns (3, 6) matrix: rows for [normal, tangent1, tangent2].
    """
    r_world = point - body_origin
    # v_contact_world = v_lin_world + omega_world × r_world
    # v_lin_world = R @ v_lin_body, omega_world = R @ omega_body
    # v_contact_world = R @ v_lin_body + (R @ omega_body) × r_world
    #                 = R @ v_lin_body + R @ (omega_body × (R^T @ r_world))
    # Hmm, simpler: in world frame,
    # v_contact = J @ v_spatial_body
    # where J[d, :3] = n_d^T @ R (linear part)
    #       J[d, 3:] = n_d^T @ (r_world × R)  hmm...
    #
    # Actually: v_point_world = R @ v_body_lin + cross(R @ omega_body, r_world)
    # dot(n, v_point) = n^T R v_lin + n^T (R omega) × r
    #                 = n^T R v_lin + (r × n)^T R omega
    #
    # J_lin = n^T @ R  (1×3 → maps body lin vel to contact normal vel)
    # J_ang = (r × n)^T @ R  (1×3 → maps body ang vel)
    # Full row: [J_lin, J_ang] = (1, 6)

    # This is NOT needed for the simplified approach below.
    pass


class PGSContactSolver:
    """Projected Gauss-Seidel LCP solver for contact dynamics.

    Args:
        max_iter     : Maximum PGS iterations.
        tolerance    : Convergence tolerance on velocity residual.
        erp          : Error Reduction Parameter (Baumgarte stabilization).
        cfm          : Constraint Force Mixing (regularization).
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

    def solve(
        self,
        contacts: list[ContactConstraint],
        body_v: list[Vec6],  # spatial velocities in body frame, per body
        body_X_world: list,  # SpatialTransform per body
        inv_mass: list[float],  # 1/mass per body (0 for ground)
        inv_inertia: list[NDArray],  # 3x3 inverse inertia per body (zeros for ground)
        dt: float,
    ) -> list[Vec6]:
        """Solve contact constraints and return spatial impulses per body.

        Returns: list of Vec6 impulses in body frame (to be applied as forces/dt).
        """
        nc = len(contacts)
        if nc == 0:
            return [np.zeros(6) for _ in body_v]

        num_bodies = len(body_v)

        # Build contact frames
        for c in contacts:
            c.tangent1, c.tangent2 = _build_contact_frame(c.normal)

        # Initialize impulses: 3 per contact (normal, tangent1, tangent2)
        impulses = np.zeros(nc * 3)

        # Precompute effective mass for each contact direction
        # For a simple approximation: W_ii ≈ 1/m_eff where
        # m_eff = 1/(1/m_i + 1/m_j + angular terms)
        w_diag = np.zeros(nc * 3)  # diagonal of Delassus operator
        for ci, c in enumerate(contacts):
            for d, direction in enumerate([c.normal, c.tangent1, c.tangent2]):
                m_eff = 0.0
                for bi in [c.body_i, c.body_j]:
                    if bi < 0:
                        continue
                    m_eff += inv_mass[bi]
                    # Angular contribution: r × n decomposition
                    r = c.point - body_X_world[bi].r
                    rxn = np.cross(r, direction)
                    # Approximate: angular contribution = rxn^T @ I_inv @ rxn
                    m_eff += rxn @ inv_inertia[bi] @ rxn
                w_diag[ci * 3 + d] = m_eff + self.cfm

        # Compute free velocity at each contact point
        v_free = np.zeros(nc * 3)
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

        # Baumgarte stabilization: bias pushes contact velocity positive (away from penetration)
        bias = np.zeros(nc * 3)
        for ci, c in enumerate(contacts):
            bias[ci * 3] = -self.erp / dt * c.depth

        # Precompute velocity change per unit impulse for each contact direction
        # v_change[ci*3+d] = effect on own contact velocity from own impulse
        # (diagonal approximation of Delassus operator)

        # PGS iterations
        # Track accumulated velocity delta from all impulses
        v_delta = np.zeros(nc * 3)

        for iteration in range(self.max_iter):
            max_delta = 0.0

            for ci, c in enumerate(contacts):
                # Current contact velocity = free + delta from all impulses
                # Normal direction
                idx_n = ci * 3
                old_n = impulses[idx_n]
                v_current_n = v_free[idx_n] + w_diag[idx_n] * old_n + bias[idx_n]
                new_n = old_n - v_current_n / w_diag[idx_n]
                new_n = max(0.0, new_n)  # Signorini projection
                delta_n = new_n - old_n
                impulses[idx_n] = new_n
                max_delta = max(max_delta, abs(delta_n))

                # Tangent directions (friction)
                max_friction = c.mu * impulses[idx_n]
                for d in [1, 2]:
                    idx_t = ci * 3 + d
                    old_t = impulses[idx_t]
                    v_current_t = v_free[idx_t] + w_diag[idx_t] * old_t
                    new_t = old_t - v_current_t / w_diag[idx_t]
                    new_t = np.clip(new_t, -max_friction, max_friction)
                    impulses[idx_t] = new_t
                    max_delta = max(max_delta, abs(new_t - old_t))

            if max_delta < self.tolerance:
                break

        # Convert impulses to spatial forces per body
        body_impulses = [np.zeros(6) for _ in range(num_bodies)]

        for ci, c in enumerate(contacts):
            # World-frame impulse
            J_world = (
                impulses[ci * 3] * c.normal
                + impulses[ci * 3 + 1] * c.tangent1
                + impulses[ci * 3 + 2] * c.tangent2
            )

            for bi, sign in [(c.body_i, 1.0), (c.body_j, -1.0)]:
                if bi < 0:
                    continue
                X = body_X_world[bi]
                f_world = sign * J_world
                r = c.point - X.r
                torque_world = np.cross(r, f_world)
                # Convert to body frame
                f_body = X.R.T @ f_world
                torque_body = X.R.T @ torque_world
                body_impulses[bi][:3] += f_body
                body_impulses[bi][3:] += torque_body

        return body_impulses
