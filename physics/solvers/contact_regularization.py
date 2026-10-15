"""Contact-model regularization, independent of ADMM's algorithmic penalty.

The ``mujoco`` policy targets MuJoCo 3.15's approximate-diagonal Euler model
for isotropic rigid contacts. MuJoCo is not a runtime dependency.
"""

import numpy as np

from ..dynamics_utils import body_velocity_jacobians
from ..joint import PrismaticJoint
from ..spatial import skew

MINVAL = 1e-15


def validate_policy(regularization, impratio):
    if regularization not in ("per_row", "mujoco"):
        raise ValueError("regularization must be 'per_row' or 'mujoco'")
    if not np.isfinite(impratio) or impratio <= 0:
        raise ValueError("impratio must be finite and positive")
    if regularization == "per_row" and impratio != 1:
        raise ValueError("impratio is only supported by regularization='mujoco'")


def normalize_parameters(solref, solimp, dt):
    """Validate standard positive solref and normalize MuJoCo solimp."""
    ref = np.asarray(solref, dtype=float)
    imp = np.asarray(solimp, dtype=float)
    if (
        ref.shape != (2,)
        or imp.shape != (5,)
        or not np.all(np.isfinite(ref))
        or not np.all(np.isfinite(imp))
        or np.any(ref <= 0)
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("mujoco regularization requires finite positive solref/dt and finite solimp")
    imp = imp.copy()
    imp[[0, 1, 3]] = np.clip(imp[[0, 1, 3]], 1e-4, 0.9999)
    imp[2] = max(0.0, imp[2])
    imp[4] = max(1.0, imp[4])
    return tuple(ref), tuple(imp)


def validate_float32_inputs(*values):
    """Validate host admission before any state upload, including conversion."""
    with np.errstate(over="ignore", invalid="ignore"):
        for value in values:
            if value is not None and not np.all(np.isfinite(np.asarray(value, dtype=np.float32))):
                raise ValueError("mujoco input must be finite and representable in float32")


def native_parameters(solref, solimp, dt, impratio):
    """Normalized GPU parameters with the same native coefficient evaluation."""
    ref, imp = normalize_parameters(solref, solimp, dt)
    validate_float32_inputs((*ref, *imp, dt, max(MINVAL, impratio)))
    native = np.asarray((*ref, dt, max(MINVAL, impratio)), dtype=np.float32)
    if np.any(native <= 0):
        raise ValueError("positive ADMM parameters underflow in float32")
    reference_coefficients(ref, imp, dt, dtype=np.float32)
    return imp


def validate_R_domain(weights, solimp, impratio, condim, dtype=np.float64):
    """Conservative native R bounds over all depths and possible body pairs.

    The smallest pair mobility is bounded by the smallest body weight (world
    weight is zero); the largest by twice the largest body weight. Require
    normal, finite outputs, avoiding CUDA flush-to-zero dependence. This can
    reject extreme models even when their particular current contacts are safe.
    """
    if len(weights) == 0:
        return
    tiny = np.finfo(dtype).tiny
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        w = np.asarray(weights, dtype=dtype)
        dmin = dtype(min(solimp[0], solimp[1]))
        dmax = dtype(max(solimp[0], solimp[1]))
        rmin = max(dtype(MINVAL), (dtype(1) - dmax) / dmax * np.min(w))
        rmax = max(dtype(MINVAL), (dtype(1) - dmin) / dmin * (dtype(2) * np.max(w)))
        bounds = [rmin, rmax]
        if condim == 3:
            ratio = dtype(max(MINVAL, impratio))
            bounds += [rmin / ratio, rmax / ratio]
    if (
        not np.all(np.isfinite(w))
        or np.any(w < 0)
        or np.any((np.asarray(weights) != 0) & (w < tiny))
        or not np.all(np.isfinite(bounds))
        or np.any(np.asarray(bounds) < tiny)
    ):
        raise ValueError("contact regularization is outside the native normal finite range")


def reference_coefficients(solref, solimp, dt, dtype=np.float64):
    """Reject unrepresentable native coefficients instead of silently flooring.

    Extreme positive solref values are outside the numerical domain if native
    evaluation overflows or underflows. This is distinct from material limits.
    """
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        tc = dtype(max(solref[0], 2 * dt))
        dr = dtype(solref[1])
        dw = dtype(solimp[1])
        b = dtype(2) / (dw * tc)
        k = dtype(1) / (dw * dw * tc * tc * dr * dr)
    if not np.isfinite(b) or not np.isfinite(k) or b < np.finfo(dtype).tiny or k < np.finfo(dtype).tiny:
        raise ValueError("reference acceleration coefficients are outside the native numerical range")
    return b, k


def impedance(depth, solimp):
    d0, dw, width, midpoint, power = solimp
    if width <= MINVAL:
        return 0.5 * (d0 + dw)
    x = min(1.0, abs(depth) / width)
    if power == 1:
        y = x
    elif x <= midpoint:
        y = midpoint * (x / midpoint) ** power
    else:
        y = 1 - (1 - midpoint) * ((1 - x) / (1 - midpoint)) ** power
    return d0 + y * (dw - d0)


def reference_body_weights(tree):
    """Reference COM mobility, including MuJoCo's simple-slider exception.

    Callers restore current FK when needed. CPU recomputes for mutable models;
    GPU snapshots these weights alongside its other static model data.
    """
    q0, _ = tree.default_state()
    H = tree.crba(q0)
    if not np.all(np.isfinite(H)):
        raise ValueError("nonfinite reference inertia")
    L = np.linalg.cholesky(H)
    B = body_velocity_jacobians(tree, q0)
    children = {b.parent for b in tree.bodies if b.parent >= 0}
    weights = np.zeros(tree.num_bodies)
    for b in tree.bodies:
        J = B[b.index, :3] - skew(b.inertia.com) @ B[b.index, 3:]
        weights[b.index] = np.sum(np.linalg.solve(L, J.T) ** 2) / 3
        parent_fixed_root = (
            b.parent >= 0 and tree.bodies[b.parent].parent < 0 and tree.bodies[b.parent].joint.nv == 0
        )
        if (
            isinstance(b.joint, PrismaticJoint)
            and b.index not in children
            and (b.parent < 0 or parent_fixed_root)
            and np.all(np.abs(b.inertia.com) < 1e-6)
        ):
            axis = b.joint.motion_subspace(q0[b.q_idx])[:3, 0]
            if np.count_nonzero(np.abs(axis) > 1e-14) == 1:
                inertia = b.inertia.inertia
                if np.any(inertia != np.diag(np.diag(inertia))):
                    raise NotImplementedError("mujoco simple slider requires a diagonal body-frame inertia")
                weights[b.index] = 1 / max(MINVAL, b.inertia.mass)
    if not np.all(np.isfinite(weights)):
        raise ValueError("nonfinite reference body mobility")
    return weights
