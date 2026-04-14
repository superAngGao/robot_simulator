"""
Collision / contact numerical tolerance configuration.

Single source of truth for the epsilons that used to be scattered across
`gjk_epa.py`, `analytical_collision.py` and GPU Warp constants. CPU code reads
from `DEFAULT_CONTACT_TOLERANCES`; GPU `wp.constant` values are initialized
from the same module-level scalars so CPU and GPU stay in lockstep.

Exposed through `ContactTolerances` dataclass for runtime override (future
per-engine tuning, currently read-only defaults).

References:
  ODE `dSafeNormalize4` epsilon (1e-6)
  Bullet `SIMD_EPSILON` / `btPersistentManifold::contactBreakingThreshold`
  MuJoCo `mjMINMU`, `mjMINIMP` — same pattern of centralized tolerances
"""

from __future__ import annotations

from dataclasses import dataclass

# Module-level scalar defaults. GPU `wp.constant(...)` initializers import these
# directly so any change to CPU defaults propagates to GPU at import time.

# Convex margin for GJK (Jolt / Bullet convex radius). Contact when
# distance < margin, depth = margin - distance. Avoids EPA numerical
# degeneracy by keeping GJK in the closest-distance regime.
CONTACT_CONVEX_MARGIN = 1.0e-3

# Cosine threshold for "face-aligned" contact in S-H clipping. When both
# shapes' support faces have dot(face_normal, epa_normal) >= this, we
# treat it as face-face and clip; below → edge-edge path.
CONTACT_FACE_ALIGN_THRESHOLD = 0.9

# Dot product threshold for coplanar-triangle merging during ConvexHull
# face-topology construction (geometry.py _build_convexhull_face_topology).
CONTACT_COPLANAR_DOT = 1.0 - 1e-6

# |axis · n| threshold below which a capsule/cylinder axis is considered
# "near-parallel" to the contact normal. Below threshold → generate 2-point
# manifold from segment endpoints; above → single-point closest.
#
# ODE uses 0.03 (~1.72°). We use 0.015 (~0.86°) for 2× higher precision —
# meaning we generate 2-point manifolds in a narrower parallel-axis cone
# and fall back to single point slightly sooner. This trades a small amount
# of robustness for fewer spurious 2-point manifolds on near-skew configs.
CONTACT_NEAR_PARALLEL_COS = 0.015


@dataclass(frozen=True)
class ContactTolerances:
    """Runtime-configurable collision tolerances. Frozen for safe sharing."""

    convex_margin: float = CONTACT_CONVEX_MARGIN
    face_align_threshold: float = CONTACT_FACE_ALIGN_THRESHOLD
    coplanar_dot: float = CONTACT_COPLANAR_DOT
    near_parallel_cos: float = CONTACT_NEAR_PARALLEL_COS


DEFAULT_CONTACT_TOLERANCES = ContactTolerances()
