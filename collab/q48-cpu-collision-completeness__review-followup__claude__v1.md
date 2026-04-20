Initiative: q48-cpu-collision-completeness
Stage: review-followup
Author: claude
Version: v1
Date: 2026-04-20
Status: implemented
Related Files: physics/gjk_epa.py, tests/integration/test_cpu_engine_shapes.py
Owner Summary: Three issues from Codex review of v1 implementation-note are fixed. Codex should verify the depth-semantics fix is correct and that the new body-body discriminators are strong enough.

## Issues Addressed

### High — depth semantics in polyhedral ground_contact_query

**Root cause**: `max_depth` was initialized to `0.0` before iterating over
penetrating vertices.  When all vertices are in the margin zone (depth < 0,
i.e. near-contact but not yet penetrating), the loop never updated `max_depth`,
so `manifold.depth` was returned as `0.0` instead of the correct negative gap.

**Fix** (`physics/gjk_epa.py`): removed the `max_depth = 0.0` initializer and
the in-loop `if d > max_depth` branch; replaced with `max_depth = max(point_depths)`
after the loop.  This is equivalent for the penetrating case and correct for the
near-contact case.

**Invariant now holds**: `manifold.depth == max(point_depths)` for all vertices,
matching the sphere fallback path and the docstring contract ("depth can be
negative when within margin but not yet penetrating").

### Medium — ConvexHull skip guard checked wrong dependency

**Root cause**: `HAS_CONVEXHULL` only tried `from physics.geometry import
ConvexHullShape`.  `ConvexHullShape.__init__` calls `scipy.spatial.ConvexHull`
at construction time, so in a scipy-free environment the tests would not skip —
they would fail at runtime.  Skip reason also said "trimesh required" (wrong).

**Fix** (`tests/integration/test_cpu_engine_shapes.py`): guard now also imports
`scipy.spatial.ConvexHull` in the try block; skip reason updated to "scipy required".

### Medium — TestBodyBodyContact assertions were too weak

**Root cause**: All three body-body tests only checked NaN and `z > 0`.  Bodies
could fall independently to the ground without ever interacting and still pass.

**Fix (v1 attempt — rejected by Codex)**: horizontal separation `|x_b - x_a| >= 2r * 0.9`
for sphere-sphere.  Flaw: initial separation was already exactly `2r`, so the
threshold `0.9 * 2r` was trivially satisfied without any contact.

**Fix (v2 — current)**: Redesigned sphere-sphere scenario: lower sphere rests on
ground at `z = r`, upper sphere dropped from directly above at `z = 3r + 0.1`.
Discriminators:
- `test_sphere_sphere_collision`: `z_upper > z_lower` (no pass-through) AND
  `sep_z = z_upper - z_lower >= 2r * 0.9` (no interpenetration).  A no-collision
  trajectory would have `z_upper → r` and `z_lower → r`, giving `sep_z → 0`,
  which fails the assertion.
- `test_box_sphere_collision`: sphere rests above box top face `z_sphere > 2*half * 0.9`.
- `test_box_box_stacking`: upper box rests above lower box top face `z_upper > 2*half * 0.9`.

## Verification

`pytest tests/integration/test_cpu_engine_shapes.py` — 12 passed in 21s.

## Remaining Known Limitations (unchanged from v1)

1. gjk_distance box-cyl/box-hull early exit — P4, deferred.
2. Body-body contact count not validated (Class 3 still does not assert contact
   point counts for body-body pairs, only geometry).
3. FlatTerrain only; HalfSpaceTerrain integration tests not added.
4. N_SETTLE = 1500 not marked slow.
