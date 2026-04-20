Initiative: q48-cpu-collision-completeness
Stage: implementation-note
Author: claude
Version: v1
Date: 2026-04-20
Status: implemented
Related Files: physics/gjk_epa.py, tests/integration/test_cpu_engine_shapes.py
Owner Summary: Q48.2 (CPU multi-point ground contact) and Q48.3 (CpuEngine integration tests) are done. ground_contact_query now returns multi-point manifolds for Box/ConvexHull. 12 new integration tests cover all shape types end-to-end. Codex should review test coverage completeness and flag any remaining gaps.

## Open Questions Addressed

- **Q48 — CPU 碰撞检测完备性**: item 2 (CPU 多点接触流形) → resolved for ground contact path. item 3 (CpuEngine 集成级别覆盖) → resolved. item 1 (gjk_distance box-cyl/box-hull 早退出) remains P4 deferred.

## REFLECTIONS.md / PROGRESS.md Impact

Not needed — this is a targeted fix + test addition, not a milestone or architectural decision.

## What Changed

### `physics/gjk_epa.py` — `ground_contact_query`

Before: single-point fallback for all shapes (support point in -Z direction).

After: polyhedral shapes (Box, ConvexHull) now enumerate all vertices via
`contact_vertices()` and return every vertex below `ground_z + margin` as a
separate contact point, with per-point depths. Sphere falls through to the
original single-point path.

This aligns `ground_contact_query` with `halfspace_convex_query`, which already
had vertex enumeration for Box/ConvexHull. The two functions now use the same
multi-point logic for polyhedral shapes.

Key invariant: `point_depths` list is populated so `manifold.depth_at(i)` returns
the correct per-vertex depth, not the global max depth.

### `tests/integration/test_cpu_engine_shapes.py` (new file, 12 tests)

Three test classes:

**Class 1 — TestSingleShapeGroundContact** (5 tests):
Each shape type (Sphere, Box, Capsule, Cylinder, ConvexHull) drops from z=0.3
and runs 1500 steps. Asserts: no NaN, body above ground, angular velocity < 1 rad/s,
resting height within 20mm of expected.

**Class 2 — TestMultiPointContactCount** (4 tests):
After settling, verifies contact point counts:
- Box flat: ≥ 2 points
- Box tilted 45°: ≥ 1 point
- Sphere: exactly 1 point
- ConvexHull flat: ≥ 2 points

**Class 3 — TestBodyBodyContact** (3 tests):
Two free bodies collide and settle:
- sphere-sphere side-by-side
- box-sphere (sphere dropped on box)
- box-box stacking

All 12 tests pass. Runtime ~22s (not marked slow; each test is ~1.5s).

## Files Touched

- `physics/gjk_epa.py` — `ground_contact_query` multi-point upgrade (~30 lines changed)
- `tests/integration/test_cpu_engine_shapes.py` — new file (~380 lines)

## Known Limitations

1. **gjk_distance box-cyl/box-hull early exit** (Q48 item 1): root cause not fixed.
   Workaround in `test_convex_margin.py` (pen = 3×margin) still in place. P4.

2. **CPU multi-point for body-body contacts**: `gjk_epa_query` body-body path
   already calls `build_contact_manifold` which does face clipping for polyhedral
   pairs. This was already working before this session. The gap closed here is
   specifically the ground contact path.

3. **N_SETTLE = 1500 steps**: box from z=0.35 takes ~1400 steps at DT=2e-4 to
   reach ground. Tests are not marked `@pytest.mark.slow` (each is ~1.5s). If
   the suite grows and these become a bottleneck, they can be marked slow.

4. **FlatTerrain only**: tests use FlatTerrain. HalfSpaceTerrain path goes through
   `halfspace_convex_query` (already had multi-point), not `ground_contact_query`.
   No regression risk, but HalfSpaceTerrain integration tests are not added here.

5. **Body-body contact count not validated**: Class 3 only checks no-NaN and
   above-ground. It does not assert contact point counts for body-body pairs.

## Commit

Not yet committed — pending this review.
