# Indexed geometry and query implementation checkpoint

Initiative: Q54 / P12.4 / C4
Stage: implementation checkpoint; amendment incomplete
Author: codex
Version: v1
Date: 2026-10-10
Status: draft
Related Files: input-precision design-request/design-review v4; full-cpu-frame gap-review v1; systematic-refactor gap-review v1
Owner Summary: Actual R0 source preservation is implemented and verified over the complete original Go2 mesh. Both scene-update sources and Warp queries have migrated. Actual Warp kernels on CPU resolve the selected old failures. CUDA rendering, binding/schema migration, native GPU validation and independent code review remain outstanding; this is not amendment or C4 acceptance.

## Scope and source identity

Approved design candidate: `056b40c3a365e8a6b4960ac2ef1dced4732994de`.
Its full fresh CLI review is archived unchanged at
`collab/q54-optical-lab-p12-4c4-input-precision__design-review__codex__v4.md`,
SHA256 `be13a105ddc8ff40c3e12a09986004a69b80706d901c66a436a5c7a0007018a5`.
Verdict: Approve with comments, design only. The review's full-Go2 environment
limitation remains explicit; it did not approve implementation or native results.

Implementation base: `8b6cb2615ae05c17a4c16b55201d5b398b638933`.
Current code checkpoint: `c055303397112770ecb8e85c5f6b40142a238414`, branch
`implementation/c4-input-precision`, local worktree `c4_input_precision`.
Original runtime/oracle checkout remains `812ca29030a45d802addf10ca5811a4354ce0eb9`.
No original CPU execution or scene-transform algorithm was changed.

Superseded v2 unfinished work was preserved in
`c4_assets/c4_input_precision_v2_unfinished_snapshot.zip`, SHA256
`8238a70a611eaa5c9ee4a68937f635c2c5de0d95ca87d99e37d472a457b90b0c`.
The proposed local-normal array and its test were replaced, not retained as a
second geometry authority.

## Implemented behavior

- Pack complete shared binary64 mesh blocks and int64 face indices/offsets;
  retain unused vertices, original source keys and all source faces. Check raw
  index types/ranges before integer conversion and reject nonfinite source data.
- Derive R0 through original whole-mesh CPU composition/transformation, original
  per-face edge/area operations, zero normals for degenerate rows, and minimal
  outward float32 bounds. Geometry is binary64; shading normals remain float32.
- Migrate Warp and CUDA scene-update inputs, allocations and wrappers to indexed
  vertices and binary64 world geometry, retaining float32 producer transforms.
  Warp's repeated scene arithmetic is consolidated into shared native functions.
  Scene contraction is disabled; CUDA scene recipe is now v3.
- Migrate Warp primary queries, camera rays, traversal distances and comparator
  to binary64 and original CPU thresholds. Remove downward float32 entry repair.
  Compatibility scene scans call the same triangle helper. Warp compatibility
  uploads preserve binary64 rays/geometry and retain source faces.
- Retain float32 shading/output boundaries and widen constructed shadow rays at
  the query helper. Shading's existing direction threshold remains separate.
- Compiler inspection exposed two implicit narrowing hazards: Python integer
  max-key literals became int32 before widening, and Python epsilon globals
  became float32 before widening. Typed Warp constants now preserve exact int64
  sentinel and binary64 policy values. The strict-endpoint and one-ULP area tests
  exercise these issues through actual compiled kernels.

The single numerical-policy module remains backend-free. NumPy is imported only
inside the host bounds helper, so the existing isolated CUDA recipe builder can
read policy constants before enabling third-party dependency imports.

## Verification performed

Local Python 3.14 CPU suite, exit 0: **30 passed** across source geometry,
scene contract, registry builder, device scene, CUDA scene wrapper and recipe
tests. Changed Python sources parse, and `git diff --check` passes.
The recipe test's old renderer hash already failed on the unchanged 812ca29
checkout; that failure was reproduced before updating the expectation to its
actual unchanged renderer bytes. This does not claim renderer numerical approval.

Remote `robot_sim` Python, with `CUDA_VISIBLE_DEVICES=`:

```text
python -B -m pytest tests/unit/optics/test_double_query_kernels.py tests/unit/optics/test_indexed_scene_kernels.py -q
```

**31 passed**, exit 0. These execute actual compiled Warp functions/kernels on
CPU: four query entry points, both visit orders, separate BVH leaves, distances
that collapse in float32, CPU tie behavior, strict physical endpoint, exact
binary64 area boundary, native camera generation, both scene-update forms and
plane update. Warp CPU compilation is not Warp GPU compilation or execution.

`verify_indexed_go2_implementation_v1.py` executed the actual
`CanonicalFixedRenderFixtureHost.from_scene_contract` route. Latest result
`c4-indexed-go2-implementation-v2.json`, SHA256
`5a59d2c7f59b5cc74263182627dedeaf4c3a02bb3d4194e2387c18746ecb212e`:

- 191,699 original source vertices; 398,432 retained source faces.
- Every world v0, edge and output normal matches the original CPU operations;
  396,778 faces meet the original CPU world-area admission rule.
- All bounds enclose the original binary64 vertices; topology covers all rows.
- All 20 prior design-review crossing keys reproduce exact original area and
  centroid-ray outcomes through actual materialization.

These probes use diagnostic fixed-transform ID `original_go2_static`. Their
fixture digests are **not** regenerated frozen workload identities. The outer
`build_canonical_fixed_fixture` digest gate has not been migrated or passed.

`verify_warp_go2_queries_cpu_v1.py` executes production Warp camera, BVH first-hit
and shadowed primary-shading kernels on CPU. It uses the frozen frame-10 camera
and original source scene, comparing saved original CPU channels at the union
of the first recorded ID/material/range/normal/RGB failures:

| Resolution | Selected pixels | Failing channels/pixels |
| --- | ---: | --- |
| 1920x1080 | 149 | Zero across all ten channels |
| 960x640 | 130 | Zero across all ten channels |

Channels include actual binary64 origins/directions, hit, range, position,
normal, instance/material IDs, RGB and intensity. Both traversal overflow
counters are zero. The latest run is `c4-warp-go2-queries-cpu-v2.json`, after the
typed-constant correction. All executions are fresh; its module uses the normal
cache from the preceding fresh JIT tests. This is **selected-pixel CPU-target
execution**, not full-frame or native GPU correctness acceptance.

CUDA scene v3 compiled successfully using the existing isolated systemd build
service and fixed toolchain, `--fmad=false`, no output import/publication:

- Job: `/data2/shared/ang.gao/c4-indexed-cuda-scene-build-v1`.
- Unit: `robot-sim-cuda-040553c8199d4fba9bdc6482a864bd5c-0.service`.
- Invocation: `5e90d5b9367d40389cbc42afe4b36692`.
- Main termination: success/exited/0; waiter exit 0; exact cgroup absent.
- Generated sources, build log, binary digest, termination and source checksums
  are retained. No CUDA rendering, GPU launch or CUDA output correctness was tested.

Raw scripts/results and build job are in local `c4_assets` and remote
`/data2/shared/ang.gao`. Initial archive
`c4-indexed-geometry-implementation-evidence-v1.tar.gz` SHA256
`b469691a315046f99a09aaf64f0fd805353a15933f5b27368a944280ed8f3881`.
Supplemented archive `c4-indexed-geometry-implementation-evidence-v2.tar.gz`
includes the initial archive unchanged plus the post-correction query result;
SHA256 `aa60d7deb11420fae95f6559c3de8ce77002139e4bbe3988aa42798246ac712c`.
All invoked sessions terminated. Owned probe/build PIDs 3779091, 3791229,
3791233, 3795130, 3798273, 3800707 and 3806133 were subsequently absent.

## Required continuation

This checkpoint is intentionally incomplete and not ready for whole-amendment
code approval. Complete the already approved v4 change before its fresh review:

1. CUDA renderer/query/camera migration and both CUDA compatibility paths.
2. Bound scalar split, asymmetric camera identity, actual f8 readback and all
   typed launch/storage consumers; regenerate frozen scene/fixture/manifests,
   gate/result/row/C3 identities and captured source/function pins. Old source
   catalog/policy pins are currently stale; captured preparation remains closed.
3. Original captured pixel regressions, new-policy test expectations, negative
   ABI/identity cases and affected setup/resource-failure checks.
4. Fresh Warp GPU JIT, complete fixed CUDA build/deployment, supervised real R0/R1
   runs and original CPU comparisons over the entire required matrix, including
   intensity, shadows, diagnostics, profiles, frames and sentinels.
5. Fresh independent review of the coherent amendment and its native evidence.

Then continue the final graph/arena/barrier integration, remaining lifecycle
matrices, C4c3/c4, C4d/B3, systematic whole-C4 refactor and independent whole
review. No speed, result, baseline, activation, shipping, backend selection or
whole-C4 completion is established by this checkpoint.
