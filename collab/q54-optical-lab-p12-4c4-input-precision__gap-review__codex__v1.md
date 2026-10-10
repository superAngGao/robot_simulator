Initiative: Q54 / P12.4 / C4 input and intersection precision  
Stage: fresh independent read-only code/gap review  
Author: Codex independent reviewer (`/root`)  
Version: v1  
Date: 2026-10-10  
Status: in_review  
Reviewer session: `01a1234e-14f7-7e00-891f-4e4bb8f5eb28`  
Related Files: input-precision design-request/design-review v4; implementation-note v1–v3; sources and tests identified below  
Owner Summary: Three concrete defects were established: a CUDA R1 ABI validation failure, incorrect grazing-hit normal orientation, and a Warp-specific slab tolerance that changes CPU/CUDA hit semantics.

**Verdict: Request changes — bounded precision implementation gap review.**

This verdict concerns the inspected implementation and counterexamples below. It is **not whole-amendment approval, numerical matrix acceptance, or C4 acceptance**. The already acknowledged unfinished stages are not presented as newly discovered findings.

I did not author this candidate, inherit an implementation conversation, spawn agents, edit source, or launch GPU work.

**Identity and repository state**

| Item | Observed result |
|---|---|
| Repository inspected | `C:/Users/AG/新建文件夹/c4_input_precision` |
| Initial and final HEAD | `ea15e152d0a40674ba2b711e3d481dae32a403b6` |
| Branch | `implementation/c4-input-precision` |
| Implementation base and observed merge-base | `8b6cb2615ae05c17a4c16b55201d5b398b638933` |
| Tracked working tree and index | Clean initially and finally |
| Full status limitation | No entries, but `.pytest_cache/` could not be inspected because access was denied |
| Base-to-candidate whitespace check | Passed |
| Configured `origin` | Local repository `C:/Users/AG/新建文件夹/robot_sim_review_workflow`; GitHub remote identity was not independently queried |
| Remote candidate/oracle state | Unverified: all SSH attempts failed before remote commands executed |

I fully read `AGENTS.md`, `collab/README.md`, `collab/REVIEW_PRINCIPLES.md`, design-request v4 and design-review v4. Implementation-note v1–v3 were read as author claims.

Git-blob SHA256 values match the reviewed design identities:

- Design-request v4: `df1035b0213fe0c48110439222c014f1a81af86f50996254d88634e520ecb9df`
- Design-review v4: `be13a105ddc8ff40c3e12a09986004a69b80706d901c66a436a5c7a0007018a5`

Worktree document bytes match those blobs after CRLF-to-LF normalization.

**F1 — High / P1: correctly constructed CUDA triangle-update stages fail ABI validation**

Location: [optics/cuda_scene.py:629](/C:/Users/AG/新建文件夹/c4_input_precision/optics/cuda_scene.py:629). Transaction validation invokes this path through [optics/cuda_timeline.py:507](/C:/Users/AG/新建文件夹/c4_input_precision/optics/cuda_timeline.py:507).

The indexed migration added `triangle_vertex_indices` after `geometry_vertices_local`. Consequently, `_triangle_abi_arguments()` now places:

```text
6: transforms.x_world_R
7: transforms.x_world_r
8: transforms.env_idx
```

However, `_validate_triangle_abi_binding()` still passes `scalar_index=7`. `_fixed_abi_arguments_match()` requires that argument to have exact Python type `int`. A valid translation tensor therefore fails even when the actual tuple is exactly the tuple produced by the current argument builder.

**Trigger:** validating a CUDA scene-update stage with any triangles.

**Consequence:** transaction preparation rejects valid R1 scene updates before dispatch, preventing the associated scene-update/LBVH-rebuild/render transaction. This does not affect the fixed R0 rendering path that bypasses scene updates.

**Fresh reproduction:** I called the actual argument builder and validator with identity-preserving placeholder objects. No Torch or CUDA execution was needed:

```python
from types import SimpleNamespace as S
from optics.cuda_scene import (
    _triangle_abi_arguments,
    _validate_triangle_abi_binding,
    _fixed_abi_arguments_match,
)

names = [
    "geometry_vertices_local", "triangle_vertex_indices",
    "triangle_instance_index", "instance_body_index",
    "instance_X_body_geometry_R", "instance_X_body_geometry_r",
]
scene = S(arrays={n: object() for n in names}, num_triangles=1)
transforms = S(x_world_R=object(), x_world_r=object(), env_idx=0)
outputs = S(**{n: object() for n in [
    "triangle_v0_world", "triangle_e1_world", "triangle_e2_world",
    "triangle_normal_world", "triangle_aabb_min", "triangle_aabb_max",
]})
args = _triangle_abi_arguments(scene, transforms, outputs)
stage = S(
    factory=S(local_scene=scene), transforms=transforms,
    outputs=outputs, _triangle_args=args,
)
```

Observed:

```text
_fixed_abi_arguments_match(args, args, scalar_index=7): False
_fixed_abi_arguments_match(args, args, scalar_index=8): True
_validate_triangle_abi_binding(stage):
ValueError: CUDA triangle scene-update ABI binding changed
```

**Why existing checks miss it:** the inspected CPU CUDA-scene tests check source structure, allocation restrictions and evidence fields. They do not round-trip the current triangle argument builder through its validator. Successful CUDA compilation and fixed R0 rendering also cannot exercise this Python R1 validation failure.

**Least correction:** change the triangle validator’s scalar position to 8. Preserve tensor identity checks and the plane validator’s existing position 8.

**Proof needed:** a positive builder-to-validator regression; negative substitutions at translation and scalar positions; then a real triangle-containing R1 transaction through scene update, LBVH rebuild and rendering.

**F2 — Medium / P2: float32 normal rounding can reverse the facing decision**

Locations:

- [optics/warp_native_render.py:935](/C:/Users/AG/新建文件夹/c4_input_precision/optics/warp_native_render.py:935)
- [optics/_cuda_direct_light_source.py:396](/C:/Users/AG/新建文件夹/c4_input_precision/optics/_cuda_direct_light_source.py:396)

The common triangle helpers correctly compute binary64 edges, area, determinant and distance. They then determine whether to flip the normal using the stored **float32 shading normal**, widened back to double.

At grazing incidence, rounding can change the sign of `normal · direction`. The original CPU helper determines that sign from the binary64 geometric normal. The result is an approximately 180-degree normal reversal, not a small output-rounding error.

**Trigger:** an admitted triangle hit for which the original and rounded normals have opposite dot-product signs against the ray.

**Fresh reproduction:** use one identity-transformed triangle:

```python
vertices = np.array([
    [0., 0., 0.],
    [-0.123456789, 1., 0.],
    [-0.3, 0., 1.],
])
origin = np.array([
    -0.26367883497322825,
    1.3257985665584018,
    0.333333333012683,
])
direction = np.array([
    0.12252657197322826,
    -0.9924652332250684,
    3.2065034296667426e-10,
])
max_distance = 2.
```

I built its query arrays through the actual indexed packer and `_derive_fixed_fixture()`, then evaluated the original CPU helper and current Warp helper body using the repository’s AST-based CPU harness.

Observed:

```text
binary64 normal · direction:  +1.1236707172973052e-09
rounded normal · direction:   -1.1236707189160028e-09

CPU and Warp helper distance: 0.9999999882523075

CPU normal:
[-0.95119902, -0.11743198, -0.28535971]

Warp helper normal:
[+0.9511989951133728, +0.11743197590112686, +0.2853597104549408]

normal L2 error: 1.99999998080983
```

I additionally evaluated the actual four Warp primary kernel bodies with this same input:

| Kernel body | Hit | Returned float32 normal orientation |
|---|---:|---|
| Brute force | Yes | Correct, negative |
| Device scene scan | Yes | Incorrect, positive |
| Device scene with AABB | Yes | Incorrect, positive |
| Device BVH | Yes | Incorrect, positive |

These were Python-body executions, not Warp JIT executions. CUDA has the same rounding-before-facing logic by source inspection; CUDA execution was unavailable.

**Consequence:** incorrect `normal_world`, with potential large RGB/intensity changes and shadow-origin displacement to the opposite side of the surface.

The affected compatibility executors are:

- `GpuDeviceSceneOpticalExecutor`
- `GpuDeviceBvhOpticalExecutor`
- `GpuDeviceBvhDirectLightOpticalExecutor`
- `CudaDeviceBvhOpticalExecutor`
- `CudaDeviceBvhDirectLightOpticalExecutor`

`GpuBruteForceOpticalExecutor` derives and orients its normal in double before output conversion and passes this counterexample.

**Why existing checks miss it:** area, nearest-hit and endpoint tests do not establish facing correctness. Some predicate tests supply zero normals and inspect only hit/distance. The four supplied R0 comparisons do not contain this synthetic counterexample.

**Least correction:** decide orientation from the existing binary64 world-edge cross product and its normalized geometric normal. Apply that sign to the retained float32 output/shading normal. No new normal buffer, shading precision expansion or ABI change is necessary.

**Proof needed:** both ray directions and triangle windings; consistent normals across all six executors; native execution in both lanes; a directional-light case demonstrating preserved RGB/intensity and shadow-side behavior. Keep existing CPU tolerances unchanged.

**F3 — Medium / P2: Warp retains an extra parallel-slab tolerance absent from CPU and CUDA**

Location: [optics/warp_native_render.py:801](/C:/Users/AG/新建文件夹/c4_input_precision/optics/warp_native_render.py:801).

When `abs(direction) <= DIR_EPS`, Warp tests the origin against:

```python
lower - TIE_EPS
upper + TIE_EPS
```

The unchanged CPU slab helper and current CUDA helper use the bounds themselves. Thus the nearest-hit tie epsilon also acts as a Warp-only geometric slab expansion.

**Trigger:** a nearly parallel ray starts just outside an exactly representable slab, and the triangle predicate accepts a subsequent intersection.

**Fresh reproduction:**

```python
vertices = np.array([
    [0., -1., -1.],
    [0.,  3., -1.],
    [0., -1.,  3.],
])
origin = np.array([-5e-13, 0., -1.])
direction = np.array([5e-13, 0., 1.])
max_distance = 2.
```

The actual outward-bounds helper produces:

```text
lower = [0, -1, -1]
upper = [0,  3,  3]
```

Observed from the original CPU and current Warp bodies:

```text
CPU triangle predicate: hit, t=1
CPU AABB predicate:     miss
Warp AABB predicate:    hit
Warp one-leaf BVH:      hit=1, instance=17, range=1
```

The CPU BVH therefore rejects this primitive at its root. CUDA’s exact parallel-slab comparison likewise rejects it by source inspection. Warp traverses and reports the hit.

This deliberately distinguishes triangle intersection from the original **CPU BVH policy**: the requested contract preserves that existing CPU policy.

**Consequence:** lane-dependent hit/ID results, and potentially different shadow occlusion because the slab helper is shared by primary and shadow traversal.

**Why existing checks miss it:** ordinary parallel rays, exact-boundary rays and nonparallel traversal cases do not exercise an origin outside the slab but inside the extra tie window. This is retained behavior that the precision migration has not reconciled with the original CPU policy.

**Least correction:** remove the `± _QUERY_TIE_EPS` expansion from the parallel-axis branch, matching CPU and CUDA. Keep nearest-hit tie handling separate.

**Proof needed:** the counterexample above in actual Warp/CUDA primary and shadow traversal, plus exact-boundary inclusion and strict physical-limit regressions.

**Inspection coverage**

I inspected packing, indexed validation, fixed materialization, native scene-update arithmetic and argument flow, world-buffer allocation, refit/rebuild interfaces, query helpers, camera/ray uploads, and all six compatibility executors.

Specific observations beyond the findings:

- R0 transforms complete source mesh blocks before gathering indexed faces; world edges use the original per-face operations.
- Both R1 source implementations widen float32 producer transforms before composition and gather original indexed vertices before computing world edges.
- Scene and renderer contraction settings are disabled in the inspected Warp options and CUDA recipes.
- World query geometry and rays are f64; shading normals, output channels and acceleration bounds retain their intended f32 layouts.
- Warp refit consumes f32 primitive bounds and unions them through the existing topology. CUDA LBVH validation likewise consumes the unchanged f32 bounds interface. I established no additional dtype mismatch at those interfaces.
- The brute-force compatibility upload preserves f64 world vertices and retains degenerate source faces for query rejection.
- Device-scene and BVH compatibility argument lists use current geometry/ray layouts. CUDA snapshot conversion preserves the incoming query-buffer types.
- The brute-force kernel retains its existing argument order without the superseded extra-normal argument.
- Git reports no changes to `optics/execution.py`, `optics/scene.py` or `physics/spatial.py` between original oracle revision `812ca29030a45d802addf10ca5811a4354ce0eb9` and the candidate.

These observations are not native runtime approval.

**Executed verification and results**

Environment: Windows PowerShell; `C:/Python314/python.exe`; Python `3.14.0`; NumPy `2.5.2`. Python commands used `-B`, with `PYTHONDONTWRITEBYTECODE=1`; test/probe execution used an empty `CUDA_VISIBLE_DEVICES`. Pytest cache writing was disabled.

Repository checks included:

```text
git rev-parse HEAD
git branch --show-current
git status --porcelain=v1
git status --porcelain=v1 --untracked-files=no
git merge-base 8b6cb2615ae05c17a4c16b55201d5b398b638933 HEAD
git diff --exit-code
git diff --cached --exit-code
git diff --check 8b6cb2615ae05c17a4c16b55201d5b398b638933 HEAD
git diff --name-only 3c1a08fd0495b6bba30d3c1666aac88faa71327b HEAD
```

I used `rg`, bounded UTF-8 `Get-Content` reads, Git diffs/blobs, and in-memory Python probes. All **50 changed Python files parsed successfully**.

The initial test command was:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider
  tests/unit/optics/test_source_geometry_precision.py
  tests/unit/optics/test_native_scalar_precision.py
  tests/unit/optics/test_native_intersection_precision.py
  tests/unit/optics/test_indexed_scene_kernels.py
  tests/unit/optics/test_double_query_kernels.py
```

Result: **89 passed, 60 failed, 2 skipped, 1 error in 13.24s**.

The failures/error came from the CUDA host-C++ fixture’s inability to access `C:/Users/AG/AppData/Local/Temp/pytest-of-AG`. They are environment failures, not 61 candidate defects. The two Warp-dependent modules skipped because Warp is unavailable. Host C++ compilation was not reached.

A subsequent command restricted execution to tests not requiring that CUDA fixture:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider
  tests/unit/optics/test_source_geometry_precision.py
  tests/unit/optics/test_native_scalar_precision.py
  tests/unit/optics/test_native_intersection_precision.py
  tests/unit/optics/test_scene_contract.py
  tests/unit/optics/test_cuda_scene_native.py
  tests/unit/optics/test_cuda_lbvh_native.py
  tests/unit/optics/test_device_scene.py
  -k "not cuda" --tb=short
```

Result: **86 passed, 79 deselected in 3.79s**. The name filter also excludes parameter cases whose IDs contain `cuda`; it is not a complete Warp-case run.

Separately:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider
  tests/unit/optics/test_cuda_scene_native.py
  tests/unit/optics/test_cuda_lbvh_native.py
  tests/unit/optics/test_native_scalar_precision.py
  tests/unit/optics/test_cuda_extension_recipes.py
  --tb=short
```

Result: **30 passed in 0.90s**. These counts overlap earlier execution and must not be added as unique coverage.

Fresh in-memory probes independently reproduced F1, F2 and F3. F2 additionally exercised all four Warp primary kernel bodies. No production helper was patched for these probes.

SSH attempts:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 tileops-dev ...
ssh -o BatchMode=yes -o ConnectTimeout=10 -p 40031 ang.gao@183.166.183.33 ...
ssh -o BatchMode=yes -o ConnectTimeout=10 ang.gao@192.168.80.101 ...
```

Results respectively: alias resolution failure; connection permission denied; connection permission denied. The intended remote commands only inspected revisions, branches and status, with CUDA hidden. None executed remotely, and no host/worktree switch occurred.

**Existing R0 evidence**

I independently checked the supplied archive:

```text
219374896 bytes
185 members, all unique
SHA256:
c4bcd8cb18e96ee65829fefd0d24e517d0b5de845d58720c0966d237bdf390a9
```

Its four comparison JSON records report all ten channels with zero failed pixels at both resolutions and both lanes.

The complete Git filename delta from executed revision `3c1a08fd0495b6bba30d3c1666aac88faa71327b` to the candidate contains documentation, cleanup-ledger, test and fixture changes, consistent with unchanged executed production source.

This was an archive/provenance inspection, **not a fresh rerender or independent recomputation of every pixel comparison**. The supplied R0 evidence is not downgraded to synthetic evidence, and it is not used as R1 evidence.

**Limitations and disposition**

Actual Warp JIT, CUDA execution, remote source-state inspection and full Go2 reconstruction were unavailable in this session. Python-body probes establish the inspected arithmetic and control-flow counterexamples; they do not establish compiled device behavior. CUDA F2/F3 conclusions additionally rely on the directly corresponding source logic.

The known remaining R1 matrix, profile/frame/failure coverage, full-amendment review and whole-C4 integration/review remain outstanding. This report adds three actionable code findings rather than restating those stages.

The least repair scope is: correct the CUDA scalar index, preserve binary64 normal-facing decisions, and align Warp’s parallel-slab comparison with the original CPU/CUDA rule. Repairs require updated source identities where applicable and fresh independent review of the new revision.

The candidate remains frozen and unchanged. No report file or implementation file was written; no agents or GPU jobs were launched. All invoked tool processes have returned.