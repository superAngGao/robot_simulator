# Native triangle precision and first-hit tie traversal

Initiative: Q54 / P12.4 / C4 numerical correctness
Stage: bounded implementation, pending independent code/native review
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: runtime-integration design-request/review codex v3; warmup-geometry-diagnostic native-review codex v1; optics/warp_native_render.py; optics/_cuda_direct_light_source.py
Owner Summary: Repair cancellation in triangle predicates and premature rejection of candidates covered by the existing hit tie rule. Keep the original input/output ABI, geometry, camera, epsilon and acceptance tolerances. Native correctness and performance remain unproved.

Base is 8cce978b73901eeac42a9e2fdc7e861a23a57890. Candidate lives separately
in c4_intersection_precision; original 5a5a17c native source/raw evidence and
the primary review checkout were preserved. This is a bounded numerical fix
within the approved CPU-oracle requirement, not a new renderer or authority.

## Reproduction and limits

Original real Go2 frame10 geometry archive:
`/data2/shared/ang.gao/c4-render-geometry-native-evidence-v1.tar.gz`, SHA256
`619167158ae9da9a99112a6ce22a191fb033645dbcf2bb714a4c810ae33e8f3a`.
The independent geometry native review approves only lifecycle/copy evidence.
Its report is in c4_assets/geometry_native_evidence_review_v1.md, SHA256
`467491b087a1ffb1bed59d13d0b13ccfa5ab34d39f857250235d857385d47aeb`.
Six substantial RGB differences are not all geometry differences: the reviewer
also found normal-only differences at1116499 and1116503. Both must be included
in subsequent full-array native comparisons.

CPU replay of exact Warp rays and exact encoded fixed geometry preserves the
wrong-first-hit counterexamples. Further scratch scripts
check_fixture_arithmetic_v1.py/v2.py use each lane's actual binary32 rays and
fixture8354993fb682d5a13a2dff62d1d7521afd7d3d46d20b876e15732b4c38415db0.
Their original JSONs are c4-fixture-arithmetic-warp-v1.json and
c4-fixture-arithmetic-cuda-v1.json, remote /data2/shared/ang.gao and local
c4_assets. CPU binary32 evaluation flips triangle edge predicates relative to
binary64; it does not emulate native instruction contraction or BVH traversal.

Examples using the actual lane rays:

| Lane / pixel / primitive | Binary64 | CPU binary32 |
| --- | --- | --- |
| CUDA901306 /346916 | u=0.00331935648, hit | u=-0.00026423694, miss |
| CUDA985873 /29189 | u=0.00002296862, hit | u=-0.00001737208, miss |
| Warp976396 /170564 | u+v=1.0002050698, miss | u+v=0.9997248650, hit |
| Warp1026413 /15840 | u=0.00000550803, hit | u=-0.00002660727, miss |

At1143492 two surfaces have effectively equal true distance and different
source keys/materials. Independently of the specific native pixel's exact
instruction behavior, both original primary traversals prune at best_t before
applying their existing 1e-5 tie rule. Thus a slightly farther lower-key hit
allowed by that rule can be discarded, depending on traversal order. The new
two-face tests reproduce this in both actual source bodies, with one leaf and
separate leaves and reversed storage order. No new tie epsilon is introduced.

## Implementation

The existing shared device triangle helper in each lane evaluates ray/vertex
subtraction, cross/dot products, determinant, barycentric tests and distance
in binary64. Casting precedes subtraction/multiplication. Native allocations,
parameters, normals and returned distance remain binary32. The determinant
threshold retains its previous binary32 value (including Warp's explicit
widening of that value). Strict barycentric and range tests stay strict.
Both primary and shadow callers use their original helper; no alternate
precision selector, fallback branch or second intersection implementation.

Primary BVH node/child and triangle clipping use
min(max_distance, best_t + existing_tie_epsilon), so the existing source-key
comparison receives candidates within its own declared tie window. Maximum
sensor range still caps traversal. Shadow range policy and hit comparator
are unchanged. Historical brute-force/non-BVH compatibility kernels are not
converted or claimed numerically accepted by this change.

This may increase GPU arithmetic cost, including shadow intersections. Its
performance effect must be measured with equal native-ready boundaries; it
is not assumed free. No threshold is relaxed to compensate. No new runtime
infrastructure, upload, allocation, synchronization, public gate or backend
selection was added.

## Author checks and required continuation

Local C:/Python314/python.exe, NumPy and existing g++ at
C:/msys64/ucrt64/bin/g++.exe, without installs. Command:
`python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_intersection_precision.py tests/unit/optics/test_warp_native_kernel_source.py tests/unit/optics/test_native_warmup_support.py`
completed with51 passed in24.66s, tool session19117 exit0.

The numerical tests execute the actual Warp function/kernel Python bodies
after removing decorators/type annotations, and compile the actual CUDA C++
helpers/primary kernel for a single host thread. A small host atomic/thread
stub permits scalar traversal; no copied intersection formula is tested.
Expected scalar intersections come from the existing CPU oracle on encoded
vertices. The captured fixture is a regression supplement, never a substitute
for real full-Go2/native validation. Host execution does not prove Warp JIT,
CUDA device lowering, FMA behavior, GPU ordering or device performance.

The same38 numerical tests against original8cce978 source in exclusive scratch
c4_assets/intersection_before_v1 produce28 failures and10 passes (exit1),
including false hits/misses and six order-dependent tie traversal failures.
Both source languages are exercised. This confirms the regressions reject
the original code, rather than merely accepting this implementation.

Native compilation, newly built fixed CUDA artifacts and source-frozen real
960/1920 geometry/RGB replays remain required. Check the full geometry arrays,
the two extra normal pixels, original CPU diagnostics, misses, all affected
profiles and shadows; preserve any residual discrepancy rather than claim
full correctness from these CPU tests. Timing collection is still pending;
old warmup files contain no resolved render duration and supervisor elapsed
time includes audit/setup overhead. Whole C4 integration/fault matrices,
systematic refactor, fresh whole review and later gates remain unfinished.

All author CPU script/test/scp/archive handles completed. No local Python,
pytest or g++ process remained at the scoped end check. No GPU job was
launched for this candidate yet; old two-case native PIDs were absent.
