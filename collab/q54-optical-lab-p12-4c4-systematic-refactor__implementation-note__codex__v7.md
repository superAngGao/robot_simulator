Initiative: Q54 / P12.4 / C4
Stage: R1 dynamic composition, work in progress
Author: Codex, implementation session
Version: v7
Date: 2026-10-10
Status: draft
Related Files: runtime-integration design-request/review v3 section6; systematic-refactor gap-review v1; implementation-note v6
Owner Summary: R1 work has begun by reusing existing scene/refit kernels and the renderer's guarded recording helper. This is an incomplete working change, not a new acceptance checkpoint or R1 approval.

The isolated c4_dynamic_graph branch starts at c3472f22f1d52bfb6c853bf95cef4040cf666316.
The M1 candidate in c4_setup_orchestration remains frozen for independent review.

WarpSceneUpdateCommandFactory.seal and WarpBvhRefitCommandFactory.seal now use
the existing _warp_command_recording core. Kernel operands, dispatch order and
evidence construction remain in their original factories. Each returned command
is retained before the next fallible operation. An internal optional context
connects the same construction body to the captured worker's original guard.
No second scene/refit executor or slot state machine is introduced.

Refit factory creation uses the existing native resource scope for its topology
ready-event dependency. A distinct slot BVH's ready-event dependency uses the
same scope before recording; registered captured calls check original ownership
and target. Legacy calls retain their signatures with an optional internal
context. Actual GPU verification of these newly scoped boundaries is pending.

CPU return-double tests exercise the actual scene/refit seal methods and actual
guard/custody code: second command failure, returned metadata/adoption failure,
first-command retention before the next leaf, original exception propagation,
scope restoration and quarantine without extra sync. Geometry/storage validators
are doubled in these tests, so they do not prove native geometry or ABI behavior.
Additional dependency tests cover wait-before-record, failed wait, foreign event
target and missing ownership. The twelve new checks pass in1.07s. The earlier
existing render-preparation/observer suites plus eight scene/refit probes passed
102 checks in36.18s. Native-command/device-scene contract checks also passed.
New test file Ruff and git diff --check pass.

Four pre-existing observer tests fail at the M1 candidate and its base because
their manually created SourceSession fixture lacks fixed_row. The independent
review reproduced that old failure. This working branch initializes fixed_row
to None in the fixture, matching its unadmitted-row setup; all four now pass.
Production does not gain a fallback for missing session fields.

All90 prospective LF source bodies pass the existing AST policy. Current draft
catalog SHA256:f41a8c69954859c7cfcaf73a9ee64da764f699a2b7e445ba31cf809ea1bb5db3.

Still required in the SAME coherent R1 implementation: guarded initial Warp
topology construction/upload using the existing median builder, independently
owned mutable slot world/bounds/render storage; CUDA existing world/LBVH rebuild
composition; exact replay/camera frame binding; shared five-phase setup/final
execution and C3/arena plans; ten observed warmups/final barrier; native and
numerical evidence and a fresh independent review. This partial change does not
open product PREPARE, measured submission, C2f or any performance/selection gate.
The full C4 goal and final systematic refactor review remain outstanding.

## Continued R1 storage and warmup composition (not yet accepted)

The original guarded input readback loop is now shared with initial topology
readback; the existing Warp staging/allocation/upload body is extracted once
and reused by a2 and derived tree arrays. The existing median BVH builder has
an internal guarded-context path. Captured metadata uses the original immutable
prepared scene allocations; legacy callers still receive their original copied
metadata semantics. Both paths use the same host topology algorithm.

A named dynamic storage record owns world, slot acceleration, render outputs,
immutable Warp topology and CUDA workspace. Warp executes initial frame0 scene
commands under setup custody, observes completion, builds the original leaf4/sort
topology, and allocates distinct mutable slot bounds. This initial snapshot is
not counted among the ten complete warmups. CUDA reuses its world/LBVH/workspace
allocators. Every returned owner is held before subsequent fallible work.

Provisional frame construction selects the exact replay index, creates actual
scene-update/refit or rebuild stages and reuses the original camera/render bind.
The existing Warp recorded executor now accepts the two fixed preceding stages;
CUDA uses its original three-stage template. The same ten-warmup driver runs
both five-phase variants with one terminal event per transaction and completion
credited only after DISARMED. Final R1 graph/C3/arena adaptation is STILL PENDING;
this partial code cannot pass the existing R0-only final admission.

Following readback/upload extraction, guarded-readback/dynamic-recording/native-
command units passed48 checks in3.30s. Existing R0 warmup/preparation/final graph
regressions are being run. A fake R0 render binding needed its real protocol's
transform_frame_index=None field after explicit replay-stage validation was
added; the fixture was corrected instead of weakening production validation.
Native dynamic execution and full frozen-candidate review remain required.

The running R0 warmup/preparation/final graph selection completed216 passed in
265.57s. The last CUDA input-view helper relocation followed that run and needs
its own regression check: the two existing helpers now live with their input
classes in cuda_direct_light, and cuda_arena re-exports them. The provisional
R1 renderer can reuse them without importing the final arena before its admitted
phase. No source-admission phase was widened for this relocation.

The current draft catalog is regenerated after these changes; the earlier
catalog digest above records the initial590e333 snapshot, not later source bytes.
Real legacy Warp scene/refit regression will run from a clean remote snapshot;
it does not replace the required captured R1 worker validation.

## Verification of working production snapshot7ae02b6

Exact production revision:7ae02b60659075470c21907cf950f52a0723d0ec.
Remote clean detached worktree:/data2/shared/ang.gao/robot_sim_dynamic_graph_7ae02b6.
The primary robot_sim interpreter ran the existing
 tests/gpu/test_optical_warp_native_commands.py
with -B -m pytest -q -rs -p no:cacheprovider and a dedicated scratch basetemp:
9 passed in16.10s, no skips. CUDA_VISIBLE_DEVICES=1 confines these legacy tests
to physical GPU1 (seen as logical0); this is NOT target1/ambient0 scope proof.
The suite covers actual canonical scene update/refit, fresh bounds/traversal,
slot bounds ownership, alias/stream drift, command reordering, empty triangles
and noncanonical topology rejection. It exercises the legacy builder interface,
not the new captured dynamic storage/warmup session.

Raw directory c4_assets/c4-dynamic-legacy-gpu-7ae02b6, mirrored from primary
/data2/shared/ang.gao/c4-dynamic-legacy-gpu-7ae02b6, contains request/result,
GPU UUID mapping and pytest log. Log SHA256
84c1baed5cf2e2469cc09b12487f7567b2d2a4ecbbcf8b3dfb244fc6312111b0;
result SHA256 e1a2c68b6395c549f2b43e34b1cc247db11b418f5fdf042c2f7cd4e0242dc42e.
Wrapper922815 and pytest922822 were observed absent after tool44914 returned0.
No other GPU processes were signaled or reset. The remote candidate stayed clean.

Fresh local CUDA arena contract/resolution, guarded readback and dynamic command
checks passed75 in6.28s after the input-view helper relocation. Native-upload core
and guarded binding initially passed88 with one AST-comparison failure because
the test assumed NumPy's import remained in the outer upload function. The test
now inlines the exact shared helper body, with strict argument/call/return shape
checks, before comparing all original allocation/copy/adoption AST nodes. It
does not exempt the helper's operations. That unchanged-behavior assertion and
binding suites now pass89 in5.23s. The repair changes tests only; production
bytes remain7ae02b6. Current production catalog SHA256
 d365c2e54c3d29534bc8c79c87c12ce9c66fdfef2caf844f10f0944b039512bd.

Next work is the R1 final factory/C3/arena composition and exact row admission,
then full captured native R1 success/failure/numerical verification and fresh
independent review as one coherent packet. C2f and subsequent full C4 acceptance
remain outstanding; do not report these legacy GPU tests as that acceptance.

## Continued final dynamic composition (working candidate)

Final row admission now accepts the frozen mandatory R0/R1 rows only. The same
final builder constructs fresh scene/acceleration/render factories, chooses
replay indices separately from camera sample indices, and seals the existing
CUDA or Warp arena. CUDA retains its existing three-stage transaction; Warp
uses the same recorded executor and NativeOpticalArena FIFO/lease controller.
World arrays and refit bounds are slot resources; immutable Warp topology and
prepared replay arrays are session resources. Both C3 adapters share a small
logical-input selector in the existing provenance module, whose validator still
independently checks the complete ordered arrows.

The shared untimed Warp validator checks actual transform roots, view shape,
strides, frame byte offset and env index against the original prepared replay.
It checks scene/refit/render physical dataflow at transaction construction and
final graph validation; acquisition remains the existing fixed-cost path.
The native observer/acceptance driver is being generalized in place to cover
dynamic storage and five phases; it is not yet verified on the real worker.

R0 arena/final graph regressions: 87 passed in187.28s on local Python3.14.
That run preceded the final repeated dataflow checks at final barrier/resolution;
those changes require the next regression run. All90 prospective LF sources
pass the existing AST policy; source catalog b9db06834328433d24ee67b697acb0a2d68d2c736e12c2c0a2eb2f9316a39dca.
Native R1 runs and numerical/failure verification plus independent review remain
pending. This is still the same unaccepted R1 packet, not R1/C4 acceptance.
