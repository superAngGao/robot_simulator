Initiative: Q54 / P12.4 / C4
Stage: R1 dynamic composition, work in progress
Author: Codex, implementation session
Version: v7
Date: 2026-10-10
Status: draft
Related Files: runtime-integration design-request/review v3 section6; systematic-refactor gap-review v1; implementation-note v6
Owner Summary: R1 work has begun by reusing existing scene/refit kernels and the renderer's guarded recording helper. This is an incomplete working change, not a new acceptance checkpoint or R1 approval.

The isolated c4_dynamic_graph branch starts at c3472f22f1d52bfb6c853bf95cef4040cf666316.
The M1 candidate c3472f2 has since received independent approval; its unchanged report is archived in code-review v5.

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

## First captured R1 primary success and raw replay

Production source revision626f6320eda3bbfb527be49390c696012873e33d now includes
final R1 composition. Subsequent3d59c2d/fea529c/0f46f64 commits changed only tests
and driver observation. Production catalog remains
b9db06834328433d24ee67b697acb0a2d68d2c736e12c2c0a2eb2f9316a39dca.
No independent R1 review has been requested yet; the full packet is unfinished.

Actual primary server, target physicalGPU1, all GPUs visible:
- CUDA atfea529cb360e55b2f4facf7a087575c823b7ef23: original driver exit0;
  worker988359/supervisor988351/parent987629 are absent. Twenty-three normal
  completed guards, twenty-four ARM operations; ten real three-stage warmups,
  three final C3 plans, one final barrier, acquire/cancel, no final submissions.
- Warp at0f46f64309f649c3daec60a2f1a0a6113cddcc5e: worker998094 exit0,
  supervisor998086/parent997982 absent. Twenty-seven normal completed guards,
  twenty-eight ARM operations; ten actual five-phase warmups, three final plans,
  one barrier, acquire/cancel, no final submissions. Original parent validator
  exited1 on an incorrect allocation-count expectation; that failure is retained,
  not relabeled as an original driver success.
Both workers ended cancelled_before_submission_exit/release completed, without
emergency synchronization, expiry or kill. Warp restores default0/no current
context; CUDA restores logical0/raw1, not a claim of raw0 restoration.
The frozen row is r1_dynamic_full_shadow, synthetic_dynamic_rigid_960x640_v1.

The raw validator now accounts for the existing nine derived topology uploads
(24 slot allocations plus9), the frozen scene's one triangle/no planes (one
scene launch), and the existing refit_topology_ready_wait before the single
slot_event_initialization_wait. None changes production execution. The input
transfer observer had wrongly applied its a2 ambient-scope receipt assumptions
to reused helpers inside the later target scope; its later-helper observations
now explicitly verify unchanged caller scope, exact stream and target1. The
original failure traceback is preserved. Another earlier driver error passed a
Go2 asset path to the synthetic case; the driver now supplies assets only for
Go2. Three earlier failed worker runs are archived with their original records.

A read-only replay on the primary Linux host ran production custody-artifact,
runtime-custody selection/attachment, storage, graph and warmup validators over
both completed runs and checked final graph/barrier hashes and all six terminal
PIDs. Both passed. This is author verification, not independent review.
Replay report: dynamic_primary_replay_v1.json; validator source SHA256
14c9adca9c8138543a6d4405b913a7000f1d8fb7218bd237d4e0ae94ca38fd24.

The final warmup frame10 RGB images are byte-identical, SHA256
26d00b19c615e79e6d954c55d9550482051048f28f78cd4f72293cd178ad1aea.
An independent CPU rendering algorithm using the frozen registry, exact decoded
camera scalars and replay frame9 checked all614400 rays:6393 hits; maximum RGB
channel absolute error2.9703586035267904e-08 for each lane; zero pixels over1e-6.
This is full-frame RGB for one setup warmup, not all-channel/all-frame numeric
acceptance or measured final sample evidence. CPU script and result are archived.

Local regressions after final barrier/resolution dataflow validation:120 passed
in187.86s (Warp arena/final graph and CUDA arena contract/resolution). Dynamic
recording/replay-root checks and C3 frame selection:32 passed in1.59s, including
both mandatory row profiles and camera sample indices0,119,120,121. Native
metadata is doubled in these CPU checks. Changed Warp modules and new test Ruff
checks pass; git diff --check passes. The whole repository is not claimed tested.

Archive c4_assets/c4-dynamic-primary-evidence-v1.tar.gz:1944165 bytes,
150 unique members, SHA256
4ebd2fe77850de49b437a0693b8751273a76da8c5127aa5345bf433f5dd6c302.
Includes the five original raw run directories, driver/replay scripts, the
corrected validator snapshot, CPU oracle results and per-file digests.
The corrected validator snapshot is archived; earlier validator source remains
available at each exact Git revision in the raw requests.

Still required for this SAME R1 packet: native second mandatory row, diagnostic
five-phase variant, numerical channels/frames and relevant failure cases; fresh
independent review after freezing the completed candidate. C2f native worker
submission/resolve/reclaim/close integration and full C4 remain outstanding.
No formal performance, product preparation or backend-selection gate is opened.
