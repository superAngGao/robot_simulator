# C4c2: closed-worker integration, Warp timeline and R0/R1 arena

Initiative: q54-optical-lab-p12-4c4c2-runtime-integration
Stage: design-request
Author: codex
Version: v3
Date: 2026-09-30
Status: in_review
Related Files: C4c parent design v7; C4c2a design v1; C4c2b design v3;
camera-resolution design/code reviews v1; optical_worker; optics/native_arena.py;
optics/cuda_arena.py; optics/warp_native_render.py
Owner Summary: Close the remaining integration decisions before implementation.
Reuse the approved ownership/protocol/renderer pieces; do not infer runtime
activation from their individual approvals. C4 remains incomplete.

Base: `2e6f9f5` (v2 proposal and archived v1 review); camera code `1bcb622`
remains the approved CPU prerequisite. This v3 supersedes v1/v2 for
implementation; both proposals and complete independent reports remain records.
Dependencies: C2a `67b4018`, C2b `54f3e1a`, C1c `6d57fa7`, C4a/C4b and
C4c0 approvals retain their original bounded scope. This request proposes the
complete integration required by C2a section8 and C2b section9. It does not
change shared review policy, historical documents or any live gate.

## Review closure and reading rules

V2 was independently reviewed by `/root/c4_c2_integration_design_review_v2`
at exact HEAD `2e6f9f5`. Its complete report is archived as review codex v2;
verdict Request changes, F1 P1 and F2 P2 blocking design clarification.

- F1: section8 now distinguishes normal COMPLETE/DISARMED from terminal
  emergency ARM/ARMED. Successful emergency sync authorizes bounded cleanup
  without any completion/disarm handshake; the original deadline stays armed
  through actual process exit. Guarded-wait v1 is unchanged.
- F2: failure disposition first checks actual pending/active/failed guard state.
  A failed leaf anywhere in an armed normal segment quarantines directly,
  without another ARM, COMPLETE or synchronization. Only a healthy IDLE guard
  may enter a single emergency segment.
- Owner instruction, 2026-09-30: each review/test round ends with a scoped
  residual-process/hang check. Section9 item11 records this operational duty;
  it is not a new numerical performance gate or permission to kill other work.

The prior v1 review closure below is retained as history, not a new assertion
that any implementation or runtime gate has passed.

The complete first review is archived as runtime-integration review claude v1,
CLI session 94ebf17a-fb5d-44c9-8dce-87e239ef4959. No implementation was made
while that review ran. This version makes the requested policies explicit.

- P1.1: section5.1 defines a distinct setup-warmup authority, two immutable
  readiness objects and the one-way observed-facts chain.
- P1.2: section5.2 defines resource coverage versus operation history and the
  exact single seal point. V1 already said "Freeze complete resource coverage
  exactly once" BEFORE its final barrier; this was not barrier-then-seal.
  The additional definition removes any inference that a barrier needs a
  second resource seal or that SEALED means GPU-complete.
- P1.3: section3.1 defines the exact versioned metadata/hash migration.
  The review's quote "No existing ... row, threshold or gate is rewritten"
  does not occur in v1 section2 or elsewhere. V1 explicitly proposed v13.
  Nonetheless, unchanged logical row content is now distinguished from the
  deliberately changed row semantic digest; no claim of unchanged gate hash.
- P2.1: the source table already names tools/optical_pipeline_lab/camera_presets.py.
  Section3 now names exact functions, CPU dependency and prerequisite ordering.
- P2.2: diagnostics is the frozen pass BOOLEAN, not an instruction to enable it.
- P2.3/P2.4: section7 makes pre-CUDA row rejection and serial EXECUTE/cancel
  dispatch explicit, including the internal-only pre-leaf failure path.
- P2.5: section4 names existing exact protocol constants and enumerates v2
  operations; guarded-wait protocol is a separate channel/version.

These evidence-based corrections do not discard findings. The new reviewer
must check both the previous report and the original/corrected text against
actual code, and decide whether the substantive ambiguity is closed.

## 1. Actual gaps and explicit decisions

- Worker PREPARE currently rejects; there is no executable token lifecycle.
- native_setup directly waits on ready events and seals guardian coverage
  before target synchronization. lifecycle.commit_target_barrier requires that
  coverage seal. Existing termination may retry after a failed normal wait.
- C0 clean close rejects every historical measured submission; resetting that
  latch is forbidden. A new post-reclaim proof is required.
- Warp factories bind readiness at construction. Warmup0 readiness cannot be
  edited to claim the ten gate-required warmup frames.
- Workload materialization still imports benchmarks and an example-owned MJCF
  importer; camera presets are tool-owned. These are not admitted worker code.
- The asymmetric rotation is now a numeric CPU contract, but no runtime ABI
  or gate/receipt camera chain exists yet. Other camera builders exist, not a
  frozen resolved sequence.
- C4b CUDA resolve calls event.synchronize directly. Worker adoption must use
  the common supervised wait, not merely add it to Warp.

This design distinguishes **coverage seal** from **execution readiness**.
Coverage becomes immutable BEFORE the final device barrier; no execution or
token release is permitted until that supervised barrier succeeds. This retains
parent v7 section6.6.3 and current C0 meaning of SEALED. It explicitly replaces
the imprecise C2b section9 phrase "then permits guardian sealing": that phrase
means admission to execution, not a second guardian seal. Never reopen a seal.

Trust remains the parent v7 closed trusted process, not a hostile-Python
sandbox. No claim that Python wrappers can prevent arbitrary CDLL calls inside
the trusted process. The parent cannot inject callbacks, pickle, modules,
reflection paths, raw handles or alternative inputs. Scope/identity validation
protects supported product paths; process isolation protects the IPC boundary.

## 2. Scope and sequence

C4c2 delivers R0 fixed-scene render and R1 native replay with one slot,
payload=none/device-only, three profiles, shadows on/off and separate
diagnostics on/off. Six frozen R0 rows and two R1 rows receive positive
construction/execution tests in both lanes. Warp uses leaf4/sort refit; CUDA
uses its already frozen native rebuild identity. This does not equate their
acceleration algorithms; R0 compares fixed render work, R1 complete native work.

All R2 rows reject before GPU allocation in THIS slice. C4c3 subsequently adds
only the device-only R2 positive path; five host R2 rows remain rejected until
B3. S, physics publication, external host holds, depth-two delivery, full image
IPC, backend selection, C4 result authority, a3b and baseline generation remain
closed. FULL profile here means device channels, not FULL host delivery.

Implementation checkpoints, each independently reviewed:

1. **C2c materialization:** sole production CPU owners and resolved parameter
   records, versioned camera/row extension and no runtime admission.
2. **C2d lifecycle adoption:** versioned closed-worker supervision, guardian and
   exact terminal custody, initially only fixed sacrificial setup tests.
3. **C2e Warp timeline/arena:** R0/R1 physical graph and exact authority, tested
   only through supervised fixed drivers, no open product dispatcher.
4. **C2f integrated worker acceptance:** actual same worker drivers for Warp and
   CUDA, complete setup/post-submit failure matrix and CPU oracle correctness.
5. **Separate activation review:** may enable only reviewed R0/R1 native worker
   preparation/submission. Result/baseline/host/shipping gates stay false.

This is a dependency order, not permission to declare C2 complete after C2c.
No numerical performance/memory threshold is introduced. Tests may collect
correctness/lifecycle diagnostics, not benchmark result records or timings used
to choose a backend. Any new policy not settled here returns for design review.

## 3. Single materialization and complete render parameters

Move definitions once, preserving old explicit re-exports/object identity:

| Current definition | Sole production owner |
| --- | --- |
| benchmarks/optical/workloads.py | optical_worker/workloads.py |
| benchmarks/optical/assets.py | optical_worker/assets.py |
| example MJCF import types, mesh/body/material/light/math helpers | optics/mjcf_visual.py |
| tools optical camera preset builders/look-at helper | optics/camera_presets.py |

Do not move example CLI/render/video operations into production. Importer
extraction includes every actual transitive helper it uses, not a copied
worker-only importer. AST comparisons of moved mathematical bodies, object
identity at old imports and all frozen scene/replay/fixture digests are required.
The existing exact asset preflight remains mandatory: canonical remote/revision,
complete tracked/untracked inventory, bytes, root constraints and manifest hash.
Worker performs it itself; parent-supplied asset evidence is not authority.
Recheck the read bytes/content against that evidence while materializing;
changed files between preflight and reading fail before any GPU preparation.
No surrogate Go2, missing-asset skip or automatic download/install.

The orbit definition is exactly
tools/optical_pipeline_lab/camera_presets.py::build_lab_video_camera and its
_look_at_camera_rotation helper; fixed front uses build_model_bounds_camera.
All three move to optics/camera_presets.py in C2c, before any C2d/e/f use.
They use CPU NumPy/math, SpatialTransform and OpticalPinholeCameraSpec; no
GPU scene/materialization is necessary. Go2 requires real asset import before
camera resolution and fails closed if missing; tests cannot replace it.
The worker camera adapter supplies the frozen CPU bounds, dimensions,120/30
frame policy and fixed mode, not a caller argparse.Namespace.

CPU materialization returns one immutable logical source and its bounds; each
lane separately uploads it through native_setup, never via the other runtime.
Use the existing packer and exact prepared/readiness storage registry. Generic
constructors with correct case labels do not become canonical by assertion.

Freeze one common **resolved render sequence** before any native plan:

- asymmetric: approved resolved camera digest
  `04ab7bc2571b8396f06d36b7baea66224868001894c4ce7a5ea12e4faecda1ef`;
  its one camera repeats, without changing existing case/content identities;
- Go2 at either resolution: actual imported visual bounds, extracted orbit
  builder, 120 frames, 30fps, frame index modulo120. Retain its exact formula,
  focal/principal point/range and look-at orientation; no new lane-local math;
- dynamic replay: fixed front camera from canonical body-triangle world bounds
  at replay frame0 (including instance transform, excluding later z offsets).
  Reuse that same camera for all120 frames. Do not reframe each moving object;
- S camera resolution/adoption remains outside this scope; S rows still reject.

Frame metadata: sample i uses camera index i mod camera_count, replay index
i mod replay_count for R1, frame_id=i+1, sim_time=i/30, env_idx=0. Warmup
uses indices0..9 on separate provisional transactions and does not offset the
first measured/test sample. Sensor ID is a fixed case-derived identifier,
not renderer-specific. R0 has no transform replay even when the camera orbits.

Lighting arrays remain the packed-scene source. Explicit scalar settings:
asymmetric ambient=(.03,.05,.07), background=(0,0,0); Go2 and dynamic
ambient=(.08,.085,.09), background=(.025,.03,.038), as in the actual Lab
render_session; shadow bias=1e-4 (existing executor default). Profile and shadows come from the frozen row. Diagnostics is the exact
pass_mode boolean: primary_unprofiled -> false; diagnostic_profiled -> true.
It is frozen in the session manifest before PREPARE, identical for every
provisional warmup and final sample in that session; no request can change it.
A diagnostic pass uses a fresh worker and the same rule for both lanes.
Diagnostics-off warmup has no diagnostic events or diagnostic shade ABI. These
resolve previously implicit defaults as an explicit proposed input decision,
not a claim that old receipts already proved these values.

`q54.p12.4.resolved-render-sequence.v1` has exactly: schema_version, case_id,
case_sha256, scene_contract_sha256, bounds_source_id, bounds_f64,
camera_resolution_ids, frames, ambient_f64, background_f64, shadow_bias_f64.
Each frame has exactly camera_index,width,height,intrinsics_f64,origin_f64,
rotation_f64,max_distance_f64,sensor_role. Arrays have fixed lengths4/3/9/2x3
as applicable; positive dimensions, finite canonical binary64 hex strings,
ordered frames0..N-1, no caller-provided digest field. Bounds source IDs are
`canonical_fixed_world_bounds_v1` or `canonical_replay_frame0_bounds_v1`.
Asymmetric bounds are descriptive only, never an alternative camera builder.

Resolve once CPU-side, serialize actual values and independently reproduce all
sequence digests before freezing the implementation's gate extension. Store
the exact canonical sequence bytes in versioned workload artifacts; both lanes
consume those bytes after source/bounds/hash validation, not separately rounded
libm results. Formula reproduction must agree with the frozen binary64 values
under the frozen environment; disagreement is preparation failure, not silent
replacement. Extracted old camera APIs retain their original behavior.

`q54.p12.4.bound-render-parameters.v1` binds sequence_sha256,camera_index,
frame_id,sim_time_f64,env_idx,sensor_id,profile,shadows,diagnostics,
width,height,sensor_role_mask and all actual scalar arguments. Native scalar
order: fx,fy,cx,cy,origin[3],rotation row-major[9],max_distance,ambient[3],
background[3],shadow_bias. Store each scalar as its eight lowercase hex binary32
bits (IEEE little-endian byte encoding) after one binary64->binary32 nearest-even
conversion. Metadata sim_time is not a shader argument. No normalization or
transposition after this conversion. Validate actual Warp packed descriptors
and CUDA fixed ABI arguments against these values, not only the record hash.

Gate/result versions increment from v12 to v13 for this binding extension;
case/content hashes, existing row workloads, thresholds and activation=false
remain unchanged. Add resolved_render_sequence_sha256 to each applicable case,
and require bound_render_parameters plus hash in stage binding/receipt. S rows
are explicitly unresolved/disabled, never given an optical replay substitute.
Canonical row-semantic hashing includes the new sequence identity. Old v12
records cannot masquerade as v13. C3 binding schema gets a new version with a
downstream reference to these independent parameters (no hash cycle).
The historical same_process_native_pair=true is replaced by independent clean
single-lane workers with common protocol and alternating paired order, as
already approved by parent v7; no comparator execution is authorized here.

### 3.1 Versioned identities, not historical hash rewriting

There are three different objects, deliberately not called simply "case":

1. CanonicalOpticalCase.to_payload(), hashed by cases.py: unchanged bytes/hash.
2. gate["cases"][case_id]: existing catalog metadata retains its "sha256"
   value and all scene/replay/fixture fields, but gains the resolved sequence
   identity. That enclosing catalog record is not hashed as CanonicalOpticalCase.
3. Mandatory row JSON and its semantic hash: the row's fields/workload remain
   unchanged, but the semantic hash includes the extended catalog and contracts
   and therefore MUST change. It is never grandfathered or kept equal.

Example: the complete logical row is identical in v12 and v13:

```json
{"row_id":"r0_asymmetric_full_shadow","mandatory":true,"boundary":"R0","case_id":"asymmetric_synthetic_7x3_v1","input_contract":"fixed_native_scene_v1","required_asset_manifest":null,"output_profile":"direct_light_full","shadows":true,"acceleration_class":"fixed_bvh_fixture","completion_endpoint":"native_renderer_completion","required_lifecycle":["native_completion_reached"],"primary_metrics":["native_renderer_completion_ms"]}
```

Its v12 semantic identity is exactly schema.py::canonical_row_semantics_sha256
on the reviewed v12 gate. Its v13 semantic preimage uses the SAME keys currently
enumerated by that function, changes schema_version from
q54.p12.4.row-semantics.v7 to v8, uses the new case_contract catalog entry and
updated native_execution_binding_contract, and adds one key
resolved_render_contract (the new schema/scalar/binding rules in section3).
All unchanged shared-contract fields remain present, not a reduced hash.
Serialization remains canonical_json_bytes; SHA-256 is recomputed. The v13
sequence digest is frozen with its actual canonical bytes in C2c, not an
invented placeholder at design time. The asymmetric CAMERA digest is not
substituted for the enclosing render-SEQUENCE digest.

C2c review requires before/after complete preimages and independent SHA-256
recomputation for this example and every mandatory row; code/test fixtures
then freeze actual v13 identities. Existing v12 artifacts remain readable as
historical v12, not executable or valid under v13. No parallel live v12 lane,
migration fallback or automatic upgrading of results. All positive integration
tests consume v13 after C2c is approved. S catalog has null resolved sequence
and explicit unsupported scope; S receipt execution remains rejected. Gate
and result schema changes do not enable producing a result.

## 4. Source admission and supervision version

Current optical_worker/protocol.py defines CLOSED_WORKER_PROTOCOL_VERSION as
q54.p12.4.closed-worker-protocol.v1. Its validate_request already lists exactly
HELLO, OPEN_FROZEN_SESSION, PREPARE_FROZEN_ROW, CANCEL_PREPARED_SAMPLE,
EXECUTE_PREPARED_SAMPLE, RESOLVE_PREPARED_SAMPLE, RELEASE_RESOLVED_SAMPLE and
CLOSE_SESSION; C0 worker rejects GPU PREPARE. Protocol v2 here means
q54.p12.4.closed-worker-protocol.v2, NOT the separate
q54.closed-worker.guarded-wait.v1 in wait_protocol.py.

Request operation names/argument keys stay as in that validate_request;
protocol-version field becomes v2. Response payloads/state transitions now
admit the reviewed prepared/pending/resolved lifecycle and render identity,
which is why this is a new version, not silent reuse of v1. V1 manifests cannot
negotiate v2 by changing one field; dispatcher/manifest/state/source hashes
must match. No mixed A/K/guarded-wait mode within a process.

The integrated runtime uses new manifest/state/evidence/runtime-contract
versions (v3), protocol v2 with the same eight request operations, trust-model
v2 unchanged, and the approved guarded-wait protocol v1. Old C0 A/K stays a
distinct terminal one-shot mode with unchanged bytes/semantics and regression
tests. The new mode uses one pre-negotiated guard channel for setup, resolve
and emergency; no fallback from failed v3 wait to C0 A/K.

Freeze literal per-lane source-path lists and source SHA-256s, including package
initializers and actual transitive repository modules, in the implementation
candidate. Reject unexpected repository imports at startup, HELLO and each
request boundary. No benchmark/example/tool import is admitted except the two
explicit production launcher/supervisor scripts. No globs or dynamic source
discovery. Dependency versions/native libraries stay environment-frozen and
trusted, not falsely characterized as Python sandboxed code.

AST policy v2 permits ctypes only in optics/warp_native_render.py's existing
read-only `_render_launch_snapshot`: module import, addressof, c_float,c_int64,
c_int32 type identities, and bytes(parameter). Bind the exact function/source
hash; reject any other ctypes reference, alias escape, dynamic native call,
new CDLL or writable pointer cast. Direct runtime.core launch/event calls and
reflection/callback ingress remain disallowed repository operations. Do not
remove packed-argument inspection to satisfy the old broad policy.

Guard configuration is explicit in the runtime manifest and shared across lanes.
Use existing ACK and watchdog budgets, validated by GuardConfiguration; these
are operational abort bounds, never performance gates. Compilation/import/
recording can block, so they also run inside an armed setup guard. Partition
setup into nonnested guarded segments; an already active segment covers any
blocking device call within it. Every NORMAL setup/resolve segment requires
ARMED before its first
potentially blocking operation and COMPLETE/DISARMED after success. Emergency
is a distinct terminal segment: ARM/ARMED before its sole sync, then bounded
release/exit or quarantine/exit, never COMPLETE/DISARMED. Timeout/EOF/invalid
ACK or any failing leaf within a normal segment is terminal; do not complete
that failed segment or replace its guard with emergency. Never restart a guard
to extend a running deadline. Parent liveness monitoring remains independent.

The supervisor arms before ACK, retains pending-ACK deadlines and fences writes
on terminal commit using the accepted C2b implementation. Parent does not use
response-write timeout as compilation/request timeout. Request-wait exceptions
close liveness and reap the supervisor. Normal close uses a bounded terminal
process-exit observation; ACK is not proof of actual exit.

## 5. Setup graph, warmup and readiness order

One session, one row, one PREPARE, all sample plans prepared before timing.
Preflight row/asset/response capacity and all source/CPU contracts before CUDA.
Guardian exists before the first CUDA-capable import/init. Every possible GPU
leaf declares device scope with CPU metadata BEFORE entering; target/ambient
scope, partial returned owners and upload sources follow C0 register-before-
next-leaf rules. Import/init is not assumed quiescent merely because it is setup.

The normative order is:

1. Create lifecycle/guardian; materialize canonical CPU inputs and all camera
   parameters; adopt host staging/module/session owners.
2. Under setup guards, initialize only the lane/target context, create/adopt
   stream, upload each array with destination adopted before copy, record and
   supervised-wait the input ready event. Freeze exact a2 physical signatures.
3. Allocate/adopt every slot output, event and acceleration resource. Create
   provisional readiness truthfully naming warmup0 and factories/bundles for
   setup only; they cannot produce prepared sample tokens or final C3 authority.
4. Run ten actual warmup transactions with the frozen row profile/shadows and
   pass diagnostics setting, all under guarded setup. Track actual calls and
   completion, not a caller count. Observe terminal completion before declaring
   warmup success. Reset final generation outputs by the normal reset path.
5. Certify NEW final readiness for the same registered exact prepared arrays
   using observed compilation/ten-warmup facts; create NEW final factories,
   record/seal all final frame/slot commands, validate ABI and ordered dataflow,
   register C3 bindings, build dormant transactions/plans/response envelopes.
   Never edit provisional readiness/factory evidence or reuse its identity as
   final authority. Retain provisional resources in the guardian until exit.
6. Freeze complete resource coverage exactly once. Run final target-device
   barrier under setup guard AFTER all final recording, JIT, event initialization
   and resource operations. Commit quiescence only after sync success AND
   DISARMED ACK, bound to exact epoch/target/final readiness/prepared graph and
   sealed coverage. Only now publish prepared tokens.

Input readiness describes its own uploads/warmup facts; it does not include the
later final graph/barrier digest. Stage facts reference final input readiness;
graph references stages; setup quiescence references graph. No upstream object
references final quiescence. Normalized C3 evidence remains an acyclic DAG.

No GPU-capable preparation after step6. Any unexpected allocation/load/event
initialization invalidates admission and terminates the session; do not open a
second setup epoch to repair a measured run. Full checks/hashing/storage reads
are untimed. Setup guards are not benchmark sample measurements.

Dirty device set only expands. Exact {target}, known scope permits one target
barrier/emergency sync. Foreign or UNKNOWN_DEVICE_SCOPE at any time permanently
requires whole-graph quarantine and process exit; target sync cannot clear it.
Actual target1/ambient0 tests in both lanes are mandatory, not ordinal mocks.

### 5.1 Explicit warmup authority and non-circular facts

Input upload-ready event is a runtime event, not a mutable readiness object.
The existing NativeReadinessEvidence has a warmup_frames_completed field;
its identity registry supports multiple distinct readiness objects pointing
to the SAME exact prepared input (register_native_readiness keys by subject
identity, not one overwriteable readiness slot per prepared input).

After input ready-event wait, direct validation and compilation, construct
R_setup with all existing true input readiness flags and
warmup_frames_completed=0. This accurately describes native input state.
It authorizes validation/recording, but is not final measurement admission.

The fixed worker setup driver alone creates a SetupWarmupExecutionAuthority
registered by exact session/guardian/target/stream/provisional bundle identity.
It is usable only in PREPARING, with an active ARMED setup guard and known
exact target scope; no sample token, C4 final lease or result capability.
Before each warmup launch/event it marks SETUP work possible, retains all
operands, and rejects use outside that exact setup scope. Raw callables remain
inside the trusted worker. This is explicit non-measured setup execution,
not execution pretending to have a final C4 lease. No public authority factory
or serializable token is added for this purpose.

The setup driver observes ten actual terminal completions under supervision
and creates immutable WarmupCompletionEvidence with schema/session/target,
prepared identity digest, R_setup hash, compiled kernel/source identities,
profile/shadows/pass_mode, ordered frame indices0..9, ten transaction/terminal
identities, successful setup guard sequence IDs and completion count10.
Those identities are actual local objects/observations as well as descriptive
hashes. Parsing a record cannot register it as observed completion.

Only that exact registered evidence allows the worker to invoke the existing
certifier with warmup_frames_completed=10, producing NEW R_final. All existing
R_setup/R_final fields are equal except warmup_frames_completed; identity and
hash differ. The worker additionally registers FinalReadinessOrigin binding
R_final, prepared, WarmupCompletionEvidence and the exact compiled kernel set.
Final factories must use that exact R_final and the same warmed kernel/ABI
variant; new numerical kernel/variant discovery invalidates preparation.
R_setup is not edited/unregistered/relabelled as R_final. Both remain retained.

Dependency DAG:
input/uploads/compilation -> R_setup -> provisional bundles ->
observed warmup completions -> R_final + origin registry ->
new final stage facts -> prepared graph -> setup quiescence.
No R_setup field references later warmup; no R_final field refers to a final
factory, graph or barrier. The final graph references FinalReadinessOrigin,
never vice versa. This fits existing readiness construction without a third
readiness state or in-place extension. Warmup authority retires before final
coverage seal; it can never execute a measured transaction.

### 5.2 Coverage is resources; quiescence is operations

The exact one-way sequence is:
CONSTRUCTING guardian -> seal_guardian() -> SEALED guardian ->
guard ARM/ARMED -> final target synchronization -> COMPLETE/DISARMED ->
commit_target_barrier() -> SETUP_QUIESCENT / prepared-token admission.

Coverage contains the resource manifest, not a trace of operations. At seal,
snapshot the current coverage_payload fields (schema_version,session_identity,
state=SEALED,records,per_device_resource_counts) and its digest, and hold it as
the immutable sealed snapshot. Records contain owners already adopted for the
barrier: device/context/module/stream, channels, guard client, terminal/setup
events and all allocations. The barrier creates no new covered owner. Its
successful operation/guard evidence belongs to later SetupQuiescenceEvidence,
NOT to the sealed resource snapshot. Existing lifecycle.coverage_payload
changes live state on quarantine/release; that does not rewrite the recorded
sealed snapshot or pretend it was a completion proof.

All fields of the final quiescence are sourced as in
NativeWorkerSetupLifecycle.commit_target_barrier plus exact successful guard
identity: session/lane/target/epoch/dirty-scope history and operation counts
from lifecycle, readiness/graph digests from already frozen objects, coverage
digest from the sealed snapshot. Quiescence additionally binds completed guard
sequence/role/session/target. The guard records do not reference the later
quiescence object.

If final barrier fails, guardian WAS SEALED and remains held; lifecycle becomes
POISONED and transfers the intact graph to quarantine. SEALED never means safe
to release. No unseal/reseal, second sync or successful-quiescence record is
permitted. Failure before seal freezes partial resources and follows section8.
This is parent v7 seal-before-barrier, not an amendment to that ordering.

## 6. Warp transaction and arena

Add production `optics/warp_timeline.py` and `optics/warp_arena.py`. Reuse
NativeOpticalArena for slot/FIFO/generation/holds/replay, not a second slot state
machine. The Warp transaction mirrors the approved C2 enqueue/finish contract:
PREPARED -> VALIDATED_OPEN -> ENQUEUING -> FINISHED, failure -> POISONED.
All stages/outputs/events/device/stream/scalars frozen before first enqueue.
Order errors, premature/repeated finish and post-finish enqueue reject without
GPU work. Failure after a possible leaf poisons all affected storage/event slots.

R0: exact a2 fixed world/BVH + packed metadata/material/light -> render.
R1: exact selected replay view + local scene -> scene update/AABB -> Warp
leaf4/sort immutable-topology refit -> render. Use C3b factories and C1 renderer,
no copied kernels or CPU rebuild in the transaction. Shared topology immutable;
mutable world/bounds/output storage independently owned by the slot. R0 fixed
fixture and shared packed arrays each upload once. Full scene/readiness/ordered
storage/acceleration-class/operation/profile checks use production provenance.

Recorded reset kernels belong to their phase: BVH primary counter reset to
first-hit, shadow primary/overlay reset to shade. R0 has exactly three timing
pairs (raygen,first-hit,shade); R1 five (+scene_update,acceleration). Empty-ray
stages still execute defined resets and phase positions. Diagnostic records are
completion-neutral; no diagnostic events exist in the off variant. Terminal
event is initialized during setup, recorded exactly once by finish after all
stages. Not re-recorded while completion/output hold is live. Profile elapsed
times read only after terminal resolve. Timing cannot authorize reuse.

Strong context owners bind indexed device and exact native stream. Scope restores
ambient device/stream on success and failure. No host wait/global sync/allocation,
bridge, dynamic kernel lookup or extension loading in enqueue/finish. Light
stability checks compare fixed identities, not whole-frame catalogs. Complete
actual packed descriptor validation occurs before timed entry.

Static templates create dormant transactions only. A successful C4 acquire
atomically registers one exact lease->transaction->active owner->slot/generation
binding. Fields derive mechanically from sealed plan/C3/camera identity, never
from caller-provided tuples. Exact type plus live identity registry, not class
names/module strings/booleans. Acquire capability is single-use. Cancel/reclaim/
poison revoke authority; every enqueue/finish scope rechecks current live lease.
No static executable capability can mint a second live authority.

Prepare may be O(resources+frames); acquire is fixed-count identity/state/cursor
work independent of frame/resource count, with no hashing or graph traversal.
All active objects, callbacks, tuples, tickets and trace buffers are prepared
before timing. Tests instrument actual operations and compare operation counts
at1/120/600 frames. Python interpreter bookkeeping is not claimed globally
malloc-free; no new per-sample containers or application buffer allocation is
allowed. This clarifies, not weakens, the existing native no-hidden-allocation ABI.

Bundles/commands/raw events stay worker-local, unreachable through IPC. Supported
raw-leaf bypass attacks reject at the dispatcher with no sample submission;
arbitrary in-worker ctypes/reflection attacks are outside the trusted-domain
claim and are never advertised as blocked by underscores.

## 7. Worker tokens, terminal observation and close

Protocol v2 retains canonical framing/limits and eight operation names. All
tokens are worker CSPRNG128-bit, one-use, session-bound. No raw resource/handle
or serialized authority crosses IPC. Parent supplies only frozen row/case IDs,
repeat index and bounded sample_count. Preparation rejects a count whose exact
prebuilt response or bounded trace cannot fit existing protocol capacities,
before any CUDA work; no arbitrary new performance limit or silent truncation.

PREPARE row admission occurs immediately after strict request/sequence checking,
before camera resolution, asset loading or any CUDA-capable import/allocation.
It compares row boundary and payload against this stage's literal allowlist:
R0/R1 + none/native_only, otherwise stage_scope_rejected. Thus every S, every
R2 (including device-only), and host payload rejects at this exact point.
S camera is unresolved but is NOT used as the mechanism for S rejection.
A standalone sequence resolver called on S raises unsupported_case; that
separate CPU API does not authorize preparing S.

Runtime states: EXPECT_HELLO -> EXPECT_OPEN -> IDLE_UNPREPARED -> PREPARING ->
PREPARED_IDLE <-> SAMPLE_ACTIVE -> DRAINED_IDLE -> CLOSE_ACK_PENDING -> EXITED.
Any protocol/transport/runtime fault -> POISONED -> one terminal disposition.
PREPARE is single-use; no worker pooling or second lane/row.

The dispatcher is serial: one request is processed to response commit before
reading another. EXECUTE consumes its prepared token, then performs acquire,
enqueue and finish within that same handler (not GPU-synchronous). Only after
successful finish is a pending token exported. Parent CANCEL cannot interleave
between acquire and first leaf. No request performs acquire automatically
before EXECUTE. Internal pre-leaf failure invokes C4 cancel directly, never
routes an acquired token through public CANCEL_PREPARED_SAMPLE.

For each token: PREPARED -> ACQUIRED -> SUBMISSION_POSSIBLE -> COMPLETION_RECORDED
-> RESOLVED_HELD -> RECLAIMED. ACQUIRED is never exported. EXECUTE accepts only
the next prepared sample in FIFO and only when slot is FREE. Before every
possible kernel/event leaf, sticky sample AND session possible-submission latches
are set; successful-return counters are diagnostics, never safety predicates.
Submit-then-raise is post-submit. Finish atomically advances C4 completion count
and issues pending token only after actual terminal record returned successfully.

RESOLVE consumes pending only once: supervisor-guarded exact terminal event
wait; after success and DISARMED, invoke the exact arena resolution transition.
It cannot accept caller-provided "already waited" evidence. Warp/CUDA use the
same trusted guarded resolver; split CUDA's current direct synchronize from
its state commit under an exact internal ticket identity. Preserve legacy
in-process behavior with a wrapper calling the same core, not duplicate logic.
RELEASE consumes resolved token, drops the worker-owned device result hold,
performs reclaim and retires authority. Host pointers are never exported.

Cancellation rule made explicit to preserve precomputed future generations:
Public CANCEL accepts only a still-unacquired prepared token; first CANCEL
starts CANCEL_ONLY mode and no later EXECUTE is allowed. If another token is
already pending/resolved, its RESOLVE/RELEASE remain mandatory and legal;
only subsequent EXECUTE is prohibited. CANCEL_ONLY closes only after these
existing generations are also reclaimed. Consume all
remaining unacquired tokens without advancing slot generation. Already acquired
but not possibly submitted internal failure burns that generation using C4a,
then terminates the session, not a continued/replanned batch. Submitted tokens
cannot cancel. CLOSE requires all tokens consumed and all existing generations
reclaimed. This avoids silently renumbering future immutable plans.

Measurement integration remains inactive. Future R0 acquire is outside its
frozen render boundary; R1 starts before acquire. Resolve is outside timed
submission, but actual native completion is still the R0/R1 endpoint: no launch
wall-clock is reported as completion latency. Event/scalar capture is prepared
before timing; this slice emits validation receipts, not result-schema records.
IPC encode/write/decode is outside the native interval for both lanes.

`PostReclaimIdleEvidence` is generated by the exact worker/arena/guard session,
not parsed data authority. Canonical fields: schema_version,session_identity,
lane,target,setup_quiescence_sha256,prepared_graph_sha256,arena_identity,
arena_lifecycle_sha256,token_registry_sha256,submitted_count,resolved_count,
reclaimed_count,cancelled_count,remaining_prepared,remaining_pending,
remaining_resolved,active_leases,device_holds,external_holds,slot_states,
guard_evidence_sha256,historical_submission_latched. Last three remaining counts
and all holds/leases must be zero; every slot FREE; submitted=resolved=reclaimed;
all accepted sample tokens accounted for. Guard must have no active/pending ACK.
Cross-check actual live identities and replayed traces before issuing authority.
Copies/self-consistent hashes alone cannot authorize close.

CLOSE accepts only exact idle proof, retains graph through complete ACK write,
then commits cleanup and one-session process exit. Historical latch remains
true if any sample submitted. No unconditional wait or new device work in close.
Parent observes/reaps actual exit separately. Failed ACK encode/write uses fault
path with graph intact. No finalizer performs ordinary unproved cleanup.

## 8. Failure custody and evidence

One lifecycle authority owns graph release for setup, arena and transport. Arena
poison invalidates operations but cannot independently clear guardian resources.
Guardian includes modules/streams/events/allocations/host upload storage,
provisional and final commands, arena/plans and all partial constructor owners.
Register-before-next-leaf covers destinations and hidden setup ownership; an
exception before owner return preserves known enclosing runtime/session owners
and marks unknown scope/coverage when exact safety cannot be proved.

Faults include EOF, parent death, broken pipe, timeout, response encode/write,
dispatcher exception, failed leaf, failed normal wait and guard failure.
Disposition rules, evaluated from actual sticky state:

The following rules are ordered. Guard/scope failure takes precedence over
the apparently clean/no-sample cases; the historical sample latch alone cannot
select a release path. These are integration rules for retained guarded-wait
v1, not changes to its state machine.
Here complete coverage means custody of all resources possibly touched so far,
including partial setup owners; it does not require that every final planned
resource has already been constructed.

| Facts at failure | Only legal terminal action |
| --- | --- |
| foreign/unknown device scope or incomplete resource coverage | whole-graph quarantine and hard exit; no target sync can repair this |
| failed guard, pending ARM/COMPLETE ACK, or any failed leaf inside an armed setup/resolve segment | retain/quarantine graph; terminal exit under the existing deadline; no COMPLETE, new ARM or synchronization |
| a fault after emergency ARM/ARMED, including failed sync or cleanup | terminal quarantine/exit; no retry, second sync, COMPLETE or DISARMED; retain remaining owned resources |
| healthy IDLE guard, no active/pending/failed guard, complete coverage; no GPU setup entered OR proven final setup quiescent with no sample possible | cancel; whole-graph release; exit |
| healthy IDLE guard, no active/pending/failed guard, complete exact-target-only coverage and possible outstanding target work | poison; one emergency ARM/ARMED; at most one target sync |
| exact emergency ARM/ARMED succeeded, sole target sync returned successfully, complete target-only coverage and zero external holds | authorize whole-graph release and terminal exit; keep emergency guard/deadline armed; no completion/disarm handshake |

### 8.1 Normal segments and emergency have different successful endings

Normal setup/resolve success is ARM -> ARMED -> guarded operation success ->
COMPLETE -> DISARMED. Both operation success and the final normal handshake
are required for its quiescence/resolution commit. Failed allocation, copy,
compile, record or warmup launch within that segment is a failed SEGMENT even
when no blocking wait was reached. In particular, submit-then-raise cannot be
followed by COMPLETE to manufacture IDLE. Mark the segment failed before generic
termination sees control. Client ARM_PENDING/ARMED/COMPLETE_PENDING/FAILED or
supervisor pending-ACK/ARMED_NORMAL is not an idle emergency entry point.

Emergency admission checks the exact live client/session and last successful
normal guard trace, not a caller-provided state/receipt. It requires actual
IDLE, no pending ACK, no active guard and no previously failed segment. This
covers a known-target sample enqueue/finish failure outside a normal guard.
It does NOT cover the same failure while doing guarded setup warmup.

The legal emergency-success trace is:
IDLE -> emergency ARM -> supervisor deadline committed -> exact ARMED received
-> one target sync -> observed sync success -> resource release authorization
-> worker terminal exit -> supervisor observed/reaped process exit.

The emergency client ticket remains ARMED and the supervisor remains
ARMED_TERMINAL until process exit or terminal fault/deadline handling.
There is NO emergency COMPLETE/DISARMED exchange, no new guard, no resumed
request loop and no idle-close proof. The earlier ARM acknowledgement proves
watchdog admission, not GPU completion. The trusted driver's observed successful
sync supplies device quiescence; a normal-completion counter never supplies it.
Do not call GuardWaitClient.complete() with an emergency ticket.

After successful emergency sync, cleanup and exit remain protected by the
original deadline: it is neither cleared nor restarted. If cleanup or exit
hangs, the supervisor terminates/reaps the worker at that deadline. If cleanup
throws, retain remaining owned resources and hard-exit without retrying cleanup
or synchronization; do not claim that already released owners are still held.
No new allocation/enqueue/resource reuse is allowed in terminal cleanup.
A successful sync/release authorization does not assert cleanup or exit actually
finished. Any final exit claim comes only from actual supervisor observation.

Preserve existing lifecycle emergency_sync_started/emergency_sync_succeeded
and resource_release_authorized facts, binding them to exact emergency
session/target/sequence and retained guard identity in the new evidence version.
Supervisor normal_complete_count excludes emergency. A worker disposition such
as aborted_synced_exit describes its authorized terminal action, NOT an observed
process exit; the final combined record additionally requires the supervisor's
actual exit record. If the worker is killed/hung before emitting its record,
retain partial evidence and supervisor outcome, never synthesize worker success.

### 8.2 Failure retention and irreversible termination

This deliberately does not reuse C0's retry after failed normal barrier.
Replace the too-narrow integration notion normal_guard_failed with sticky
guarded_segment_failed, including failure in non-wait leaves, handshake
uncertainty and partial submission inside guarded setup. This state never
clears. Such failures use their original guard deadline; no second emergency
attempt or successful normal-completion record is permitted.

Quarantine retains the complete available graph BEFORE exception unwinding,
then hard-exits without Python per-resource cleanup. This applies at all
pre-release failures; the already-quiescent terminal cleanup exception above
retains all remaining owners and makes no false full-retention claim.
No external host lease can exist in C2; require zero rather than assume it.
C4d/B3 still owns the dual proof for future external holds. Unexpected nonzero
hold forbids ordinary release. Evidence distinguishes release authorization,
completed cleanup and observed process exit; no field substitutes for another.

Version termination/setup evidence for new operations. Extend current replay
validators, not separate self-consistent summary booleans: replay setup adoption,
coverage seal, guards, sample latch, exact C4 lifecycle, token transitions and
close commits; derive/cross-check final state, counts and disposition. Bind
guard sequence/role/session/target, actual arena generation/frame and quiescence
epoch. Unknown/duplicate transitions and forged summary fail. Diagnostic event
counts never contribute terminal completion counts. The evidence chain has no
backreferences from inputs/stages to final lifecycle.

## 9. Same-slice mandatory validation

Tests run against actual production drivers in source-frozen, supervisor-owned
sacrificial processes. Fixed test fault selectors live in test-only launchers;
no product callback/plugin or arbitrary operation channel. Product activation
stays false until independent code review covers both lanes and all faults.

1. CPU moves: old/new identity, AST algorithm equality, all case/content digests,
   real Go2 assets, all120 camera values and first/last/wrap indices; unknown
   case/changed asset/bounds/sequence/hash reject. Import-attempt and actual
   repository ledger checks with negative controls, no benchmark/example load.
2. Camera chain: same hash/wrong matrix, wrong row, changed ABI scalar, same-shape
   reordered arrays, set_()/Warp pointer drift fail before first leaf. Independent
   CPU rays/geometry/IDs/material/RGB/intensity for asymmetric and dynamic;
   validate real Go2 fixture instead of a surrogate. Existing C1 parity tolerances
   apply unchanged; counters use existing exact invariants, not cross-kernel
   counter equality. Binary32 input agreement is not assumed GPU arithmetic.
3. Authority: C4b original forged class, duplicate mint, frame0 lease/frame1
   transaction, fake owner, empty binding, cancelled/reclaimed authority, repeated
   finish, sealed-field/packed-command drift. Pre-enqueue reject => no sample
   leaf/event, no completion ticket, unchanged C4 state. Post-latch failure =>
   poison/retained graph, not the same zero-work assertion.
4. All six R0 + two R1 rows in both lanes; all three profiles, both shadow modes,
   diagnostic3/5 pairs and off=zero, exactly one terminal event; sentinel reset
   across generations. R2/S/host reject before allocation; no result exports.
5. Setup faults at import/init, first allocation/copy, ready-event wait, compile,
   record, every warmup and final barrier; returned-owner and pre-return failure;
   guardian sentinels never destruct early. Guard ARMED/DISARMED loss/hangs with
   parent alive, failed barrier no emergency retry, normal resolve hung after
   all launches succeeded, final cleanup/ACK failure.
6. Disconnect/kill parent in PREPARED, acquired-before-first-leaf, each leaf
   including submit-then-raise, pending, resolved-held, reclaimed-idle and close
   ACK. Verify actual exit, source coverage, owners retained until safe
   disposition and no reuse. Supervisor stall/no ACK => sync callable never runs.
7. Real dual-GPU Warp/CUDA target1/ambient0, foreign touch before helper return,
   unknown-device failure, normal device restoration and stream isolation;
   no target-only sync authorizes foreign resource release. No mocks/skips
   count as this mandatory acceptance. Primary183, fallback101 per policy.
8. CPU trace replay attacks: duplicate operations, invented state/disposition,
   summary counters, old quiescence, wrong terminal generation, modified camera
   chain, resolved-without-wait, reclaimed-with-live-hold. Repeat close/cancel/
   release, cross-session tokens, simultaneous requests and reentrancy fail
   closed without duplicate transitions. Single serial dispatcher owns calls.
9. Explicit emergency composition: idle post-submit failure -> exactly one
   ARM/ARMED and sync -> release -> observed exit, with zero COMPLETE/DISARMED;
   cleanup/exit hang after sync success must expire the same terminal deadline.
   Failed allocation/compile/record/launch inside armed setup and pending-ACK
   failures -> zero further ARM/COMPLETE/sync, intact pre-release graph and
   terminal quarantine. Preserve emergency-COMPLETE rejection tests. Include
   submit-then-raise and distinguish worker disposition from actual exit.
10. Regress C4a/C4b, C0 A/K, C1c S/V/E, C2b guarded-wait CPU tests and real
    subprocess hangs. Full binding preflight stays untimed; acquire work-count
    independent of replay length. Run code checks and inspect clean frozen HEAD.
11. Round-end process hygiene (owner instruction): record the launched test,
    reviewer CLI, worker and supervisor process identities/parentage and exits;
    check scoped OS process state and GPU-process records after each round,
    including exceptions and timeouts. Correlate PID/start identity with the
    launch ledger; a matching process name alone is not ownership evidence.
    Preserve logs/stacks before handling an actual stall; use the reviewed
    supervisor to reap owned children where applicable. Do not kill unrelated
    processes, clear another session's resources or reset a shared GPU. Report
    any unresolved residual explicitly and do not start another GPU round on
    top of it. A clean check is scoped evidence, not proof of no possible hang.

## 10. Completion and cleanup

Each code review reports exact delivered subset and closed gates. C2 finishes
only after C2f real-device acceptance and independent review, not CPU contract
approval. C4c3 static interop/R2, C4c4 closure, C4d/B3 joint lifecycle and final
activation reviews remain required; C4 cannot be declared done here.

Durable owners: selected-lane worker supervision, canonical inputs/camera
sequence, common C4 control and selected runtime arena. P12.4e removes losing
lane, candidate selector, unused aliases, comparison-only factories/artifacts
and sacrificial transitional harness after equivalent retained coverage. No
second source of kernels/materializers/authority or permanent dual-backend
product abstraction is introduced. Update cleanup ledger per implementation
slice; do not edit collab/README or historical approval meanings.
