# C4c2: closed-worker integration, Warp timeline and R0/R1 arena

Initiative: q54-optical-lab-p12-4c4c2-runtime-integration
Stage: design-request
Author: codex
Version: v1
Date: 2026-09-29
Status: in_review
Related Files: C4c parent design v7; C4c2a design v1; C4c2b design v3;
camera-resolution design/code reviews v1; optical_worker; optics/native_arena.py;
optics/cuda_arena.py; optics/warp_native_render.py
Owner Summary: Close the remaining integration decisions before implementation.
Reuse the approved ownership/protocol/renderer pieces; do not infer runtime
activation from their individual approvals. C4 remains incomplete.

Base: `c3d2a37` (camera evidence-only archive, approved code `1bcb622`).
Dependencies: C2a `67b4018`, C2b `54f3e1a`, C1c `6d57fa7`, C4a/C4b and
C4c0 approvals retain their original bounded scope. This request proposes the
complete integration required by C2a section8 and C2b section9. It does not
change shared review policy, historical documents or any live gate.

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
render_session; shadow bias=1e-4 (existing executor default). Profile, shadows
and diagnostics come from the frozen row/pass, not caller overrides. These
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

## 4. Source admission and supervision version

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
blocking device call within it. Every segment requires ARMED before its first
potentially blocking operation and DISARMED after success. Timeout/EOF/invalid
ACK/failure at any point is terminal. Never restart a guard to extend a running
deadline. Parent liveness monitoring remains active independently.

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

Runtime states: EXPECT_HELLO -> EXPECT_OPEN -> IDLE_UNPREPARED -> PREPARING ->
PREPARED_IDLE <-> SAMPLE_ACTIVE -> DRAINED_IDLE -> CLOSE_ACK_PENDING -> EXITED.
Any protocol/transport/runtime fault -> POISONED -> one terminal disposition.
PREPARE is single-use; no worker pooling or second lane/row.

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
first CANCEL starts CANCEL_ONLY mode; no later EXECUTE is allowed. Consume all
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

| Facts at failure | Only legal terminal action |
| --- | --- |
| no GPU setup entered, or proven final setup quiescent and no sample possible | cancel; whole-graph release; exit |
| known exact target dirty / sample possible; no previous failed guarded wait | poison; ARM emergency/ACK; at most one target sync |
| emergency sync AND completion handshake succeeded, zero external holds | authorize whole-graph release; exit, never reuse |
| guard arm/ACK/disarm failure, normal wait failure/hang, emergency failure, foreign/unknown scope or incomplete coverage | whole-graph quarantine; hard exit; no second synchronization |

This deliberately does not reuse C0's retry after failed normal barrier. Mark
`normal_guard_failed` before control can reach generic terminate. If worker is
hung it cannot publish a final receipt: supervisor deadline termination and
actual exit observation are the evidence; never fabricate a worker finish.
Quarantine retains the complete graph BEFORE exception unwinding, then hard
exits without Python per-resource cleanup. No external host lease can exist in
C2; require zero rather than assume it. C4d/B3 still owns the dual proof for
future external holds. Unexpected nonzero hold forbids ordinary release.

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
9. Regress C4a/C4b, C0 A/K, C1c S/V/E, C2b guarded-wait CPU tests and real
   subprocess hangs. Full binding preflight stays untimed; acquire work-count
   independent of replay length. Run code checks and inspect clean frozen HEAD.

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
