Initiative: Q54 / P12.4 / remaining C2f and whole C4 refactor
Stage: Author diagnostic, not implementation or acceptance
Author: Codex
Version: v1
Date: 2026-10-10
Status: draft
Related Files: runtime-integration design-request v3 sections6/7/9; worker-intake gap-review v1; native-state-reader design-request v1; tests/unit/optics/test_warp_arena.py; tests/unit/optics/test_integrated_worker_lifecycle.py
Owner Summary: Actual worker EXECUTE still replays growing history. Existing bare-arena constant-work evidence does not establish the complete worker contract. This records a concrete refactor prerequisite without creating a new performance gate or claiming a renderer speed comparison.

## Bounded executable evidence

Production source cf7f5a6d9ff9b3971144410a7c3ad156a806646c; author HEAD
836700f8d098b062770476c5687a92e7d46082f3 differs only in evidence documents.
Scratch probe `c4_assets/test_worker_record_work_probe_v1.py` uses the existing
integrated-worker fixtures, actual production dispatcher/arena/controller and
CPU native leaf doubles. It wraps only the module-level replay_lifecycle call
to count input trace length, then delegates exactly once to the original.
Each of two samples runs actual EXECUTE, RESOLVE and RELEASE. Only EXECUTE's
replay work is included below; this is neither GPU work nor wall-clock timing.

| Lane | First EXECUTE | Second EXECUTE | Record calls per EXECUTE |
| --- | --- | --- | --- |
| Warp | 1596 events replayed (531+532+533) | 1626 (541+542+543) | 3 |
| CUDA | 1338 events replayed (445+446+447) | 1368 (455+456+457) | 3 |

The number of record calls is constant, but their work grows with previously
recorded lifecycle history. The different lane totals reflect different setup
histories and do not compare renderer speed. The probe's two diagnostic cases
passed in6.87s; it asserts the current defect, not the desired fixed-work
acceptance criterion. It is intentionally scratch evidence, not a regression
test that would permanently require inefficient behavior.

Command/outputs live in `c4_assets/worker_record_work_probe_v1/execution.json`,
stdout.txt, stderr.txt and per-case record_work.json. Started11:15:36Z;
driver22024 waited pytest39904 to exit0. Both are absent. Guard-peer teardown
belongs to the reused fixture; no GPU/backend acquisition or native child ran.
No production or tracked test code changed for this diagnostic.

## Concrete implementation gaps to close together

The parent contract already requires fixed work independent of frame/resource
count and prepared application objects/trace storage at its measured boundary.
It does not require a new benchmark or arbitrary hardware-memory ceiling.

- NativeGuardedSetupLifecycle._record builds event/detail/guard dictionaries,
  copies the full list and replays the entire trace. Existing _Replay.apply is
  the single semantic reducer, but it changes fields before some validations;
  live incremental use must first preserve failure atomicity. replay_lifecycle
  also checks post_reclaim_idle's actual prefix hash outside apply; that check
  must not disappear during consolidation. Version transitions must retain
  the original v3/v4/v5 meanings.
- NativeOpticalArena._validate_plan_for_slot creates a new expected-field
  dictionary on acquisition; _record_transition creates/validates a dataclass
  and appends it. Its FIFO also grows by append. Exact original plan/lease and
  generation checks must survive preallocation; cloning slot control is not a
  solution.
- _SourceSession.bound_controller builds both configuration payload dictionaries
  whenever it verifies the original binding. WholeGraphResourceGuardian's
  contains_exact_owner is a linear identity scan. Only call sites actually in
  the claimed sample boundary should be changed; untimed final serialization
  or isolated setup checks are not automatically hot-path defects.
- ClosedWorkerSession._update_native_idle scans all prepared samples at each
  relevant release/cancel. PREPARE bounds its token response but does not yet
  prove the complete trace-capacity requirement before GPU preparation.

Existing test_acquire_work_does_not_scale_with_frame_catalog counts real Python
lines for1/120/600 prepared plans, but only in warp_arena.py/native_arena.py and
only for bare acquire. It usefully covers that scope; it does not cover source
binding, worker lifecycle replay, enqueue/finish/resolve/reclaim or allocations.
The parent requires evidence proportionate to the actual final implementation.

## Next coherent assessment, not another approval gate

Fresh independent read-only session01a12584-473c-7593-a8e3-968b8632e8c6 is
assessing the remaining final-worker scope/CPU-oracle, fixed-work/preallocation
and fault/close requirements together against frozen cf7f5a6. It received the
accepted requirements, code and completed evidence, not author conversation.
Candidate `c4_worker_acceptance_gap_review`; scratch
`c4_assets/worker_acceptance_gap_scratch_v1`; tool12466; CLI42716,
launcher13204; start11:13:08Z. Packet SHA256
e85b1fbc0ac97eabf0603faa41ede8f7de53869b49f24999c3ce5fcc4199646f.
The review completed with Request changes for the remaining integrated scope,
not withdrawal of the preparation/repair approvals. Its complete report is
archived verbatim as worker-acceptance gap-design-boundary-review v1, SHA256
f6b5c260bea9fe5e8d4a088860ccf340d01b2940b0c53ae80671c2e14e0658f4.
Tool12466 returned exit0; CLI42716 and launcher13204 are absent. It identifies
one narrow final-observation policy supplement and existing required work in
final numerical staging, complete fixed-work/capacity, close supervision and
full journal recovery. It cannot grant hardware success or a later gate.

A separate CPU encoded lower bound is retained in
c4_assets/worker_trace_capacity_lower_bound_v1.json. The current PREPARE response
formula admits119830 tokens under4194304 bytes, while only the three mandatory
EXECUTE records per sample already require at least35949000 bytes. This excludes
all setup/resolve/reclaim/close/summary/coverage and separators. _journal uses
the same4194304-byte envelope ceiling. This is a capacity counterexample, not
an actual native run or a proposal to allow that sample count.

No reader regression, two-sample normal exit or this diagnostic completes C2f
or C4. The original whole objective, full failure coverage, systematic cleanup,
independent acceptance and later C4 stages remain required.

## Integrated implementation in progress (2026-10-10)

The following author changes belong to the one G1–G5 packet, not independent
stage completion or reviewed native code. D1 final-observation design v2 is in
fresh review session01a1259f-a957-7071-9142-7b4593996629 at208f32c;
tool71384, reviewer45132/launcher20728. V1's full report is archived verbatim.
Its F1 corrected scope inflation (all120 CPU camera values do not mandate all120
full-resolution numeric images); F2 requires exact sequence-bound request
deadline retirement/expiry arbitration. That new diagnostic aperture is not yet
implemented. Separate full reader acceptance reviews frozen1286cbb, production
cf7f5a6, session01a125a2-0c60-7aa3-8736-a5b6c3a977f7, tool72080,
reviewer40760/launcher46608. Neither candidate includes these moving author edits.

- Shared reducer prerequisite: rejecting a bad descriptor or bad coverage hash
  previously changed epoch/guardian state. Two direct original-core repros failed
  before the change. Epoch and guardian changes now follow validation; declared
  ordinals validate before mutation. The actual preceding-prefix hash remains
  computed at its untimed boundary, while apply itself now requires the binding
  for post_reclaim_idle. Live _record still reconstructs/replays; switching it
  awaits reserved storage and full failure-atomic publication. No fixed-work or
  allocation-free claim is made yet.
- Journal: original short-write repro retained97of555bytes while returning
  normally. The writer now advances over positive short writes, fsyncs only a
  complete frame, and returns incomplete on zero/error/interrupt/fsync failure
  without replacing the original fault or retrying a failed frame. Raw partial
  bytes remain. Full enclosing request/journal-stall supervision and F3 coverage
  remain part of the packet; this loop alone does not supply that supervision.
- Existing fixed worker supervisor now checks actual CLOSE ACK-to-exit timeout
  and client failure, fences command transport (close fallback on shutdown error),
  and enters the existing core's terminal path. It records ACK, timeout/fence,
  actual exit and reaping separately. Existing earlier guard deadlines are not
  renewed. OS child tests cover this exact policy/core with modeled ACKs; they
  are not yet the required actual integrated native CLOSE fault matrix. Legacy
  C0 proxy behavior is not claimed changed or integrated by this work.

Author evidence in local c4_assets:
- worker_journal_before_v2.txt: actual555-byte frame truncated to97by old writer.
  The earlier draft used an overlong string rejected before writing; it was
  corrected before the meaningful repro and is not short-write evidence.
- reducer_atomic_before_v1.txt:2original invalid-transition mutation failures.
- worker_journal_after_v1.txt:63passed/no skips in79.37s (journal + existing
  integrated worker tests), before the later reducer change.
- worker_completion_control_v1.txt:77passed/no skips in92.85s with original
  dispatcher, both lanes' CPU native leaves, reducer prefix and journal checks.
- worker_completion_reducer_regression_v1.txt:311passed/11Windows platform skips
  in8.82s. The11 historical CUDA codec cases require a POSIX parser host and a
  pinned retained input archive; they still need an actual run for these changes.
- worker_completion_io_v2:16passed/no skips in0.75s after transport-close fallback;
  actual owned CPU subprocesses were waited/reaped by the tests.
- worker_reducer_mutations_v1:12passed/no skips in0.21s, including malformed fields
  at every step in ten independently authored failure/cleanup stories; complete
  reducer state stays equal on rejection and original valid stories still match.
- Three new test files pass Ruff; this is not a clean whole-repository lint claim.
  All91 prospective LF source catalog entries pass the unchanged AST policy;
  updated catalog SHA256e214c22f1a8a7a749f9f9295bb3598e1c786820a8aea8824bad8ff08edf84f45.

Outstanding G3 still includes pre-GPU exact capacity admission, preallocated
event/arena/authority/scope/ticket storage, atomic incremental record publication,
bounded identity checks and remaining-sample bookkeeping, then full1/120/600
operation/allocation proof. G2 final-output staging/oracle and G4/G5 complete
actual worker/fault/transport evidence are also open. These author changes have
not been deployed for native use or independently accepted. Preserve the complete
packet and subsequent C4 objective; do not promote these local passes to a gate.

## D1 approval and subsequent implementation progress

Fresh design review session01a1259f-a957-7071-9142-7b4593996629 completed
Approve with comments for208f32c86c25b26bda6a31f66c21f9aeb0fe1ffb.
Tool71384 exited0. Full report is archived verbatim as final-observation
design-review v2, working-byte SHA256
28f422d64ee7fdc4f29765b999b3a1d36930b34afac01dc8114b0a7dba1f9d75.
Author confirms the intended candidate is that exact committed HEAD/document.
Its committed LF digest is81111b1e548c438689e44a04129407273379184b62c69624480835964a71b7a6;
the packet's06862ff... was the Windows CRLF working-byte digest. Actual Git bytes
and working bytes were independently compared and differ only by CRLF. The
review expressly binds the inspected LF bytes, so no unreviewed alternate text
is substituted. Keep this identity correction with the original report/packet.
F1/F2 are closed at design level. C1 native event initialization before seal and
C2 three successful sentinel reuses/actual post-resolution failure state remain
mandatory implementation evidence, not waived comments.

After that approval, the existing fixed request client/supervisor gained the
sequence-bound observation handoff. Publication precedes send; timely success
is timestamped before persistence and only the original supervisor thread
retires it. Equality/late responses and stale observations cannot clear an
expired obligation. Existing guard deadlines tick before retirement. The parent
retains original partial/error response bytes and first exceptions even if
response persistence also fails. Fast normal worker exit still drains the
buffered CLOSE response and supervisor retirement handshake before joining the
client. This is the approved fixed correctness driver, not a product callback
or native-query implementation. Final reader/staging wiring remains unfinished.

New author tests:19 request/close scheduling and actual child-disposal cases
passed in0.65s;12 actual fixed-client socket cases passed in2.42s (partial reply,
error reply and wrong sequence, each with successful/failed persistence, both
CPU fixture lanes). Original native worker dispatch is not claimed by these
modeled peer responses. Full integrated native fault/close tests still remain.

Linux CPU verification of snapshotc29a197cf217ece399fbb4703c24d934ba3e18f8:
26new control/journal/reducer cases passed;11historical compilation-codec cases
errored in shared fixture setup at NativeInputManifest.from_payload with
"native manifest version" while reading the pinned fc86182 archive. Exact
unchangedcf7f5a6 baseline reproduces all11 same errors. This is not a new
reducer regression and not a pass: the retained historical fixture/current
schema compatibility gap remains to reconcile without relabeling old records
as current or weakening current manifest admission. Raw logs/launch/exit are
downloaded as c4-worker-control-cpu-c29a197-v1-* and
c4-worker-control-baseline-cpu-cf7f5a6-v1-* in c4_assets. Both runs block
torch/warp/cupy imports and perform no native/GPU work. Six recorded owned CPU
PIDs3026067,3026072,3026253,3026254,3034182,3034187 are absent; closure JSON is
retained. These are diagnostic/check outcomes, not full refactor acceptance.

## Incremental live reducer implementation (author work, after925d606)

The live controller now retains one `_Replay` and applies only the new event.
It no longer builds a copy of the entire trace and reconstructs the reducer on
each record. Full replay remains in `GuardedLifecycleEvidence.from_payload` at
the untimed evidence boundary. The post-reclaim event still verifies the hash
of the actual preceding prefix, computed locally rather than accepted from the
submitted descriptive proof. Existing v3/v4/v5 serialized schemas are unchanged.

Trace append happens before reducer application, under the original event lock;
allocation failure therefore precedes state mutation. On rejected application,
the appended event is removed and the prior reducer version restored. Reducer
setup branches stage growing collections, descriptor hashes and counters before
publishing their state. Three injected allocation failures reproduce partial
mutation in the925d606 reducer and preserve all fields after this change.
This does not claim atomicity against arbitrary asynchronous interpreter faults.

The same reducer now tracks remaining samples, decrementing only on validated
cancel/reclaim transitions. Worker idle updates use that count rather than
walking the whole prepared batch after every release. Close still checks the
actual worker/arena records and the full replayed sample states; the counter
does not independently authorize resource release.

Author counterexamples and verification:
- Isolated baseline checkout `../c4_incremental_baseline_925d606` retains exact
  production925d606 with four copied current test/fixture files. From that
  checkout, the original one-sample Warp EXECUTE invokes the reducer1581times
  (1578more than the three new facts), failing the new test. Raw output:
  `worker_incremental_baseline_v2.txt`. The earlier v1 invocation started from
  the main checkout, so imported main production modules and is invalid as
  baseline evidence. Its exact owned Python PID464/parent45024 was stopped;
  identity and raw output are retained, tool13953closed with exit-1.
- Baseline allocation-failure counterexamples:3failed/12deselected in0.37s,
  `worker_incremental_atomic_baseline_v1.txt`. Current reducer tests:15passed
  in0.20s, including the three counterexamples and prior malformed-event cases.
- Initial incremental regression before the remaining-count/fixture changes:
  282passed/no skips in61.01s. This is not the final snapshot verification.
- Scale-probe draft v1 failed fixture admission because its requested count
  differed from the originally captured row. The shared final-graph fixture now
  takes its count before original row admission; no captured row is rewritten.
- Scale-probe v2:Warp1/120passed, Warp600 errored during setup on guard EOF.
  The original CPU peer's operation budget is5000ms. The scale test now declares
  60000ms before constructing either guard endpoint, leaving all production
  deadlines and default fixtures unchanged. This is an operation-count probe,
  not elapsed-time or native setup-time acceptance.
- Final regression:473passed/no skips,6scale cases deselected in131.43s;
  `worker_incremental_regression_v1.txt`, tool36489exit0. This covers lifecycle,
  shared reducer/fixtures, integrated dispatcher, final graph and guard core.
  The separate scale run ended with5passed/1error/75deselected in699.57s:
  CUDA600 failed during final-graph preparation at guard COMPLETE with an
  aborted connection. Python38428/parent47260 and tool24882 are terminal.
  `worker_incremental_counts_v3.txt` is retained (8562bytes, SHA256
  66a17d33640cecfd4636ac49c5a652d35c92f4bc630efe6d4a9b09d58bd32e9d).
  This changed the next action to profiling and eliminating repeated batch
  validation, rather than increasing the guard budget again. Repair is below.
  Source catalog refresh validates all91 prospective LF sources under
  the unchanged AST policy; current catalog SHA256
  2706cbe713011091db8354c996c39d516813c0f1abe9d555eec8f0828365fa92.

Remaining scope is explicit: live event dictionaries/list growth, arena trace
records, authorities/scopes/tickets and other per-sample allocations still need
preparation; exact pre-GPU trace/journal capacity admission is unfinished. The
count tests instrument the lifecycle reducer inside the actual dispatcher with
CPU native leaves; they do not prove whole-dispatch fixed work/allocation limits
or native speed. This is continuing G3 implementation, not its acceptance or a
substitute for the coherent G1-G5 packet, independent review, or full C4 goal.
No changed production source has been deployed for native execution.

## Arena preparation/identity consolidation (continuing G3, afteraabc375)

Work used isolated `../c4_arena_fixed_work` while the originalaabc375 scale run
was still live, preserving its executing source snapshot. This is author
implementation in the same G1-G5 packet, not another product/acceptance gate.

Actual CPU profiling of the original dispatcher established the missing
preparation cost. At120 CUDA samples, `_validate_target_evidence` ran57960times:
four per-plan operations each revisited all120 targets, plus three whole-graph
checks. `worker_preparation_cuda_120_v1.json` has187650bytes, SHA256
7437c6064a69abe50e95f6a18ea4a8d2dde8e55b62e9ec3f9f34489764651ddd.
The1/120 preparation probes passed in63.54s total (profiling overhead included).
Four first-EXECUTE profiles at1/120 samples across both CPU fixture lanes also
passed; their CPU elapsed times are not native performance measurements.
Scratch drivers and outputs are `worker_preparation_profile_v1*`,
`test_worker_preparation_profile_v1.py`, `worker_hotpath_profile_v1*`,
`test_worker_hotpath_profile_v1.py`, and `worker_hotpath_{lane}_{count}_v1.json`.

Changes:
- CUDA preparation validates the selected target on plan selection/provenance/
  transaction preparation. Full target validation remains at arena seal and
  both final-barrier admission edges. Target/executable maps require immutable
  mapping proxies; the arena owns a separate immutable target identity index.
  Selected and full checks both bind that original index, and original target
  storage/event/stage validation remains. No unrelated target scan is performed
  per plan, and no native pointer check is waived.
- Shared arena preparation performs metadata/hash validation before acquire.
  Acquire checks exact prevalidated evidence/lease objects plus the current
  FIFO slot/generation. Frozen spec/manifest and privately copied read-only
  owner-map identity replace repeated owner traversal. Copying equal evidence
  cannot replace its original live object. Reflective in-place mutation of
  frozen objects is outside the existing trusted-worker claim; public mutation
  remains rejected and replacement attacks are covered.
- Prepared plans retain their original weak-reference registration key.
  Acquire reuses that key; no new weak reference or hash payload is created.
  The callback is prepared once per arena. Discarded speculative plans still
  disappear, and competing/foreign/stale plans retain their rejection behavior.

Verification (local Python3.14, CPU leaves, no native deployment):
- Originalaabc375 fails four new counterexamples: acquire calls hashing;
  equal copied plan/lease evidence is accepted; acquire line work grows with
  owner count (311/549/1509lines at1/120/600 owners). One original plan-copy
  rejection passes; the new registration-key case is inapplicable to baseline
  and explicitly deselected. Raw `arena_fixed_binding_baseline_v2.txt`:
  4failed/1passed/1deselected in1.35s,13226bytes, SHA256
  2ebd386ff9168033a8c692f44a2627e1a20e7c202f0f9f60fa8374dc4bc50d58.
  Baselinev1 ran from the changed checkout and is invalid as baseline evidence;
  its six passes are not attributed to original code. V2 ran from exactaabc375.
- Six new prepared-arena checks pass in0.33s, including no-hash/payload/weakref
  tripwires, equal-copy/foreign-key rejection, and constant resource-count work.
  The preexisting native/Warp/CUDA/dynamic arena set passed83cases in14.50s
  before the selected-target preparation change.
- Shared arena/CUDA contract/final graph/worker regression:204passed/no skips,
  six scale cases deselected in123.88s, `arena_selected_target_regression_v1.txt`
  SHA2564493d5c7b1e3b243db5716f4f14177594098e34c59ea6093b44b40c75319fced.
- Scale tests:9passed/no skips/77deselected in169.12s. Both lanes completed
  actual1/120/600-sample dispatcher lifecycles with CPU leaves, and each EXECUTE
  applied exactly its three new reducer facts. CUDA preparation checked exactly
  7/840/4200targets for1/120/600samples. The same60000ms test guard budget was
  retained; no additional timeout increase. `arena_worker_scale_v1.txt`:
  3582bytes, SHA256cc08638c90d584562501c3ad06b24113554c5df5dfb53ca7fe10a8d29f256203.
- After adding the final full-audit original-index identity cross-check,
  the affected final-barrier normal/fault cases passed20checks/48deselected
  in21.45s. Earlier204/9 runs used the immediately preceding loaded code;
  do not claim they executed this last additional comparison.
  `arena_final_admission_identity_v1.txt` SHA256
  ad5d6400912ba1030d940130eddb40fb8323fb04e7e0c577565e38b0a4ac5909.
- All tool sessions24882/63068/66568/76165/84016/57645/85007 are terminal.
  Final OS query found no remaining matching profile/scale/regression Python
  process. Source refresh passes unchanged AST policy for all91 LF sources;
  current catalog SHA25625090cbdeedef29c78a8675797a683f5582a6a7936f702cf5c1777979a236f8e.

The complete worker still needs prepared trace buffers/events/authorities/
scopes/tickets, exact capacity admission and whole-path allocation proof, plus
D1 final physical/numerical validation and complete G4/G5 fault/persistence
evidence. Arena checks and the successful scale probes do not complete those
requirements, C2f or C4, and do not approve renderer speed comparisons. The
coherent implementation still awaits independent review/native preflight.

## Prepared CUDA execution objects (continuing G3, afterf22a8b9)

The previous status turn freshly checked the pending authority/completion
changes:111 related CPU cases passed,9 scale cases deselected,56.68s
(`../c4_assets/cuda_prepared_authority_v2.txt`, tool27930 terminal0). This was
progress evidence for that intermediate source, not whole-path acceptance.

CUDA preparation now creates the exact dormant execution authority and its
enqueue/finish scopes. Acquire activates that original authority against its
exact prevalidated lease; it no longer constructs an authority, rehashes lease
evidence, or inserts new authority-registration tuples. Prepared registrations
alone do not authorize execution. Retirement removes the live strong owner
registration; dropping a dormant preparation clears its weak lease entry.
Copied capabilities, early finish, scope copies/reentry and retired scopes
are rejected without granting another execution.

Each transaction prepares its stage generators, original Torch device/stream
scope objects, terminal scope, completion and optional deferred profile.
`enqueue_stage` and the original stage body both check authority/order: raw
dormant access cannot consume a prepared generator, and retrieving a scope
does not authorize entering it later. Scope entry checks the original stream
identity/handle and indexed device value. The original native context managers
still save ambient state at entry and restore it on unwind. Diagnostic event
pairs are prepared once. Completion stays unpublished/unrecorded until the
original terminal event record succeeds; failures retain the original error.

Before this change, read-only inspection of the pinned remote Torch source
confirmed that device/StreamContext save prior state in `__enter__`, not their
constructors. StreamContext's constructor device index only controls its
unavailable-device branch. This inspection did not import Torch or execute
candidate code on the remote host. Actual GPU restoration still requires the
pending integrated native preflight/campaign.

Verification used local Python3.14 CPU native leaves:
- The same scratch driver `test_prepared_cuda_allocation_regression.py` fails
  both counterexamples against exact productionf22a8b9 in
  `../c4_arena_fixed_work`: new authority creation on acquire and new scope
  creation on enqueue. Only its two current fixture/test modules were copied
  into that baseline tree; CWD was the baseline root. Those copies were then
  restored, leaving the baseline tracked tree clean. Candidate:2passed2.29s;
  baseline:2failed2.76s. Raw files `cuda_prepared_allocation_baseline_v1.txt`
  (9480bytes,SHA25657946b7e0ac55c4c302ac736e83a89aab95fe7098ded2960fffb59c014119fac)
  and `cuda_prepared_allocation_candidate_v1.txt`
  (1262bytes,SHA2569c4df2522ebcdf77eb4bb2590792ddedc02abab4a9f32d4ce57afa0cf1701195).
- Related preparation/final-graph/worker regression:287passed,9 scale cases
  deselected,130.55s. `cuda_prepared_scopes_v1.txt` (88338bytes,
  SHA256f5ca753bc3d470602b3d1c02ad4146421b2c44af1783893776ed97b49b8ba656).
- Actual1/120/600 CPU-leaf worker cycles pass both lanes, retaining the same
  three-new-facts and7N preparation assertions. CUDA cycles additionally forbid
  late authority/scope/completion/profile constructors and context activation
  factories.9passed,77 deselected,161.69s. `cuda_prepared_scopes_scale_v1.txt`
  (3584bytes,SHA25667f10a57b799da820376cd420a490036578be5cdb224352a11c2368f9f0d01dd).
- Warmup original authority, normal/diagnostic, record/launch/wait/ACK failures:
  58passed29.37s. `cuda_prepared_scopes_warmup_v1.txt` (18196bytes,
  SHA256771aa13ce31aed9f714fe465ae087d71472b66b6c7870f72fad12d7447b3fea3).
- After line wrapping/import cleanup,52 affected timeline/resolution checks
  pass3.35s; `cuda_prepared_scopes_final_v1.txt` (204bytes,
  SHA256e10476d94ba9be56598cd6bc1d4e527a451f907a7b1c47a8c276df846dd66dc3).
  The287/9/58 runs preceded those formatting edits; production behavior did
  not change. Ruff and Git whitespace checks pass. The91-source LF catalog
  passes the unchanged AST policy; SHA256
  65e48904d47c31e1e7af42ca94c957f1be91f87780c87e13f9095a0a849eb981.
  All listed logs are under `../c4_assets`; tools61572/50791/74296 are terminal0.

This removes specific application-object creation, not every whole-worker
allocation. Active/stage stability signatures, lifecycle/arena trace records,
guard tickets, response/trace capacity admission, D1 physical observations,
G2 numerical readback and the complete fault/persistence matrix remain open.
No renderer timings, native deployment, integrated independent approval or
later gate activation are claimed. Continue the coherent G1-G5 implementation
and mandatory review/native campaign toward the unchanged full C4 objective.

## CUDA binding stability without runtime manifest rebuilding (afterde0dd3f)

The previous turn committedde0dd3f and completed its stated CPU evidence; it
was progress, not C4 acceptance. The main tree was clean at that revision
before this continuation. No remote execution or pending native job was used.

Inspection found that R1 stage checks still recomputed scene/workspace storage
manifests and SHA256 values, while render and active-sample checks rebuilt
nested signature containers. These checks now share a prepared, readonly
`_CudaBindingSnapshot` in the existing timeline module. It records original
owner/field bindings and tensor metadata after the original complete ABI and
provenance validation. Runtime checks compare the original objects, data
pointer, shape, stride, dtype, indexed device value and byte extent. Scene
mapping entries are also checked; tensor metadata checks are deduplicated by
original tensor identity during preparation. No renderer/kernel algorithm or
execution authority is implemented by this helper.

Snapshot owner identity prevents a copied stage from validating a different
object through the original snapshot. Fixed argument/resource sequences must
be actual immutable tuples before capture; a mutable list cannot be accepted
and later changed behind an identity-only comparison. Hashes remain in the
untimed ABI/provenance construction and evidence paths. The obsolete transform
signature builder was removed. Active-sample checks retain original event
handle observations, including diagnostic events, without rebuilding tuples.

Evidence (Python3.14, CPU arrays/native-call doubles):
- The unchanged two selected counterexamples fail against productionde0dd3f
  in `../c4_arena_fixed_work`: the R1 scene stability check reaches the original
  manifest hash, and active enqueue rebuilds its configuration signature.
 2failed/45deselected1.31s; `cuda_binding_baseline_v1.txt` (10066bytes,
  SHA25612c89417ee89d6b2376f3c896378a726ed659d458bf7159c7e511e7f04065f1a).
  Baseline CWD/revision were checked, only two test modules were copied; those
  copies were removed/restored after the run, leaving its tracked tree clean.
- Initial focused drift/no-rebuild checks:34passed/117deselected3.32s;
  `cuda_binding_counterexamples_v1.txt` (236bytes,
  SHA25650005c99153abfd3217142f4643c26af8e9693f0d6ff5e154a6fbf76a38cdef5).
- Complete selected regression, including both1/120/600 worker cycles, actual
  production R1 transaction with CPU leaves, warmup normal/diagnostic/failure,
  final graph/barrier and worker lifecycle:429passed/no skips312.28s;
  `cuda_binding_regression_v1.txt` (128236bytes,
  SHA25660d22e8b4d461106fe0488949489425e70444fc5413f8468313f2faef656893f).
  The CUDA worker cycles forbid late binding-snapshot construction in addition
  to the preceding authority/scope/completion/profile constructor checks.
- After adding the preparation-only immutable-tuple requirement, the affected
  dynamic binding, arena resolution, render preparation and warmup set passes
  215checks41.35s. `cuda_binding_immutable_v1.txt` (532bytes,
  SHA256266cbe2e8b3a215f77ef3714569257b871b5d6ded1624e400b18cfbc5a1926b4).
  The429 run precedes this added admission check; do not attribute it to the
  final bytes. A subsequent one-line wrap has no behavior change.
- New/helper/timeline/arena/worker-test Ruff checks and Git whitespace pass.
  The three older CUDA stage modules plus render-preparation test have the
  same60 preexisting Ruff findings asde0dd3f, with no new diagnostic after
  normalizing path/line movement (`cuda_binding_lint_{baseline,candidate}_v2.json`).
  This is not a claim that whole-repository lint is clean. The unchanged AST
  policy accepts all91 LF sources; final catalog SHA256
  ff32bf78307afda62d9d365ce5ff10089e40f3f97a4d518694556ef01a31bdbe.
  Logs are under `../c4_assets`; tools11034/12787/49730 are terminal0 and a
  scoped OS query found no remaining matching test Python process.

Tensor contents may change as intended, but tensor/field/storage replacement
and shape/stride/dtype/device drift reject. Equal indexed device values remain
valid. These CPU tests do not prove native numerical correctness or globally
allocation-free execution. Remaining runtime work includes the first-hit
argument/callable binding, other arena/template signatures, Warp/worker/guard
records and capacity admission. D1/G2 observation/readback, complete G4/G5
fault/persistence evidence and the integrated independent/native review remain
mandatory. No speed comparison or later gate has opened.

## Prepared first-hit ABI and direct stream metadata checks (after7304975)

Current author work binds the CUDA first-hit callable and its35 positional
arguments during stage preparation, like the existing raygen/shade paths.
Enqueue uses those exact retained objects. Untimed full validation compares the
original module entry and argument ABI; the prepared stability snapshot detects
later field/tuple changes. The shared argument comparison rejects mutable lists
and equal-valued scalars of the wrong type. The native diagnostic collector now
observes the actual bound callable, not a fresh module namespace lookup.
Context validation compares the already indexed device value and exact integer
stream handle without constructing another device or coercing changed metadata.

The C++ declaration supplies the independent35-name ABI-order check. Actual
prepared graph/transaction CPU tests forbid late callable lookup, argument
construction and device construction. Against unchanged production7304975 in
`../c4_arena_fixed_work`, both late-work counterexamples fail; only their two
test modules were copied, then restored, leaving the baseline tracked clean.
The focused candidate run passes170checks14.31s. These are CPU native-call
doubles, not GPU numerical or performance results.

The wider first run stopped with325passed/1failed113.92s: the existing warmup
test constructed `render_graph_binding.json` without `phase_launch_counts`.
That same failure reproduces at production7304975 (1failed3.07s). The test's
R0 graph fixture now supplies its expected raygen/first-hit/shade launch counts;
the replay validator is unchanged. No field is made optional or defaulted.

Raw logs under `../c4_assets`:
- `cuda_fixed_first_hit_v1.txt`:532bytes, SHA256
  47be7bb8ccf8769fe4cce9a806dc83558a7940360471e434e2f449a8cc1a638a.
- `cuda_fixed_first_hit_baseline_v1.txt`:11374bytes, SHA256
  6d5a24707a04d5360ce56bd5a5f6b649d5bcc97b7e728a3cb506d33409c72ad0.
- `cuda_fixed_first_hit_regression_v1.txt`:113252bytes, SHA256
  55bc6d216174a1086088bb31c666a9d7193eaa10b92fb90d70d6dc040621dcf4.
- `warmup_fixture_baseline_7304975.txt`:16536bytes, SHA256
  46d77695a0a6289b1256c20e191506066ecff5607ccd52530eaabf3c1d337fcb.

Ruff comparison of the six affected Python modules/tests reports no added
diagnostics (128baseline,126candidate; two preexisting long lines removed).
This is not whole-repository lint acceptance. Source catalog remains subject
to the unchanged91-source AST policy. Full packet acceptance, native final
observations/numerical validation, complete fixed-work/capacity and faults,
independent review and later C4 stages remain open.

After the fixture repair, the complete selected regression passes421checks,
no skips,327.06s, including both1/120/600 worker cycles, binding/timeline,
warmup/final graph and actual graph/warmup evidence-helper tests. Log
`cuda_fixed_first_hit_regression_v2.txt`:126878bytes, SHA256
ee2a83e634afefea9c93c634a315ddcd30556374cb09df80c5d71ac20f44aa43.
Tool37192 is terminal0. Catalog SHA256
f6758b695b4804034ad5cdec53d5ed23e9a477033037f16f84a0ba64ba9e070b.
No native job was launched. The scope remains unreviewed author work in the
required coherent G1-G5 packet, with the full C4 objective still active.
