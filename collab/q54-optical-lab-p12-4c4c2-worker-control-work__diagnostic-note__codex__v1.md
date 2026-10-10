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
