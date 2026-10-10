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
Its report is pending. It may identify a narrow design supplement if existing
guard/source or sealed-resource constraints conflict with final validation;
it cannot grant hardware success, product activation or a later gate.

No reader regression, two-sample normal exit or this diagnostic completes C2f
or C4. The original whole objective, full failure coverage, systematic cleanup,
independent acceptance and later C4 stages remain required.
