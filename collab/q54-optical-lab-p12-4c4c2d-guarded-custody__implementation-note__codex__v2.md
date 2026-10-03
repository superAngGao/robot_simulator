# C2d1 repair candidate: terminal admission and compatible histories

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: implementation-note
Author: codex
Version: v2
Date: 2026-10-03
Status: in_review
Related Files: implementation-note v1; code-review v1; accepted repair-design v2
Owner Summary: Implement F1/F2 repairs and RD1/RD2 design boundaries. This is a
CPU custody candidate pending fresh independent code review, not GPU admission.

Base: `be3a4616fb2e5507bd9074e7ecc271b3eb739ae1`; prior failing code `4ee33ab`.
Requirements: original custody design v2 plus approved repair-design v2 at
`2ea88c8`, SHA256 `01a464293f4474f1665ee0ae3d2eda5477df43cec2f2ebb7374dcf87195656e2`.
Complete independent reports remain unchanged; no shared policy edits.

## Actual changes and gaps

- One reducer owns the new once-only terminal_operation_failed event and exact
  summary boolean. A failure preserves the actual guard/guardian state; it is
  not relabelled as sync/cleanup that never began. New admission/success after
  it is rejected. The unpublished v3 schema changes explicitly; C0 is unchanged.
- Terminal entry, sticky failure, custody publication, ARM/sync/release markers,
  success observations and final intent use the same controller RLock. ACK,
  native sync and actual guardian reference release occur outside it. Admission
  is conservative before-call ordering, not atomicity with GPU execution.
- A fault before admission prevents ARM/sync/release; one observed after a
  genuinely started operation prevents success. Full versus remaining custody
  follows actual release admission/state. Exact existing guardian release code
  remains the sole implementation. In-flight guards never retry or COMPLETE.
- Joint evidence matches pending roles too, not only received ACKs. Terminal,
  failed or compromised prefixes bound supervisor ARM/COMPLETE to actual worker
  attempts. Only a healthy-IDLE termination prefix can permit its one not-yet-
  observed next emergency. Descriptor/owner scope is derived before notification.
- Source catalog repins only the two changed production modules. No inventory,
  AST policy, C0 driver/protocol, bootstrap admission or activation changes.

Tests add real socket delayed-ACK thread/context faults, deterministic terminal
entry lock schedules, pre-sync/pre-release/post-release failures, and threads
inside sync/destructors to check that no controller lock spans those calls.
Evidence cases include original contradictory pairs; pending/failed ACK positive
uncertainty; healthy lagging and terminating prefixes; early foreign/unknown
descriptor, returned-owner and explicit-scope prefixes; duplicate fault and
forbidden post-fault success. Scheduling wrappers in entry/boundary tests are
test instrumentation, not production callbacks or a copied controller.

## Author evidence before remote candidate freeze

Local `C:/Python314/python.exe -B -m pytest`:

- existing controller+handwritten fixtures after initial code edit:57 passed;
- expanded controller, fixture and wait-core suite:238 passed in3.56s;
- those three plus native source-policy suite:403 passed in11.88s, actual exit0.

Ruff check/format and git diff --check pass. Initial four owner-unknown fixture
failures were an incorrect literal in the test; changed to the existing formal
UNKNOWN_DEVICE_OWNER constant, without relaxing the production validator.
Initial Ruff formatting findings were corrected. Initial catalog repin script
used the wrong JSON member name and exited before editing; corrected to the
existing source_digests field. Catalog packaging uses Git LF bytes; production
verification still compares exact bytes without newline normalization.

Reran original external `c2d1-review-runner.py`: runner24968(parent15484),
child32272(parent24968), actual exit1 at the now-rejected conflicting-role
artifact. The unmodified script expects the old acceptance and therefore stops
at that rejection; it is not reported as a passing full script. Before that
point both original delayed-ACK cases now show zero sync, no release authority
and full quarantine. The second F2 case is covered by explicit pytest cases.
Exact process query afterwards found neither PID. Local pytest command handles
returned actual exit0; their short-lived OS PIDs were not sampled.

Primary183 SSH40031 again timed out. Authorized fallback101 has clean prior
candidate worktree4ee33ab; original dirty checkout's four paths are preserved.
Unrelated training GPU PID2336351 was observed and left untouched. No GPU work
or installation in this slice. Remote real-process and broad source/C0 tests
must run on the committed candidate; results are not preclaimed here.

## Review and still-closed work

Fresh independent code review must close original F1/F2 using the archived
counterexamples and the actual new tests. CPU-only fixtures prove control and
replay, not CUDA/Warp completion or resource-release safety. Prior counts are
not evidence that the repaired candidate passes remote acceptance.

Still pending: integrated C2d source-frozen role/module custody before runtime
imports, real Warp/Torch and multi-GPU fault matrices, complete graph/barrier/
warmup/readiness, C2e/C2f arenas and acceptance. Product GPU preparation, a3b,
result/baseline, R2/host/S, later C4/B3 and shipping activation stay closed.
Cleanup/adoption obligations remain unchanged. This is not C4 completion.
