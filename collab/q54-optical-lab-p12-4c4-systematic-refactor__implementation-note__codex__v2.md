Initiative: Q54 / P12.4 / C4
Stage: execution and setup consolidation, integration still incomplete
Author: codex
Version: v2
Date: 2026-10-10
Status: in_review
Related Files: runtime-integration design-request codex v3; systematic-refactor gap-review codex v1; input-precision implementation-note codex v4
Owner Summary: Consolidate existing event initialization and frame binding; remove diagnostic-dependent submission latching; delay CUDA lease resolution until supervised wait acknowledgement. These changes do not implement final graph admission or provide a speed result. C4 remains active.

## Candidate and changes

Base: c9d537778659db8ac79ba509bdf2f73d971a8c23, local c4_final_graph,
branch implementation/c4-final-graph. Candidate commit is the commit containing
this note; exact revision is supplied separately to the independent reviewer.
The prior turn's uncommitted work is included and requires review with this delta.

- M2: guarded setup, standalone CUDA arena, and CUDA compatibility event builders
  share `_prepare_native_events` and existing `_native_resource_scope`. Terminal
  initialization is recorded last and its wait covers all diagnostic records.
  Captured setup still requires its exact custody context and adopts each return
  before the next fallible operation. No event factory/callback framework.
- M3: each actual CUDA native launch now calls `before_native_launch`; setup
  latching no longer hides in `record_diagnostic_start`. Triangle and plane
  updates each latch their own call; an empty phase issues no launch latch.
  Kernel sources, extension recipes and numerical algorithms are unchanged.
- M5: CUDA active-sample resolution optionally uses the exact guarded controller.
  It retains the transaction/output holds throughout the physical wait and
  COMPLETE/DISARMED exchange, then commits the existing arena lease. Failed
  wait/acknowledgement/binding checks poison the generation without resolving it.
  The standalone arena retains its existing wait-and-resolve behavior.
- M6: named setup warmup bookkeeping replaces positional lists, and the required
  ten-frame count has one definition. Identity registries and retirement remain.
- M4 groundwork: the frame binder is shared; the CUDA C3 adapter lives in its
  production owner with exact compatibility aliases; camera sample and replay
  selection are distinct. The final graph consumer and Warp arena are still
  missing. This groundwork is not accepted as final graph admission.

Source catalog v17 is refreshed over exact LF release bytes, 88 entries, SHA256
`ad4801eb4eb0b511e0eb711c66234c382a009c53cd383557df665e012a312094`.
The production CUDA C3 adapter joins the CUDA-only inventory; current bootstrap
phases still do not import/admit final arena execution. No source policy or
closed product dispatcher is weakened.

## Author verification before independent review

Local C:/Python314/python.exe, no GPU runtime execution:

- Initial resolution/arena checks: 59 passed.
- Launch-boundary change plus existing warmup/timeline/resolution checks:
  68 passed in 73.84s.
- Explicit launch-entry and guarded lifecycle checks: 124 passed in 7.56s.
- Broader intermediate check: 482 passed, 4 skipped, 1 failed in 147.71s.
  Failure was the literal CUDA source inventory expectation, which lacked the
  newly relocated production adapter; its exact expected set was updated.
  Skips cover unavailable real Go2 assets and POSIX-only checkout verification.
- After event-loop consolidation: preparation/timeline/resolution checks,
  105 passed in 13.87s. They check terminal-last ordering through actual shared
  builders and include real guard/arena state transitions with CPU native doubles.
- Catalog generation validates every inventoried source against the unchanged
  AST policy. `git diff --check` passed (Windows LF checkout warning only).

Counts overlap and are not additive. The full post-consolidation command and
independent review will be recorded separately. CPU doubles prove coordination,
not CUDA event/device behavior. No GPU, native image, or performance result is
claimed for this candidate. Existing precision evidence belongs to 5e79749.

## Required continuation

Finish the coherent both-lane final graph over warmed resources, including exact
final-readiness origin, C3/arena plans, immutable coverage and acknowledged final
barrier; then R1 and full lifecycle/failure validation. Complete the outstanding
precision GPU counterexamples, dynamic update/rebuild/render and native review.
M1/M4 and the whole-C4 refactor/review remain open. No product activation,
baseline, timing claim, shipping, backend selection or later gate is issued.
