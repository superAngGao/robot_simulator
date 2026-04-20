Initiative: test-speed-slow-marking
Stage: review
Author: codex
Version: v1
Date: 2026-04-20
Status: draft
Related Files:
- collab/test-speed-slow-marking__implementation-note__claude__v1.md

## Findings

1. [medium] `OPEN_QUESTIONS.md:1095` — Q40 itself was not updated to reflect the partial resolution claimed in the implementation note. The note says the commit gate dropped from about 15 minutes to about 3 minutes, but the tracked open question still says `pytest -m "not slow"` takes about 5 minutes and still frames the problem entirely in its pre-change state. For an initiative explicitly scoped around Q40, leaving Q40 stale makes the durable project record lag behind the actual change.

2. [medium] `OPEN_QUESTIONS.md:57` and `OPEN_QUESTIONS.md:1201` — the same file now says both that GPU ConvexHull support is unresolved ("GPU 端未实现 ... 见 Q41") and that Q41 is resolved. That contradiction should be cleaned up in the same change that closes Q41, otherwise future readers will get conflicting status from one document.

3. [low] `OPEN_QUESTIONS.md:1151` and `OPEN_QUESTIONS.md:1201` — closing Q41 and Q44 is unrelated to the narrow `test-speed-slow-marking` initiative and makes this review thread harder to reason about. Bundling unrelated open-question resolution into a test-speed change increases review surface and makes revert / blame less clean.

## Coverage Gaps

- The collection-only verification is good for confirming marker placement, but this review did not execute the newly marked tests. Their behavior remains covered only by prior assumptions and future full-suite runs.
- There is no durable repo-level check that Q40's new split is reflected consistently in long-lived docs. In practice that means the implementation note, `CLAUDE.md`, and `OPEN_QUESTIONS.md` can drift, which already happened for Q40/Q41.
- Marking `tests/gpu/solvers/test_solver_backends.py::TestCrossSolverAgreement::test_single_sphere_all_solvers_agree` as slow moves all cross-solver agreement coverage for that file out of the commit gate. That may be acceptable, but it should be treated as an explicit tradeoff rather than an invisible consequence.

## Residual Risks

- The selected 20-second threshold is heuristic and hardware-dependent. Tests near the boundary may oscillate between acceptable and frustrating on slower GPUs.
- The commit gate is now defined in `CLAUDE.md`, but no shared project-level document currently records the same split. If that guidance broadens beyond Claude's workflow, it should eventually move to a neutral repo doc.
- Q40 should remain open until the team validates the new timings across a few real development sessions, as the implementation note already suggests.
