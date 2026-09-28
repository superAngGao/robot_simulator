# C4c1b2 recording — native stream ownership repair

Initiative: q54-optical-lab-p12-4c4c1b2-recording
Stage: implementation-note
Author: codex
Version: v2
Date: 2026-09-28
Status: in_review
Related Files: recording implementation note v1 and independent review v1
Owner Summary: Reject borrowed Warp stream handles and revalidate the retained native owner before recording or accepting a bundle.

The independent v1 report is preserved verbatim. Its F1 is addressed by
requiring an exact Warp Stream with `owner is True` before module loading or
recording. Ownership joins the frozen stream facts and is checked on every
untimed full preflight. This is native Warp ownership only; external-stream
interop is not silently added to this checkpoint.

Three GPU regressions reject an external-handle wrapper before loading or
recording, reject same-object ownership drift before recording/validation,
and check target-one/ambient-zero nondefault stream restoration when recording
raises. Render bundle execution remains forbidden in these tests. The test
restores the genuine owner flag before ordinary cleanup.

Author verification on 101, robot_sim, scratch c4_binding_work: the same six
test files from v1 now report **203 passed in 6.54s**. On the clean v1 candidate
c09d4e7, the full non-slow regression completed **1994 passed, 39 skipped,
117 deselected, 1 warning in 373.27s**. That full run predates this narrow
repair and is not represented as a full v2 run. Fresh-session v2 review follows.

No shared workflow, stage activation, product worker, compatibility driver,
result/baseline or performance gate is changed. Scope remains recording and
preflight, not C4c1c supervised execution or full C4c1 approval.
