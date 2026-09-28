# C4 completion work ledger

Initiative: q54-optical-lab-c4-completion
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-09-28
Status: in_review
Related Files: C4 parent design v1/v2, C4c design v7, C4c1 design v1/v2/v3
Owner Summary: Continue through C4 using fresh independent reviews at each checkpoint.

The owner now requests continuous progress until C4 is complete, following the
shared review/fix/re-review workflow. This ledger schedules the existing approved
designs; it does not change any numerical, lifetime, timing or activation rule.

Already accepted: C4a control/evidence, C4b CUDA arena, C4c0 closed worker
foundation, C4c1a readiness, b1 kernel source extraction and b2 storage checkpoint
at aa97f09. Latest evidence-only HEAD at start: 48bbc20.

Remaining ordered checkpoints:

1. C4c1b: deterministic reset, fixed four-entry shade ABI, camera/evidence,
   immutable factory and recorded bundle. Public compatibility stays on its
   existing lifecycle and never drives the unfinished recorded bundle.
2. C4c1c: supervised sacrificial setup/recording/execution, ACK-before-wait,
   whole-resource retention and fault matrix; independent CPU oracle.
3. C4c2: Warp transactions, exact C3/C4 live authority, depth-one arena and
   R0/R1 binding, with prepared O(1) acquire and same-slice supervised worker
   post-submit disconnect tests before any submission transition is enabled.
4. C4c3: production-owned static Warp/CUDA-LBVH binding for device-only R2.
5. C4c4: aggregate real worker lifecycle/instrumentation closure for both
   candidates within the approved worker trust boundary, including the full
   eligible construction and transport-failure matrices.
6. C4d: normal drain/reuse/backpressure, poison/quarantine, complete resource
   manifests and C0 guardian handoff, respecting the external-hold addendum.
7. Independent aggregate C4 gap review against its acceptance inventory.

Any missing detailed design is reviewed independently before implementing its
new policy. Each code candidate is frozen and tested, reviewed by a fresh
session, repaired and re-reviewed until that exact scope passes. Reports remain
verbatim and approval cannot be inherited by an untested later stage.

B3 host delivery, joint result activation, a3b, shipping and baseline outlets
remain separate from this C4 completion request. No default backend selection,
performance result, push or losing-lane deletion is implied.

Primary 183 SSH was rechecked and timed out. Work continues in the existing
isolated 101 branch and local mirror, preserving all original dirty worktrees.
The original b1 commit objects still require reconciliation when 183 returns;
recovered content is not claimed to have the unavailable original ancestry.
