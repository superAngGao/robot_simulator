# C2d1: preserve conservative scope intake in failed setup handlers

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: implementation-note
Author: codex
Version: v5
Date: 2026-10-03
Status: in_review
Related Files: custody/repair designs v2; code-review v4; implementation-note v4
Owner Summary: Repair F6 in the actual prepare exception composition. Failed
normal work stays closed while owner-thread failure metadata can tighten scope.
Fresh independent review required; no runtime or result gate opens.

Base6b5080fbfa00309c757ee3467fa6cb53b78fd57d. Full independent v4 report archived
verbatim SHA25623e785818ebe171f65db37bea1429d1b8ab94e70e94f145ddf5858dbd313eb53;
it closes original F1–F5 but requests changes for F6. Previous candidate's author
751-pass run is separately archived in test-evidencev4, not new-candidate proof.

## Changes

Owner-thread unknown-scope reporting before termination is conservative metadata,
not work admission. It can follow a recorded guard failure or compromise, never
clear it. Validate source/reason before mutating inherited state. The existing
scope implementation and reducer remain sole truth; no added schema/booleans.
Wrong-thread and terminal/post-intent calls still reject and observe failure.
The actual unchanged prepare_* handler can now mark unknown then invoke _abort,
which retains its exception and immediately commits full terminal quarantine.

Scope declarations, inherited notifications, permit entry and returned-owner
adoption now use the same existing controller RLock for their CPU bookkeeping.
A concurrent fault cannot split descriptor/owner facts from their notification
chain and prevent the subsequent conservative report. No native call, ACK,
sync, copy, allocation or reference release occurs inside these metadata methods;
the actual device leaf executes only after the method returns. Existing no-lock-
across-native/destructor and terminal-admission tests remain applicable.

Protected C0 files, upload loops, worker protocols, bootstrap admission, source
inventory/policy, product/result gates and shared workflow unchanged. Only the
lifecycle source catalog digest is repinned. No separate guardian or cleanup.

## Verification added

- Both actual Warp/CUDA preparation functions, with a fixed CPU import finder
  that raises at the selected runtime root before importing any runtime. Healthy
  import-failure controls and deterministic wrong-thread failures during import
  exercise the real mark-unknown-before-_abort ordering. This is test-only fixed
  instrumentation, never a production plugin/callback or GPU simulation claim.
- Caught/propagated abort variants assert exact terminal evidence before outer
  exit, original ImportError cause/source retention, unknown=true, full custody,
  no sync/new ARM/COMPLETE/release, and unchanged single termination afterward.
- Four real supervisor process cases use that same fixture. Their inner snapshot
  must match final worker evidence; no runtime import, destructor or atexit,
  actual returncode74 and observed reap required.
- Wrong-thread, terminal-sync and post-intent scope reporting negatives; sealed
  guardian failure metadata; deterministic thread scheduling at descriptor and
  owner facts proves the CPU notification chain is not interrupted. Scheduling
  wraps a test observation point, not a copied production state machine.

Local Python3.14 lifecycle/evidence/wait-core309 passed before the two sealed
variants; final suite with source-policy476 passed8.65s, exit0. Initial Ruff
E402 on a multiline test import was fixed by placing noqa on the import line.
Ruff/format, remote regression and fresh review are independently checked at
the subsequent frozen checkpoint, not inferred from previous counts.

Still closed: source-frozen guarded runtime/module custody and real Warp/Torch
multi-GPU faults; full setup graph/barrier/warmup/readiness; C2e/f arenas;
product preparation; a3b/results/baseline; R2/host/S and later C4/B3/shipping.
CPU proofs do not establish GPU memory-release safety. C4 remains incomplete.
