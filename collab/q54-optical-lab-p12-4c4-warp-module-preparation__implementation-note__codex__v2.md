# Warp compilation integration candidate

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: implementation in progress, not acceptance
Author: Codex
Version: v2
Date: 2026-10-09
Status: draft
Related Files: warp-module-preparation design-request/design-review v2; implementation-note v1
Owner Summary: Connected the existing source session to the private Warp cache,
  guarded imports, three serial module loads and retained executable set. Reused
  the existing supervisor and native driver for author verification. Native
  execution, integrated independent code review and whole C4 remain unproved.

Base: 179f247d2c40984bb0c7e23b2556c8fdc346afc6. This pending candidate continues
the approved design v2; no new process manager, renderer, source loader, resource
guardian or public product entry is added. The sole new production helper owns
Warp compilation configuration and facts used by the source session.

The session configures a fresh private cache before Warp initialization. After
input validation it imports the three existing optical modules under one setup
guard and compiles each under its own existing guard. Module/definition/exec
mapping owners are retained before loading; the returned executable is retained
before adoption. Successful records bind generated source, binary, metadata,
module options/hash, executable and completed guard identities. Aggregate issuance
uses the actual IDLE boundary. Later validation rejects copied module sets and
changed options, definitions, executable mappings, handles or artifact bytes.

Source failure retention was factored from termination, removing the previous
source/admission recursive callback. Default source failures retain the existing
failed-guard policy. The new preparation's CPU entry/issuance failures validate
the original runtime binding and use the existing lifecycle's disposition from
the actual state: one emergency target synchronization for complete target-only
custody, quarantine for uncertain custody. Successful guards are not relabeled.

Local Python3.14 author checks:

- Session80444: guarded runtime binding, source catalog and source policy tests:
  239 passed,4 skipped,1 failed in62.74s. The failure was the expected Warp-only
  catalog set lacking the newly added warp_setup.py; corrected that assertion.
- New real-protocol/simulated-Warp entry tests initially had2 fixture mistakes
  (a nonexistent coverage API and a stronger-than-production check_live double).
  Corrected the fixtures and added explicit compromised-scope handling before
  runtime validation to preserve actual IDLE history.
- Focused rerun:11 passed,40 deselected in2.09s. Tests cover import/compile order,
  retained partial roots, original guarded failures, CPU entry/issuance/repeat
  failures, unknown scope, missing guardian owner and changed binding identity.
- Broader session42604 is still running at this record's creation. Do not infer
  completion or native behavior from these CPU tests.

The existing sacrificial driver has a Warp compilation case that observes real
module loading and same-session reuse. It extends the existing acceptance runner;
it creates no alternate upload/compiler/supervisor implementation. Input-phase
custody retains its original meaning; compilation observations are downstream
raw evidence only. This native case has not yet executed at this record cut.

Pending: real Go2 at both resolutions, target1/ambient0 restoration, compiler
input/flag verification, cache and executable negatives, partial-publication/
adoption/CPU-boundary failures and original-guard hang termination; independent
code review and acceptance. Then continue timeline/arena, graph, ten warmups,
final readiness/seal/barrier, rendering correctness/lifecycle matrices, C4c3/c4,
C4d/B3 and systematic whole-C4 refactor/review. No product PREPARE, performance,
result/baseline or backend-selection gate is opened by this candidate.
