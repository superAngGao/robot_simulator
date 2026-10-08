Initiative: q54-optical-lab-p12-4c4c2d-native-global-publication
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: native-global-publication design-request/design-review/implementation-gap-review v1; runtime-adoption design-request v2
Owner Summary: Implement the approved exact native interval observation and direct custody relation. The first independent implementation gap review requested changes. This revision repairs its three reproduced findings, but actual Go2 input preparation and whole C2d3 acceptance remain unfinished.

## Scope and review history

Design snapshot `80387e8d8199e9dc6741924b95ff1aeeae047b45` was independently
approved for implementation only; report archived at `8b7c9b8`. Implementation
candidate `4933b7ca46eb594f51c924ce88f8bd124bb14ba8` was independently gap-reviewed
by `/root/native_global_implementation_gap_v1` with Request changes. Its complete
report was archived unchanged at `1b597b5`; raw report SHA-256 is
`b3ba499896359d140d66fefefea46c8ae2a4d58019a1fd0e383d4a3ca20db32b`.

The original implementation advances the environment schema to v4, adds only
the three approved literal native-global relations to the new Go2 profile,
and retains historical profiles. Existing finder/loader boundaries issue
identity-bound calls, hold exposed globals before validation/adoption, and admit
them only on their own successful exec return. No exact creator or native hidden
ownership edge is asserted. The descriptive native-global record remains a
component, not the required final aggregate runtime-custody schema.

## Repairs after independent findings

- F1: The private call binding now retains the original exact spec, and frame
  and index projections must match it. A substituted spec is held before
  rejection. Pending return, completed audit and evidence issuance each reject
  the preserved coherent-copy counterexample with sticky failure.
- F2: Event issuance keeps immutable facts and exact record/module/frame roots
  before exposing the mutable projection. Completion, segment audit and healthy
  issuance validate that history without reconstructing missing events. Paired
  token mutation, changed checkpoint, removal, reorder and copied events reject;
  issuance failures poison the session and preserve actual custody.
- F3: Every replayed call phase must equal its configured module phase. An
  exec-return observation closes the ancestor interval for subsequent nested
  events, including failed prefixes. Valid request-entry nesting and unrelated
  sibling return orders remain accepted. Parsed data still grants no authority.

The source policy/bootstrap class hashes and active v10 catalog are repinned to
the exact staged Git LF bytes. No admission table, watchdog bound, product gate,
backend selection criterion or later phase is changed by these repairs.

## Actual supervised Go2 failure on the first candidate

Primary host: tileops-dev (183.166.183.33:40031).
Checkout: `/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`.
Interpreter: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`.
Invocation:

```text
timeout 420 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/run_c2d3_go2_guarded_author_v9.py
```

Fixed author invoker SHA-256:
`c95986ef4e21546079bb9dab2344248140ac876a09ebc8b1921b28caa3c8f038`.
It calls the existing guarded driver; no second importer/uploader is present.
Actual frozen revision: `4933b7ca46eb594f51c924ce88f8bd124bb14ba8`.
Artifact directory:
`/data2/shared/ang.gao/c2d3-4933b7c-go2-960-guarded-author-v9`.
Local byte copy retained outside the checkout under `../c4_assets/`.

Warp target1, real frozen Go2 960x640, original 1000ms ACK and 120000ms operation
deadline. Parent/supervisor/worker: 4118656/4118679/4118680. Setup sequence2
(materialization) expired; supervisor exited normally after killing/reaping
worker -9, summary EXITED/complete=true/deadline_expired=true, two earlier normal
completions only. Invoker exit1 correctly rejects this as success. No successful
canonical materialization, upload, readback, native-global captured admission or
worker cleanup proof is present. The retained worker prefix predates the failed
segment and cannot override the supervisor trace.

CPU suites overlapped on the host, so timeout cause is not isolated and this is
not performance evidence. No deadline extension is made. Scoped ps checks after
completion found no owned parent/supervisor/worker or author pytest processes;
NVML listed only pre-existing MPS server2238410 at60MiB. Nothing was signaled by
the author outside the existing supervisor protocol.

Raw artifact SHA-256:

| File | SHA-256 |
| --- | --- |
| request.json | 16aaa05dc900f33b035c4397d88eb643971d48593bda89cbcad7d7ff16d5b30b |
| identity.json | 5a2d5a92a0a4ec23fa4a7be647a5e183c9c3f5c440aa286cf41aed03c09b6740 |
| supervisor_stdout.json | 84944b1fddc74594450cf32112990c3cff856024cf7535504d6a96129a16e8c4 |
| custody_artifact.json | c60b2e4e2505c693f148c70242badb4ce72742aca2f517ef6b80eaeadedd95d5 |
| prefix.json | 92d700303a0d5e981485129c972c850f72693d60b255019327229f8a9faea3d6 |
| external_environment.json | cbf2569620b3546fa2a1ea00fda18373b60da47cd3f9038b53f7be3f9c66d4e7 |
| setup_definitions.json | 04b2c24873ffd613d5ebdcc783c9ef7aa5708537d7896840aa90b5959344c18c |
| supervisor_stderr.txt | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |

## Validation and unfinished acceptance

On the first candidate, author remote CPU regression passed538/skipped3; exact
independent commands and separate215 and306/3 results are in the full gap report.
These are CPU contract evidence only. Repair validation is recorded with its
exact frozen revision in the next review/evidence record.

Still required: successful captured both-lane canonical/real Go2 materialization
and actual upload/readback, import/init/allocation/copy/event/adoption/audit
faults and original-deadline hangs, real dual-device ambient/foreign/UNKNOWN
cases, sole supervised terminal sync/cleanup and failures, aggregate versioned
runtime-custody schema/replay, and whole-stage independent code review.

C2d3 and C4 are incomplete. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b,
result/baseline, host payload, R2 and shipping stay closed. No optical sample or
final backend selection is performed here.
