Initiative: Q54 / P12.4 / C4
Stage: implementation-note
Author: codex
Version: v4
Date: 2026-10-10
Status: in_review
Related Files: runtime-integration design-request v3 sections 5-8; systematic-refactor gap-review v1; code-review v2; implementation-note v3
Owner Summary: R0 final preparation now connects observed warmups, new final factories and C3 plans, immutable resource sealing, and an acknowledged target barrier. Both lanes have real Go2 primary-pass success and injected final-sync failure evidence at the revisions below. Later acquisition/submission restrictions have CPU verification only. Independent approval, diagnostic native coverage, final sample execution and full C4 remain pending.

Scope and identity

Base is evidence commit 3d77448291643ebc1a512fe8c4a9d1be2e78341d; last
approved code is 99eb1f3aff0436134346d08b2be4f235a910b19d. The implementation
starts at c59edc74d7d766eb71bd4b7f81d94045440cb889. Combined lifecycle replay
is repaired at e31b83a7b03bc84df8cf63ba7ad6ff64898b3ee3. Fixed native failure
support is f05235f14c75c0c75b2bef7f473d66df8d1ec8b8; its optics/optical_worker
source bytes are identical to e31b83a. The exact subsequent review candidate is
recorded in the independent review request, avoiding a self-referential commit.
This note records author observations, not independent approval.

Implementation

- Warp setup and final execution share one recorded phase/reset/event loop.
  The new Warp adapter delegates slots, FIFO, generations and resource holds to
  existing NativeOpticalArena. It adds no kernel, upload path or slot state machine.
- Both lanes share the existing guarded terminal wait. Binding checks remain
  before waiting and after actual DISARMED, before committing lease resolution.
- The sole native arena spec constructor moves from benchmarks/optical/arena.py
  into optical_worker/arena.py. The benchmark retains its full gate validation
  and delegates construction. A captured session admits its exact frozen R0 row
  before materialization/runtime initialization. Fixed native drivers use three
  samples; the protocol numeric bound alone does not satisfy future C2f response
  and trace capacity budgeting.
- Final graph construction consumes ten actual registered warmup completions,
  certifies NEW readiness, creates NEW factories and frame/C3/dormant plans, and
  reuses existing physical outputs/events. Provisional readiness stays unchanged.
  The origin refers only to upstream input, warmup and compiled-module facts.
- Captured CUDA C3 consumes that exact locally observed origin instead of doing
  another unguarded device-to-host validation per frame. Standalone C3 keeps its
  original validation. One controlled capability per slot owns shared events;
  every frame still has a distinct template and plan.
- The source session finishes construction, seals complete resource coverage
  once, runs an explicit target/stream synchronization under the normal setup
  guard, waits for actual DISARMED, rechecks the graph and immutable coverage,
  then commits quiescence and acquisition admission. Barrier failure retains
  the sealed graph and quarantines without an emergency synchronization retry.
- Lifecycle v4 describes the final seal/barrier separately. Historical v3 keeps
  its meanings. Runtime custody and combined supervisor validation replay the
  explicit nested lifecycle version. The input-custody readiness flag continues
  to describe the input stage and is not final execution admission.
- Existing CUDA configuration signatures now also freeze native event handles;
  Warp retains and checks native event/stream bindings in final validation,
  execution and resolution. Object identity alone did not catch handle drift.
- After the native revisions below, acquisition also rejects a newly compromised
  device scope. The common lease begin-enqueue path now calls the captured
  controller's submission latch, whose implementation remains closed. Thus
  acquiring/cancelling a plan does not open measured execution through a raw
  arena API. Standalone arenas retain their existing contract. These last two
  changes are currently CPU-verified, not freshly exercised on a GPU.

Verification and discovered defects

CPU fixtures use actual source-session, guard, readiness/C3 and arena control,
with doubles for native arrays/modules/commands. They cannot prove real native
allocation, packed-command, GPU synchronization or native scope properties.

- Initial composition caught a hidden CUDA input readback and per-frame
  capabilities conflicting on shared events; both were removed as described above.
- A new CUDA event-handle mutation test initially failed because only object
  identity was retained. The repair covers terminal and diagnostic handles. An
  intermediate arbitrary-event replacement raised AttributeError; signature
  collection now treats absent metadata as a protocol mismatch. The original
  resolution regression suite then passed all 21 cases.
- At c59edc7, source/policy/final graph/CUDA resolution/Warp arena checks passed
  262 tests with 4 local platform skips. Earlier runtime custody/schema/source
  policy checks passed 192 with 70 native-fixture skips. Counts overlap.
- Combined-artifact version propagation plus historical lifecycle fixtures passed
  98 tests. After the final acquisition/submission restrictions, final graph,
  native arena, Warp arena and CUDA resolution passed 97 tests in 39.60 seconds.
- The final graph suite was then expanded to both primary_unprofiled and
  diagnostic_profiled modes: all 32 cases passed in 77.43 seconds, including
  both-lane sync failure, lost DISARMED, stream/event rebinding, forbidden early
  acquisition, closed submission, late unknown scope and versioned replay.
- The earlier Warp operation-count test checks actual Python lines for 1/120/600
  prebuilt frame catalogs. This is fixed-work evidence, not GPU timing.
- Ruff passes for the new arena source and final graph tests. All 90 literal
  release bodies pass the AST policy using exact LF packaging; production
  verification does not normalize source bytes. Current catalog SHA256 is
  618eb6a69be829f7d153e1a4680a25c22ab9ae4de7b71527ff8fdc1dceffba6e.

Actual native runs

Host tileops-dev, Python /home/ang.gao/miniforge3/envs/robot_sim/bin/python,
Warp 1.12.0 / Torch 2.9.1+cu128, target GPU1 with ambient0. The existing Go2
asset and frozen R0 960x640 render_only/shadows row were used. The isolated
remote tree is /data2/shared/ang.gao/robot_sim_final_graph_wip.

1. c59edc7 Warp completed ten warmups, final graph construction, barrier and
   acquire/cancel, but top-level acceptance FAILED: the combined-artifact
   validator defaulted to lifecycle v3 for the new v4 trace. The original
   supervisor error and worker records are preserved. This is not a passing run.
2. e31b83a Warp and CUDA each passed the complete fixed-driver acceptance:
   real preparation, target/stream observations and ambient restoration,
   successful final barrier and DISARMED, acquisition/cancellation, full terminal
   artifact replay, actual worker exit0 and supervisor reaping. Normal guard
   completions were 27 for Warp and 23 for CUDA. No final sample was submitted.
3. f05235f Warp and CUDA each passed the fixed injected failure check. The real
   target synchronization returned, then the wrapper raised before production
   could record barrier success. Neither lane issued quiescence or admission;
   both retained sealed coverage, transferred full graph custody to quarantine,
   performed zero emergency sync attempts and exited74 under their supervisor.
   The fixed driver returned0 only after checking those expected failure facts.
   This is failure-injection evidence, not a claim of an actual hardware fault.

The driver observes production code through the existing worker/supervisor; it
adds no execution engine. These cases do not prove final sample rendering or
performance, diagnostic native behavior, hangs, lost-ACK native behavior, the
full frozen row matrix, or all output-channel correctness at this revision.
Prior full-channel CPU comparisons retain their own recorded revision/scope.

Raw archive (remote /data2/shared/ang.gao, local c4_assets):
c4-final-graph-native-evidence-v1.tar.gz, 28,937,648 bytes, 189 members,
SHA256 3f88013319638b1b47d60e49c598a9cc97e66b2c76366819365b1375f6539c57.
It includes all five original run directories, both exact driver scripts,
environment contract, archiver and closure record. The original live Warp
compilation caches and unchanged CUDA deployment are referenced by their raw
records; this archive does not copy those caches or the deployment binaries.

All 15 recorded parent/supervisor/worker PIDs were explicitly absent when the
archive was made: 75699/75759/75765; 126947/127296/127308;
102476/102798/102800; 160220/160601/160605; 170404/170845/170851.
Every author pytest/tool session in this packet returned a terminal result.
No install, dependency change, broad signal, GPU reset, product activation or
performance measurement was performed. The remote tree remains at f05235f;
subsequent local admission restrictions must not be represented as that run.

Required continuation

Freeze and independently review this complete delta, repair findings and obtain
fresh independent re-review as required. Finish diagnostic/native failure and
scope obligations at the appropriate exact revision; complete shared setup and
failure routing where required by M1. Continue actual guarded sample execution,
R1/C2f tokens and lifecycle integration, precision/counterexample/compatibility
and full native failure matrices, C4c3/c4 and C4d/B3, then systematic whole-C4
review. Product PREPARE, performance/results/baselines, backend selection and
shipping remain closed. The active objective is still full C4, not this packet.
