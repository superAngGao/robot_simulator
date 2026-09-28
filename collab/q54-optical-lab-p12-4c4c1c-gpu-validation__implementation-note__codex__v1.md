# C4c1c fixed sacrificial GPU validation candidate

Initiative: q54-optical-lab-p12-4c4c1c-gpu-validation
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-09-28
Status: in_review
Related Files: C4c1 design v2/v3; supervision review v3; recording review v2
Owner Summary: Implement the fixed, test-only recorded-Warp validation child;
do not open product preparation or any result outlet. Final GPU verification
is outstanding after both authorized SSH hosts became unreachable.

## Candidate and scope

Base: `7de02e89c3e868e7554e5b1c385cfbbcf06570e1` (evidence-only successor
of the approved CPU supervision checkpoint `3c81fa1`). The candidate commit
containing this note is the exact review HEAD; it is not a release or push.

Changes are test/support code and test-only supervisors. The existing S/V/E
event loop is shared between the CPU protocol child and the new fixed Warp
child. The production C4c0 A/K protocol, its allowlist and all activation
gates are unchanged. No ordinary executor drives the recorded bundle.

The child is a one-shot `python -I -S` program with fixed configuration enums,
a frozen repository source closure, and an import ledger. This is trusted
test code, not an arbitrary-Python sandbox or native capability boundary.
The complete benchmark package import closure is explicitly included only in
this test manifest; it does not widen the product worker's allowlist.

Guardian creation precedes site-packages exposure and CUDA-capable imports.
Setup leaves transfer their returned owners before the next leaf. Prepared
input uses the production exact-type registration and the canonical source.
All setup waits use S ACK-before-call. After final setup quiescence, the
guardian seals, and every recorded command gets its irreversible pre-call
possible-submission latch. V guards successful validation drain; E guards
eligible failure-only cleanup. Unknown ownership or failed V goes directly
to process-terminal quarantine, never a second synchronization.

The fixed parent-loss and missing/wrong-ACK scenarios use the actual external
supervisor, not a timer inside the blocked child. Journal encoding/write
failure hard-exits instead of unwinding a possibly active GPU resource graph.
The journal is diagnostic and cannot authorize cleanup.

## Correctness and lifetime checks

- Three output profiles, shadow on/off and diagnostics on/off use the same
  production recorded kernels and compare physical outputs with the independent
  CPU executor. Integer IDs/materials are exact; existing float tolerance is
  absolute/relative `1e-5`; rays use `1e-6`. These were not loosened after GPU
  observation.
- The canonical 21-triangle scene is unchanged. The correctness-only camera
  is 31x13, off-center, unequal focal lengths, with nonidentity rotation.
  CPU-only coverage inspection found 119 hit pixels, all 21 instance/material
  IDs, and seven pixels changed by shadows. This is **not** a benchmark row or
  a changed gate camera. Earlier 7x3 testing had only one hit and was not
  sufficient coverage of this property.
- Resident-ray counts 0/1/21 exercise two-/four-command paths; camera uses
  five commands. Physical output sentinels check deterministic reset.
- Counter checks cover scalar dtype/shape, nonnegativity, stack bounds,
  decomposition of rays, occlusion totals, scene-size upper bounds and zeros.
  The earlier independent shade-review probe is promoted into a tracked
  production-kernel regression with 56 hand-derived shade/counter runs and
  repeated empty-scene first-hit checks. It is not a recorded-bundle driver.
- Every command position has pre-latch, post-latch and submit-then-raise
  coverage. Setup faults include import, unreturned allocation/recording
  owner, first upload, compile and warmup. S/V/E hangs and wrong ACKs,
  parent loss across setup/submission/validation, and post-submit journal
  failures are explicit cases.
- Actual Warp allocation owners have weakref collection observers (atexit
  callbacks disabled). Pre-release custody checks force GC and inspect exact
  guardian owner identity. This supplements, not replaces, the approved CPU
  graph-retention and quarantine tests.

## Observed evidence and limitations

On 101, `robot_sim` (Python 3.12.13, Warp 1.12.0, Torch 2.9.1+cu128;
conda CUDA 12.8 toolchain), intermediate draft runs completed:

1. 12 camera/profile/shadow/diagnostics cases passed with the smaller camera.
2. First GPU fault matrix: `52 passed in 214.67s`.
3. Extended ACK/parent-loss matrix plus source tests:
   `74 passed in 274.33s`.

These are development evidence, **not proof for this final candidate**. The
31x13/two-device/microfixture run had started but its tool session became
unavailable. Subsequent SSH checks to both 183:40031 and 101:22 timed out;
its final result is unknown. The final journal-error cases were added locally
after that interruption and have not run on Linux/GPU yet.

On the local Windows Python 3.14 environment, the final CPU-source/lifecycle
selection ran `41 passed, 21 skipped`; the skips are POSIX subprocess cases,
not substitute runtime evidence. Ruff check, Ruff format check and diff check
pass. The GPU files collect 74 tests locally; collection is not execution.

The intermediate dirty 101 workspace additionally produced `71 passed,
7 failed` for CPU/C4c0 regressions. All seven failures were the expected
clean-repository manifest precondition. They must be rerun from a clean
candidate checkout; they are not recorded as passing regressions.

Required final clean Linux commands:

```
python -m pytest tests/gpu/test_warp_sacrificial_render.py tests/gpu/test_warp_shade_microfixtures.py tests/unit/optics/test_sacrificial_sources.py -q
python -m pytest tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py -q
```

Rerun the affected recording/ABI/readiness suites and inspect source/guardian
coverage independently. Do not approve a missing real-device property from
Windows skips or an older draft's green result.

## Remaining boundaries

This candidate has no C4 slot/generation/live-lease authority. C4c2-c4, C4d,
B3, product GPU preparation, host payload, shipping, a3b, result generation
and baseline access remain closed. C4 is not complete. The full dual-backend
worker setup/transport matrix remains a later required activation review.

Primary 183 was rechecked first; fallback 101 was previously reachable but is
now also timing out. Original dirty worktrees and all historical branches
were preserved. No dependency installation, push or main merge occurred.
