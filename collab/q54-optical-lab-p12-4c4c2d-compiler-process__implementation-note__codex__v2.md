Initiative: q54-optical-lab-p12-4c4c2d-compiler-process
Stage: implementation-note / R1 repair and fresh code preflight
Author: Codex
Version: v2
Date: 2026-10-09
Status: in_review
Related Files: implementation-note v1; independent review v1; compilation-admission design v3
Owner Summary: Fixed fixture validation now requires the intended actual exit
  mechanism. The ineffective-signal/natural-exit model rejects; real OS execution
  still requires approval of this exact repaired candidate.

Base candidate: ef6c8df717f574684e88cadbeb48222e7916d3a6.
The separate evidence-only archive1039f55 preserves full review v1 unchanged,
SHA256 7d41ca9bae8ed77103dbd97b37191fd5f11d13000fa0064a6a3a5b985b7afeba.
Exact repaired HEAD is provided in the fresh review handoff. The accepted design,
15-case scope, fixed deadlines, production code, source catalog/policy/profile,
ABI source and campaign invocation remain as in implementation-note v1.

R1: validate_result previously accepted a recorded descendant SIGKILL attempt
followed by natural exit. The production replay legitimately permits this;
the fixture's stronger expected outcome belongs in the test runner. Its common
_validate_case_exits boundary now joins every actual child_reaped wait status to
the intended case result, checks exact worker/child/grandchild roles, separates
worker-reaped intermediates, checks the exact signal target set and ordering,
requires absence of natural fallback except the explicit signal-denial case,
and checks expected worker error artifacts.

Expected outcomes are explicit: normal/separate-PGID workers exit0; leaders exit0
and their remaining children/grandchildren exit-9; fork-during-cleanup intermediate
exits0 and its grandchild-9; double-fork intermediate is actually waited by worker.
Residual COMPLETE worker exits74 and its child-9. Missing COMPLETE, post-deadline
proc denial and parent loss workers exit-9. Handle/stale/disappearing acquisition
faults happen before first ARM, close transport, and the worker exits74. Signal
denial records one failed attempt and the worker's declared natural exit0.
Actual OS outcomes still must verify these expectations; no runtime pass is claimed.

New CPU oracle regression uses actual production core/obligation records and the
real validate_result, with explicitly synthetic process observations. For all five
leader/descendant cases it accepts the matching SIGKILL status and rejects the
ineffective-signal/natural-exit variant. Additional attacks reject natural-fallback
markers and erased grandchild roles while keeping production records fixed. This
is test-discrimination evidence, not process execution proof.

Windows Python3.14, -B, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, pytest -q
-p no:cacheprovider on compiler_job, compiler_transport, compiler_process_unwind
and compiler_process_oracle:41 passed in1.09s. Fresh independent target CPU checks
and original retained counterexample rejection are requested before execution.

Independent v1 ABI evidence remains at
/data2/shared/ang.gao/compiler_process_preflight_v1.zTQ8bY69/abi.json, SHA256
037752b995cd2f1f82caa698ad655ca4b6aa8f7a5da00bd037905fd7ee19656b.
The C source, compiler, transitive headers, libc and compiled binary are unchanged
inputs and have retained hashes/raw output in that directory. Fresh reviewer
should verify these unchanged facts and ctypes layout; another C compilation is
unnecessary unless changed evidence warrants it. This ABI result is not itself
OS execution permission. Native job setup and the one fixed15-case campaign stay
closed until this candidate's fresh review approves them. All CUDA/backend,
readiness, measurement and parent C4 gates remain closed as before.

While v1 review ran, the author also refreshed the read-only production toolchain
inventory at ef6c8df: /data2/shared/ang.gao/cuda-toolchain-ef6c8df-inventory-v1.json,
SHA256 ec4b2cfc004ee3e387c1c6bce1251e718825bf9b96889221b8f2beba8e74e65d,
identical to the earlier52fb36a inventory. Actual scanner PID964811 exited0 and
was verified absent;17023 files,60 roots,1169 links,3420577 output bytes,
12.42496703704819seconds. It imported no backend or executed any toolchain binary;
this scan duration is not rendering performance evidence.
