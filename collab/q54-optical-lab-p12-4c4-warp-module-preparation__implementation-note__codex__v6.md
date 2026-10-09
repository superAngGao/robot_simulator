# Warp native fault diagnostics: preserve the primary exception

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: native acceptance support review repair
Author: Codex
Version: v6
Date: 2026-10-10
Status: draft
Related Files: warp-module-fault-support code-review v1;
  warp-module-preparation implementation-note v5; native_runtime_process.py
Owner Summary: A nested diagnostic catch preserves the original compile error
  if post-quarantine inspection or evidence I/O fails. Missing roots evidence
  still fails acceptance. Production/runtime policy is unchanged.

Base8648067ba08bb826c50b1322a75bdff5c4db75d5. Independent support reviewv1
requested one P2 F1: the post-quarantine diagnostic block could replace the
original error with its own inspection/write failure. It reproduced this by
executing the exact candidate AST, with the native observer as a passing control.
Full unchanged report archived as warp-module-fault-support code-review v1,
SHA2561be6b15ad4a51a8d538ec830c2451a0606c57c38085d540ed7cdcdaeeac8c8d5.
Its57 existing regressions passed; no new native selectors were executed.

The repair wraps only the CPU roots diagnostic in the same nested catch used
by the existing native scope diagnostic. Diagnostic exceptions are retained in
the session; the outer bare raise always propagates the original exception.
No native operation, retry, adoption or synthetic success record is added.
The runner still unconditionally reads and validates warp_failure_roots.json
for each non-hang case, so missing/invalid evidence cannot pass acceptance.

New test_warp_failure_diagnostics.py executes the actual driver's AST try block
(located by its sole prepare_warp_modules call rather than a fixed line number).
It covers successful diagnostics, inspection failure and write failure. All
three propagate the identical original RuntimeError. Failed diagnostics retain
their OSError and emit no roots record; successful diagnostics emit the actual
record and retain no diagnostic error. Dependency metadata is a CPU fixture;
no native behavior is claimed. This directly covers the independent finding.

Author Python3.14 -B -m pytest -q -p no:cacheprovider on that file plus
test_warp_module_preparation.py/test_cuda_artifact_runtime.py:60 passed in10.30s,
tool38323 exit0. git diff --check passes. Fresh independent re-review is required
before executing the five new native selectors. Production remains approved
beedab6 and the remote six-case beedab6 batch remains unchanged/live in10124;
v11-v14 passed, v15 positive1920 was last observed live. No complete native or
rendering/performance claim. Finish full native acceptance and the approved
graph/warmup/readiness/rendering/lifecycle/C4c3-c4/C4d-B3 work, then systematic
refactor and fresh whole review. Product/result/baseline/selection gates closed.
