# Remaining Warp compilation admission observations

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: bounded test support, not production activation
Author: Codex
Version: v7
Date: 2026-10-10
Status: in_review
Related Files: warp-module-preparation design-request v2 and code-review v4;
  warp-module-fault-support code-review v2; native_runtime_process.py;
  cuda_artifact_runtime_acceptance.py
Owner Summary: Extend the existing supervised test entry with the remaining
  post-compilation admission mutations. Production code, compiler, source
  catalog and process supervision are unchanged. This removes no acceptance
  requirement and introduces no new process manager or product abstraction.

Base: 63051811cc84495862d7da43d5049bc4297180c8, clean before this delta.
The existing three ABI/entry cases now share one explicit expected-error table
with twelve additional selectors: backend configuration, module options, private
cache permissions, binary bytes, binary symlink redirection, executable mapping,
copied module set, changed module-set session, copied compilation record,
target binding, stream binding, and captured source cache mutation.

Each reaches actual Go2 input preparation and three real Warp compilations.
Inside the original next setup guard, it first admits the unchanged objects,
then mutates one fact and calls the composed production checks: runtime binding,
source/environment audit, module-set validation. An admitted mutation raises an
AssertionError and cannot pass the runner's exact expected-error assertion.
The runner also requires the original failed guard, full quarantine, no native
leaf after the marker, zero emergency synchronization, no successful module
binding, actual worker exit74 and supervisor reaping. The existing production
artifact/runtime-custody replay remains mandatory.

Artifact mutations affect only this worker's own generated CUBIN. Exact original
bytes are first preserved with exclusive creation in its raw evidence directory.
The redirect case replaces that owned cache file with a symlink to the saved
copy outside the cache; no shared deployment, repository source or external
dependency file is changed. Changed objects remain retained through the existing
session/issued records. Target/stream mutations use CPU object substitution and
must reject before any native call; they do not prove a second GPU was used.
The changed-session case tests module-set projection integrity, not possession
of a genuine module set issued in another live session.

Author verification: Python3.14, python -B -m pytest
tests/unit/optics/test_warp_module_preparation.py
tests/unit/optics/test_warp_failure_diagnostics.py
tests/unit/optics/test_cuda_artifact_runtime.py -q:
60 passed in10.22s; tool4018 terminal exit0. Both support files compile; all86
fault selectors are unique, including15 post-compilation admission selectors.
git diff --check passed. These CPU checks do not establish the new native cases.

The remote checkout is separately frozen at reviewed552d1b7 while the preceding
load-failure batch runs (tool82019, parent3074957). It has not received these
changes. New selectors require fresh independent support review before running.
Remaining acceptance includes pre-compilation cache/executable admission,
source-consumption evidence, fixed-CUDA/input regressions and independent native
acceptance. Whole C4 still requires actual graph/timeline/arena, warmups/final
readiness/seal/barrier, rendering correctness/lifecycle, C4c3/c4,C4d/B3 and the
owner-requested systematic refactor with fresh whole review. No performance,
product preparation, result/baseline or backend-selection gate is opened.
