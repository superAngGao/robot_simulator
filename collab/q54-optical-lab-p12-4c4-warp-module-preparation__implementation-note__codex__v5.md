# Warp compilation: original-guard native failure observations

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: native acceptance support after bounded code approval
Author: Codex
Version: v5
Date: 2026-10-10
Status: draft
Related Files: warp-module-preparation design-request v2; code-review v4;
  test-evidence v3; tests/support/native_runtime_process.py;
  tests/support/cuda_artifact_runtime_acceptance.py
Owner Summary: Extend only the existing fixed acceptance driver to observe
  ARMED before optical import, original-guard load/publication/adoption failure
  and hang, post-unwind device restoration and actual retained partial roots.
  No production code, compiler manager or product entry is added.

Base9787594249c84e00bb5776f51ed298667deda646; production code remains the bounded
approved beedab6143e702a1a50ef0e2686fd0138e194376. This support delta implements
the supplement's still-missing native import/load failure observations. Its
five new selectors are not yet native execution evidence at this cut.

The fixed driver observes actual captured module adoption immediately before
the loader executes each of the three optical bodies. It requires the original
live setup ticket and records ARMED/started/module/guard facts. The import fault
raises after that actual adoption but before module execution. Remaining faults
occur in the first real compile: before public load, after actual ModuleExec
publication before public return, before returned executable adoption, and a
stall after publication. Original120s operation timeout remains unchanged.

The load observer no longer appends an extra executable to the session's owner
list. Its assertions inspect the production-held Module and execs mapping;
ordinary stack/exception references are not claimed absent. The adoption fault
acts on the actual executable after production has appended it to its session,
before the guardian registration. For non-hang compile errors, the outer wrapper
samples the actual native state after ScopedDevice/ScopedStream unwind, while
the original guard remains active. A diagnostic failure cannot replace the
original error; missing restoration evidence fails the acceptance runner.

After prepare_warp_modules unwinds and quarantines, CPU-only inspection records
actual module/mapping/executable identities, guardian/session membership,
completed records, absence of an issued module set and exact load attempts.
There is no post-failure native query, resource adoption or custody repair.
The runner requires the original error/guard, exact6-or7 original completions,
no next leaf/extra ARM/sync, restored ambient0/current-context-none and retained
partial graph. Hang instead requires supervisor timeout/kill,worker-9 and no
fabricated worker terminal record; its available worker prefix is preserved.
No failure-restoration claim is made for a killed/hung process.

Author local Python3.14 -B -m pytest -q -p no:cacheprovider:
tests/unit/optics/test_warp_module_preparation.py and test_cuda_artifact_runtime.py
completed57 passed in10.12s, tool1627 exit0. Both support modules compile, fault
names are unique and git diff --check passes. These existing CPU regressions do
not execute the new native selectors. Fresh independent review precedes their
native execution; native acceptance remains unapproved.

In parallel, the unchanged reviewed beedab6 remote checkout is running the
already-approved six-case IDLE/positive batch using the existing run_case loop.
Tool10124, parent3048547; v11 healthy issuance has passed, v12 incomplete entry
was live at the last observation. Do not update that remote checkout until the
batch is observed terminal. Exact results will be recorded separately, never
inferred from this plan. Current local edits do not affect those remote bytes.

Remaining acceptance also includes source/cache/options/artifact/target/stream/
foreign-record negatives and retained fixed-CUDA/input regressions. Whole C4
still requires the approved graph/timeline/arena, actual warmups/final readiness,
CPU-oracle rendering and lifecycle, C4c3/c4,C4d/B3, systematic refactor and fresh
whole review. Product PREPARE, performance/results/baselines and selection stay
closed. This support work is required failure coverage, not another runtime layer.
