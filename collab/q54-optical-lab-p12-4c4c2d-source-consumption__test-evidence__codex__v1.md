# C2d0 approved CPU source-consumption checkpoint

Initiative: q54-optical-lab-p12-4c4c2d-source-consumption
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-03
Status: accepted
Related Files: code-review codex v1/v2; review-followup codex v1; PROGRESS.md
Owner Summary: Record exact CPU-only approval and executed tests. No C4 product,
GPU, lifecycle or result gate is opened by this evidence-only record.

## Exact revisions and independent review

Original base `29b2055061d38211546774f510e70545701bc0e9`.
Initial implementation `0fae0a4c6df4f5527196009ce994aee3f0b7385d` was rejected
for two P2 identity-audit omissions despite passing existing tests. Repair and
approved code: `1f0576bfc607d24d2f5a04838051497f90a5592a`.

Fresh reviewer `/root/c4_c2d0_source_bootstrap_code_review_v2`, without the
implementation context, issued Approve for exactly that CPU subset. Complete
report archived verbatim as code-review v2, SHA-256:
`d35bcf191fc20e7730214421d7817833c84ab6177f512fb2589340d36a3fe3c6`.

Independent results: remote323passed/no skips/135.56s; local268passed/6skips/
134.32s. The local skips (POSIX and real Go2 assets) passed remotely. Original
three counterexamples now reject; ten additional both-lane probes prove that
restoring altered fields/owners cannot revive the failed session. The full
report records actual commands, scoped process identities and limitations.

## Author revised execution

Primary183:40031 was retried and timed out; authorized fallback101 was used.
No port, host configuration or environment installation was changed. The
original dirty checkout was preserved. Remote clean detached worktree:
`/data3/ang.gao/robot_sim_c2d0_code_v2`, exact approved code, robot_sim Python3.12.

```sh
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2 timeout --signal=TERM --kill-after=15s 300s /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/benchmarks/test_optical_benchmark_contracts.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py -q
```

Actual exit0: **323 passed in135.65s**, no skips. Includes actual POSIX direct
entry for both catalogs, valid stale-pyc attacks, identity/re-execution failures,
live-parent verification/response stalls, real Go2 and C0/contract regressions.
No GPU correctness/performance acceptance is inferred from these CPU tests.

The unchanged independent-v1 scratch repro was additionally piped to robot_sim
against the revised root. Parent2206267 (parent2206263) printed children2206302,
2206337,2206367. All three returned exit0 with REJECTED, not success reports:
spec name/package => module metadata drift; main replacement => bootstrap owner
identity/metadata drift. The outer command exited0. This verifies original
counterexamples separately from tests added by the author.

Local revised bootstrap/source-policy run:219passed/2POSIX skips/95.04s,
actual exit0. Ruff for the three changed production modules and bootstrap
tests, plus Git diff/show checks passed. Historical first-round local missing
Torch failures and corrected Windows timeout-output assertions are retained in
the repair note; they are not reclassified as passing runs.

## Round-end process checks

Remote timeout2204201 (parent2204190), pytest2204222 (parent2204201), sampled
Go2 CPU child2205979 (parent2204222), and all original-repro PIDs above were
absent in exact postchecks. Scoped pytest/native_bootstrap/closed-worker scans
returned no remaining test/worker. GPU compute-process queries returned no rows.
Local revised pytest33632 (parent21360) and sampled child23996 had exited;
final local scoped scan was empty. Process starttimes were not retained for
these author PIDs; the independent report includes its stronger ancestry/start
sampling and limitations. No broad kill, cache deletion or GPU reset was used.

These are scoped round-end observations, not proof that arbitrary workloads can
never hang, nor diagnosis of the old unidentified101 hang. Intentional stalls
were isolated test children, killed/reaped by their bounded launcher.

## Approved and remaining scope

Approved: CPU direct source bootstrap, six-core captured-byte loader/registry,
bootstrap-owner and module metadata audit, native AST v3/catalog v2, descriptive
CPU report, regression tests and existing cleanup obligation. Initial bootstrap
execution remains the trusted interpreter/script boundary, not retroactive
loader authorization. This is not a hostile same-process Python sandbox.

C0 files/protocol and shared collab rules are unchanged. C2d integrated guarded
lifecycle/termination and custody before GPU import/JIT remain unfinished;
C2e/f, R0/R1 activation, C4c3/c4, C4d/B3 and full C4 completion remain pending.
Product GPU preparation, R2/S/host shipping, a3b and benchmark/result/baseline
outlets stay closed. This separate evidence commit changes none of those gates.
