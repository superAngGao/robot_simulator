# C2d1 candidate v4 author regression evidence

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: test-evidence
Author: codex
Version: v4
Date: 2026-10-03
Status: in_review
Related Files: implementation-note v4; code-review v3; accepted custody/repair v2
Owner Summary: Exact6b5080f author remote regression751 passed, no skips, actual
exit0 and no owned residual. Pending independent review; no GPU/C4 gate opens.

Candidate6b5080fbfa00309c757ee3467fa6cb53b78fd57d. Local frozen workspace
C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design; remote clean detached
/data3/ang.gao/robot_sim_c2d1_code_v4. Authorized fallback101 after primary183
transport failure. Original dirty checkout preserved. No installs/push/reset.

## Command/results

SSH BatchMode/ConnectTimeout8/ServerAliveInterval5/ServerAliveCountMax2 using
C:/Users/AG/.ssh/tileops_ed25519, ang.gao@192.168.80.101. Remote candidate:

```text
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2
timeout --signal=TERM --kill-after=15s 420s
/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest
tests/unit/optics/test_guarded_lifecycle.py
tests/unit/optics/test_guarded_evidence_fixtures.py
tests/unit/optics/test_guarded_custody_process.py
tests/unit/optics/test_guarded_wait_core.py
tests/unit/optics/test_guarded_wait_process.py
tests/unit/optics/test_native_bootstrap.py
tests/unit/optics/test_native_source_policy.py
tests/unit/optics/test_native_sources.py
tests/unit/optics/test_sacrificial_watchdog.py
tests/unit/optics/test_sacrificial_sources.py
tests/unit/optics/test_sacrificial_lifecycle.py
tests/unit/optics/test_sacrificial_leaf_descriptors.py
tests/unit/benchmarks/test_closed_worker_protocol.py
tests/unit/benchmarks/test_optical_closed_worker.py
tests/unit/benchmarks/test_optical_benchmark_contracts.py
-q -s -p no:cacheprovider
```

Actual command exit0, 751 passed171.12s, no skips/outer timeout; session90373
completed. Only Warp array deprecation warning. Local297 targeted tests and
separate462-test source-policy run passed8.08s/exit0; Ruff checks pass. Original
v3 external probe actual exit0, four abort cases now return terminal evidence
before context exit, and prior F3/F4 cases pass. Implementation-notev4 records
fixture boundaries and the earlier combined-shell empty-process-query issue.

Protected C0 lifecycle/native_setup/worker/launchers no-diff versus8cbf864,
explicit exit0. Candidate clean and git diff --check passed.

## Process/hang evidence

Remote11:00:15: timeout2441306(parent2441297), pytest2441330(parent2441306),
sampled cold Go2 child2445062(parent2441330). Exact return/reap observed by tool
and subprocess harness, not inferred from disappearance.

Fixed fault ledger: supervisor / worker / actual worker returncode. All
supervisor returncodes0; harness validates worker parent and final process exit.

```text
cpu_success              2441546 2441551 0
cpu_cleanup_hang         2441555 2441556 -9
cpu_exit_hang            2441569 2441570 -9
normal_success           2441634 2441635 0
normal_leaf_raise        2441636 2441637 74
inner_abort              2441638 2441643 74
foreign_inner_abort      2441647 2441648 74
unknown_inner_abort      2441649 2441651 74
invalid_permit_inner_abort 2441652 2441653 74
pre_leaf_raise           2441654 2441655 74
foreign                  2441660 2441664 74
unknown                  2441665 2441670 74
sync_raise               2441671 2441672 74
sync_hang                2441673 2441674 -9
cleanup_raise_before     2441700 2441701 74
cleanup_raise_after      2441702 2441703 74
normal_no_ack            2441704 2441709 74
normal_wrong_ack         2441729 2441730 74
normal_ack_eof           2441731 2441732 74
normal_stalled_disarm    2441737 2441741 74
terminal_no_ack          2441742 2441744 -9
terminal_wrong_ack       2441754 2441759 74
terminal_ack_eof         2441763 2441764 74
target_cleanup_hang      2441768 2441769 -9
target_exit_hang         2441846 2441849 -9
```

Final exact ps lookup of all53 recorded IDs empty; scoped pytest/guarded-custody/
closed-worker command scan empty. Short-lived other child IDs were not all
sampled. GPU query before/after only unrelated training2434311, preserved.
No GPU test or manual/broad kill/reset. Prior candidate round observed another
trainingPID2386045; these are unrelated workloads, not test residue. Source
transfer command21372 separately completed exit0. No unresolved owned process.

## Limits

Fresh reviewer v4 independently checks F5 and neighboring failure facts. Passing
counts are not approval. Source-frozen runtime adoption/module custody, real
Warp/Torch multi-GPU faults, complete graph/barrier/warmup/readiness, C2e/f,
product preparation, a3b/results/baseline, R2/host/S and later C4/B3 remain
closed. C4 incomplete; historical unidentified101 hang remains undiagnosed.
