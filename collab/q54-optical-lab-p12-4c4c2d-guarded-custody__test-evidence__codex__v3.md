# C2d1 candidate v3 author runtime evidence

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: test-evidence
Author: codex
Version: v3
Date: 2026-10-03
Status: in_review
Related Files: implementation-note v3; code-review v2; approved repair-design v2
Owner Summary: Exact candidate6f87d23 broad remote CPU/source/process regression
completed730 passed, no skips. This is author evidence, not independent approval
or GPU resource-safety proof. All later product/result gates remain closed.

Candidate: `6f87d23a5ce2c1acefc8795caf79709fa7ea33e2`.
Local frozen clean workspace: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design.
Remote isolated clean detached worktree: /data3/ang.gao/robot_sim_c2d1_code_v3.
Primary183 transport timed out on40031 and22; authorized fallback101 used.
No install/push/merge/reset or original dirty checkout edit. H200 is GPU2;
GPUs0/1 are A100. This CPU slice does not authorize a real GPU workload.

## Actual command and result

SSH used BatchMode, ConnectTimeout8, ServerAliveInterval5, ServerAliveCountMax2,
key C:/Users/AG/.ssh/tileops_ed25519, ang.gao@192.168.80.101. From the isolated
candidate, with ROBOT_SIM_TEST_GO2_MODEL_DIR set to
/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2:

```text
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

Actual exit0, **730 passed in169.54s**, no skips, no outer timeout. Observed
Warp deprecation warning about warp.types.array only. No new result/baseline.
Command handle75540 completed; no lost-SSH timeout treated as successful exit.

Local independent-of-remote author checks before freeze:444 passed19.51s;
Ruff check/format and diff check pass. Afterwards git show --check pass, tree
clean. Archived v2 review bytes match6c555b14a883bb342df9bc1632d55638ba2794ef48b25ac163742f1c50346289.
See implementation-notev3 for the earlier new-fixture corrections.

## Actual process observations and cleanup

Remote start10:41:03. timeout2408769(parent2408676), pytest2408790(parent2408769).
Sampled cold Go2 source children2411492 and2412628, both parent2408790, made
progress in separate Warp/CUDA CPU materialization subprocesses.

Process-case ledger: supervisorPID / workerPID / actual worker returncode.
Every supervisor actual returncode0, with worker parent identity and observed
reap validated by the actual subprocess tests.

```text
cpu_success             2409030 2409031 0
cpu_cleanup_hang        2409032 2409033 -9
cpu_exit_hang           2409052 2409053 -9
normal_success          2409071 2409076 0
normal_leaf_raise       2409080 2409083 74
inner_abort             2409086 2409087 74
pre_leaf_raise          2409088 2409089 74
foreign                 2409090 2409095 74
unknown                 2409096 2409097 74
sync_raise              2409101 2409103 74
sync_hang               2409106 2409107 -9
cleanup_raise_before    2409168 2409172 74
cleanup_raise_after     2409174 2409175 74
normal_no_ack           2409179 2409180 74
normal_wrong_ack        2409192 2409193 74
normal_ack_eof          2409194 2409195 74
normal_stalled_disarm   2409196 2409201 74
terminal_no_ack         2409205 2409211 -9
terminal_wrong_ack      2409216 2409217 74
terminal_ack_eof        2409222 2409223 74
target_cleanup_hang     2409227 2409241 -9
target_exit_hang        2409261 2409265 -9
```

After actual exit, exact ps lookup for all48 recorded timeout/pytest/cold-child/
supervisor/worker PIDs returned no process. Scoped ps scan for pytest,
guarded_custody_process and optical_closed_worker also empty. Earlier ps-C
python3.12 did not identify pytest because the executable was named python;
direct parent/child and command scans supplied actual identity evidence.
Short-lived other test child IDs were not individually sampled.

GPU process query before/after showed only unrelated training2386045
(/home/lin.li/ChainLoRA/chainlora-conda/bin/python), preserved. Local scoped
Python test/repro process query empty. No manual kill, broad kill or GPU reset.
Original remote dirty checkout retained its exact four preexisting paths.

## Scope

Fresh code-reviewv3 is still required. No integrated guarded runtime/module
adoption, actual Warp/Torch multi-GPU faults, complete setup graph/readiness,
C2e/f arenas, product preparation, a3b/result/baseline, R2/host/S, later C4/B3
or C4 completion is proved or opened here. Historical unidentified101 hang
is not retrospectively diagnosed by this clean round.
