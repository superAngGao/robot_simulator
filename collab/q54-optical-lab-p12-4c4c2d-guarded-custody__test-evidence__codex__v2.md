# C2d1 repaired-candidate author regression

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: test-evidence
Author: codex
Version: v2
Date: 2026-10-03
Status: implemented
Related Files: implementation-note v2; repair-design v2; pending code-review v2
Owner Summary: Actual author regression on frozen90f50bb. Passing tests do not
substitute for independent approval or authorize GPU/product/C4 activation.

Candidate: `90f50bb3ef67dd99182e4da0e5bc8479cd76e53b`, clean detached remote
`/data3/ang.gao/robot_sim_c2d1_code_v2`. Primary183:40031 timed out; approved
fallback101 used. Original dirty worktree unchanged; no resets/installs/pushes.

Local403 tests passed in11.88s, including controller/evidence/wait core and
source policy; Ruff and diff check passed. Local tool session39018 was polled
to actualexit0 rather than treating the initial observation yield as completion.
Original external repro showed both F1 cases now zero-sync/no-release/full
quarantine; script then exited1 at the first newly rejected F2 artifact as
documented in implementation-note v2. Both F2 cases have explicit pytest tests.

Remote command from candidate, robot_sim Python3.12, real Go2 fixture:

```text
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2 timeout --signal=TERM --kill-after=15s 420s /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_guarded_lifecycle.py tests/unit/optics/test_guarded_evidence_fixtures.py tests/unit/optics/test_guarded_custody_process.py tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_leaf_descriptors.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/benchmarks/test_optical_benchmark_contracts.py -q -s -p no:cacheprovider
```

Tool session38087: **689 passed in169.23s**, actualexit0, no skips. Complete
stdout is tool-transcript evidence, not a separately claimed persistent log.
timeout2362121(parent2362109), pytest2362142(parent2362121), both started10:12:42.
At10:14:52 Go2 source/materialization child2365634(parent2362142) was genuinely
running; the same test handle was continued, not restarted. No outer timeout
was credited as a successful observed exit.

22 fault cases: supervisor / worker / observed worker exit (supervisor exit0):

```text
cpu_success             2362335 2362336 0
cpu_cleanup_hang        2362341 2362342 -9
cpu_exit_hang           2362354 2362356 -9
normal_success          2362417 2362418 0
normal_leaf_raise       2362422 2362423 74
inner_abort             2362425 2362429 74
pre_leaf_raise          2362434 2362435 74
foreign                 2362436 2362437 74
unknown                 2362438 2362442 74
sync_raise              2362444 2362445 74
sync_hang               2362450 2362451 -9
cleanup_raise_before    2362469 2362471 74
cleanup_raise_after     2362472 2362477 74
normal_no_ack           2362478 2362479 74
normal_wrong_ack        2362484 2362485 74
normal_ack_eof          2362491 2362494 74
normal_stalled_disarm   2362497 2362501 74
terminal_no_ack         2362502 2362505 -9
terminal_wrong_ack      2362512 2362513 74
terminal_ack_eof        2362514 2362515 74
target_cleanup_hang     2362519 2362521 -9
target_exit_hang        2362582 2362583 -9
```

Harness verified each exact worker's parent identity and process_exited evidence.
Final queried timeout/pytest/Go2 and sampled first/last worker/supervisor PIDs
were absent; ps for python3.12/pytest/timeout returned none. GPU compute query
contained only unrelated ChainLoRA PID2336351, left untouched. Candidate tree
was still clean. This proves scoped cleanup, not global idle or absence of all
possible hangs. No new GPU setup/real-device lifecycle proof is claimed.

All integrated C2d, arenas/C2e/f, final readiness, productGPU/a3b/results/
baseline/R2/host/S/laterC4/B3 activation remain closed. Independent review may
still identify defects; this evidence records execution, not a verdict.
