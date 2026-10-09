# Warp preparation: real IDLE matrix and cold positive reruns

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: bounded native author evidence before complete acceptance
Author: Codex
Version: v4
Date: 2026-10-10
Status: draft
Related Files: warp-module-preparation code-review v4; test-evidence v3;
  implementation-note v5/v6; warp-module-fault-support code-review v1/v2
Owner Summary: The reviewed code passes real healthy/unknown/incomplete IDLE
  failure disposition and cold Go2 positives at both resolutions. Processes
  exited and native artifacts were retained. Complete native acceptance and
  actual rendering remain outstanding.

Exact executed code beedab6143e702a1a50ef0e2686fd0138e194376 in clean detached
tileops-dev:/data2/shared/ang.gao/robot_sim_warp_modules_wip. The unchanged
reviewed run_case executes through the existing supervisor; the external author
script only sequences fixed cases, checks the same revision/clean tree, records
GPU mapping and waits for each owned child before the next. No deadline change,
native installation, compile manager or runtime retry was introduced.

Command: /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -B
/data2/shared/ang.gao/run_warp_module_remaining_idle.py.
Tool10124 terminal exit0. Parent3048547. Warp1.12/Python3.12.13/H200 target1,
ambient0; same v4 environment digest02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb.
Original ACK1000ms/operation120000ms applies separately to each admitted guard.
All inputs are real Go2, not a surrogate.

| Raw directory suffix | Case/fault | Worker/supervisor | Completions/arms | Exit |
| --- | --- | --- | --- | --- |
| v11 | 960x640 healthy CPU issuance failure | 3048611/3048603 | 10/11 | worker74,expected negative accepted |
| v12 | 960x640 incomplete entry custody | 3051866/3051857 | 6/6 | worker74,expected negative accepted |
| v13 | 960x640 incomplete issuance custody | 3054550/3054549 | 10/10 | worker74,expected negative accepted |
| v14 | 960x640 unknown entry scope | 3060945/3060944 | 6/6 | worker74,expected negative accepted |
| v15 | 1920x1080 cold compile and same-session reuse | 3063653/3063642 | 11/12 | worker0 |
| v16 | 960x640 cold compile and same-session reuse | 3067589/3067582 | 11/12 | worker0 |

Full raw directory prefix: /data2/shared/ang.gao/c4-warp-module-author-.
Each result passed production guarded artifact replay, runtime-custody
selection/attachment and exact case assertions. No ordinary worker atexit.
All supervisors observed EXITED/complete without deadline expiration or kill.

v11 observes original RuntimeError("fixed Warp CPU issuance failure") after
three real completed compiles, with exactly one bounded emergency sync and
completed release; no fabricated failed normal guard. v12/v13 deliberately
remove exact stream guardian membership while the actual stream stays retained
by the source-session graph. Production failure routing detects incomplete
custody and quarantines without synchronization or new guard. v14 unknown scope
also quarantines without sync, preserving the six completed guards. Together
with v9/v10 in test-evidencev3, these cover all six fixed IDLE fault selectors.
They do not by themselves prove load/adoption failure restoration or hang.

v15 and v16 each compile all three existing optical modules and reuse the same
actual ModuleExec identities in that worker. Three outer compile observations
have exact before/after equality, ambient default0/current-context-none;
six inner load/reuse observations bind target1 and its held stream. Fresh cache
paths differ: /tmp/robot-sim-warp-a_rl7e1h and /tmp/robot-sim-warp-iufhvmk9.
After observed batch exit, the existing inspector verified each private0700
cache and all9 recorded file lengths/SHA256, ELF64 NVIDIA/sm90 CUBIN headers
and generated builtin.h includes. No numerical determinism across compilers,
optical output, warmup readiness or timing/performance is claimed.

Exact remote/local archive c4-warp-module-author-evidence-v3.tar.gz SHA256:
0cdea96fa3467aea46fc0ff9a5d3b1d7dc079e8bfb72dc3d57fa827a0c75329c.
Remote base /data2/shared/ang.gao; local base C:/Users/AG/新建文件夹/c4_assets.
Contains all6 raw directories, both positive caches'18 recorded files, selected
environment and exact author launch/inspect/archive scripts. Adjacent JSON and
c4-warp-idle-batch-v1.json record the terminal cases. Inspector output was
truncated in the tool display; complete post_exit_artifacts.json files are in
the raw archive. No execution was repeated or artifact overwritten to repair
that display limitation.

Each case's worker/supervisor PID was checked absent before the next case. After
the batch, all13 distinct owned PIDs (one parent plus12 children) were checked
absent again by the archive script. Final nvidia-smi output listed only unrelated
MPS server2238410; it was not touched. No cache or historical evidence deleted.
The remote checkout remained unchanged through the entire terminal batch.

While this batch ran, only local test support was changed for the next five
guarded import/load/adoption/hang cases. Production code did not change. The
separate support review found and prompted repair of diagnostic exception
masking; no new selector executed before its fresh re-review. That code/support
review is not complete native acceptance. Continue the remaining negative
admission/failure tests and fixed-CUDA/input regressions, then the approved
one-slot graph/timeline/arena, warmups/final readiness/barrier, CPU-oracle
rendering/lifecycle,C4c3/c4,C4d/B3, systematic refactor and fresh whole review.
Product PREPARE, performance/results/baselines and backend selection stay closed.
