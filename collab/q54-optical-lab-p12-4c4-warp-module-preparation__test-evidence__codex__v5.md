# Warp guarded import/load failure and deadline evidence

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: native author evidence, not whole acceptance
Author: Codex
Version: v5
Date: 2026-10-10
Status: draft
Related Files: warp-module-fault-support code-review v2;
  warp-module-preparation test-evidence v4; warp-admission-support code-review v1
Owner Summary: All five reviewed import/load/adoption/hang selectors completed
  their expected real Go2 failure paths. The existing supervisor reaped every
  worker, including the load hang. Production code is unchanged. Remaining
  admission/regression evidence and integrated rendering are still pending.

Executed clean detached552d1b78f90d3b358710174bed2c58b6ccdd571f at
tileops-dev:/data2/shared/ang.gao/robot_sim_warp_modules_wip.
Command: /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -B
/data2/shared/ang.gao/run_warp_module_load_faults.py.
Tool82019 terminal exit0, parent3074957. Python3.12.13/Warp1.12/H200,
target1/ambient0, real Go2 960x640, selected v4 Warp environment digest
02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb.
Unchanged ACK1000ms/operation120000ms; sequential cases, no retry or timeout
extension. A live observation poll is not treated as a failed test or restart.

| Directory suffix | Fault | Worker / supervisor | Complete / ARM | Exit |
| --- | --- | --- | --- | --- |
| v17 | captured import before body | 3075005 / 3075004 | 6 / 7 | 74 |
| v18 | before public load | 3077753 / 3077745 | 7 / 8 | 74 |
| v19 | after actual publication, before return | 3080525 / 3080524 | 7 / 8 | 74 |
| v20 | returned executable adoption | 3083448 / 3083446 | 7 / 8 | 74 |
| v21 | hang after actual publication | 3086490 / 3086486 | 7 / 8 | -9 |

Raw directory prefix /data2/shared/ang.gao/c4-warp-module-author-.
All passed the existing run_case assertions, production guarded-artifact replay
and runtime-custody selection/attachment validation. For v17-v20 the original
RuntimeError is preserved, the original guard fails once, no later native leaf
is entered, no emergency synchronization/new guard occurs, and full custody is
quarantined. No completed compilation record or issued set is fabricated.

v17 observes the actual first captured module adopted under ARMED setup guard6
before its body executes, with its Python owner retained and no load attempted.
v18-v20 observe all three imports under guard6. Their compile failure retains
the Module and its original executable mapping under guard7. v19 retains the
actual published executable through that mapping despite the load wrapper not
returning; v20 additionally retains the executable in the source session before
the injected adoption failure. Native observations before/after the real scope
unwind match, restoring default device0/current-context-none. These observations
do not assert destruction or absence of incidental stack/traceback references.

v21 reaches a marker after actual publication with the mapping already held.
The original setup guard deadline expires; supervisor records kill_requested,
worker -9 and EXITED/complete. There is no terminal worker.json/failure.json.
The attached worker record is the older durable input prefix (six completed
guards), not a terminal worker claim; supervisor has seven completed guards.
No failure restoration or ordinary cleanup is claimed for a killed worker.

After batch exit, archive_warp_module_author_v4.py checked all11 distinct owned
PIDs absent, archived the five raw directories plus exact launch/archive scripts,
batch results and selected environment. Final nvidia-smi listed only unrelated
MPS2238410, which was not touched. All four exception cases have no deadline
expiration/kill; only the intentionally stalled fifth case does. No cache was
deleted, unrelated process signalled or remote checkout changed during the run.

Remote/local archive c4-warp-module-author-evidence-v4.tar.gz SHA256:
91d08111f6434ca734b202efcb4ce4370e9c311c205dce0c6657510317393c65.
Remote base /data2/shared/ang.gao; local base C:/Users/AG/新建文件夹/c4_assets.
Adjacent JSON and c4-warp-load-batch-v1.json preserve summaries. Failure caches
remain remote; this archive contains raw observations, not a claimed inventory
of every cache produced before failure. Transfer SHA256 matches remote.

Additional fresh CPU checks at4fe6b3a (production still unchanged sincebeedab6):
- test_native_bootstrap.py -k 'captured_inspect or captured_source_audit':
  6 passed,78 deselected in17.06s, tool65511 exit0. Actual captured loader/inspect
  consumes held bytes despite stale linecache and on-disk changes in private
  test copies, and rejects clear/replace/line/mtime mutations. No native codegen
  claim is derived from these CPU tests.
- External scratch test_warp_preexisting_admission.py:3 passed in1.21s, exit0.
  Actual compile_module rejects preexisting mapping, failed-build and module-cache
  directory before its native leaf with retained mapping and no sync. Dependency
  objects are CPU stand-ins; this does not prove native compilation behavior.
  Script SHA256 ce6adc5d0039132add895b8961a1d75ae343750ecb9be296bf561209f8cd1e4f.

Fresh independent support review approves4fe6b3a for the fifteen post-compilation
admission selectors, with no substantive findings; full unchanged report archived
separately. Its60 CPU tests and synthetic assertion probes are not native results.
Continue those real cases, required CUDA/input regressions, fresh native acceptance,
then the approved physical graph/timeline/arena and whole-C4 obligations. Product
PREPARE, performance/results/baselines and backend selection remain closed.
