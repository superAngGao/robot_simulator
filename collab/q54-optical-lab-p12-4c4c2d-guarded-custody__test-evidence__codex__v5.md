# C2d1 author verification at 854c2ea

Date2026-10-03. Candidate854c2ea6a8b6ec4021cb35b330542b56dc5a1935,
F6 repair8b1918d086343dd16e92bdd7017433dd4aabfa6d. Not independent approval.
CPU/control scope only; all runtime/product/result gates remain closed.

Local Python3.14: lifecycle109 passed6.87s; lifecycle/evidence/wait-core/source-
policy476 passed10.34s, actualexit0. Ruff/import-format passed after ordering
correction. git diff/show --check and explicit no-diff protectedC0 files passed.
Local scoped Python process check found no owned pytest/guarded residue.

Earlier8b1918d remote run actually completed8 failed761 passed172.28s, exit1.
Failures all cold-import precondition in an already runtime-loaded pytest
process. Intermediate isolation attempt importedpytest under-I and failed
locally8failed101passed because pytest lives in usersite. Final fixed harness
has only stdlib/CPUproduction dependencies; no environment changes/moduleunload.
Those failed runs are not erased or called successful.

Authorized fallback101 isolated /data3/ang.gao/robot_sim_c2d1_code_v5b;
originaldirty /data3/ang.gao/robot_sim_c4c1b2 preserved unchanged(fourpaths).
Primary183 SSH40031/22 timedout this turn. No installs/push/reset/GPUwork.

Actual command (SSH identityC:/Users/AG/.ssh/tileops_ed25519):

```
cd /data3/ang.gao/robot_sim_c2d1_code_v5b
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2 timeout --signal=TERM --kill-after=15s 420s /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_guarded_lifecycle.py tests/unit/optics/test_guarded_evidence_fixtures.py tests/unit/optics/test_guarded_custody_process.py tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_leaf_descriptors.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/benchmarks/test_optical_benchmark_contracts.py -q -s -p no:cacheprovider
```

769 passed174.35s, no skips; actualexit0, toolsession68900. Warpdeprecation
warning only. Test timeout did not fire. Live parent supervision remains real.
timeout2492835(parent2492749), pytest2492856(parent2492835).
Coldprobe children2493057/2493059/2493064/2493067/2493073/2493079/2493084/2493089:
each parent2492856, checked exact PID/PPID, actualreturn0.
Sampledsourcebootstrap2493826(parent2492856).

Supervisor/worker/actualworkerexit ledger, eachsupervisor actualexit0:
```
cpu_success2493124/2493125/0
cpu_cleanup_hang2493126/2493128/-9
cpu_exit_hang2493144/2493152/-9
normal_success2493174/2493175/0
normal_leaf_raise2493218/2493219/74
inner_abort2493258/2493277/74
foreign_inner_abort2493278/2493280/74
unknown_inner_abort2493281/2493284/74
invalid_permit_inner_abort2493288/2493289/74
cuda_import_failure2493293/2493298/74
warp_import_failure2493299/2493301/74
cuda_import_thread_failure2493302/2493310/74
warp_import_thread_failure2493312/2493313/74
pre_leaf_raise2493318/2493323/74
foreign2493324/2493326/74
unknown2493327/2493328/74
sync_raise2493329/2493333/74
sync_hang2493337/2493342/-9
cleanup_raise_before2493354/2493355/74
cleanup_raise_after2493360/2493364/74
normal_no_ack2493369/2493370/74
normal_wrong_ack2493372/2493373/74
normal_ack_eof2493377/2493382/74
normal_stalled_disarm2493383/2493385/74
terminal_no_ack2493386/2493387/-9
terminal_wrong_ack2493398/2493399/74
terminal_ack_eof2493401/2493402/74
target_cleanup_hang2493403/2493404/-9
target_exit_hang2493469/2493470/-9
```

Finalexact ps query for allaboveIDs returnedheaderonly. Scopedps grep for
pytest/guarded_custody_process/guarded_wait_process/native_bootstrap.py/
optical_closed_worker returnedempty. Treeclean. nvidia-smi before/after only
unrelated2483521(GPU-aa67d96f-c377-7912-8efb-2ac8f0ddcfe4), preserved; usage
changed7170->9058MiB, not attributed totheseCPUtests. Other shortlived suite
descendants not individually sampled. This is scoped hygiene, not proof of
globalidleness or diagnosis of the original unknownhang.
