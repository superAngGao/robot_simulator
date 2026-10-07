# C2d2 author evidence (not independent approval)

Date: 2026-10-07 Asia/Shanghai. Base af509cb5e3ed4103a1035a1298cd20bb652f45e4.
Candidate f7692b4b7bfd50521956c5c83fcb2bc94049906a.
Local C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design.

Primary183 SSH alias tileops-dev is reachable again, hostname hd03-gpu2-0031,
eight NVIDIA H200 reported. Existing /data2/shared/ang.gao/robot_simulator
is clean main at f771e612beaeea126e26926738097512cd2d809c; not edited.
Current candidate cloned from a full local Git bundle into NEW isolated
/data2/shared/ang.gao/robot_sim_c2d2_code_v1. No reset, push or package install.
101 not used this round. Original local candidate was dirty from paused work;
all changes preserved and completed into the explicit candidate.

Local C:/Python314/python.exe:

- pytest test_native_source_custody.py test_native_source_custody_process.py
  -q -x -p no:cacheprovider: 26 passed,30 POSIX skips,123.51s,actual exit0.
  Process session31684; final scoped CIM query found no owned Python process.
- pytest test_guarded_lifecycle.py test_guarded_evidence_fixtures.py
  test_guarded_wait_core.py test_native_source_policy.py -q -p no:cacheprovider:
  476 passed10.52s,actualexit0,session62913.
- Ruff changed Python files and git show --check passed; candidate tree clean.

Primary robot_sim interpreter /home/ang.gao/miniforge3/envs/robot_sim/bin/python
Python3.12.13,pytest9.1.1. Author source-process suite:

```
cd /data2/shared/ang.gao/robot_sim_c2d2_code_v1
timeout --signal=TERM --kill-after=10s 220s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_native_source_custody_process.py -q -x -s -p no:cacheprovider
```

Actual1failed9passed32.04s,exit1,session42206. source_no_ack accepted safety
outcome was NOT established by this failing run: supervisor returned0 and
child was killed/reaped(-9), but test demanded74. Independent reviewer must
check allowed evidence/endpoint before author changes test. No production
timeout/gate changed. All output below is observed, not implied success:

| fault (Warp) | supervisor | child | child exit |
|---|---:|---:|---:|
|success|2905635|2905636|0|
|prelude_raise|2905711|2905716|74|
|module_adopt|2905804|2905805|74|
|compile_raise|2905959|2905965|74|
|body_raise|2906055|2906057|74|
|caught_nested|2906125|2906128|74|
|audit_raise|2906145|2906150|74|
|import_hang|2906202|2906206|-9|
|parent_loss|2906388|2906389|-9|
|no_ack|2906704|2906706|-9|

All supervisors returned0 and exact identities were checked by the tests;
last no_ack assertion failed after actual exit/reap validation. Final ps query
showed no owned pytest/source/guarded process. GPU inventory consistently
contained only pre-existing MPS2238410 (60MiB); untouched. No GPU work launched.

Frozen Go2 archive relayed from local c4_assets/go2-frozen-crlf-checkout.tar.gz,
SHA25678013271088bfe33f09e44605a3f7234ba907ca6a667865069a45fd22d014e52,
into NEW /data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie. Git HEAD
da76818e269b82289eba39808e2fb91d679d6994. Archive member paths checked before
extraction; existing sources not overwritten. core.autocrlf=true configured
only in this new asset copy; no byte rewriting. Diff numstat/summary empty,
status lists text paths M and allowed untracked q54_asset_manifest.json.
Formal asset preflight, not this narrative, must decide validity.

First real_go22case probe under45s outer bound yielded one dot then tool
returned1 without pytestsummary(session94498). No owned residual found.
Do not credit this as a passing run. Verbose150s rerun session18137 completed
2passed39deselected46.03s, actualexit0, both lanes. Formal preflight and actual
CPU materialization succeeded; outer bound did not fire. The first45s bound
was shorter than the successful run duration. Environment:
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2.

Explicit protected-file diff af509cb..f7692b4 is empty for C0 lifecycle,
native_setup/native_contract/worker, wait_protocol/wait_supervision,
guarded_lifecycle/guarded_evidence, product supervisor, gate.json and shared
collab rules. Full bundle SHA256
8e4f44319ac34b4ef93c17e65a09e9f777590f0df6afd29e06a26d85fc7adda5.

Read-only shell diagnostics initially had Windows-to-SSH quoting errors
(local id substitution and remote grep alternation / Python-c quoting).
Corrected to literal single-quoted command and grep -e; no writes resulted.

All tests are CPU scope. No product runtime/preparation, C4 completion,
a3b/result/baseline, GPU safety or historical hang fix is asserted.

An additional local benchmark-contract attempt failed25/passed16 in4.44s.
Short -x --tb=short repro failed1/passed1 in0.64s: runtime hook resolver raises
ImportError for torch.empty (local Python3.14 has noTorch/Warp). This is a test
environment prerequisite failure, not evidence of product behavior. No install;
Repeat in primary robot_sim, session20591:41passed8.58s,actualexit0. Command:
timeout --signal=TERM --kill-after=10s 90s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest tests/unit/benchmarks/test_optical_benchmark_contracts.py -q -p no:cacheprovider.

## F1 repair author round

New frozen candidate0dd684867a8d60a69978e878b26d98169d2b9b1c in isolated
/data2/shared/ang.gao/robot_sim_c2d2_code_v2. Originalv1 preserved. Candidate
delta is tests plus verbatim review/implementation note, no production code.
Ruff and git show --check passed, localcollection32cases exit0.

Actual command from new worktree:
timeout --signal=TERM --kill-after=10s 250s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest tests/unit/optics/test_native_source_custody_process.py -q -s -p no:cacheprovider

32passed118.81s,actualexit0,session32280. No outertimeout. Every supervisor
returned0, recorded exact child parentage, validated artifact and reaped child.
Actual supervisor/child/childexit ledger:

```
warp success2936770/2936771/0 prelude2936825/2936826/74
module_adopt2936910/2936912/74 compile2936940/2936941/74
body2937029/2937227/74 nested2937280/2937284/74 audit2937448/2937449/74
import_hang2937518/2937520/-9 parent_loss2938043/2938047/-9
no_ack2938512/2938514/-9 worker_exit_no_ack2938541/2938542/74
wrong_ack2938571/2938572/74 ack_eof2938630/2938635/74
stalled_disarm2938701/2938702/74 cleanup_hang2938755/2938756/-9
exit_hang2939090/2939091/-9
cuda success2940974/2940983/0 prelude2941418/2941419/74
module_adopt2941499/2941505/74 compile2941647/2941649/74
body2941677/2941686/74 nested2941827/2941830/74 audit2941911/2941914/74
import_hang2942480/2942488/-9 parent_loss2942710/2942711/-9
no_ack2943389/2943392/-9 worker_exit_no_ack2943496/2943497/74
wrong_ack2943651/2943660/74 ack_eof2943716/2943717/74
stalled_disarm2943771/2943773/74 cleanup_hang2943804/2943805/-9
exit_hang2944185/2944189/-9
```

Actualpytest command exit observed; launcherPID not sampled this run. This
is not a complete OS descendant ledger. Round-end scoped ps/exact noACK and
last supervisor/child lookups showed noownedresidual; GPUonlypreexistingMPS.
Fresh reviewer v2 independently repeats tests; authorpass is not approval.

## Final archive-round hygiene

After the approved v2 report was archived, local/remote candidate and original
worktree checks remained clean; local scoped Python and remote ang.gao
Python/timeout inventories were empty. GPU inventory had newly started
unrelated pytest-xdist workers2962673/76/79/82/85/88/93/96, parent2962518.
Exact ps identified UID232071 (current ang.gao UID1006), parent command
python3 -m pytest -q tests --ignore=tests/ops/test_softmax.py -m smoke -n8,
and a running roofline-oracle case. These are not this project's launch ledger
or residuals; none were signalled. ExistingMPS2238410 also remained untouched.
The final machine is not globally idle; this round's owned processes are gone.
