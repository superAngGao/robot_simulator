# C2c4 author regression evidence

Initiative: q54-optical-lab-p12-4c4c2c-render-binding-migration
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-03
Status: implemented
Related Files: implementation-note v1/v2; independent review v1/v2
Owner Summary: Author validation at 4b7bf4c passed 886 remote tests and 276 local CPU tests. This records execution evidence, not independent approval or C4 completion.

## Exact candidate and environment

- HEAD: `4b7bf4cee0d1671e5d60c16c697d37caeec6d8a3`.
- Tree: `027623ff65495caddc6283ff8da8c7aa7ac7ed85`.
- Author checkout: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Remote detached checkout: `/data3/ang.gao/robot_sim_c2c4_4b7bf4c`.
- Primary `tileops-dev` / `183.166.183.33:40031` timed out. Owner-authorized fallback `ang.gao@192.168.80.101` used the existing `robot_sim` environment, not a new installation.
- Interpreter/CUDA_HOME: `/data3/ang.gao/anaconda3/envs/robot_sim`; its bin/lib prepended to PATH/LD_LIBRARY_PATH; exact checkout PYTHONPATH; MAX_JOBS=4. Python3.12.13, Torch2.9.1+cu128, Warp1.12.0, CUDA12.8.
- CUDA_VISIBLE_DEVICES: `GPU-22a6a96e-63d9-b348-e2db-ff3e4813a989,GPU-aa67d96f-c377-7912-8efb-2ac8f0ddcfe4,GPU-3cf5c216-4235-4739-5ed0-818737fae8d7`. Logical0 is H200 NVL, with two real A100 devices available for second-device checks.
- Normal runtime/JIT caches were allowed. Every result here comes from a fresh pytest invocation/log, not cached test outputs. No assets/dependencies installed, no GPU reset or unrelated process termination.

## Completed final author rounds

Local Python3.14 (NumPy/pytest/Ruff available, no Torch/Warp):

```text
C:/Python314/python.exe -m pytest -q tests/unit/optics/test_native_contract_extraction.py tests/unit/optics/test_resolved_camera_contract.py tests/unit/optics/test_sacrificial_sources.py tests/unit/benchmarks/test_optical_render_migration.py tests/unit/optics/test_render_parameters.py
```

**276 passed in12.06s**. Additional guarded-wait core/process selection: **120 passed,10 skipped in1.12s**; the skipped POSIX process tests are not claimed as Windows coverage. Changed-source Ruff and `git show --check` passed. Local test/SSH/scp processes were checked after their rounds.

Remote command (through the owned bounded monitor):

```text
python -m pytest tests/unit/benchmarks tests/unit/optics/test_native_contract_extraction.py tests/unit/optics/test_cuda_timeline.py tests/unit/optics/test_render_parameters.py tests/unit/optics/test_captured_asset_bytes.py tests/unit/optics/test_cpu_materialization_ownership.py tests/unit/optics/test_resolved_camera_contract.py tests/unit/optics/test_optical_pipeline_lab.py tests/unit/optics/test_sacrificial_sources.py tests/gpu/test_warp_render_recording.py tests/gpu/test_cuda_native_arena.py tests/gpu/test_optical_cuda_scene.py tests/gpu/test_optical_warp_native_commands.py tests/gpu/test_warp_sacrificial_render.py -vv -ra --tb=short --durations=30 -o faulthandler_timeout=120 --junitxml=/data3/ang.gao/c2c4_4b7bf4c_validation_logs/junit.xml
```

**886 passed in902.94s**, no skips/failures. Monitor elapsed904.652s, exit0, `watchdog_reason=null`; finished2026-10-02T21:21:58.898113Z. Full sacrificial oracle/fault/hold coverage reached its intended states. Intentional setup/validation/emergency hang injections were supervised successes; no unexpected whole-round hang or deadline occurred.

Tests exercise actual Warp initial/post-seal scalar bytes, actual CUDA timed acquire/enqueue/finish, pre-enqueue scalar/storage rejection, real two-device scope restoration, independent CPU numerical oracle, recorded command reuse, hidden runtime-operation rejection, partial submission, setup/record/warmup failures, ACK loss, parent loss and guardian retention. This is targeted affected-surface regression, not an assertion that the entire repository or complete C4 product path was tested.

## Reproducible artifacts and process hygiene

External monitor source: `C:/Users/AG/新建文件夹/c4_assets/c2c4_revision2_monitor.py`, copied to `/data3/ang.gao/c2c4_revision2_monitor.py`. No-output deadline600s, whole-round7200s; owns only the new pytest process group. No deadline fired and no termination was needed.

Logs, command/environment metadata, JUnit, snapshots, summary and pytest PID: `/data3/ang.gao/c2c4_4b7bf4c_validation_logs`, copied unchanged to `C:/Users/AG/新建文件夹/c4_assets/c2c4_4b7bf4c_validation_logs`.

`pytest.log` SHA-256: `efe8fe8d3e8c8d3d16e1302e492a209be4e25413a588000c7efc428f3ccc9ff9`.

Pytest1939353 and monitor1939325 exited and were reaped; SSH session49105 completed exit0. The subsequent read-only hygiene checker1970298 found neither PID, no candidate/sacrificial process and no GPU compute record. Local SSH/scp query was empty. Both candidate trees remained clean at the exact revision. GPU clearance was then handed to the independent reviewer; its later processes are separate from this completed author round.

The dirty shared `/data3/ang.gao/robot_sim_c4c1b2` and all previous worktrees were preserved. No cleanup-by-name or broad kill occurred. This round observed no unexpected hang; it does not establish the root cause of the earlier historical unidentified hang.

## Failed intermediate rounds retained

- Earlier WIP at base9c49722 plus author changes:103 passed/5 failed. Three error-message assertions, one stale sample-index fixture and an old2x1 shape assertion were corrected before8b1d13c. Logs: `c4_assets/c2c4_wip_round1_logs`.
- Frozen8b1d13c expanded round:708 passed/13 failed in94.71s, no watchdog timeout. Twelve sacrificial oracle cases failed at source admission; one catalog invariant was not migration-aware. Logs: `c4_assets/c2c4_8b1d13c_validation_logs`; pytest.log SHA-256 `394abe9d4d3ff6f5b5b9b1ae7f2a4c425c68030f709696a64e0c6567e4127a6c`.
- Additional local8b1d13c selection:45 passed/2 failed, historical provenance AST/gate assertions. These and independent F1/F2 counterexamples were real candidate issues, not accepted coverage.
- First repair CPU round:275 passed/1 failed because isolated local Python omitted user-installed NumPy. The cold test now admits explicit installed paths outside the repository; it still validates repository imports and rejects an unlisted module. No package was installed. Final276-pass round supersedes that test execution, not its historical record.
- An earlier broad local test attempt lacked Torch/Warp runtime dependencies and is not used as runtime-hook evidence. Independent v1's bounded incomplete run and reproduced defects remain in its separate unchanged report.

## Approval and scope

Only the independent report can approve the frozen implementation. This evidence does not change any gate. Final closed-worker source admission, C2d lifecycle adoption, C2e arena, C2f integrated acceptance, C4c3/c4 and C4d/B3 remain required. Product preparation, a3b, result/baseline, host payload and shipping remain closed. Historical migration fixtures are validation data, never a live fallback or executable paired lane.
