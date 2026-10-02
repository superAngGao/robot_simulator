# C2c2 asset custody: scoped acceptance and round-end evidence

Initiative: q54-optical-lab-p12-4c4c2c-asset-custody
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-03
Status: accepted
Related Files: asset-custody implementation-note/review v1; runtime-integration design v3
Owner Summary: Exact candidate536b815 passed independent C2c2 review. This records tests and observed cleanup only; no additional implementation or activation authority.

Reviewed code: `536b815ee65db351c502922ce8b1cf9ddf3c7fd0`.
Base: `6b882f6b87e16189c334e1e41df50c86f5b1abf5`.
The fresh reviewer `/root/c4_c2c2_asset_custody_review_v1` returned Approve;
its full report is archived unchanged beside this record. Independent191 tests
and its own real-Go2 probe passed; those are separate from author results below.
This evidence-only delta changes neither reviewed source nor stage requirements.

## Environment and preservation

Primary `tileops-dev` (183.166.183.33:40031) still timed out with ConnectTimeout8.
The owner-authorized101 fallback was reachable. Its existing common worktree
`/data3/ang.gao/robot_sim_c4c1b2` contained unrelated modified/untracked files;
they were preserved. A bundle was fetched into the common Git object database
and a fresh detached worktree `/data3/ang.gao/robot_sim_c2c2_536b815` was created
at the exact candidate. That tree was clean before and after all runs.

Existing `/data3/ang.gao/anaconda3/envs/robot_sim`: Python3.12.13,
Torch2.9.1+cu128, Warp1.12, CUDA12.8 toolchain. No dependency installation.
CUDA_HOME and PATH used this environment, MAX_JOBS4, PYTHONPATH the clean tree.
GPU0 in the process was the owner's H200 NVL (physical2,
`GPU-22a6a96e-63d9-b348-e2db-ff3e4813a989`); physical A100 GPUs0/1 followed it
in CUDA_VISIBLE_DEVICES. This is not a claim that every test exercised all GPUs.

## Author executions

Local Python3.14: custody17passed/2skipped (real parser and POSIX checks deferred
to Linux), ownership/camera88passed; changed-file Ruff passed. The implementation
note records the corrected initial Windows descriptor/path ctime test failure.

Remote clean candidate, robot_sim:

```text
python -m pytest tests/unit/benchmarks tests/unit/optics/test_captured_asset_bytes.py tests/unit/optics/test_cpu_materialization_ownership.py tests/unit/optics/test_resolved_camera_contract.py tests/unit/optics/test_optical_pipeline_lab.py tests/unit/optics/test_sacrificial_sources.py tests/gpu/test_warp_render_recording.py tests/gpu/test_optical_cuda_scene.py tests/gpu/test_optical_warp_native_commands.py tests/gpu/test_warp_sacrificial_render.py::test_recorded_camera_matches_independent_cpu_oracle -vv -ra --tb=short --durations=30 -o faulthandler_timeout=120 --junitxml=/data3/ang.gao/c2c2_536b815_validation_logs/junit.xml
```

**516 passed in113.99s**, no skips/failures. This is a targeted regression,
not an entire-repository test run. External monitor exit0,116.616815923s,
watchdog_reason=null. Operational7200s overall/600s no-output limits were not
reached; they are diagnostic supervision bounds, not performance/product gates.

Real-Go2 probe ran separately with a300s timeout and exited0. Its fresh process
blocked benchmark/example/tool/Torch/Warp imports. Legacy versus captured-byte
importer bounds and398432 triangles matched exactly. Both resolutions' scene
and fixed-fixture digests and1080p replay matched existing frozen values;
the unchanged values are also independently reproduced in the review report.
Asset aggregate:479dcbdb52c64ca86243d6b1d3b7167bb7d3820af1e60e355168b3c17ddf3ed0.
Canonical assets were not modified, replaced or mocked for this acceptance.

Artifacts (metadata, command/environment, junit, pytest log, snapshots, summary,
Go2 log): `/data3/ang.gao/c2c2_536b815_validation_logs`, mirrored at
`C:/Users/AG/新建文件夹/c4_assets/c2c2_536b815_validation_logs`.
Monitor/probe sources are retained outside the repository at
`C:/Users/AG/新建文件夹/c4_assets/c2c2_validation_monitor.py` and
`C:/Users/AG/新建文件夹/c4_assets/c2c2_real_go2_probe.py`.

- pytest.log SHA-256:d9d20a18af7aa9a10ae1a39fdfbe022bbd3de69ee44b46f122b63c8090d4d046
- go2.log SHA-256:726ae955e042d517f53ad3162f9aec4a62bf190b96d88958247487820479781a

## Round-end no-hang checks

Author monitor1786059/pytest1786087 started at host03:19:21/22;
probe timeout1788971/Python1788972 at03:20:16;
sampled supervisor1790208/worker1790221 at03:20:56. These owned processes
were absent after their successful exits. Monitor SSH, probe SSH and evidence
SCP handles all completed. Local matching Python/SSH/SCP process query was empty.
The independent report records its own PID/start/exit ledger and final checks.

After reviewer completion, the author additionally checked PIDs1786059,1786087,
1788971,1788972,1790208,1790221,1797201,1797226,1808792,1808793:
`ps` returned only its heading. `nvidia-smi --query-compute-apps` returned no
entries. Remote candidate remained clean at536b815. No unrelated process was
killed; no GPU reset or unrequested cleanup was performed.

This proves observed scoped cleanup, not absence of all possible system hangs.
Snapshots do not enumerate every short-lived child. The earlier unidentified
hang remains unroot-caused; this passing round does not retrospectively resolve it.

## Approval boundary and next work

Only C2c2 CPU captured-asset custody is approved. Resolved render sequences,
native scalar binding, v13 row/schema/binding migration and final worker source
admission are still incomplete C2c work. C2d/e/f, R2/S/host/depth-two, C4d/B3,
product GPU preparation/submission, a3b, result/baseline and C4 completion remain
closed. No performance comparison or backend selection result was produced.
