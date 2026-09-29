# C2c1 approved revision and author regression evidence

Initiative: q54-optical-lab-p12-4c4c2c-cpu-owners
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-09-30
Status: accepted
Related Files: CPU-owners review codex v2; review-followup v1; runtime-integration design v3
Owner Summary: Record exact 4cac1be C2c1 approval and completed bounded regression. This record changes no source, requirement, verdict or activation gate.

## Approved scope

Independent fresh session `/root/c4_c2c1_cpu_owners_review_v2` approved
`4cac1be29efde9912c3ea1ebcf0781ef37423db5` for C2c1 only. Its full report is
archived unchanged as review v2. It independently closed the original four
Python3.12 AST failures, verified all36 fixture hashes from historical c969296,
ran local88/remote117 tests, and recomputed real Go2 content and camera parity.
The earlier Request changes report and failed-run evidence remain preserved.

## Author validation of the approved code

Local Python3.14 ownership/camera: 88passed in2.57s; Ruff and Git checks passed.
Remote clean detached checkout:
`/data3/ang.gao/robot_sim_c4_c2c1_4cac1be` on authorized fallback101.
Environment robot_sim Python3.12.13, Torch2.9.1+cu128, Warp1.12, CUDA12.8;
H200 UUID GPU-22a6a96e-63d9-b348-e2db-ff3e4813a989 mapped to logical0.
Two A100 devices remained visible for existing multi-device tests. No installs.

Command, using that environment's Python:

```text
python -m pytest tests/unit/benchmarks tests/unit/optics/test_cpu_materialization_ownership.py tests/unit/optics/test_resolved_camera_contract.py tests/unit/optics/test_optical_pipeline_lab.py tests/unit/optics/test_sacrificial_sources.py tests/gpu/test_warp_render_recording.py tests/gpu/test_optical_cuda_scene.py tests/gpu/test_optical_warp_native_commands.py tests/gpu/test_warp_sacrificial_render.py::test_recorded_camera_matches_independent_cpu_oracle -vv -ra --tb=short --durations=30 -o faulthandler_timeout=120 --junitxml=/data3/ang.gao/c2c1_4cac1be_validation_logs/junit.xml
```

Result: **497passed in112.86s**, no failures/skips. Monitor exit0 after114.62s,
watchdog_reason=null. This is targeted regression, not a new full-suite claim.
Exact command, revision, environment and before/running/after process/GPU
snapshots are in `/data3/ang.gao/c2c1_4cac1be_validation_logs`, mirrored locally
under `C:/Users/AG/新建文件夹/c4_assets/c2c1_4cac1be_validation_logs`.

Artifact SHA-256:

- pytest.log: `0f36364b544fecc63e828b01fb9d84cda4c9b275fe4899d224e13c4053cdcfbe`
- metadata.json: `651daf89e91b970a1696d9ddb764bd2f4288e9b3e3f51b301c36d4c53b330bbc`
- summary.json: `d43b24c73487958f2692fb426e1ed4e66936340b5c6c0eee5290a09c8357bc87`

## Round-end process checks

Author monitor2409347 (parent2409331, start04:37:42 server-local) and
pytest2409378 (parent2409347, start04:37:43) exited. Sampled supervised
children2414643/2414644 (start04:38:36) were also absent in explicit ps after
the run. Worktree-path worker/supervisor scan found none. Local SSH/scp tool
sessions completed and matching author command scan returned no entries.
Sampling does not identify every short-lived child; process snapshots, normal
tool exit and scoped residual scans are the available evidence.

Reviewer independently recorded its own process checks. Its initial60s Go2
attempt stopped after partial output; the raw timeout status was not captured,
so this record does not attribute an exact termination cause or count it as a
pass. Both recorded PIDs were absent before a240s bounded rerun, which completed
all checks and exited0. Full details and limitations are in review v2.

Final nvidia-smi showed only unrelatedPID2000951 on H200, untouched. No GPU
reset or cleanup of unrelated work occurred. No unresolved owned hang/residual
was observed. The earlier unidentified killed-process incident remains
unroot-caused; a clean round is not proof that future hangs are impossible.

C2c remains incomplete. Resolved parameters/artifacts, v13 bindings, asset-byte
custody and later lifecycle/worker integration still require review. All
R2/S/host/depth-two, C4d/B3, product preparation/submission, a3b/result/baseline
and C4 parent-completion gates remain closed. No benchmark result was generated.
