# C2c3 scoped acceptance and operational evidence

Initiative: q54-optical-lab-p12-4c4c2c-render-sequences
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-03
Status: accepted
Related Files: render-sequences implementation-note/review codex v1
Owner Summary: c8b86e7 CPU sequence delta independently approved;634 author targeted regressions passed. One accidental diagnostic-command pytest was identified/stopped and excluded. C4 remains incomplete.

Base:b0a6f82e4541e8c9334795e54e04f419766a5d7c.
Reviewed candidate:c8b86e70d717c2787152b967c08dbb4cc1440a87, including
source derivation bb85f027b35120d0dd3875f3765960d41b7f947e.
The full independent report from fresh session
`/root/c4_c2c3_sequence_code_review_v1` is archived verbatim. It independently
passed288 tests and real-Go2/four-artifact reproduction, one-import/one-pack
identity checks, all120 camera frames and AST mutation controls. This evidence
commit adds no source, test, design, gate or workflow-policy change.

## Author commands and observed outcomes

Primary183:40031 ConnectTimeout8 expired; authorized101 fallback reachable.
Existing dirty common worktree was preserved. Clean detached test checkout:
`/data3/ang.gao/robot_sim_c2c3_c8b86e7`. No dependency installation.
Existing robot_sim Python3.12.13, Torch2.9.1+cu128, Warp1.12, CUDA12.8;
GPU0 in process is H200 NVL physical2,
GPU-22a6a96e-63d9-b348-e2db-ff3e4813a989; the two A100 devices followed it.
Full command/environment/revision and before/after status are in metadata.json.

Local focused parameter/ownership/camera tests:206passed; changed-file Ruff
passed. On the clean Linux candidate:

```text
python -m pytest tests/unit/benchmarks tests/unit/optics/test_render_parameters.py tests/unit/optics/test_captured_asset_bytes.py tests/unit/optics/test_cpu_materialization_ownership.py tests/unit/optics/test_resolved_camera_contract.py tests/unit/optics/test_optical_pipeline_lab.py tests/unit/optics/test_sacrificial_sources.py tests/gpu/test_warp_render_recording.py tests/gpu/test_optical_cuda_scene.py tests/gpu/test_optical_warp_native_commands.py tests/gpu/test_warp_sacrificial_render.py::test_recorded_camera_matches_independent_cpu_oracle -vv -ra --tb=short --durations=30 -o faulthandler_timeout=120 --junitxml=/data3/ang.gao/c2c3_c8b86e7_validation_logs/junit.xml
```

634passed in113.64s; no skips/failures. External monitor exited0,
watchdog_reason=null, elapsed116.593429326s. Its600s no-output/7200s overall
bounds were not reached; these are operational bounds, not performance gates.
This was targeted regression, not a whole-repository suite or baseline run.

Source generation probe at bb85f02 and final resolve probe at c8b86e7 each
completed within240s with exit0 and no forbidden benchmark/tool/example/Torch/
Warp imports. Final probe resolved the actual four sources and verified exact
frozen artifact bytes; sequence and scene digests match the independent report.
No mocked Go2, new math baseline or benchmark result record was used.

Artifacts: `/data3/ang.gao/c2c3_c8b86e7_validation_logs`, mirrored at
`C:/Users/AG/新建文件夹/c4_assets/c2c3_c8b86e7_validation_logs`.
pytest.log SHA-256:15920870475f1d0fdc08ce8e3add4adc89cbf6b0e94c535b66d2270bb457076d.
External author scripts remain in that local c4_assets directory:
`c2c3_validation_monitor.py`, `c2c3_sequence_probe.py`, `c2c3_final_probe.py`;
remote counterparts are under/tmp. Actual commands/probe output also appear in
the session tool log. These external diagnostics do not expand the worker IPC.

## Diagnostic-command incident (not candidate evidence)

An author residual-check command used an SSH shell pipeline with a quoted regex.
PowerShell/native SSH argument transfer removed the regex quoting. Remote rg
was also unavailable. The resulting pipeline accidentally executed a bare
`python -m pytest` in `/data3/ang.gao` using default Python3.13.5, outside the
candidate and outside robot_sim. This was an author command error, not a product
hang, candidate failure, intended full test or valid verification result.

Exact inspection identified PID1844108, parent1844082, start04:00:26 host time,
with command `python -m pytest`, cwd/data3/ang.gao; parent command showed the
malformed diagnostic pipeline. It had no children at inspection. Only this
confirmed owned PID was sent SIGINT. It left collection with KeyboardInterrupt;
the SSH handle33469 was fully drained/reaped (exit1). Its collection output
(3527 errors/42 skips under the wrong environment/root) is excluded entirely.
No related GPU compute was observed. No broad process kill or GPU reset occurred.

Ordinary import/pytest cache files may have been created during that invalid
collection; absence of all side effects is not claimed. No sweeping deletion
was attempted. Candidate Git status remained clean, and the original common
worktree retained exactly its four pre-existing modified/untracked paths. Final
checks switched to exact PID arguments without a remote regex/pipeline. The
independent reviewer was informed and performed its own clean scoped CPU run.

## Round-end cleanup

Author monitor1838751/pytest1838786, generation timeout1829842/probe1829863,
final probe timeout1842845/probe1842867, sampled GPU child1843842, and accidental
shell1844082/pytest1844108 were all absent on exact final ps checks. Normal test,
probe, evidence SCP and SSH tool sessions all completed. Reviewer-owned
1846077/1846103/1847470/1847492/1847493 were also absent on the author's final
check, agreeing with its independent ledger. nvidia-smi compute-app query was
empty; local matching c2c3/c8b86e7 SSH/Python/SCP query was empty. Source trees
remained at the expected revisions. No test watchdog fired in the planned run.

These sampled observations do not prove every possible hang absent. The prior
unidentified hang remains unroot-caused. This round's accidental command is
explicitly distinguished from both that historical incident and planned tests.

## Scope still closed

Only C2c3 CPU materialization/sequence artifacts/descriptive scalar records are
approved. Complete C2c still requires v13 gate/result, v8 row preimage migration,
C3 binding and real native parameter verification plus final source admission.
C2d/e/f, R2/S/host/depth-two, C4d/B3, product GPU preparation/submission,
a3b, result/baseline and C4 completion remain closed. This checkpoint does not
permit a caller-supplied record/hash to stand in for actual execution authority.
