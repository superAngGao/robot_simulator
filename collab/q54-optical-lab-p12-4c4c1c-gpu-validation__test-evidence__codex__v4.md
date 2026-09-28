# C4c1c native generic event audit: author clean-checkout evidence

Initiative: q54-optical-lab-p12-4c4c1c-gpu-validation
Stage: test-evidence
Author: codex
Version: v4
Date: 2026-09-29
Status: in_review
Related Files: acceptance-closure review-followup v4; independent review v7
Owner Summary: 239 author tests pass on the frozen candidate; independent
review is separate and is not replaced by these results.

Candidate: `6d57fa789cb50e4191239a4d00f628f25824bb46`.
Tree: `5b22a629b5cf0787743b79cf8f522a9fe2a20bb7`.
Runtime checkout: `/data3/ang.gao/robot_sim_c4_verify_6d57fa7`, detached and clean
before and after all three groups. Primary `tileops-dev` (183:40031) SSH was
retried and timed out; owner-authorized 101 fallback was used. No other existing
worktree was edited, cleaned or reset.

All commands ran from that checkout using:

```text
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim env CUDA_HOME=/data3/ang.gao/anaconda3/envs/robot_sim python -m pytest -q -p no:cacheprovider <selection>
```

Python 3.12.13 and toolkit NVCC 12.8.93 were rechecked. Installed runtime is
Torch 2.9.1+cu128 and Warp 1.12.0. The system NVCC 11.5 was not used. These are
functional tests, not performance samples. The three disjoint author groups
overlapped; independent GPU review started only after all author groups exited.

## Group 0: 131 passed in 195.45 seconds

Selection:

```text
tests/gpu/test_warp_sacrificial_render.py::test_recorded_camera_matches_independent_cpu_oracle
tests/gpu/test_warp_sacrificial_render.py::test_recorded_render_targets_second_real_gpu_and_restores_ambient
tests/gpu/test_warp_sacrificial_render.py::test_exact_recorded_bundle_reused_with_two_complete_sentinels
tests/gpu/test_warp_sacrificial_render.py::test_resident_and_empty_recorded_abi
tests/gpu/test_warp_shade_microfixtures.py
tests/unit/optics/test_sacrificial_lifecycle.py
tests/unit/optics/test_sacrificial_watchdog.py
tests/unit/optics/test_sacrificial_sources.py
tests/unit/optics/test_sacrificial_leaf_descriptors.py
tests/unit/benchmarks/test_native_input_observations.py
```

## Group 1: 64 passed in 413.37 seconds

Each node below belongs to `tests/gpu/test_warp_sacrificial_render.py`:

```text
test_real_recorded_leaf_rejects_forbidden_runtime_before_native_submit
test_torch_wait_negative_uses_explicit_second_device_stream
test_exception_restores_exact_current_stream_on_real_devices
test_stream_drift_quarantines_without_target_only_sync
test_reuse_partial_failure_retains_both_snapshots
test_every_command_partial_submission_lifetime
```

## Group 2: 44 passed in 288.20 seconds

Each node below belongs to the same GPU test module:

```text
test_setup_recording_and_warmup_failure
test_setup_and_drain_failures_have_process_terminal_disposition
test_real_gpu_graph_watchdog_rejects_missing_or_wrong_ack
test_parent_loss_kills_real_gpu_child_without_unwinding
test_post_submit_journal_failure_is_hard_exit_not_python_unwind
test_input_readback_resources_remain_in_guardian_across_failures
```

All three commands exit 0, no skips. Total is 239, including every one of the
135 current sacrificial GPU cases, the independent shade microfixture test and
103 CPU/POSIX tests. This is the scoped C1c selection, not the whole repository.

## Logs and source checks

Logs remain under `/data3/ang.gao/` and byte-identical local copies under
`C:/Users/AG/新建文件夹/c4_assets/`:

| File | SHA-256 |
| --- | --- |
| c4-6d57fa7-author-group0.log | 33ec9a7a3c7ad9be8b45ba3eb90de3f68b1f19a40911db9d68685052f4dbc47e |
| c4-6d57fa7-author-group1.log | 6891c992707551e34b7cf060e43277825e32f8e3e5a06725fbac8616d3c1dddd |
| c4-6d57fa7-author-group2.log | c907007dc7b2f24dc0206f79496dc0e5e8155321cdead15438d13ef50f29a7f1 |

Local Python 3.14: all 16 Python files in `git diff --name-only 7de02e8 HEAD
-- '*.py'` pass Ruff check and format check. Final local descriptor tests:
15 passed in 0.56 seconds. `git diff --check 7de02e8 HEAD` and remote
`git show --check HEAD` pass; HEAD/tree remained unchanged and both trees clean.

Routine ignored Python/pytest/Warp caches are permitted. This run does not
claim cold-cache compilation. No packages, product gate, baseline, result or
shipping artifact was installed/enabled/generated. No push or merge occurred.
The earlier four-failure draft and its repaired 24-test run are documented in
follow-up v4 and are not counted as final-candidate evidence.

C4c1c approval is pending independent review. C4c2-c4, C4d/B3, product GPU
preparation, a3b, baseline/results and host/shipping paths remain closed.

Subsequent independent review v8 approved only C4c1c at the same candidate,
with 472 independent passes and calibrated native counterexample closure.
Its complete report is archived separately without alteration; no broader
scope or gate is implied by recording that verdict.
