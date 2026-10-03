# C2d1 author verification, 2026-10-03

Initiative: q54-optical-lab-p12-4c4c2d-guarded-custody
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-03
Status: implemented
Related Files: implementation-note v1; code-review v1; repair-design v2
Owner Summary: Record actual author tests, original bug reproduction and scoped
process cleanup. This evidence does not approve the implementation.

This is author evidence, not independent approval. Reviewed code remains
`4ee33ab7a09720f0dc08f3e86bb199af59e58e40`; design-only descendants do not
repair its two reproduced defects. No GPU preparation or result gate opened.

## Final candidate regression

Remote fallback `/data3/ang.gao/robot_sim_c2d1_code_v1`, exact clean HEAD above.
Primary SSH183.166.183.33:40031 timed out on this turn;101 transport succeeded.
Environment `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python` (robot_sim).

With real Go2 fixture at
`/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2`, ran
`timeout --signal=TERM --kill-after=15s 420s <python> -B -m pytest ... -q -s -p no:cacheprovider`:

- tests/unit/optics/test_guarded_lifecycle.py
- tests/unit/optics/test_guarded_evidence_fixtures.py
- tests/unit/optics/test_guarded_custody_process.py
- tests/unit/optics/test_guarded_wait_core.py
- tests/unit/optics/test_guarded_wait_process.py
- tests/unit/optics/test_native_bootstrap.py
- tests/unit/optics/test_native_source_policy.py
- tests/unit/optics/test_native_sources.py
- tests/unit/optics/test_sacrificial_watchdog.py
- tests/unit/optics/test_sacrificial_sources.py
- tests/unit/optics/test_sacrificial_lifecycle.py
- tests/unit/optics/test_sacrificial_leaf_descriptors.py
- tests/unit/benchmarks/test_closed_worker_protocol.py
- tests/unit/benchmarks/test_optical_closed_worker.py
- tests/unit/benchmarks/test_optical_benchmark_contracts.py

Tool session65897 returned actual exit0, `628 passed in 167.81s` with no skips.
Output retained by tool transcript; no separate persistent stdout log claimed.
timeout2314618(parent2314536) and pytest2314639(parent2314618), start09:42:52;
sampled bootstrap2316046(parent2314639), start09:43:40. All absent on final
exact PID query. Process-name query showed only unrelated2568 and626988;
GPU compute query empty.22 process cases were supervised and observed/reaped,
including deliberate sync/cleanup/exit hangs; none credited outer timeout.

## Original independent counterexample reproduced by author

Ran existing externally bounded `c2d1-review-runner.py` using local
`C:/Python314/python.exe -B`, unchanged code under the design-only HEAD.
Runner19140(parent31156), child6152(parent19140), actual child and command exit0.
The runner's15-second bound was not reached. Both F1 variants reproduced:
CPU branch authorizes release after fault; dirty-target branch calls sync after
fault. Both F2 contradictory histories were accepted. Exit0 here means bug
reproduction succeeded, not correctness. Exact final process queries returned
neither runner nor child. No runtime GPU call was used by this CPU repro.

## Preservation and interpretation

Remote original `/data3/ang.gao/robot_sim_c4c1b2` still has its pre-existing four
dirty paths; unchanged and never reset. Host GPU inventory is A100 GPU0/1,
H200 NVL GPU2. CPU tests do not establish H200 correctness, historical-hang
cause, complete C2d integration or C4 completion. No broad kills/GPU reset,
dependency installation, push or merge performed. All product/results gates
remain closed pending their required implementation and independent review.

Later round-end query after design approval again found all owned sampled PIDs
absent. A new unrelated ChainLoRA training tree2336113 ->2336215 ->2336351 was
then present; GPU compute PID2336351 belonged to that other user's command and
was left untouched. Thus the final observation is no owned test residual,
not global GPU idleness. The earlier empty GPU query remains an earlier fact.
