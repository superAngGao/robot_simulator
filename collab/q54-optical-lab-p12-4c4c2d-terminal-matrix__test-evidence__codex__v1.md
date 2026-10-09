# Both-lane terminal matrix actual result

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-09
Status: implemented
Related Files: terminal-matrix implementation-note/code-preflight-review v1;
runtime-adoption whole-code-review v1
Owner Summary: All12 approved terminal cases passed on7cde887. This completes
that old-candidate matrix; whole review remains Request changes.

Actual remote clean detached candidate7cde8879b307090d54350ca3511529bfd8e5d7c8
at /data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip, primary tileops-dev.
Production bytes still equal e585956eb29e3a33483d8cb0c7444f893b390ac4.
Interpreter /home/ang.gao/miniforge3/envs/robot_sim/bin/python.
Existing test_native_runtime_cold_warp.py with ROBOT_SIM_TEST_C2D3_TERMINAL=1
ran both lanes, asymmetric/replay inputs and success/sync-raise/sync-hang.
Result: 12passed950.62s, exit0; four ordinary exit0, four quarantined exit74,
four original-supervisor deadline/reap exit-9. No author outer-timeout/signals.

Archive c2d3-7cde887-terminal-matrix-v1.tar.gz is4265371bytes, SHA256
fdd875571e59a4b33092c1c220a3a2213a4ed98c0bb0c50f2f26221c50ffcde0.
It is retained remotely under /data2/shared/ang.gao and downloaded/hash-verified
under ../c4_assets. It contains original per-case requests, raw outputs,
lifecycle/custody/native records, same-run external identity chains, stdout,
author summary and residual check. All12 chains were observed, no observer
errors. All26 owned PID references were absent from /proc and GPU query;
runner3162992/pytest3163009 absence was subsequently rechecked by exact ps.
Runner source ../c4_assets/run_c2d3_terminal_matrix_v1.py SHA256
aa2302b18d3a0b20162b9d955d2a621be991572a55480030e1a542a558018c58.

The unchanged whole review deliberately inspected seven then-complete cases.
This later result does not rewrite its evidence cut or close its three findings.
F1/F2 repairs need their own native validation and fresh whole review. C2d3/C4
remain incomplete; all later product and performance gates remain closed.
