# Captured input and CUDA factory regressions after Warp compilation support

Initiative: Q54 / P12.4 / C4 guarded compilation
Stage: author native regression evidence
Author: Codex
Version: v7
Date: 2026-10-10
Status: in_review
Related Files: warp-module-preparation test-evidence v4–v6;
  tests/support/cuda_artifact_runtime_acceptance.py
Owner Summary: Existing Warp input and both CUDA Go2 resolutions passed at
  4fe6b3a. All owned processes exited. Compilation's independent native review
  is pending; these runs do not exercise the newer fixed output-storage code.

Exact clean revision: 4fe6b3ae71ac6aa1586cad5f4f7ac66c6717a2ec in the isolated
remote /data2/shared/ang.gao/robot_sim_warp_modules_wip. Command: robot_sim Python
-I -B /data2/shared/ang.gao/run_warp_module_regressions.py. Parent3149177,
tool89112 observed terminalexit0. Same H200 target1/ambient0 and unchanged
1000 ms ACK / 120000 ms operation budgets. Warp profile v4 and fixed CUDA
environment v2 were verified by the exact SHA values in the archived launcher.

| Raw directory suffix | Case / selector | Worker / supervisor | Normal completes / ARMs |
| --- | --- | --- | --- |
| v37 | Warp Go2 960, runtime_scope_success | 3149403 / 3149401 | 6 / 7 |
| v38 | CUDA Go2 960, runtime_artifacts_success | 3152181 / 3152172 | 7 / 8 |
| v39 | CUDA Go2 1920, runtime_artifacts_success | 3160311 / 3160310 | 7 / 8 |

All workers exited0 and supervisors recorded EXITED without deadline expiry or
kill. Each performed one successful terminal emergency target synchronization
and whole-graph release. This is the existing sacrificial input/factory disposal,
not final preparation quiescence or sample execution. The reviewed runner replayed
runtime/native-scope/custody records and checked exact phase counts, absence of
failure/ordinary-atexit markers, and actual worker/supervisor disappearance.

After parent exit the archive script verified all seven distinct owned PIDs
absent. It preserved the three raw directories, launcher/archive scripts, batch
result and both environment profiles. Remote shared-root and local ../c4_assets
copies: c4-warp-module-author-evidence-v6.tar.gz and adjacent .json; also
c4-warp-regression-batch-v1.json. Locally verified archive SHA256:
bdcd27b71a0c9c00cf527c6aa0e09268ab44443c7c5b27803d6f4467fe3f8e5d.
The final GPU-process snapshot contained only unrelated MPS2238410, untouched.
No dependency change, GPU reset, unrelated signal or source-tree update occurred.

The newer local allocation/storage code at12c1619 is not covered by these runs.
Compilation native acceptance is assigned to a fresh independent reviewer at
the immutable4fe6b3a source/evidence checkpoint. Full graph, real optical output,
warmups, final readiness/barrier, full C4/refactor, performance and product gates
remain incomplete.
