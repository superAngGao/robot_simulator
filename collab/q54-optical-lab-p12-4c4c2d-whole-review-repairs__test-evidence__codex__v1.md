# Repair CPU evidence and bounded native preflight

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-09
Status: implemented
Related Files: whole-review-repairs implementation-note/code-preflight-review v1;
runtime-adoption whole-code-review v1
Owner Summary: The repaired candidate has independent bounded preflight approval.
CPU replay closes the original incomplete-admission counterexample; real-device
reruns have started but no result or whole-stage acceptance is claimed here.

Reviewed code fc8618268dd8f3fb10e29e32626219500751c41a, repair base
bca0d5345244014d054f94b0ae5ec4954a54b61e. Fresh independent session
/root/whole_repairs_preflight_v1 returns Approve with comments for the repair
and exact16/85/12 native rerunners only. Full unchanged report SHA256
79e3c55e08f87f89d40fc92fb089f2ce56374ccc9d1f8dbc56a4cb7ef61d74c2.
F1/F2/F3 and C1/C2 are addressed at code/preflight scope, not whole acceptance.

Primary tileops-dev uses clean detached fc86182 at
/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip and
/home/ang.gao/miniforge3/envs/robot_sim/bin/python. No dependency change.
Exact scripts in ../c4_assets were fed via Get-Content -Raw -Encoding UTF8
through ssh to python -I -B -. Pytest children use -B -m pytest, disabled plugin
autoload/cacheprovider and fresh external basetemp directories.

| Original run | Actual result | Archive bytes / SHA256 |
| --- | --- | --- |
| run_c2d3_foundation_cpu_v3.py; c2d3-fc86182-foundation-cpu-v3 | 1058passed45failed160.24s; exit1 | 16428721 / 1519637cc29bf02b2bddc421f982af96e3e67a04e89f71ef6dd5823774b29f9f |
| run_c2d3_cold_importer_cpu_v1.py; c2d3-fc86182-cold-importer-cpu-v1 | 262passed20.75s; exit0 | 60524 / 78c4b455dfbc235c8a3de12d80bcd83d746714d0c0110be432f796d56e46dae5 |
| run_actual_record_mutations_v6.py; c2d3-fc86182-actual-record-mutations-v6 | 500passed36skipped832.34s; exit0 | 6894 / 9484af240e3daaad30d6e2e02bdea282502a87d1bd91c98f05d7bd0d1fc4da3e |

Each archive is named as the run plus .tar.gz, retained under remote
/data2/shared/ang.gao and downloaded/hash-verified in ../c4_assets. Archives
contain raw stdout/stderr, commands, summaries and scoped residual evidence.
These overlapping selections are not additive unique-case totals.

Preserve the failed combined invocation honestly. Its45 failures all occurred
before replacement scenarios: collection of test_cuda_arena_contract imports
optics.cuda_arena -> cuda_timeline and installed Torch, conflicting with the
cold importer fixture's required baseline. A separate persisted diagnostic in
the cold-trio archive reproduces that collection import with cuda._initialized
false. The fresh interpreter's complete importer/global/replacement trio passes
all262, including all45 replacement cases. No production check or fixture was
weakened. Independent review classifies this as nonblocking orchestration N1:
keep cold importer fixtures separate from backend-importing collection.

The mutation suite reads eight original actual positives. All48 new deletion
mutants ran.36skips are explicitly inapplicable:12nonGo2 cross-case,12missing
component occurrences,12sources without empty arrays. All168 original source
artifact files retain their before/after hashes. This is descriptive CPU replay,
not new execution or reconstructed native authority.

The unchanged original whole_review_v1_replay.py separately validates all113
originals (16native,85fault,12terminal), their exact selection/attachment and
same-run identity joins. All eight NumPy-completion-deletion counterexamples
now reject with incomplete external admission; no NumPy/Torch/Warp import.
Local whole_repairs_v1_replay.json rawSHA256
3730a827dddaebe80aab9927bcf693124563f806733009aba0de3126d00673c5.
The old native records remain old e585956/7cde887 execution evidence.

Scoped residual checks find all70 observed foundation,2mutation and2cold-trio
PID references absent from /proc and GPU query (rc0). Procfs polling may miss
very short-lived CPU children; this is bounded evidence. Exact ps also finds
replay3242917/parent3242897 and diagnostic3259957/3262423 absent. No signals,
GPU reset, alternate host or unrelated-process action was used.

After reading and archiving the completed independent report, the approved
run_c2d3_native_scope_v2.py started on the unchanged remote candidate. Its
runner3293204/pytest3293209 and output c2d3-fc86182-native-scope-v2 identify
the NEW run. At this evidence cut it is running, not passed. Native/fault/
terminal results and fresh whole C2d3 re-review remain required.

C4 remains active. C2d3, C2d and C4 are not accepted as complete. C2e/f,
C4c3/c4, C4d/B3, final graph/warmup/readiness, product GPU preparation, a3b,
result/baseline, R2/host/shipping and backend selection remain closed.
