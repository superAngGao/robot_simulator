# Repaired native fault and terminal matrices — final evidence

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: test-evidence
Author: codex
Version: v2
Date: 2026-10-09
Status: accepted evidence for the whole-C2d3 approval recorded below
Related Files: whole-review-repairs code-preflight-review v1; whole-code-review
v1/v2/v3; array-adoption-join implementation-note v2 and test-evidence v1;
native-scope-aggregate test-evidence v2
Owner Summary: The repaired full85 and separately approved12 terminal matrices
pass, as do both current-parser replays. Fresh independent whole review v4
accepts exact4da C2d3 under evidence HEADe5299e3; C4 remains active.

## Exact source and approval boundaries

Native runtime tree on primary tileops-dev remains clean detached
fc8618268dd8f3fb10e29e32626219500751c41a at
/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip.
The fresh whole reviewer audits local clean e5299e3fe1bc7bbb10a3c89c046e7ced8ab60e96,
code/test4da1fa75634d716d4018d616327c84096e047432,
productionf1ca45aa0e6b178804828acedb2ad2539712a5ef.
The new named-array join/source pin/alias test fixture evidence is separate.
No fc runtime is relabelled as a later source execution. Current parser replay
uses the clean CPU tree /data2/shared/ang.gao/robot_sim_c2d3_array_join_wip at4da.

Independent preflight /root/whole_repairs_preflight_v1 approved the exactfc
repair and hashed16/85/12 scripts. Whole v4 independently assesses source
equivalence and evidence reuse, including current committed cold admission.
The candidate stays frozen throughout that audit. All commands use primary183,
the pinned robot_sim interpreter, disabled pytest plugin autoload/cacheprovider,
and fresh test output directories. No dependency/toolkit changes or alternate
host occurred. Original supervisor deadlines/ACK/guard/reaper remain in charge;
no author outer timeout or remote termination is added.

## Complete85 fault matrix

Script: c4_assets/run_c2d3_runtime_fault_matrix_v5.py
SHA2567551827e828a4119d3d232ec2e0f9327a6e56a661015d972b94c393b996bea11.
Actual result: **85 passed in4495.91s (1:14:55)**, pytest returncode0,
timed_out=false, observer_errors=[], observed_cases85. Raw intentional worker
outcomes:55 exit74,30 exit-9. These are expected fault outcomes validated by
the actual tests, not inferred successful workers.

Remote directory/archive:
/data2/shared/ang.gao/c2d3-fc86182-runtime-fault-matrix-v5[.tar.gz].
Archive18392236bytes, SHA256
6586b2e3e8434d4506bc6dfd56b60f0e30fc53e49bf9f2e599e57cc83b2f9608.
Downloaded c4_assets same name; local SHA matches remote. Original stdout,
stderr, summary,85 external live identity chains and residual record are in
the archive. Runner3342425 and pytest3342430 have exited. Scoped residual check
finds all172 owned PID references absent from OS and GPU process lists.
Other-user/MPS processes are not owned and were not touched.

Current CPU replay:
c4_assets/replay_fc86182_completed_matrix_v1.py, SHA256
4431c436223aba91b52e6bf4ac9311446879afc3bd91ac59c0e793a3645eafa0,
run via pinned remote Python -I -B - fault. All85 original encoded records
roundtrip and validate selection/attachment/same-run identity with4da parser.
79 current selections plus6 explicit prior_validated_cut selections;28 prefix
and57 terminal-shaped worker records, the latter including6 prior cuts.
No earlier cut is described as the last observed worker tail or full execution.
All1552 original evidence file hashes unchanged; no NumPy/Torch/Warp imports.
Output c4_assets/replay_fc86182_fault_v1.json, raw UTF8-BOM SHA256
48fff967537f6c269c9e7eeb6d3c14392b3dfbf6316221ca3411ac2a37dc8925.
CPU replay3497168/parent3497167 subsequently absent OS/GPU; exactCPUtreeclean.

After remote actual exit, residual check, archive and download verification,
the local original SSH transport remained open without new output. Exact
ssh.exe36068, PowerShell parent39696 and its run_c2d3_runtime_fault_matrix_v5.py
command were rechecked before closing only that local transport. Tool46892
then returned local transport exit1. Remote actual pytest0 and raw files are
unchanged; no remote worker/pytest/supervisor received an author signal.

## Complete12 terminal matrix

Script: c4_assets/run_c2d3_terminal_matrix_v2.py, unchanged SHA256
3538f7403dd945798ddc3872f2930d22d659c15a49227cdeb1309e80d8c8eb7f.
Only launched after the85 scoped residual check passed. Runner3495623,
parent3495622,pytest3495628; tool34460. Exact cleanfc tree asserted by runner.
Output /data2/shared/ang.gao/c2d3-fc86182-terminal-matrix-v2.
Actual result: **12 passed in950.95s (0:15:50)**, pytest returncode0,
timed_out=false, observer_errors=[], observed_cases12. Four success workers
exit0, four terminal-sync exceptions exit74 and four deliberate terminal-sync
hangs exit-9 under the original watchdog. All12 live identity chains are present;
all26 owned PID references are absent afterward from scoped OS/GPU queries.

Archive4266879bytes, SHA256
97ab2e1e0a05cdea6815fb7899648b724fc9118049f57eb0b757df2a3f4d8727.
Remote same output path with.tar.gz, downloaded localc4_assets same name/hash.
Same CPU replay script in terminal mode, exact4da: all12 original records
roundtrip and validate selection/attachment/identity,12current selections,
8terminal/4prefix. All252 original evidence files hash unchanged; no backend
imports. Raw UTF8-BOM c4_assets/replay_fc86182_terminal_v1.json SHA256
f328aa81e5c64e772c3a609e10f7402c20ab93212dbce694dcb197949bdf04b6.
Replay3526668/parent3526664 subsequently absent from scoped OS/GPU checks.

After completed remote exit/archive/download verification, local ssh.exe44140
remained open. Revalidated exact PowerShell parent20268 and terminal-runner
command before closing only that local transport. Tool34460 then returned
local transport exit1; remote actual pytest0/raw artifacts remain unchanged.
No remote worker/pytest/supervisor signal was sent by the author.

## Remaining scope

Whole reviewer /root/c2d3_whole_review_v4 independently audited complete85,
including exact expected(selector,lane,case) coverage, raw result/attachment,
deadlines/ACK, nativefcsource provenance,liveancestry/start/command and priorcut
diagnostics, then independently audited the completed12 matrix and issued
**Approve** for whole C2d3 at4da (productionf1), under inspected HEADe5299e3.
The complete report is archived unchanged as runtime-adoption whole-code-review
v4, raw SHA25610d0971656853fa88b86360c319d582b50cdb425a5ebda7a5ccc00fc336979ec.
This evidence-only archive changes no code, requirements, approval scope or gates.

C2d3 is accepted for the exact bounded scope in that report. C4 remains active
and incomplete. Compilation/render/warmup/final graph/seal,
R_setup/R_final and prepared tokens, C2e/f activation, C4c3/c4,C4d/B3, product
GPU preparation,a3b,result/baseline,R2/S/host/shipping/backend selection remain
closed under their existing gates. Future compilation scratch does not alter
current scope or authorize a compiler/native action.
