Initiative: q54-optical-lab-p12-4c4-systematic-refactor
Stage: R1 preparation and final graph, F1 repair
Author: Codex (implementation)
Version: 9
Date: 2026-10-10
Status: in_review
Related Files: runtime-integration design-request v3; systematic-refactor gap-review v1; code-review v6; implementation-note v8
Owner Summary: The recorded Warp scene/refit commands now use the existing renderer's native argument inspection. The original wrong-frame and wrong-output counterexamples reject before final admission. Both lanes' valid preparation and numerical checks pass. Fresh independent review is required; C2f and whole C4 remain incomplete.

## Candidate and repair

Review the complete R1 scope from approved M1 base
c3472f22f1d52bfb6c853bf95cef4040cf666316, including the F1 repair at
13d17dd293f9c79028260b856e99e39d5f344da5. The primary clean native checkout is
/data2/shared/ang.gao/robot_sim_dynamic_graph_13d17dd. The local branch is
implementation/c4-dynamic-graph in c4_dynamic_graph. This note and the associated
PROGRESS update are evidence-only descendants of that tested revision.

The prior fresh review (code-review v6, session
01a1246a-22f1-7702-b7be-618ec7caa515) requested changes for one blocking P2:
mutating actual scene/refit command operands did not change the Python graph,
so final admission accepted inconsistent native work. Its full original report
is archived unchanged; the original scratch repro remains available.

The existing recording helper now retains descriptors from the exact input
tuple used to record each scene/refit command, after adopting the returned
native owner. Both bundles snapshot actual packed arguments, kernel and launch
dimensions using the unchanged _render_launch_snapshot/_WarpRenderDescriptor
implementation already used by render commands. Scene and refit validation
compare those facts with the frozen bundle, including during untimed final
validation. Acquire keeps its existing fixed-cost checks. There is no new ABI
inspector, kernel, slot state machine or generic backend adapter.

AST comparison with ba394c6 confirms all three scene and two refit kernel
bodies are unchanged. Source catalog v17 still has 90 entries, SHA256
e532beb68548a1dbc515cf7af59028053ac9292edbd70401b90db4c138363097.
The existing source admission policy and renderer inspector are unchanged.

## Observed verification

Local Python3.14: 278 tests passed in306.10s across dynamic command preparation,
dynamic arena binding, Warp native command contract/arena, final graph, render
warmup, upload core, CUDA arena contract and resolution. Earlier focused tests
passed45. These CPU tests include doubles; they do not replace native proof.

Primary host tileops-dev uses Python3.12.13 robot_sim, Warp1.12.0 and
Torch2.9.1+cu128. Captured matrix target is physical GPU1 H200,
GPU-3a3112ff-837c-3b87-38d1-cb203d656589, all GPUs visible, ambient logical0.
No performance measurement or result record is produced by these runs.

New matrix /data2/shared/ang.gao/c4-dynamic-matrix-13d17dd293:
14/14 driver cases passed. Six separate Warp mutations cover scene replay
pointer, refit output pointer, scene scalar/env index, scene dispatch, refit
dispatch and scene kernel. Each mutates the actual native command after graph
construction, before normal final admission; the test does not invoke a
validator itself. Production rejects with the expected exact error, worker74,
zero affected final submissions and no final admission/quiescence. The healthy
idle failure path performs one emergency target sync, releases the graph and
exits under the same terminal guard. This differs legitimately from the
previous armed-warmup failures, which quarantine without another sync.

Eight positive cases cover both R1 rows, both lanes and primary/diagnostic
passes. They complete ten setup warmups, three dormant final plans, the final
barrier and acquire/cancel. Independent CPU executor comparison of all channels
and counter invariants passes all eight using the unchanged checker and C1
tolerances. As before, this checks replay frame9/setup warmup10, all614400 rays;
it does not prove final sample execution, all replay frames, occlusion or
internal-node traversal in this one-triangle canonical fixture.

The legacy real Warp scene/refit suite was freshly run at13d17dd:
`CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -rs -p no:cacheprovider tests/gpu/test_optical_warp_native_commands.py --basetemp=/data2/shared/ang.gao/c4-r1-repair-13d17dd-native-tests`
Result: 9 passed in3.07s, command exit0. This uses physical1/logical0 and is
not dual-GPU ambient0/target1 proof. The pytest PID was not individually sampled;
the synchronous SSH command completed. Matrix parent/worker/supervisor PIDs are
recorded separately: all29 are absent, all14 supervisor records show observed
exit with no deadline expiry or kill, and none appears among GPU processes.

Archive /data2/shared/ang.gao/c4-dynamic-matrix-evidence-v2.tar.gz:
184751064 bytes,717 members, SHA256
94e1af07deba62234ba48295bd4113abf2611d5e0ec213955ddafb36d4cd3058.
It contains all14 raw case directories, CPU numeric report, runner/checker,
per-file digest manifest and dynamic_matrix_process_closure_v2.json. A local
copy is retained in c4_assets. Previous archives and original failed runs are
unchanged. Reproduce a fresh campaign with run_dynamic_matrix_v1.py REVISION
command-repair; never overwrite the existing revision-named evidence folder.

## Remaining work

Fresh independent review must resolve F1 against both original mutations and
the repaired coherent R1 candidate; these author results are not approval.
The accepted next step is C2f's complete worker prepare/execute/guarded-resolve/
release/reclaim/close path, with all six R0 and two R1 rows and the design's
failure and numerical matrix. Whole-C4 systematic refactor and C4c3/c4/C4d/B3
remain required. Product activation, formal speed comparison, backend choice,
a3b/result/baseline and host/shipping remain closed. The prior v8 limitations
and temporary-comparator cleanup obligations still apply.
