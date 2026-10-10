Initiative: q54-optical-lab-p12-4c4-systematic-refactor
Stage: R1 dynamic preparation and final graph composition
Author: Codex (implementation)
Version: 8
Date: 2026-10-10
Status: draft
Related Files: runtime-integration design-request v3; systematic-refactor gap-review v1; code-review v5; implementation-note v7
Owner Summary: Both backends now compose dynamic scene update, acceleration and rendering through the existing setup and arena machinery. Native correctness/failure evidence is complete for this preparation checkpoint and awaits independent review. This does not complete C2f or C4 and provides no speed comparison.

## Candidate and approved requirements

The code-review base is approved M1 revision
c3472f22f1d52bfb6c853bf95cef4040cf666316. Production last changed at
626f6320eda3bbfb527be49390c696012873e33d; later commits change tests and evidence.
The native candidate is 98315dac353490161742c44062f5794d31d4720a, branch
implementation/c4-dynamic-graph, local c4_dynamic_graph and primary-host
/data2/shared/ang.gao/robot_sim_dynamic_graph_98315da (detached, clean).
Source catalog v17 has 90 sources, SHA256
b9db06834328433d24ee67b697acb0a2d68d2c736e12c2c0a2eb2f9316a39dca.

Requirements come from
[q54-optical-lab-p12-4c4c2-runtime-integration__design-request__codex__v3.md](q54-optical-lab-p12-4c4c2-runtime-integration__design-request__codex__v3.md)
and the accepted R0 -> R1 -> C2f delivery order in
[q54-optical-lab-p12-4c4-systematic-refactor__gap-review__codex__v1.md](q54-optical-lab-p12-4c4-systematic-refactor__gap-review__codex__v1.md).
The earlier author chronology and original failed native runs remain in
[implementation-note v7](q54-optical-lab-p12-4c4-systematic-refactor__implementation-note__codex__v7.md).
Neither that note nor this one is independent approval.

## Final implementation

R1 admits the two frozen mandatory dynamic rows. R0 remains supported; later
boundaries stay rejected. Warp uses actual views of the prepared replay, the
existing scene-update kernels and immutable leaf4/sort topology refit. CUDA
uses its existing scene update and LBVH rebuild. Mutable world, acceleration
bounds and rendering outputs belong to the physical slot. Immutable topology
and prepared source allocations remain session-owned.

The original topology builder reuses guarded readback/upload helpers for its
initial preparation; there is no CPU rebuild in a transaction. Scene and refit
recording use the same returned-owner custody helper as render recording.
The common frame binder and five-phase executor serve both setup warmups and
final plans. Warmup and final execution retain distinct authorities and factory
identities. Actual replay roots, offsets and selected frame are checked before
final authority. Existing C3 validation checks the ordered storage/dataflow.

Ten actual setup terminal completions produce the existing final-readiness
origin. New final factories bind the same warmed storage and numerical variants,
then the existing arena constructs dormant plans and performs one guarded final
barrier. Both arenas reuse NativeOpticalArena for FIFO, generations and holds;
no second slot state machine, numerical implementation or general backend bridge
was added. The product worker still rejects GPU PREPARE/sample dispatch.

## Verification and limits

Local CPU checks of recording failures, replay root/frame rejection and C3
binding passed 32 tests. Existing Warp arena/final graph and CUDA arena contract
and resolution regressions passed 120 tests. Previous real Warp scene/refit
regressions passed nine tests on GPU1. These runs and their exact limits are in
v7; they are not substitutes for native integrated execution or full-repo tests.

Primary host: tileops-dev, Python3.12.13 robot_sim environment, Warp1.12.0,
Torch2.9.1+cu128; physicalGPU1 H200
GPU-3a3112ff-837c-3b87-38d1-cb203d656589, driver595.71.05. All GPUs visible,
ambient logical0, target1. CUDA restores raw device1 as observed; no raw0 claim.
Cases run concurrently for correctness only; elapsed campaign times are not
performance measurements.

Original matrix at f15aa62b11d210eeed9dd62161b8a0bc17479139:
/data2/shared/ang.gao/c4-dynamic-matrix-f15aa62b11. Sixteen of twenty driver
cases passed: four primary row/lane cases and twelve faults. Four diagnostic
cases failed because the test used getattr for counters stored in tuples.
Their original failures are retained. Commit98315da uses the existing native
arrays/tensor_items inventories; no backend or acceptance rule changed.

The twelve passing faults cover each lane after actual scene, acceleration,
render and terminal-wait return, plus final-barrier return and storage-owner
adoption. Warmup failures retain the graph under the original failed guard,
exit74, and perform no emergency synchronization. The final-barrier fault never
publishes quiescence. All41 owned first-round parent/worker/supervisor PIDs were
absent, including from GPU process records, before the second GPU round began.

Corrected matrix at98315da:
/data2/shared/ang.gao/c4-dynamic-matrix-98315dac35. All eight row/lane/pass
combinations passed: both frozen R1 rows, Warp/CUDA, primary and diagnostic.
Every worker exits0 after ten warmups, three final C3 plans, one final barrier
and acquire/cancel; zero final sample submissions. Diagnostics have five pairs;
primary has none. Each corrected case passes the driver's raw custody/graph validators.
CPU numerical comparison passes all eight corrected cases (and the four
original primary cases). Hits/IDs/materials match exactly; maximum observed
absolute floating error across checked channels is7.298419912871168e-08.
Both shadow diagnostics count6393 directional/unoccluded rays; no-shadow
counters are zero. Primary overflows are zero; observed stack depth is1,
within the existing native bound32. This frozen one-triangle scene exercises
no shadow occlusion/internal BVH nodes; do not infer those cases are covered.

Full-frame CPU comparison uses the existing independent CPU optical executor,
exact decoded native camera scalars, frozen registry and replay frame9. Existing
C1 tolerances remain unchanged: rays1e-6 and output floats1e-5 absolute/relative,
exact hit mask and IDs/materials, plus original counter invariants. This checks
all614400 rays of setup warmup frame10. It does not claim all replay frames,
final sample execution, real physics publication or shipping correctness.

Archive on the primary host (and copied to local c4_assets):
c4-dynamic-matrix-evidence-v1.tar.gz,245895053 bytes,1195 members, SHA256
7fa37ad6cad63cc79e9a1f66d74457003db12830e2bf73307351130451073849.
It contains both original matrix directories, raw channels, worker/supervisor
traces, original driver outcomes, CPU reports, scripts and a per-file digest
manifest. dynamic_matrix_process_closure_v1.json records all58 distinct owned
matrix parent/worker/supervisor PIDs absent and no owned GPU processes. None
of the four original failed driver cases is relabeled as passing.

Reproduce using the existing fixed driver from a clean exact revision:
`python -B /data2/shared/ang.gao/run_dynamic_matrix_v1.py REVISION positive-only`
(omit positive-only for all20 cases; it creates a new revision-named directory).
Read-only numeric replay:
`python -B /data2/shared/ang.gao/check_dynamic_matrix_cpu_v1.py ROOT MATRIX_DIR 8`
(use4 for the original matrix). The scripts reside in the archive and beside
the remote evidence. They are author scratch tools, not product entrypoints.

## Remaining scope and cleanup

Fresh independent review of this complete R1 candidate is required. C2f must
still join the worker tokens, actual final submission, guarded resolution,
release/reclaim and post-reclaim close/ACK/observed-exit proof, exercising all six
R0 and two R1 rows with the parent's full correctness/failure matrix. Whole-C4
refactor review, C4c3/c4 and C4d/B3 remain outstanding. No performance result,
product preparation, a3b, result/baseline, host payload or backend choice opens.

The sacrificial driver and comparison-only scratch programs remain transitional:
P12.4e removes the losing lane and temporary comparison machinery after its
selection and equivalent retained regression coverage. Shared production
recording, transfer, frame-binding and arena control are retained building
blocks. No new permanent test framework is introduced by this packet.
