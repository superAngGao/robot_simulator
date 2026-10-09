# Real Go2 intersection repair evidence and remaining numerical gap

Initiative: Q54 / P12.4 / C4 numerical correctness
Stage: author native evidence, pending independent review
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: intersection-precision code-review v3/release-review v4; parent runtime-integration v3; original geometry native review
Owner Summary: All four real Go2 runs complete normally. Both resolutions now agree on every hit/instance/material; large1920 discrepancies are removed. One cross-lane normal/RGB pixel and original-CPU normal errors remain. Do not declare optical correctness or C4 complete.

## Exact execution and artifacts

Native source: clean detached812ca29030a45d802addf10ca5811a4354ce0eb9 at
tileops-dev:/data2/shared/ang.gao/robot_sim_intersection_precision_wip.
Numerical code is identical to independently approveda42dbe0. Fixed CUDA
artifacts retain that build revision; existing verifier accepts the identical
build source under812ca29. Main integrates both reviewed deltas and reports;
no later product/result/performance gate is opened.

Existing run_case and runtime_render_warmup_geometry selector run sequentially
via /data2/shared/ang.gao/run_intersection_geometry_native_v2.py, using fixed
Python3.12.13 -I -S -B and the existing target1/ambient0 supervision. Cases:

| Output suffix | Lane | Resolution | Worker / supervisor |
| --- | --- | --- | --- |
| c4-intersection-geometry-author-v2 | Warp | 1920x1080 | 3639235 /3639232 |
| v3 | CUDA | 1920x1080 | 3643200 /3643199 |
| v4 | Warp | 960x640 | 3650069 /3650063 |
| v5 | CUDA | 960x640 | 3654592 /3654590 |

All workers exit0, original strict lifecycle/storage/graph/warmup validators
pass, and parent3639064 plus all eight worker/supervisor PIDs are absent.
Normal guards complete25 per Warp case and21 per CUDA case; terminal disposal
uses the original single emergency sync and release. No failed operation is
restarted or credited as completed. Local tool81205 is terminal exit0.

Archive c4-intersection-geometry-native-evidence-v2.tar.gz, SHA256
02ed978f890352fc5a52c88f3915db6605295ffc12b6c347de40045ccf987698,
contains four raw directories, scripts/profiles/batch and both actual Warp
private caches'18 source/binary/metadata files. Each cache artifact size/hash
was checked against its real module binding. This includes actual compilation
of the new Warp double arithmetic and wp.cast usage, not a host substitute.
Archive companion JSON records strict replays and PID absence.

Build archive and source-pin failure are separately retained as described by
release-followup v4. The first failed v1 output was not overwritten. The only
observed preexisting GPU process remains the ambient0 MPS service; it was not
touched. No performance conclusion follows from supervisor wall time.

## Full cross-lane arrays and CPU diagnoses

compare_intersection_geometry_v1.py validates all original readback sizes and
digests, compares every pixel/channel, and accounts for expected miss sentinels.
The1e-4 RGB list is a diagnostic filter, never an acceptance threshold.

| Scope | 1920x1080 | 960x640 |
| --- | --- | --- |
| Hit mask / instance / material differences | 0 /0 /0 | 0 /0 /0 |
| Normal differences | one pixel782417 | 0 |
| Maximum RGB absolute difference | 0.0006107091903686523 | 2.384185791015625e-7 |
| Maximum normal difference | 0.0016724737361073494 | 0 |
| Expected misses | 132022 per lane | 80946 per lane |
| Nonfinite values on hits | 0 in every channel | 0 in every channel |

The previous1920 maximum RGB discrepancy was1.2056253775954247. All original
six RGB pixels now agree on instance/material, and normal-only1116499/1116503
now agree across lanes. Tiny ray/range/position differences remain; full
counts/maxima are retained in comparison JSON, not summarized as exact parity.
Warp's complete origin/direction byte arrays are identical to its old run.

Existing CPU executor diagnostics ran freshly on all eight original problem
pixels, using each lane's actual rays, both exact lossless binary32-to-double
directions and ordinary normalized CPU directions. Both CPU runs select the
same IDs as the repaired corresponding GPU. However, maximum CPU RGB errors
are4.112343581408329e-5 and4.113131022057126e-5. These are not full-image CPU
checks, and do not pass the parent's fixed absolute/relative1e-5 everywhere.

At pixel782417, CPU intersection on the exact frozen encoded R0 fixture selects
primitive53234 for the actual Warp ray and28312 for the actual CUDA ray. It
reproduces the GPU normal difference. CPU shading on each actual ray likewise
reproduces the cross-lane color difference. The canonical double camera ray
and that ray rounded to float32 both hit28312 in the diagnostic. Current Warp
raygen normalizes in camera coordinates with reciprocal sqrt division; CUDA
uses rsqrtf; the CPU camera rotates then normalizes. No camera values were
changed, and no new ray algorithm has been implemented or approved here.

At pixel901306 /primitive346916, both GPU lanes return exactly the stored
fixture normal [0.9007425904273987,0.17109498381614685,-0.3992357552051544].
Original CPU normal is
[0.900718743935694,0.17145981755928835,-0.39913315483111816], maximum difference
0.00036483374314150163. Recomputing from the existing encoded edges in double
still differs0.00036476589415207483. Computing transformed edges in double from
the existing packed local vertices reduces this to7.027663032155296e-5, still
outside the fixed normal tolerance. Thus another arithmetic patch in the
intersection helper cannot restore precision already lost upstream.

The actual materializer _derive_fixed_fixture computes world vertices, their
differences/cross/normalization in float32; PackedOpticalSceneHost already packs
local vertices/transforms as float32. The immutable fixture identity is still
8354993fb682d5a13a2dff62d1d7521afd7d3d46d20b876e15732b4c38415db0.
Diagnostics do not modify it, the source registry, oracle or tolerances.

CPU diagnostic archive c4-intersection-cpu-diagnostics-v1.tar.gz, SHA256
d61c2382ffb15cd065499a7d560e5e65b6163096c172cba557ef7747c559dda5,
contains original CPU results/NPZ, full comparison summaries/extracts and the
exact analysis scripts. It records the additional arithmetic exploration as
diagnosis only, not an admitted replacement materializer or correctness proof.
CPU tool21782/73791/77993/78470/72668 all terminate exit0. No CPU timing is
reported as rendering performance.

## Required continuation

Independently review the raw native evidence and these remaining numerical
facts against the approved logical-source/oracle/ABI/fixture requirements.
Determine the minimal coherent input-precision repair and which frozen design
identities it would require revising before implementation. Do not relax
tolerances, silently replace the oracle, refreeze a different workload to
manufacture a pass, or add pixel-specific renderer branches. No such change
is authorized by this evidence note alone.

Then continue complete native correctness/profile/shadow/R0/R1 validation and
the already approved final-graph/arena/barrier integration, C2f fault/token
matrices, whole-C4 systematic refactor and fresh whole review. A successful
build, these four positive runs or eight CPU pixels do not complete C4.
