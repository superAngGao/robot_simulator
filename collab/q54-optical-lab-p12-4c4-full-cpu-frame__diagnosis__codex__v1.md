# Full CPU frames expose shared native numerical differences

Initiative: Q54 / P12.4 / C4 numerical correctness
Stage: author diagnosis and evidence preservation
Author: codex
Version: v1
Date: 2026-10-10
Status: draft
Related Files: input-precision design-request/review v2; intersection-precision native-gap-review v1; optics/execution.py; optics/scene_contract.py
Owner Summary: Both native lanes share errors against the original CPU renderer. The source-normal/ray amendment alone cannot close them: native source filtering and nearest-hit rules differ from CPU. Consolidate the numerical repair around these actual discrepancies before adding further implementation. This is author evidence, not independent approval or performance evidence.

## Revisions and scope

Original CPU and original native evidence use the unchanged, clean source
812ca29030a45d802addf10ca5811a4354ce0eb9. Remote source is
`/data2/shared/ang.gao/robot_sim_intersection_precision_wip`, rechecked clean.
The approved representation design is v2 at
db062cd603464b9b3088739a2aca68acb2a44769; its complete independent approval
is archived at02e5c1c. That approval permits the specified implementation,
not numerical acceptance. Its constraints preserve native filtering and tie
rules, which the new evidence below shows cannot remain the complete remedy.

The separate `implementation/c4-input-precision` worktree retains unfinished
source-normal packing/materialization and seven passing local checks. It is
not integrated, code-approved, native-validated or an answer to these new
failures. No native policy, oracle, tolerance or gate changes in this record.

## Original full-frame evidence

`run_original_cpu_frame_v2.py` invokes the original
`CpuDirectLightOpticalExecutor` with original binary64 scene, frozen camera,
lighting and shadows, for frame10 at both frozen Go2 resolutions. It saves
all ten channels in contiguous512-pixel chunks. The subset direction arrays
are restored from the original camera rays to avoid another normalization.
Completion records and per-chunk hashes cover2073600 and614400 pixels.
Chunking and the comparison scripts still require independent review.

Archive, on both local `c4_assets/` and remote `/data2/shared/ang.gao/`:
`c4-original-cpu-frames-v1.tar.gz`,184911247 bytes,5263 regular files,
SHA256 `f6698bee3e11b7e0a2bb24b7abc28a6e74531b88b369bd10ae103051a6e92936`.
Local download completed normally and its SHA256 matches the remote record.
It contains both CPU frames, comparisons, exact run/comparison scripts and
the proposed-camera arithmetic CPU probe. It does not contain the separate
geometry isolation below. Host elapsed times are not renderer timings.

Script SHA256 values:

- `run_original_cpu_frame_v2.py`:
  `aed02bfb27e65209a34c5d3fe604cc815e718a7a7e51583c2278f45438bd8b82`.
- `compare_original_cpu_frame_v1.py`:
  `e8c31fbb237239a634320ac00302ab932055693f8bd9e8dadcca982875d2e2b7`.

Failed pixels against the CPU frame (counts can overlap):

| Resolution | Lane | Instance | Material | Range / position | Normal | RGB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
|1920x1080|Warp|36|1|3 / 3|2375|507|
|1920x1080|CUDA|36|1|3 / 3|2375|507|
|960x640|Warp|13|2|4 / 4|576|144|
|960x640|CUDA|13|2|4 / 4|576|143|

Hit masks match; original and native ray values pass the existing ray
tolerance. This does not make ray rounding harmless at sensitive boundaries.
Native readback did not include intensity, so intensity is explicitly
uncompared. The comparison checks saved bytes, shapes and coverage; its
equal-NaN comparison is not separate proof of valid miss sentinels. Other
frames, profiles and the remaining correctness matrix are not covered here.

## Geometry isolation and directly observed policy mismatch

`probe_full_frame_geometry_failures_v2.py`, SHA256
`19e8b3a6fdc19229f0221948f94cf979ee69a75d8561a4b05bbb8d36f449c153`,
and `c4-full-frame-geometry-failures-1920-v2.json`, SHA256
`1391ac726bdd89124192b139323d30ec35e2a1578f80eaa3f9c074c56c3a918a`,
are preserved separately in local `c4_assets/` and remote shared scratch.
The v1 attempt failed on its assumption that CPU/native primitive lists were
identical; it produced no successful result. The v2 process exited normally.

The CPU snapshot contains396778 triangles; the native packer contains376139.
20639 CPU source faces are absent from the native materialization. Actual
source checks use1e-12 in CPU scene/execution and1e-8 in native packing and
triangle determinant rejection. Native nearest-hit epsilon is1e-5, versus
the CPU executor's1e-9; native distances also round to float32 before ordering.

The isolation checks all39 pixels comprising the1920 instance/range failures.
It maps source keys explicitly and refits CPU bounds over encoded native
geometry, making omitted faces degenerate. This altered snapshot is a
diagnostic, never a replacement oracle or a native BVH execution.

- Original geometry with original rays reproduces saved CPU IDs/ranges exactly.
  Original geometry with actual Warp rays has no ID/range failures in this set.
- Encoded geometry under CPU traversal has no ID failures with either ray set,
  but fails ranges at999527 and1016744. Their original nearest source faces,
  keys21474879918 and21474891830, are absent from the native packer. Their next
  retained hits are about0.104264m and0.035143m farther respectively.
- All36 instance-error pixels have an immediately following hit with a lower
  source key and different instance, between1e-9 and1e-5 farther away. For
  example891790 has instance6 at1.6056378750401445m and lower-key instance5
  at1.6056450124999637m. The differing tie thresholds support a concrete
  explanation; this is not yet a replay proving every native traversal choice.
- Pixel970674 still needs diagnosis: its nearest face is retained and the
  encoded CPU traversal finds it. Determinant rejection/native traversal
  remain possibilities, not established causes in this record.

These observations rule out treating only source normals and binary64 rays
as a complete numerical repair. A bounded amendment must explicitly resolve
the filtering/predicate/ordering conflict with the original CPU rules, and
verify sufficient geometry precision for the previously omitted faces. Do not
silently change those policies under the narrower v2 approval. Preserve the
original oracle, assets, workload, tolerances and historical artifacts.

Full numerical correctness, fair speed evidence, final graph/lifecycle work,
the owner's systematic whole-C4 refactor and independent whole review remain
outstanding. C4 and all later product/result/selection gates remain incomplete.
