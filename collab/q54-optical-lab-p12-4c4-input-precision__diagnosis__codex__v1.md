# Source representation and authoritative camera precision diagnosis

Initiative: Q54 / P12.4 / C4
Stage: author CPU diagnosis, not implementation or acceptance
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: intersection-precision native-evidence v1; runtime-integration design-request v3; optics/scene_contract.py; optical_worker/render_parameters.py
Owner Summary: Original-source normals can be retained accurately without making all shading double precision. Independently, rounding authoritative camera rays to float32 changes a triangle hit. A local intersection hotfix or standardized float32 ray algorithm cannot close both problems.

## Unchanged candidate and scope

Native candidate remains812ca29030a45d802addf10ca5811a4354ce0eb9.
These scripts run on CPU outside its tracked files using the existing183
robot_sim interpreter and frozen Go2 assets. They change no source scene,
fixture, camera sequence, CPU oracle, tolerance, ABI or gate. The preceding
goal turn explained status only; this continuation produces new numerical
evidence and narrows the required repair.

The scripts and exact JSON results are retained in both
`C:/Users/AG/新建文件夹/c4_assets` and `/data2/shared/ang.gao` on tileops-dev.

## All-triangle source-normal experiment

`probe_source_representation_v1.py` imports the original static scene, matches
packed triangle IDs to their source mesh faces, and evaluates all376139
retained triangles. All instances are asserted static. The reference computes
world vertices and their normal from original binary64 source geometry and
transforms. Candidate normals are rounded to float32 before comparison.

| Arithmetic representation | Maximum normal absolute error | Triangles above diagnostic1e-5 |
| --- | --- | --- |
| Existing packed float32 local vertices, double edge/transform arithmetic | 0.0037789534950759818 | 49212 |
| Original local edges rounded once to float32, double transform/cross | 0.00003980613605267247 | 21 |
| Original local unit normal rounded once to float32, double transform/normalize | 0.00000006130744889532025 | 0 |

The diagnostic threshold does not create a new acceptance rule. The result
establishes that source-normal retention is a promising compact representation;
it does not prove intersections, shadows, full images or moving geometry.
Replacing vertices with float32 edges alone still leaves21 counterexamples.
No new packed normal array has been implemented or admitted.

Script SHA256:
`569cdb6e8929d54ebe41adcd3166e2457899477efa6fa8b39f57377291b9e7a4`.
Result `c4-source-representation-probe-v1.json` SHA256:
`753283734c27e78929c06336b612c56e261d8ce78b90ac0f5bfc0114d0c7492b`.
Author exec8226 completed exit0.

## Corrected authoritative-camera experiment

The phrase "canonical double camera ray" in the earlier native-evidence v1
needs qualification: that older diagnostic builds a double CPU camera from
already-rounded native float32 scalar arguments. It is not the original
binary64 camera sequence. Preserve that older evidence without treating its
ray or hit as the authoritative-camera result.

`check_authoritative_camera_pixels_v1.py` instead decodes the exact frozen
binary64 camera, ambient/background and shadow-bias values, invokes the
existing `build_pinhole_camera_rays`, and executes the original CPU scene/BVH
and `CpuDirectLightOpticalExecutor`. It checks the eight previously identified
pixels using (a) the exact camera rays and (b) those same origins/directions
rounded once to float32 and promoted losslessly. It does not substitute the
second variant for the oracle or claim full-image acceptance.

At1920x1080 pixel782417, the original camera selects the face with normal
`[-0.009871492917164241, -0.18423474580451124, 0.9828326979023101]`
(source primitive53234). Merely rounding its origin/direction selects the
adjacent face normal
`[-0.008200014979111304, -0.1841965693519502, 0.9828552200570105]`
(source primitive28312). The RGB maximum difference is approximately0.00061033.
Both retain instance6/material5, so ID agreement does not detect the error.
The other seven selected pixel normals agree across these two CPU runs.

Consequently, widening camera scalar arithmetic but storing the intersection
rays as float32 is not sufficient for the current per-pixel CPU requirement.
This is a discontinuity at a geometric boundary, not evidence that ordinary
small ray component error guarantees small color error.

Script SHA256:
`a3e71da94bc61971569301ed9ae794199a88ac4491efc6c228adc5be788647b0`.
Result `c4-authoritative-camera-pixels-v1.json` SHA256:
`08302ac9b42de6a3f8b544dc5c063e717be3a719ce11db028a291ba989be081b`.
Author exec50772 completed exit0. Sequence SHA256 remains
`d8519b3977b6d0a144884166ffa943747a389ee61f574ac5cb84e1ae9920250e`.

A separate two-face CPU intersection matrix confirms this boundary behavior
for original binary64 vertices, the current encoded fixture, source world
vertices rounded to float32, and source world anchor/edges rounded to float32:
the exact camera ray selects53234 in all four; the rounded ray selects28312
in all four. This counterexample requires retaining query precision but does
not establish that all geometry arrays must become double precision.
`probe_boundary_representations_v1.py` SHA256:
`2cd036837a014ff2e0f5326cde447f65c4ad9b14bf5cd11d7c70f8bbf4387f71`.
Result `c4-boundary-representations-v1.json` SHA256:
`8cd3f389bb736840168c9a6795588064dba47f156074505068697217c339f764`.
Author exec63732 completed exit0. This is a two-face diagnosis, not a
whole-scene nearest-hit or image acceptance run.

## Consequence for the next implementation

Treat source geometry, internal query precision and source-normal retention
as one numerical design problem. Do not add a pixel-specific branch or another
downstream cast and assume it restores upstream information. Preserve the
original workload, source assets, camera formula/sequence and CPU acceptance.

The existing scene schema/materializer, fixed fixture and bound native camera
arguments explicitly freeze float32 representations. A coherent repair must
identify and independently review its representation/ABI and derived-identity
changes before implementation. This evidence alone authorizes no such change
and does not establish that every array or all shading must become float64.
Both lanes must still own independent native allocations derived from the
same logical source with equal preparation/timing boundaries.

Native full correctness, performance, final graph/arena integration, lifecycle
matrices, whole-C4 refactor and whole-C4 review remain incomplete.
