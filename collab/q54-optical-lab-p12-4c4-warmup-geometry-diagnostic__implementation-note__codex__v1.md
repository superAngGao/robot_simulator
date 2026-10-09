# Warmup geometry diagnostic for the observed 1920 RGB discrepancy

Initiative: Q54 / P12.4 / C4 render integration
Stage: bounded test-driver implementation awaiting independent review
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: runtime-integration design-request v3 sections 5 and 9; warmup-readback-checkpoint code-review v1; tests/support/native_runtime_process.py; tests/support/cuda_artifact_runtime_acceptance.py; tests/unit/optics/test_native_warmup_support.py
Owner Summary: Actual 1920x1080 RGB has six substantial cross-backend discrepancies. Read existing ray/first-hit outputs through the same guarded diagnostic copy path to locate the defect before changing numerical kernels or tolerances. This adds no production execution capability.

## Observed problem and exact scope

Base: 9067e114ed609928904bf3cd95b871554bb15d26, production/support source
7fe32a58b9e4bee1b6d354f42ceed942f6e9cb92. Candidate is developed separately
in c4_render_execution_next; the primary review checkout and ongoing native
warmup campaign stay frozen. No production source, catalog, kernel, numerical
parameter, supervision deadline, product entry or gate changes.

The successful native Warp/CUDA primary 1920 runs are directories
/data2/shared/ang.gao/c4-render-warmup-author-v5 and v6 on tileops-dev.
Their frame10 full RGB maximum absolute difference is 1.2056253775954247;
six pixels exceed 1e-4. That number is an investigation filter, not a new
acceptance tolerance. Indices are 782417,901306,976396,985873,1026413,1143492.
An original CPU executor diagnostic of those pixels and their neighbours
(54 rays, selected after seeing the discrepancy) finds material differences
against CUDA at four of these indices and Warp at two. It does not identify
the cause, establish unbiased full-frame CPU correctness or authorize an
epsilon/algorithm change. The original 960 pair maximum difference is only
2.384185791015625e-7; that earlier result did not cover 1920 correctness.

Author diagnostics/scripts remain under c4_assets:
warmup_rgb_1920_cpu_outliers_comparison_v1.json,
warmup_cpu_outliers_1920_v1/cpu_sample_diagnostic.npz,
check_warmup_rgb_cpu_outliers_1920.py. The remote CPU54 command used the
existing robot_sim interpreter, frozen source and frozen Go2 asset; it exited0.
No additional GPU rendering or altered scene was used for that CPU check.

## Change

The existing fixed sacrificial driver gains runtime_render_warmup_geometry,
an unprofiled ten-frame case. After the same physical completion, its existing
diagnostic readback guard copies the original origins, directions, hit_mask,
range_m, position_world, normal_world, numeric_instance_id and material_index.
One local readback body now serves RGB and geometry. Every host result is
retained before conversion/serialization and before the next copy. Files and
metadata identify the same last frame and guard; final custody checkpoint
still follows successful DISARMED. Partial failure does not emit a complete
geometry manifest or continue to later channels.

The original acceptance runner additionally validates this selector's exact
eight-channel set, native shapes/dtypes/byte counts/hashes, matching RGB
frame/guard, and ordered nine real D2H declarations followed by DISARMED.
Old RGB-only cases keep their exact one-copy validation. Neither path claims
that well-formed raw geometry proves numerical correctness.

This is temporary fixed-driver diagnostic support, with the same later cleanup
obligation as the existing comparator/test drivers. It does not create another
renderer, input uploader, schema authority, arena or synchronization path.

## Validation and remaining work

Run with C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider
tests/unit/optics/test_native_warmup_support.py. Tests exercise both lanes'
actual source-session/guarded lifecycle and original warmup transactions with
native doubles: preserved RGB cases, geometry bytes, altered channel bytes,
wrong frame guard and a mid-readback copy failure that stops later work and
quarantines without another synchronization. These are support/code checks,
not real geometry readbacks. Final author run: 10 passed in 23.41s, exit0
(tool session15693 terminal); git diff --check passed. No author test handle
remains running; short-lived local pytest PIDs were not separately sampled.

After independent code review, run the fixed 1920 case in both lanes and
compare the six discrepant pixels' original camera/geometry/shading inputs to
the CPU diagnostic. Existing positive warmup campaign, its native review,
required failure/hang matrices, full CPU/all-channel/sentinel validation,
systematic whole-C4 refactor, final graph/arena/barrier, R1 and all remaining
C4 gates remain separate unfinished obligations. No speed claim is made.
