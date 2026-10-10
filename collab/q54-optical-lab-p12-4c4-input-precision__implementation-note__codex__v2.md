# CUDA renderer and scalar binding checkpoint

Initiative: Q54 / P12.4 / C4
Stage: implementation checkpoint; amendment incomplete
Author: codex
Version: v2
Date: 2026-10-10
Status: draft
Related Files: input-precision design-request/design-review v4; implementation-note v1
Owner Summary: CUDA queries and camera generation now use the reviewed double representation. Both bindings use the shared mixed-precision scalar codec. Host regression checks and an actual CUDA renderer compilation passed. Frozen content/source identities and complete native validation remain unfinished; C4 and subsequent gates remain closed.

## Candidate and changes

Base `373131daaa13b3df86327d74a8f7bd4f7009e561`; renderer/binding code
`05f04d0c184466e5913f0c37537588118791a0ca` in `c4_input_precision` and
`/data2/shared/ang.gao/robot_sim_input_precision_wip`. Both were clean at the
successful renderer compilation. The subsequent delta recorded with this note
updates the existing validation readback scripts, as described below.

- CUDA camera/query scalars, ray arrays, world geometry/planes, hit distances
  and traversal distances become double; output/shading/bounds remain float.
  Camera generation rotates in world space before double normalization.
- CUDA consumes the common CPU numerical constants, rejects degenerate world
  triangles, compares unrounded distances, and removes the old float box-entry
  quantization repair. Primary and shadow queries use the same triangle helper.
  Existing float shadow construction and shading thresholds remain unchanged.
- Actual Torch checks/data pointers and compatibility ray uploads migrate with
  the kernel. Recipe `robot_sim_cuda_direct_light_v10` uses `--fmad=false`.
  Current recipe consumers are updated; captured source pins are still pending.
- Bound-render-parameters v2 splits 17 double camera/query scalars and seven
  float shading scalars. `render_parameters` owns the codec; preparation and
  Warp/CUDA argument evidence use it. Old/mixed/noncanonical encodings reject.
- Resolved asymmetric camera v2 preserves every numerical field and changes
  only schema/native encoding. New digest:
  `3cee982411e6a7c02b81dda2929ead53ea5895ed00513b8e67d3d91dcf1d46ef`.
  The old artifact is retained in `optical_worker/data/resolved_cameras/v1/`.
- Existing warmup diagnostic readback adds intensity and requires eight-byte
  double ray arrays. Sacrificial Warp resident-ray upload and both inspection
  buffers now preserve double rays. No new execution/control-plane route.

## Verification and limitations

Local `C:/Python314/python.exe -B -m pytest`, exit 0:

- `test_native_intersection_precision.py`, `test_native_scalar_precision.py`,
  `test_scene_contract.py`, `test_source_geometry_precision.py`,
  `test_device_scene.py`, `test_cuda_scene_native.py`,
  `test_cuda_extension_recipes.py`, `test_registry_builder.py`: **167 passed**.
  These run actual CUDA C++ camera/query bodies through a host compiler and
  current Warp Python bodies, including old edge/traversal cases, both visit
  orders, double distances that collapse in float32, exact area threshold,
  physical cutoff and binding changes below one float32 ULP. This is not GPU
  execution; earlier actual Warp CPU-kernel evidence remains scoped as in v1.
- `test_resolved_camera_contract.py -k 'not actual_canonical_cpu_scene_observability'`:
  **66 passed, four deselected**. Camera source numbers and old-format rejection
  are checked. The four canonical scene cases still require regenerated inputs.

`test_render_parameters.py -q --maxfail=1` and
`test_native_warmup_support.py -q --maxfail=1` each fail in initial fixture setup:
`materialized scene does not match the frozen workload digest`. This is the
known incomplete manifest migration, not a passed binding/warmup integration.
The new readback delta is AST-checked only pending that integration. No digest
check was removed or bypassed. Changed Python ASTs and `git diff --check` pass.

Actual remote CUDA renderer build through the existing isolated build service:

- Directory `/data2/shared/ang.gao/c4-double-cuda-renderer-build-v2`.
- Unit `robot-sim-cuda-71c749c7cc2a442c96ad39daeb4861b8-2.service`;
  invocation `c123fc4c656943bc894e7030b2b90d64`.
- Exit status 0, service success, waiter exit 0, cgroup absent; launcher PID
  3829881 absent afterwards. CUDA devices hidden during compilation.
- Artifact `robot_sim_cuda_direct_light_v10.cpython-312-x86_64-linux-gnu.so`,
  526224 bytes, SHA256
  `6f3b737275df3b5edaeb87d74c4d0ea9531482d69898996a899d7ba1f5702a45`.
- No import, publication, GPU launch or performance measurement. This is a
  diagnostic binary, not an admitted deployment.

An initial synchronization command requested a branch not present in the
bundle (the bundle exports `HEAD`). Its build directory `...-build-v1` therefore
compiled the old renderer at 373131d. It exited successfully and was retained
but **is excluded from new-code evidence**. After that process/group ended,
the correct bundle HEAD was fetched, fast-forwarded and verified clean at
05f04d0 before the fresh v2 build. Initial PIDs 3827730/3827733 are absent.

Both build directories, exact launcher and source bundle are archived locally
and remotely as `c4-cuda-renderer-precision-evidence-v1.tar.gz`, SHA256
`28dd4af27175f44dd433aa6e42776d1d554313ac540ce6ba275d00db0f857a35`.
The downloaded archive hash matches the remote archive. All owned build and
pytest sessions have terminated; no GPU process was launched in this checkpoint.

## Required continuation

Regenerate the canonical scene/fixture/sequence/native manifests from original
assets with explicit before/after numerical custody; add native f8 dtype metadata
and native-input v4, update gate/result v15, row semantics v10, C3 binding v8 and
the captured source catalog. Preserve replay samples and original camera/light
numbers. `cases.py` still contains historical float32 ray/scene declarations;
the combined migration must reconcile their meaning with the reviewed new native
representation without silently changing source cases or numerical workloads.

Run actual binding/readback integration, real native R0/R1 and all required
original CPU channels, then freeze the complete amendment for a fresh independent
code/native review. No partial checkpoint substitutes for that review or opens a
speed/result/product gate. Continue the full C4 graph/arena/barrier, lifecycle,
C4c3/c4, C4d/B3 and systematic refactor/whole-review obligations listed in v1.
