# Q54 Optical Lab P12 Open Questions

Owner: Codex
Date: 2026-07-21
Status: Open

## Context

P11 is now usable as the public Optical Lab workflow surface:

- physics body-triangle preset runs through `run_optical_lab_preset(...)`;
- Go2 static preset runs through the same public workflow;
- `go2_backend.py` has been deleted;
- P11 examples exist for physics video/debug, physics observation, and Go2
  static video/debug.

P12 is expected to focus on rendering backend support as an incremental layer
under the now-stable P11 workflow/API surface.

## Open Questions

1. **When should examples get a usability pass?**

   Defer until after P12 render backend behavior is stable.

   Candidate follow-ups:

   - add a concise artifacts guide for `scenario_config.json`,
     `frame_timing.csv`, and video/debug product outputs;
   - align example flags with the final P12 backend selection API;
   - add `--help` / smoke coverage for backend-specific example options;
   - document recommended conda environments for Go2 static vs physics smokes;
   - keep examples user-facing and avoid exposing low-level benchmark-only flags
     before the backend API settles.

2. **What is the P12 backend selection surface?**

   Decide whether backend choice belongs in:

   - preset config;
   - `ArtifactOutput` / run options;
   - explicit `run_optical_lab_preset(..., backend=...)` style keyword;
   - a reviewed backend registry used by preset/runtime/product builders.

3. **How much backend variability should P11 examples expose?**

   The likely answer is "only stable, reviewed backend choices." Examples
   should show the public workflow, not become benchmark matrices.

4. **How broad should `optics.device_channel.channel_is_device(...)` become?**

   P12.5a keeps device detection intentionally narrow:

   - Torch tensors use `tensor.is_cuda`;
   - non-Torch arrays use a conservative `.device` string heuristic.

   This is enough for the current Torch/Warp CUDA paths. If future backends
   return CuPy, JAX, DLPack-only wrappers, or other device arrays, revisit this
   helper and add explicit framework-aware detection instead of spreading
   backend-specific checks across staging, delivery, and async readback.
