# Q54 Static Asset Source Naming Cleanup Implementation Note

Author: Codex
Date: 2026-05-22
Status: implemented locally, not pushed

## Summary

Moved generic static asset source construction out of `go2_backend.py` and into
`tools/optical_pipeline_lab/static_asset_source.py`.

This closes the naming mismatch where a generic static asset render source
builder was named `build_go2_static_asset_render_source(...)`, even though it
also builds non-Go2 static/synthetic lab sources. Go2 remains a concrete
Menagerie asset instance and CLI/reporting wrapper, not the name of the static
render source component.

## Code Changes

- Added `static_asset_source.py`.
  - `build_static_asset_render_source(...)`
  - `scene_from_static_asset_render_source(...)`
  - `build_static_asset_scene_for_preset(...)`
  - `base_gpu_frame_for_static_asset_scene(...)`
  - `configure_dynamic_video_frame_inputs(...)`
  - synthetic triangle static/dynamic frame helpers
- Updated `go2_backend.py` to call `static_asset_source`.
- Removed `build_go2_static_asset_render_source(...)` from `go2_backend.py`.
- Updated tests to import and patch `static_asset_source` directly.
- Updated `MANIFEST.md` and `GPU_OPTICAL_PIPELINE_DESIGN.md`.

## Boundaries

- `go2_backend.py` still owns Go2/Menagerie CLI, camera, video/reporting glue,
  and example wrapper behavior.
- Static asset source construction now uses generic static-asset vocabulary.
- No `adapter` terminology was introduced. `adapter` remains reserved for
  future external renderer/backend integration.
- No physics runner behavior changed.

## Verification

Focused static asset naming and Go2 wrapper tests:

```bash
conda run -n env_tilelang_20260119 \
  python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q \
  -k "static_asset or go2_backend_configures or render_options_map_args or run_scenario_smoke"
```

Result:

```text
4 passed, 103 deselected
```

Static checks:

```bash
conda run -n env_tilelang_20260119 \
  python -m ruff check tools/optical_pipeline_lab/go2_backend.py \
    tools/optical_pipeline_lab/static_asset_source.py \
    tests/unit/optics/test_optical_pipeline_lab.py
python -m compileall -q \
  tools/optical_pipeline_lab/go2_backend.py \
  tools/optical_pipeline_lab/static_asset_source.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
All checks passed
compileall passed
```

## 关键思考

### 非显而易见的技术决策

1. Why not rename `go2_backend.py` itself?

   The module still owns concrete Go2/Menagerie CLI, camera presets, output
   reporting, and example-wrapper behavior. Renaming the whole module would
   create a much larger diff and blur the distinction between the Go2 wrapper
   and the generic static asset source builder. The narrow cleanup is to move
   only the generic static source construction into `static_asset_source.py`.

2. Why not keep a transitional alias?

   `build_go2_static_asset_render_source(...)` was an internal lab helper, not a
   public API. Keeping the alias would preserve exactly the misleading name this
   cleanup removes. The test now asserts that the old name is absent.

3. Why move synthetic triangle helpers too?

   The synthetic triangle static/dynamic smoke is not Go2. Keeping its source
   and frame-input helpers in `go2_backend.py` made the module look like the
   owner of generic static/dynamic lab source construction. Moving those helpers
   keeps `go2_backend.py` focused on the concrete Menagerie wrapper.

### 调试困难与诊断

1. The old tests monkeypatched `go2_backend._build_scene_for_preset` and
   `_base_gpu_frame_for_scene`.

   After the split, those monkeypatches had to move to `static_asset_source`.
   This was a useful check that the source-builder ownership really changed,
   instead of only renaming a call site.

2. Import ownership needed care.

   `go2_backend.py` still needs Menagerie camera helpers, while
   `static_asset_source.py` needs Menagerie scene import. Separating those
   imports keeps scene/source construction independent from video camera
   construction.
