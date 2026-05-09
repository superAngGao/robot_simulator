Initiative: q54-optical-pipeline-lab-c1-go2-backend
Stage: implementation-note
Author: codex
Reviewer: claude
Version: v1
Date: 2026-05-09
Status: implemented-c1-go2-backend-extraction
Related Files: tools/optical_pipeline_lab/go2_backend.py, tools/optical_pipeline_lab/runner.py, examples/mujoco_menagerie_gpu_preview.py, tests/unit/optics/test_optical_pipeline_lab.py, MANIFEST.md
Owner Summary: Applied Claude's C1 requirement before matrix work. The Go2 Menagerie GPU implementation is now in `tools/optical_pipeline_lab/go2_backend.py`; the example is a thin CLI wrapper and the lab runner calls the backend directly, eliminating the C0 dependency on `examples.mujoco_menagerie_gpu_preview._render_many_views`.

# Q54 Optical Pipeline Lab C1 Go2 Backend Extraction

## Motivation

Claude accepted C0 but required this before C2 matrix work:

```text
extract the Go2 runner helper before adding matrix / multiple baseline presets
```

The concern was that C0 had:

```text
tools.optical_pipeline_lab.runner
  -> examples.mujoco_menagerie_gpu_preview._render_many_views(...)
```

which would make an example-private function a de facto public dependency if
matrix support were added on top of it.

## What Changed

Moved the Go2 Menagerie GPU implementation into:

```text
tools/optical_pipeline_lab/go2_backend.py
```

Renamed the shared entrypoint to:

```python
render_many_views(args)
```

Updated:

```text
tools/optical_pipeline_lab/runner.py
  -> imports render_many_views from .go2_backend

examples/mujoco_menagerie_gpu_preview.py
  -> thin CLI wrapper around tools.optical_pipeline_lab.go2_backend.main
```

No render kernel, BVH builder, output profile, readback, camera raygen, or
timing behavior was intentionally changed.

## Tests

Updated the runner smoke test to monkeypatch:

```text
tools.optical_pipeline_lab.go2_backend.render_many_views
```

instead of faking:

```text
examples.mujoco_menagerie_gpu_preview._render_many_views
```

The previous immediate follow-up test also remains:

```text
video_raygen=gpu + video_ray_cache=precompute rejects early
```

## Validation

Ran:

```bash
ruff check tools/optical_pipeline_lab examples/mujoco_menagerie_gpu_preview.py tests/unit/optics/test_optical_pipeline_lab.py
python -m py_compile tools/optical_pipeline_lab/*.py tests/unit/optics/test_optical_pipeline_lab.py examples/mujoco_menagerie_gpu_preview.py
python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q
python -m pytest --collect-only -q tests/unit/optics tests/unit/sensing tests/gpu/test_optical_warp_executor.py tests/gpu/test_optical_gpu_runtime.py
python examples/mujoco_menagerie_gpu_preview.py --help
python -m tools.optical_pipeline_lab run --help
conda run -n env_tilelang_20260119 python -m tools.optical_pipeline_lab run \
  --preset go2_video_ordered_static \
  --width 160 --height 120 --frames 1 \
  --readback none --no-fail-on-overflow \
  --out out/optical_pipeline_lab/smoke_160_readback_none_c1 \
  --progress-every 1
```

Results:

```text
ruff: passed
py_compile: passed
lab tests: 15 passed
Q54 collect-only: 149 tests collected
example --help: passed
lab run --help: passed
160x120 GPU smoke: passed in env_tilelang_20260119
  render_execute_ms ~= 1.29 ms
  frame_total_ms ~= 1.44 ms
```

## Next

Now C2 can safely add:

```text
matrix subcommand
baseline preset suite
```

without depending on an example-private API.
