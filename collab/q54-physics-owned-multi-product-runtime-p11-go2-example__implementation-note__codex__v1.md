# Q54 P11 Go2 Example Implementation Note

Owner: Codex
Date: 2026-07-21
Related Files:
- `examples/optical_lab/go2_video_ordered_static.py`
- `examples/optical_lab/README.md`
- `tests/unit/optics/test_optical_pipeline_lab.py`
- `MANIFEST.md`

## Summary

Added the user-facing Go2 static example now that P11.6b proved and preserved
the public workflow path:

```python
run_optical_lab_preset(
    "go2_video_ordered_static",
    frames=...,
    products=("video", "debug"),
    output=ArtifactOutput(...),
)
```

The example does not import `menagerie_static_runner.py` or the deleted
`go2_backend.py` shim. It calls only the P11 preset workflow API.

## Behavior

`examples/optical_lab/go2_video_ordered_static.py` supports:

- `--dry-run`;
- `--frames`;
- `--out`;
- `--device`;
- `--fps`;
- `--warmup-renders`;
- `--video-readback-delivery`;
- `--video-readback-ring-depth`.

The actual run path constructs an `ArtifactOutput` and delegates to
`run_optical_lab_preset(...)`.

## Verification

```bash
python examples/optical_lab/go2_video_ordered_static.py --dry-run
```

Result:

```text
Would run: preset=go2_video_ordered_static frames=120 products=('video', 'debug') out=runs/examples/go2_video_ordered_static
```

```bash
ruff check examples/optical_lab/go2_video_ordered_static.py \
  examples/optical_lab/physics_body_triangle_video_debug.py \
  examples/optical_lab/physics_body_triangle_observation.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
All checks passed!
```

```bash
ruff format --check examples/optical_lab/go2_video_ordered_static.py \
  examples/optical_lab/physics_body_triangle_video_debug.py \
  examples/optical_lab/physics_body_triangle_observation.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
4 files already formatted
```

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py \
  -k "optical_lab_examples_dry_run"
```

Result:

```text
1 passed, 186 deselected
```

Real Go2 example smoke:

```bash
conda run -n env_tilelang_20260119 env PYTHONPATH=. python \
  examples/optical_lab/go2_video_ordered_static.py \
  --frames 1 \
  --out out/optical_pipeline_lab/go2_example_smoke \
  --warmup-renders 0 \
  --video-readback-delivery sync \
  --video-readback-ring-depth 1
```

Result:

```text
video_frame 1/1: total=51.761ms, render=19.634ms, readback=31.826ms, fps=19.32
artifacts={'root': PosixPath('out/optical_pipeline_lab/go2_example_smoke')}
products=['debug', 'video']
```

