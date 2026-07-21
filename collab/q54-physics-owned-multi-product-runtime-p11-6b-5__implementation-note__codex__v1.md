# Q54 P11.6b-5 Implementation Note: Delete Go2 Backend Shim

Owner: Codex
Date: 2026-07-21
Related Design: `collab/q54-physics-owned-multi-product-runtime-p11-6b-go2-static-p11-workflow__design-request__codex__v3.md`
Previous Slice: `collab/q54-physics-owned-multi-product-runtime-p11-6b-4__implementation-note__codex__v1.md`
Related Files:
- `tools/optical_pipeline_lab/go2_backend.py`
- `tools/optical_pipeline_lab/menagerie_static_runner.py`
- `tests/unit/optics/test_optical_pipeline_lab.py`
- `tests/gpu/test_optical_gpu_runtime.py`
- `examples/optical_lab/README.md`
- `GPU_OPTICAL_PIPELINE_DESIGN.md`
- `MANIFEST.md`

## Summary

P11.6b-5 completes the Go2 backend exit by deleting the deprecated
`tools/optical_pipeline_lab/go2_backend.py` compatibility shim.

The deletion is now safe because P11.6b-4 proved the original Go2 static video
path through the public API:

```python
run_optical_lab_preset(
    "go2_video_ordered_static",
    frames=1,
    products=("video", "debug"),
    output=ArtifactOutput(...),
)
```

## Implementation

### 1. Deleted Shim

Removed:

```text
tools/optical_pipeline_lab/go2_backend.py
```

There is no replacement shim. Legacy imports now fail fast with
`ModuleNotFoundError`.

### 2. Migrated Tests to Canonical Module

Unit and GPU tests that were still importing the deprecated shim now import:

```python
import tools.optical_pipeline_lab.menagerie_static_runner as menagerie_static_runner
```

This preserves coverage of the current canonical module instead of testing a
deleted forwarding layer.

The migrated coverage includes:

- render profile helpers;
- video render/delivery request helpers;
- Menagerie static runner private video loop wrappers;
- `OpticalLabRenderSession` / `OpticalLabRenderPipeline` compatibility paths;
- GPU collection references in `tests/gpu/test_optical_gpu_runtime.py`.

### 3. Added Deletion Guard

Added:

```python
def test_go2_backend_module_is_deleted():
    path = Path(__file__).resolve().parents[3] / "tools/optical_pipeline_lab/go2_backend.py"

    assert not path.exists()
    assert importlib.util.find_spec("tools.optical_pipeline_lab.go2_backend") is None
    with pytest.raises(ModuleNotFoundError):
        __import__("tools.optical_pipeline_lab.go2_backend")
```

This makes the deletion explicit and prevents accidental reintroduction.

### 4. Documentation / Manifest

Updated:

- `MANIFEST.md`
  - removed the `go2_backend.py` row.
- `GPU_OPTICAL_PIPELINE_DESIGN.md`
  - changed the P11.6a status note from "deprecated shim" to "shim removed".
- `examples/optical_lab/README.md`
  - removed the stale "No Go2 P11 example is included yet" wording;
  - documents that Go2 static now runs via `run_optical_lab_preset(...)`.

## Verification

```bash
ruff check tests/unit/optics/test_optical_pipeline_lab.py \
  tests/gpu/test_optical_gpu_runtime.py \
  tools/optical_pipeline_lab/menagerie_static_runner.py
```

Result:

```text
All checks passed!
```

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py \
  -k "go2_backend_module_is_deleted or menagerie_static_runner or render_profile_row or video_render_request or run_video_benchmark or lab_render_pipeline"
```

Result:

```text
17 passed, 170 deselected
```

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
187 passed
```

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/gpu/test_optical_gpu_runtime.py --collect-only
```

Result:

```text
37 tests collected
```

Real Go2 static P11 smoke after deleting the shim:

```bash
conda run -n env_tilelang_20260119 env PYTHONPATH=. python -c "..."
```

The smoke used:

```python
run_optical_lab_preset(
    "go2_video_ordered_static",
    frames=1,
    products=("video", "debug"),
    output=ArtifactOutput(
        root=Path("out/optical_pipeline_lab/p11_go2_static_preset_after_shim_delete"),
        frames=1,
        fps=30.0,
        warmup_renders=0,
        video_readback_delivery="sync",
        video_readback_ring_depth=1,
    ),
)
```

Result:

```text
video_frame 1/1: total=53.615ms, render=19.696ms, readback=33.596ms, fps=18.65
artifacts {'root': PosixPath('out/optical_pipeline_lab/p11_go2_static_preset_after_shim_delete')}
products ['debug', 'video']
video 1 [1]
debug 1 [1]
```

## Not Done

- Did not remove historical collab references to `go2_backend.py`.
  Those are design/review history, not live imports.
- Did not add a dedicated Go2 user example script. The P11 workflow is proven
  and documented, but an example file can still be added as a small follow-up.

