# Review Request: Static Asset Builder Naming + Tests

Commit under review:

```text
251b099 Add static asset builder naming tests
```

## Context

We clarified the naming boundary around the Optical Pipeline Lab:

```text
Physics/simulator systems provide PublishedFrame.
Static asset builders provide OpticalWorldRegistry for non-simulated assets.
Only external renderer/backend integration should use the word adapter.
```

This commit is a small naming/test cleanup after that decision.

## Files Changed

```text
tools/optical_pipeline_lab/render_session.py
tools/optical_pipeline_lab/go2_backend.py
tests/unit/optics/test_optical_pipeline_lab.py
GPU_OPTICAL_PIPELINE_DESIGN.md
MANIFEST.md
```

## Main Changes

1. `OpticalLabRenderSource` docstring changed:

```text
Before:
Scene/source adapter output consumed by the lab render pipeline.

After:
Lab-local bundle of optical registry and base frame identity.
```

Reason: `adapter` should not describe the asset/static scene side. It is reserved
for external renderer/backend integration.

2. Go2 static asset builder rename:

```text
build_go2_render_source
  -> build_go2_static_asset_render_source

_scene_from_render_source
  -> _scene_from_static_asset_render_source
```

Reason: Go2/Menagerie is a static asset builder for non-simulated benchmark
assets, not a render pipeline/backend and not a physics frame provider.

3. Go2 render source metadata now includes:

```python
"source_kind": "static_asset"
```

4. Unit tests added/updated:

```python
def test_lab_render_source_naming_does_not_use_adapter_language():
    doc = render_session.OpticalLabRenderSource.__doc__ or ""

    assert "adapter" not in doc.lower()
    assert "registry" in doc
    assert "base frame" in doc
```

Go2 builder test now asserts the old function name is gone:

```python
assert not hasattr(go2_backend, "build_go2_render_source")
source = go2_backend.build_go2_static_asset_render_source(...)
assert source.metadata["source_kind"] == "static_asset"
assert go2_backend._scene_from_static_asset_render_source(source) is scene
```

5. Docs updated:

`GPU_OPTICAL_PIPELINE_DESIGN.md` now says C3 uses:

```text
build_go2_static_asset_render_source(...)
```

`MANIFEST.md` test count updated:

```text
222 tests
145 unit optics/lab + 40 unit sensing + 37 GPU optical
```

## Validation

```text
PYTHONPATH=. pytest tests/unit/optics/test_optical_pipeline_lab.py -q
72 passed

PYTHONPATH=. pytest tests/gpu/test_optical_gpu_runtime.py -q -k "physics_published_frame_drives_dynamic_render"
1 skipped, 31 deselected
# skipped because current environment has no usable Warp/CUDA

ruff check tools/optical_pipeline_lab/go2_backend.py \
           tools/optical_pipeline_lab/render_session.py \
           tests/unit/optics/test_optical_pipeline_lab.py
All checks passed

git diff --check
clean
```

## Review Focus

- Is `build_go2_static_asset_render_source` the right name, or should it be shorter/clearer?
- Is it correct to remove the old `build_go2_render_source` name entirely instead of keeping a transitional alias?
- Does the `OpticalLabRenderSource` docstring now express the right boundary?
- Is `source_kind="static_asset"` useful metadata, or should this be modeled more formally later?
- Any risk that `adapter` remains in an asset-side context where it should now be reserved for external renderer/backend integration?
