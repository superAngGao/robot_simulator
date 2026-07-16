# Q54 Physics-Owned Multi-Product Runtime P11.2 Implementation Note

Author: Codex
Date: 2026-07-16
Status: implemented locally, not pushed

## Summary

Implemented P11.2: the preset-to-runtime factory layer for Optical Pipeline Lab.

P11.2 adds a narrow user-facing factory that maps reviewed lab preset names to
live physics runtimes. This is the first implementation slice of the P11 preset
workflow design, and intentionally stops before product string resolution,
`run_optical_lab_preset(...)`, or examples.

## Code Changes

- Added `tools/optical_pipeline_lab/preset_runtime.py`.
- Added `create_runtime_for_lab_preset(...)`.
  - Supports the reviewed `physics_body_triangle_video_smoke` preset.
  - Creates the runtime through `create_physics_body_triangle_lab_runtime(...)`.
  - Forwards `device` only when explicitly supplied, so the underlying factory
    can retain its default device behavior.
  - Forwards advanced runtime keyword arguments such as `initial_height`,
    `dt`, `metadata`, and related factory options.
  - Rejects unregistered presets with a clear `NotImplementedError`.
- Added `supported_runtime_presets()` as the companion discovery helper.
- Exported both helpers through `tools.optical_pipeline_lab` lazy exports.
- Updated `MANIFEST.md` with the new module entry.

## Tests

Added focused coverage in `tests/unit/optics/test_optical_pipeline_lab.py`:

- `test_create_runtime_for_lab_preset_builds_reviewed_physics_runtime`
  - monkeypatches the reviewed runtime factory;
  - verifies the supported preset path;
  - verifies `device`, `initial_height`, and `metadata` pass-through.
- `test_create_runtime_for_lab_preset_uses_factory_default_device`
  - verifies no `device` keyword is injected when `device=None`;
  - confirms factory defaults remain owned by the runtime factory.
- `test_create_runtime_for_lab_preset_rejects_unregistered_preset`
  - verifies fail-fast behavior for unsupported presets;
  - verifies `supported_runtime_presets()` returns the reviewed registry.
- `test_optical_pipeline_lab_exports_p9_product_contracts`
  - verifies top-level lazy exports include `create_runtime_for_lab_preset`;
  - verifies top-level lazy exports include `supported_runtime_presets`.

## Verification

Focused lint/format:

```bash
ruff check tools/optical_pipeline_lab/preset_runtime.py \
  tools/optical_pipeline_lab/__init__.py \
  tests/unit/optics/test_optical_pipeline_lab.py
ruff format --check tools/optical_pipeline_lab/preset_runtime.py \
  tools/optical_pipeline_lab/__init__.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Focused optical lab unit coverage:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py \
  -k "runtime_for_lab_preset or optical_pipeline_lab_exports_p9_product_contracts"
```

Full optical lab unit coverage from the original implementation pass:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result at implementation time:

```text
164 passed
```

## Boundaries

- No product string resolver was added; that remains P11.3.
- No `run_optical_lab_preset(...)` preset workflow was added; that remains
  P11.4.
- No examples were added; those remain P11.5.
- No runtime inference from arbitrary `OpticalLabScenarioConfig` values was
  introduced.
- No new render backend behavior was introduced.

## Notes

The registry is intentionally explicit. P11 should be easy to call, but not
magical: a preset becomes runnable only after its runtime factory has been
reviewed and registered.
