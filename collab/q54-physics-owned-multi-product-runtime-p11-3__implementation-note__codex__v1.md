# Q54 Physics-Owned Multi-Product Runtime P11.3 Implementation Note

Author: Codex
Date: 2026-07-17
Status: implemented locally, not pushed

## Summary

Implemented P11.3: preset product selection for Optical Pipeline Lab workflows.

This slice adds the user-facing resolver that turns small product strings into
reviewed product specs while preserving the P10 advanced path for explicit
`ProductSpec` and `FrameProduct` values.

## Code Changes

- Added `tools/optical_pipeline_lab/preset_products.py`.
- Added `resolve_lab_product_specs(...)`.
  - Resolves `"debug"` to `DebugProductSpec()`.
  - Resolves `"video"` to a preset-specific `VideoProductSpec(...)` only for
    the reviewed `physics_body_triangle_video_smoke` preset.
  - Passes through explicit `ProductSpec` and `FrameProduct` values after the
    existing P10 product input validation.
  - Rejects `"observation"` with a message requiring explicit robot metadata
    and `ObservationProductSpec.from_scenario(...)`.
  - Rejects unknown product strings with a clear fail-fast error.
- Added `supported_lab_product_strings(...)` as the companion discovery helper.
- Exported both helpers through `tools.optical_pipeline_lab` lazy exports.
- Updated `MANIFEST.md` with the new module entry.

## Tests

Added focused coverage in `tests/unit/optics/test_optical_pipeline_lab.py`:

- `test_resolve_lab_product_specs_builds_reviewed_video_and_debug_specs`
  - verifies `"video"` resolves to the reviewed video dependencies;
  - verifies `"debug"` resolves to `DebugProductSpec`;
  - verifies supported string discovery for the reviewed preset.
- `test_resolve_lab_product_specs_passes_through_explicit_products`
  - verifies explicit specs and frame products are not wrapped or rebuilt.
- `test_resolve_lab_product_specs_requires_explicit_observation_spec`
  - verifies `"observation"` does not create a guessed observation product.
- `test_resolve_lab_product_specs_rejects_unknown_product_string`
  - verifies unknown product strings fail fast.
- `test_resolve_lab_product_specs_rejects_unregistered_video_preset`
  - verifies `"video"` is preset-registered rather than inferred.
- `test_optical_pipeline_lab_exports_p9_product_contracts`
  - verifies top-level lazy exports include the P11.3 helpers.

## Verification

Focused lint/format:

```bash
ruff check tools/optical_pipeline_lab/preset_products.py \
  tools/optical_pipeline_lab/__init__.py \
  tests/unit/optics/test_optical_pipeline_lab.py
ruff format --check tools/optical_pipeline_lab/preset_products.py \
  tools/optical_pipeline_lab/__init__.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Focused optical lab unit coverage:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py \
  -k "resolve_lab_product_specs or optical_pipeline_lab_exports_p9_product_contracts"
```

Result:

```text
6 passed, 163 deselected
```

Full optical lab unit coverage:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
169 passed
```

## Boundaries

- No `run_optical_lab_preset(...)` workflow was added; that remains P11.4.
- No examples were added; those remain P11.5.
- No observation defaults were inferred.
- No render backend behavior was changed.

## Notes

P11.3 intentionally makes the common path concise without making the API
ambiguous. String products are accepted only where the lab has reviewed default
dependencies; richer products still use explicit specs.
