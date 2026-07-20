# Q54 Physics-Owned Multi-Product Runtime P11.4 Implementation Note

Author: Codex
Date: 2026-07-20
Status: implemented locally, not pushed

## Summary

Implemented P11.4: the public preset workflow API.

This implementation follows the P11 Go2 backend exit amendment and keeps
`preset_workflows.py` as a thin facade:

```text
run_optical_lab_preset(...)
  -> resolve_lab_product_specs(...)
  -> create_runtime_for_lab_preset(...)
  -> run_optical_lab_products(..., owns_runtime=True)
```

The first P11.4 attempt was reverted because it duplicated P10 product
materialization and workflow setup, which broke concrete `FrameProduct`
pass-through and left runtime cleanup gaps before workflow entry. This version
reuses the P10 workflow boundary instead of reimplementing it.

## Code Changes

- Added `tools/optical_pipeline_lab/preset_workflows.py`.
- Added `run_optical_lab_preset(...)`.
  - Accepts a reviewed preset name, frame count, product selections, output
    root/output options, optional device, and runtime kwargs.
  - Resolves string products through `resolve_lab_product_specs(...)` before
    creating the runtime.
  - Creates the live runtime through `create_runtime_for_lab_preset(...)`.
  - Delegates scenario config writing, product materialization, concrete
    `FrameProduct` pass-through, execution, result shaping, and cleanup to
    P10 `run_optical_lab_products(..., owns_runtime=True)`.
- Exported `run_optical_lab_preset` through `tools.optical_pipeline_lab`.
- Updated `MANIFEST.md` with the new module entry.

## Tests

Added focused coverage in `tests/unit/optics/test_optical_pipeline_lab.py`:

- `test_run_optical_lab_preset_delegates_to_p10_workflow`
  - verifies runtime factory arguments;
  - verifies debug results;
  - verifies `scenario_config.json` and artifact root;
  - verifies runtime close on success.
- `test_run_optical_lab_preset_accepts_frame_product_instances`
  - verifies concrete `FrameProduct` pass-through reaches P10 correctly.
- `test_run_optical_lab_preset_rejects_products_before_creating_runtime`
  - verifies invalid product strings fail before runtime construction.
- `test_run_optical_lab_preset_closes_runtime_on_p10_setup_error`
  - verifies runtime cleanup when P10 output/config setup fails.
- `test_run_optical_lab_preset_does_not_import_go2_backend`
  - verifies the P11 public workflow does not load the legacy Go2 backend.
- `test_optical_pipeline_lab_exports_p9_product_contracts`
  - verifies top-level lazy export wiring.

## Verification

Focused lint/format:

```bash
ruff check tools/optical_pipeline_lab/preset_workflows.py \
  tools/optical_pipeline_lab/__init__.py \
  tests/unit/optics/test_optical_pipeline_lab.py
ruff format --check tools/optical_pipeline_lab/preset_workflows.py \
  tools/optical_pipeline_lab/__init__.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Focused optical lab unit coverage:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py \
  -k "run_optical_lab_preset or optical_pipeline_lab_exports_p9_product_contracts"
```

Result:

```text
6 passed, 169 deselected
```

Full optical lab unit coverage:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
175 passed
```

## Boundaries

- No static Go2 P11 workflow support was added.
- No examples were added; that remains P11.5.
- No `go2_backend.py` imports or references were added to P11 modules.
- No render backend behavior changed.
- No P10 workflow semantics were duplicated in P11.

## Notes

P11.4 is intentionally small. The public preset workflow should be easy to call,
but the durable execution boundary remains P10. This keeps future product types
and concrete product instances on the same tested path.
