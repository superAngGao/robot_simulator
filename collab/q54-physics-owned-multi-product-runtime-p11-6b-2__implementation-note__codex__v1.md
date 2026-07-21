# Q54 P11.6b-2 Implementation Note: Generic Lab Product Scenario Runner

Owner: Codex
Date: 2026-07-21

## Summary

Implemented the first P11.6b execution slice from the approved V3 design:
`run_optical_lab_workflow(...)` no longer routes every product workflow through
the physics-only scenario validator.

New generic entry points:

- `ProductRunResult = PhysicsProductRunResult`
- `ProductWorkflowRunner = PhysicsOwnedProductWorkflow`
- `run_lab_product_scenario(...)`
- `validate_lab_product_scenario(...)`

Existing physics-specific entry points remain:

- `run_physics_product_scenario(...)`
- `run_physics_product_preset(...)`

The physics-specific path still validates
`frame_source='physics_published_frame'` and
`clock_owner='external_physics_runtime'` before delegating to the generic lab
runner.

## Design Alignment

This resolves the first V3 blocker:

```text
run_optical_lab_preset(...)
  -> run_optical_lab_products(...)
  -> run_optical_lab_workflow(...)
  -> run_lab_product_scenario(...)
```

Static asset presets can now enter the generic product workflow with a runtime
that exposes `step_tick(...)`. The static video/runtime implementation is not
added in this slice.

## Changed Files

- `tools/optical_pipeline_lab/product_workflow.py`
  - added generic result/workflow aliases;
  - added `run_lab_product_scenario(...)`;
  - added `validate_lab_product_scenario(...)`;
  - changed `run_optical_lab_workflow(...)` to call the generic runner;
  - kept `run_physics_product_scenario(...)` as a physics-validating wrapper.
- `tools/optical_pipeline_lab/__init__.py`
  - lazy-exported the new generic aliases and `run_lab_product_scenario(...)`.
- `tests/unit/optics/test_optical_pipeline_lab.py`
  - asserted the new lazy exports;
  - added a static-asset workflow test through `run_optical_lab_workflow(...)`;
  - strengthened the physics-specific wrapper test to verify owned runtime
    cleanup on validation failure.
- `MANIFEST.md`
  - updated the product workflow module description.

## Tests

Focused coverage should include:

```bash
PYTHONPATH=. pytest -q tests/unit/optics/test_optical_pipeline_lab.py \
  -k "lab_product_scenario or run_optical_lab_workflow_accepts_static_asset_runtime or run_physics_product_scenario_requires_physics_owned_clock or exports_p9_product_contracts"
```

Full optical lab unit coverage should still pass.

## Non-Goals

- No `frame_providers.py` yet.
- No `VideoFrameProduct` generalization yet.
- No static runtime factory yet.
- No Go2 video product registration yet.
- No `go2_backend.py` deletion yet.
