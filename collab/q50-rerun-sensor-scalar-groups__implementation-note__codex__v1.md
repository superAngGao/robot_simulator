Initiative: q50-render-backend
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-28
Status: implemented
Related Files: OPEN_QUESTIONS.md#Q50, rendering/backends/rerun_backend.py, tests/rendering/test_rerun_backend.py
Owner Summary: Rerun sensor scalar timeline logging now has a small group filter so training/debug runs can choose only the signal families they need without changing `RenderScene.sensor_data`.

## Summary

`RerunBackend` now accepts:

```python
RerunBackend(sensor_scalar_groups=("contact", "joint"))
```

Allowed groups are:

- `contact`
- `joint`
- `force`
- `imu`

The default remains all four groups, so existing recordings keep the same
behavior. A single string such as `"contact"` is accepted as one group, and
unknown group names raise `ValueError` during backend construction.

This is deliberately a backend-side selection layer. It does not expand or
redefine the `RenderScene.sensor_data` contract.

## Why This Step

The first scalar timeline pass proved that Rerun can consume the narrow
numeric/state sensor payload, but large training runs can produce noisy entity
trees if every scalar family is always emitted.

This pass gives callers a small control surface before we invest in richer
Rerun blueprints/layouts.

## Verification

Verified with:

```bash
PYTHONPATH=. pytest tests/rendering/test_rerun_backend.py -q
PYTHONPATH=. pytest tests/rendering tests/unit/rendering tests/integration/test_render_scene_integration.py tests/integration/test_published_frame_render_backend_integration.py -q
```

Current results after review follow-ups:

- `15 passed` in the Rerun-enabled environment
- `73 passed, 1 skipped` for the default render subset

## Remaining Work

- Rerun blueprint/layout for geometry + scalar panels.
- Preset constructors or named profiles if callers converge on common sets.
- DebugExporter parity remains a separate decision.
