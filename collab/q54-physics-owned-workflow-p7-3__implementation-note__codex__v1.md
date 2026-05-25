# Q54 Physics-Owned Workflow P7.3 Implementation Note

Author: Codex
Date: 2026-05-25
Status: implemented locally, not pushed

## Summary

Closed P7 by keeping the plain `run_scenario(...)` physics-runtime path
guarded.

P7.1 and P7.2 added explicit physics-owned entries:

```text
run_physics_video_scenario(...)
run_physics_stepped_video_scenario(...)
```

P7.3 confirms that the generic lab CLI-style runner still does not construct a
physics engine or own physics lifecycle. That remains a P8 design topic.

## Code Changes

- Strengthened
  `test_run_scenario_physics_runtime_requires_explicit_runtime_inputs`.
  - The test now checks the guard message names both explicit physics entries.
  - The test now verifies `run_scenario(...)` exits before creating the output
    directory/config for physics runtime configs.
- Updated `GPU_OPTICAL_PIPELINE_DESIGN.md`.
  - Marked P7.3 complete.
  - Replaced the P7 next-slice item with the P8 design question: who should own
    physics engine construction, action source, lifecycle, and cleanup policy.

## Verification

Focused unit test:

```bash
conda run -n env_tilelang_20260119 \
  python -m pytest tests/unit/optics/test_optical_pipeline_lab.py -q \
  -k "run_scenario_physics_runtime_requires_explicit_runtime_inputs"
```

Broader verification still to run before commit:

```bash
conda run -n env_tilelang_20260119 python -m pytest tests/unit/optics -q
ruff check tests/unit/optics/test_optical_pipeline_lab.py
ruff format --check tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
1 passed, 109 deselected
183 passed
ruff check clean
ruff format --check clean
```

## Boundaries

- No `run_scenario(...)` physics execution was enabled.
- No CLI physics engine construction was added.
- No action/control API was introduced.
- No render/session/provider behavior changed.

## 关键思考

1. Why make P7.3 a guard hardening slice?

   P7.1/P7.2 made the explicit physics entries useful enough that future callers
   might expect plain `run_scenario(...)` to work too. P7.3 locks the opposite
   decision: generic scenario execution still cannot own physics lifecycle.

2. Why assert the output directory is absent?

   The guard should fail before any runtime side effect. Checking that the
   output directory is not created proves the function exits before writing
   `scenario_config.json` or entering any backend path.

3. Why defer CLI engine construction to P8?

   Constructing a physics engine requires choices about scene construction,
   action/control source, reset policy, ownership of cleanup, and eventually RL
   observation products. None of those decisions belong in the narrow P7 bridge
   from physics-owned frames to render/video/delivery.
