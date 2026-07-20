# Q54 Physics-Owned Multi-Product Runtime P11.5 Implementation Note

Author: Codex
Date: 2026-07-20
Status: implemented locally, not pushed

## Summary

Implemented P11.5: public examples for the P11 Optical Lab preset workflow.

The examples are intentionally written as user code that calls
`run_optical_lab_preset(...)`. They do not import or call legacy
`go2_backend.py` paths.

## Code Changes

- Added `examples/optical_lab/README.md`.
- Added `examples/optical_lab/physics_body_triangle_video_debug.py`.
  - Runs `physics_body_triangle_video_smoke` with `("video", "debug")`.
  - Supports `--dry-run`, `--frames`, `--out`, and `--device`.
- Added `examples/optical_lab/physics_body_triangle_observation.py`.
  - Shows explicit `ObservationProductSpec.from_scenario(...)` usage.
  - Supports `--dry-run`, `--frames`, `--out`, and `--device`.
- Updated `MANIFEST.md` with the new example entries.

## Tests

Added `test_optical_lab_examples_dry_run` in
`tests/unit/optics/test_optical_pipeline_lab.py`.

The test runs:

```bash
python examples/optical_lab/physics_body_triangle_video_debug.py --dry-run
python examples/optical_lab/physics_body_triangle_observation.py --dry-run
```

This validates import paths and argument parsing without creating a live
physics runtime or requiring GPU work.

## Verification

Focused lint/format:

```bash
ruff check examples/optical_lab/physics_body_triangle_video_debug.py \
  examples/optical_lab/physics_body_triangle_observation.py \
  tests/unit/optics/test_optical_pipeline_lab.py
ruff format --check examples/optical_lab/physics_body_triangle_video_debug.py \
  examples/optical_lab/physics_body_triangle_observation.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Focused dry-run coverage:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py \
  -k "optical_lab_examples_dry_run or run_optical_lab_preset"
```

Result:

```text
6 passed, 170 deselected
```

Full optical lab unit coverage:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
176 passed
```

## Boundaries

- No Go2 P11 example was added.
- No static-asset P11 workflow support was added.
- No legacy `go2_backend.py` path is used by the new examples.
- No full GPU execution is required by unit tests.

## Notes

Go2 should get a P11 example only after it can call the same public workflow
without a `go2_backend.render_many_views(...)` bypass.
