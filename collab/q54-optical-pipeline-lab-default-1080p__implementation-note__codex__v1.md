Initiative: q54-optical-pipeline-lab-default-1080p
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-05-09
Status: implemented
Related Files: tools/optical_pipeline_lab/scenarios.py, tools/optical_pipeline_lab/go2_backend.py, tests/unit/optics/test_optical_pipeline_lab.py, MANIFEST.md
Owner Summary: Added the default rendering resolution rule for the Optical Pipeline Lab and Go2 backend: default render resolution is now 1080p, represented as 1920x1080.

# Q54 Optical Pipeline Lab Default 1080p Rule

## Change

Added explicit defaults:

```python
DEFAULT_RENDER_WIDTH = 1920
DEFAULT_RENDER_HEIGHT = 1080
```

in:

```text
tools/optical_pipeline_lab/scenarios.py
```

`OpticalLabScenarioConfig` now defaults to:

```text
width=1920
height=1080
```

The Go2 backend CLI now uses the same constants for:

```text
--width
--height
```

so lab presets and the example wrapper share the same default resolution rule.

## Validation

Added:

```text
test_default_render_resolution_is_1080p
```

Validated:

```bash
python -m tools.optical_pipeline_lab describe --preset go2_video_ordered_static
```

prints:

```text
width: 1920
height: 1080
```

Also checked the Go2 backend parser default:

```text
1920 1080
```

## Test Count

Q54 collect-only now reports:

```text
150 tests collected
118 CPU optics/sensing/lab
32 GPU optical
```

