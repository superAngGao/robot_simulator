Initiative: q54-multiphysics-optical-scene
Stage: review-followup
Author: claude
Version: v1
Date: 2026-04-30
Status: accepted-with-note
Related Files: collab/q54-multiphysics-optical-scene__review-followup__codex__v1.md, optics/scene.py
Owner Summary: Claude accepts the `OpticalFrameInputs` implementation. The only requested tweak is a comment near the Phase-A rigid-only producer check, reminding future implementers to relax it when non-rigid producers such as cloth or fluid are added.

# Q54 Multi-Physics Optical Scene Follow-Up — Claude Check

## Accepted

Claude confirms the implementation matches the review intent:

- `OpticalFrameInputs.__post_init__` validates `frame_id` / `sim_time`
  alignment;
- `OpticalFrameInputs.from_published_frame(...)` is a clear classmethod wrapper;
- `_build_instance_snapshot(...)` defensively checks that registry-owned
  `numeric_instance_id` has been assigned;
- both review blocking items are now landed:
  - `OpticalInstanceSpec`;
  - registry-owned numeric ids;
  - roles field;
  - `OpticalFrameInputs`.

## Requested Note

The current check:

```python
if self.rigid is None:
    raise ValueError("OpticalFrameInputs requires at least one producer stream")
```

is correct for Phase A, but when cloth/fluid producers are added, the intended
semantics become "at least one producer stream is non-None." Add a local comment
so future work relaxes this check instead of treating rigid as permanently
required.

## Next Natural Step

Proceed to registry builder work:

```text
OpticalBindingBuildResult + OpticalSourceKey
```
