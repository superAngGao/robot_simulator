Initiative: q54-optical-registry-builder
Stage: review
Author: claude
Version: v1
Date: 2026-04-30
Status: accepted
Related Files: collab/q54-optical-registry-builder__implementation-note__codex__v1.md, optics/builder.py, tests/unit/optics/test_registry_builder.py
Owner Summary: Claude accepts the Phase-A optical registry builder implementation. No required fixes were found. The review specifically validates the builder's fallback order, diagnostic behavior, source provenance design, and token sanitization.

# Q54 Optical Registry Builder — Claude Review

## Conclusion

Accepted. Registry builder Phase A can be committed.

## Positive Findings

- `_add_shape_geometry(...)` uses the right fallback order:
  `HalfSpaceShape -> MeshShape -> face_topology()`. This avoids accidentally
  treating half-spaces as mesh-like geometry.
- Unsupported or incomplete shapes are converted to diagnostics rather than
  crashing the builder. This keeps the builder robust for incomplete models.
- `_safe_token(...)` uses an allowlist, which is the right sanitization style
  for stable ids.
- `OpticalSourceKey` exists both in the build result maps and in
  `OpticalInstanceSpec.source_key`, so the registry remains explainable even
  after the build result is out of scope.

## Checked Concern

Potential issue considered:

```python
SpatialTransform.from_rpy(
    float(shape_instance.origin_rpy[0]),
    float(shape_instance.origin_rpy[1]),
    float(shape_instance.origin_rpy[2]),
    r=np.asarray(shape_instance.origin_xyz, dtype=np.float64),
)
```

If `ShapeInstance.origin_rpy` could be `None`, this would fail. It is not a real
issue because `ShapeInstance.origin_rpy` has a `default_factory` that produces a
zero vector.

## Suggested Next Step

Commit this work. After that, choose between:

- `visual_preferred` registry-builder policy;
- executor Phase B internal split and result-schema tests.
