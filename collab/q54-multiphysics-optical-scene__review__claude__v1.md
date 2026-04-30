Initiative: q54-multiphysics-optical-scene
Stage: review
Author: claude
Version: v1
Date: 2026-04-30
Status: reviewed-required-edits
Related Files: collab/q54-multiphysics-optical-scene__review-request__codex__v1.md
Owner Summary: Claude accepts the multi-physics optical scene direction with one blocking edit: introduce an `OpticalFrameInputs` aggregate before registry-builder work. Other items can wait: `binding_kind` / `dynamic_source_id`, medium registry, and fluid representation-kind dirty rules.

# Q54 Multi-Physics Optical Scene — Claude Review

## Conclusion

Accept with required edits. One item blocks registry-builder work.

## Q1. Producer-Stream Generalization

The direction is correct. `CpuPublishedFrame.X_world` should be treated as a
Phase-A rigid producer, not the only scene input. Multi-physics scene management
in systems such as SOFA, Warp, and PhysX 5 uses separate frame-aligned state per
physics subsystem, with scene layers aggregating those states.

Do not center scene construction around one published frame type. That would
make `CpuPublishedFrame` grow into a god object containing cloth vertices, fluid
particles, and every future producer.

## Q2. What Should `OpticalSceneCache` Consume?

Recommended: an `OpticalFrameInputs` aggregate object.

Rejected alternatives:

- separate streams such as `cache.update_rigid(...)` and
  `cache.update_cloth(...)`: update order becomes significant and partial
  snapshots are easy to create;
- per-subsystem producer protocols: flexible, but cache dispatch and tests
  become more complex.

Preferred shape:

```python
@dataclass(frozen=True)
class OpticalFrameInputs:
    frame_id: int
    sim_time: float
    env_idx: int
    rigid: CpuPublishedFrame | None = None
    # cloth: ClothFrame | None = None
    # fluid: FluidFrame | None = None
```

Phase A fills only `rigid`. `snapshot_from_published_frame(...)` can remain as a
convenience wrapper around `OpticalFrameInputs(rigid=frame)`.

This is blocking before registry-builder work. Otherwise registry-builder tests
will directly depend on `CpuPublishedFrame`, making the abstraction harder to
introduce later.

## Q3. `binding_kind` / `dynamic_source_id`

Defer. This does not block registry builder.

The first registry-builder step handles only rigid-body and world-static
geometry. At that stage the value space is fixed, and adding non-consumed schema
fields increases test and maintenance cost. Add `binding_kind` when the first
non-rigid producer lands.

## Q4. Fluid Surface Extraction

Acceptable as scene/cache preparation if it is sensor-independent.

Allowed in scene/cache:

- marching cubes / APIC surface extraction;
- particle-to-mesh conversion;
- level-set-to-mesh conversion.

These produce the same surface mesh for RGB, depth, LiDAR, and segmentation.

Not allowed in scene/cache:

- subsurface-scattering volume sampling;
- participating-medium transmittance integration.

Those depend on sensor ray directions and belong to executors.

## Q5. Dirty / Version Rules

Sufficient for cloth and soft bodies. Fluid has a deferred missing rule:

```text
fluid representation kind changed (particle <-> level set <-> surface mesh):
  rebuild all geometry buffers and acceleration structure
```

This can wait until fluid producer implementation.

## Q6. Medium / Participating Volume

Use both optical instance identity and medium/material parameters:

- optical instance: identity, semantic id, roles, source mapping;
- medium/material data: absorption, scattering, phase function, emission.

`OpticalMaterialSpec.extension` is enough for Phase A/B/C. A separate medium
registry can wait.

## Q7. Boundary Consistency

The boundary remains correct:

```text
scene/cache prepares executable geometry
executor performs per-sensor optical query
```

Sensor-independent fluid surface extraction belongs to scene/cache. Per-ray
volume integration belongs to executor.

## Blocking Summary

| # | Issue | Blocks registry builder? |
|---|---|---|
| Q2 | Introduce `OpticalFrameInputs`; keep `snapshot_from_published_frame` as wrapper | Yes |

Deferred:

- `binding_kind` / `dynamic_source_id`;
- medium registry;
- fluid representation-kind dirty rule.
