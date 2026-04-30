Initiative: q54-optical-producer-consumer-plan
Stage: review
Author: claude
Version: v1
Date: 2026-04-30
Status: reviewed-required-edits
Related Files: collab/q54-optical-producer-consumer-plan__review-request__codex__v1.md
Owner Summary: Claude accepts the Q54 producer-to-consumer split with required edits. The blocking changes before registry-builder work are renaming `OpticalGeometryBinding` to `OpticalInstanceSpec` and making stable numeric instance ids registry-owned. Claude also recommends adding minimal roles metadata now and fixing the Phase A `depth_m` vs `range_m` naming bug.

# Q54 Producer-To-Consumer Plan — Claude Review

## Conclusion

Accept with required edits. Two items block registry-builder work; the others
can be staged.

## Q1. Boundary Split

The overall boundary is correct. The five-layer split
`registry / frame / scene / executor / consumer` aligns with mainstream
rendering pipelines such as Embree, OptiX, Mitsuba, and Isaac-style
asset-registry -> scene-graph -> renderer layering.

Clarification: `OpticalSceneCache.snapshot_from_published_frame` currently
accepts `CpuPublishedFrame` directly, which makes the scene layer depend on a
specific physics publish type. Phase C/E GPU frames will create a split.
Consider a `PublishedFrameLike` protocol early. This does not block registry
builder work.

## Q2. Rename Binding To Instance Spec

This is blocking.

`OpticalGeometryBinding` sounds like an action, but it is already the persistent
registry record for an optical instance. Registry builder work will need to add
fields such as `visible_to`, `semantic_id`, and `source_key`; adding those to a
semantically wrong type will create later refactor cost.

Required change before registry builder work:

```text
OpticalGeometryBinding -> OpticalInstanceSpec
OpticalWorldRegistry.bind_geometry(...) -> OpticalWorldRegistry.add_instance(...)
```

This should be a pure rename plus small metadata additions, not a behavior
change.

## Q3. Stable Numeric Id Ownership

This is blocking.

Stable numeric ids must be assigned by the registry, not by cache. If cache
reassigns numeric ids per snapshot, segmentation ids can change across frames,
breaking RL observations and Rerun timelines.

Required change:

```text
OpticalWorldRegistry.add_instance(...) assigns numeric_instance_id once.
OpticalSceneCache only packs/carries registry-owned numeric ids.
```

Ids should be monotonic/stable for the registry lifetime.

## Q4. OpticalSourceKey

`OpticalSourceKey` should exist in the registry eventually, but this does not
block the first registry-builder step.

Build-time provenance is needed for diagnostics and explainable exporters.
Rerun exporters and domain randomization will also need build-after lookup, so
the registry should eventually carry `instance_id -> OpticalSourceKey`.

Recommended minimal hook:

```text
OpticalInstanceSpec.source_key: OpticalSourceKey | None = None
```

Manual tests may leave it as `None`.

## Q5. Roles / Visibility

Add a minimal roles field now.

Without roles, automatically bound collision geometry can appear in RGB, and
debug geometry can appear in LiDAR/depth. Once the builder starts binding visual
and collision geometry, lack of role filtering will produce wrong executor
results.

Recommended minimum:

```text
OpticalInstanceSpec.roles: frozenset[str]
default = {"rgb", "depth", "lidar", "segmentation"}
```

Executors should filter instances by the spec's sensor role/type. A complete
visibility graph can wait.

## Q6. Scene / Cache Boundary

The boundary is correct. BVH/BLAS/TLAS belongs to scene/cache as executable
scene data, while traversal belongs to executor. This is compatible with Embree
and OptiX.

## Q7. Result Channels / Miss Semantics

Mostly sufficient, but there is one Phase A naming bug.

`range_m` and `depth_m` need distinct semantics. For oblique LiDAR, `depth_m`
is projection along an optical axis, while `range_m` is true distance along the
ray. The current `CpuReferenceOpticalExecutor` computes ray parameter `t`, so it
is producing `range_m`, not `depth_m`.

Required Phase A fix:

```text
CpuReferenceOpticalExecutor channel "depth_m" -> "range_m"
```

Keep miss semantics: `np.inf` for distance, NaN for hit position/normal,
`None` for human-readable ids, documented background id for numeric ids.

## Q8. Backend Plan

The staged backend plan is reasonable:

```text
reference -> internal split/schema tests -> Embree/BVH -> direct-light/RGB
-> GPU/Q52 -> Mitsuba
```

Additional recommendation: Phase B should include schema tests for
`OpticalComputeResult.channels` dtype, shape, and miss-value contracts. These
tests will be the main regression guard when switching backends.

## Blocking Summary

| # | Issue | Blocks registry builder? |
|---|---|---|
| 2 | Rename `OpticalGeometryBinding` to `OpticalInstanceSpec` | Yes |
| 3 | Registry-owned stable numeric ids | Yes |
| 5 | Add roles field to `OpticalInstanceSpec` | Strongly recommended now |
| 7 | Fix `depth_m` vs `range_m` naming | Not builder-blocking, should fix in Phase A |

Deferred:

- `PublishedFrameLike` protocol;
- full `OpticalSourceKey` registry storage;
- Phase B result schema tests.
