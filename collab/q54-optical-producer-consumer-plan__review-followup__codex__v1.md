Initiative: q54-optical-producer-consumer-plan
Stage: review-followup
Author: codex
Version: v1
Date: 2026-04-30
Status: required-edits-applied
Related Files: collab/q54-optical-producer-consumer-plan__review-request__codex__v1.md, collab/q54-optical-producer-consumer-plan__review__claude__v1.md, optics/registry.py, optics/scene.py, optics/execution.py, sensing/optical.py, tests/unit/optics/test_optics_phase_a.py
Owner Summary: Claude's required Q54 edits have been applied. The persistent registry record is now `OpticalInstanceSpec`, numeric instance ids are registry-owned and stable, minimal roles support has been added, and the reference executor now reports `range_m` for ray distance instead of mislabeling it as `depth_m`.

# Q54 Producer-To-Consumer Plan Review Follow-Up

## 1. Applied Required Edits

### 1.1 `OpticalGeometryBinding` -> `OpticalInstanceSpec`

Applied.

Changes:

- renamed the persistent registry record to `OpticalInstanceSpec`;
- changed `OpticalWorldRegistry.bind_geometry(...)` to
  `OpticalWorldRegistry.add_instance(...)`;
- changed scene construction to iterate `registry.instances`;
- updated tests and exports.

Rationale from review: this object is not a transient binding action; it is the
long-lived optical instance record that downstream registry-builder work will
extend.

### 1.2 Registry-Owned Stable Numeric Instance Ids

Applied.

Changes:

- `OpticalInstanceSpec.numeric_instance_id` is optional at construction time;
- `OpticalWorldRegistry.add_instance(...)` assigns it once when omitted;
- explicit numeric ids are accepted if unique;
- ids are monotonic and stable for the registry lifetime;
- `OpticalSceneSnapshot` carries `numeric_instance_id`;
- `CpuReferenceOpticalExecutor` returns a `numeric_instance_id` result channel;
- miss/background numeric instance id is currently `0`.

Rationale from review: cache-assigned per-snapshot ids would break temporal
stability for segmentation, RL observations, and Rerun timelines.

### 1.3 Minimal Roles Field

Applied.

Changes:

- `OpticalInstanceSpec.roles` defaults to
  `{"rgb", "depth", "lidar", "segmentation"}`;
- `OpticalRaySensorSpec.sensor_role` defaults to `"depth"`;
- `CpuReferenceOpticalExecutor` skips instances that do not include the spec's
  role;
- unit coverage verifies role filtering.

This is intentionally not a full visibility graph. It is the minimum needed to
keep visual/collision/debug geometry from being accidentally consumed by the
wrong sensor family once registry builders start auto-binding assets.

### 1.4 `depth_m` vs `range_m`

Applied.

Changes:

- `CpuReferenceOpticalExecutor` now returns `range_m` instead of `depth_m`;
- tests were updated to assert `range_m`;
- Q54 docs clarify:
  - `range_m` is true first-hit distance along a normalized ray;
  - `depth_m` is projected depth for camera/optical-axis sensor models and is
    not produced by the generic ray reference executor.

Rationale from review: the current executor computes ray parameter `t`, which
is range, not projected camera depth.

## 2. Deferred Review Items

Deferred as requested/non-blocking:

- `PublishedFrameLike` protocol for scene-cache input abstraction;
- concrete `OpticalSourceKey` dataclass and registry provenance storage;
- Phase B schema tests for `OpticalComputeResult.channels` dtype, shape, and
  miss-value contracts.

The code now has a `source_key: object | None` hook on `OpticalInstanceSpec` so
builder work can fill provenance later without changing the instance record
shape again.

## 3. Verification

Commands:

```text
PYTHONPATH=. pytest tests/unit/optics -q
ruff check optics sensing tests/unit/optics
```

Results:

```text
11 passed
All checks passed
```

Broader sensing/optics verification should be run before final handoff if more
code changes are added in the same worktree.
