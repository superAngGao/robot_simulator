Initiative: q54-optical-registry-builder
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-30
Status: implemented-phase-a
Related Files: optics/builder.py, optics/registry.py, optics/scene.py, optics/execution.py, tests/unit/optics/test_registry_builder.py
Owner Summary: Phase-A optical registry builder is implemented for `RobotModel.geometries` with `collision_only` policy. It outputs `OpticalBindingBuildResult`, stable provenance maps via `OpticalSourceKey`, and diagnostics for unsupported shapes. The builder intentionally avoids visual asset assumptions and does not silently approximate smooth shapes.

# Q54 Optical Registry Builder Implementation Note

## 1. What Landed

Added `optics.builder`:

- `OpticalSourceKey`
- `OpticalBindingDiagnostic`
- `OpticalBindingBuildResult`
- `build_optical_registry_from_robot_model(...)`

The builder converts:

```text
RobotModel.geometries
  -> OpticalWorldRegistry
  -> OpticalInstanceSpec records
  -> source/instance provenance maps
  -> diagnostics
```

## 2. Supported Policy

Phase A supports only:

```text
source_policy="collision_only"
```

This is deliberate. Visual-preferred and explicit optical asset policies need a
real visual/optical asset pipeline and should not be faked from collision data.

Default collision-derived roles:

```text
{"depth", "lidar", "segmentation"}
```

RGB is intentionally omitted for collision-derived instances. If a caller wants
collision geometry to drive RGB, it can pass a custom `default_roles` value.

## 3. Supported Geometry Conversion

Supported:

- `BoxShape`, `ConvexHullShape`, `CylinderShape`, and other shapes exposing
  `face_topology()` are triangulated into `OpticalTriangleMeshGeometry`;
- `MeshShape` with loaded `vertices` and triangle `faces`;
- `HalfSpaceShape` as `OpticalPlaneGeometry`.

Deferred with diagnostics:

- `SphereShape`;
- `CapsuleShape`;
- `MeshShape` without loaded triangle faces;
- any shape with no triangle/plane representation.

The builder emits `OpticalBindingDiagnostic(severity="warning",
code="unsupported_collision_shape", ...)` instead of creating a hidden
low-fidelity approximation.

## 4. Identity And Provenance

For each supported collision shape:

- `geometry_id` is stable and derived from model/body/shape source;
- `instance_id` is stable and derived from model/body/shape source;
- `OpticalWorldRegistry.add_instance(...)` assigns stable
  `numeric_instance_id`;
- `OpticalSourceKey` records provenance:
  - model name;
  - body name/index;
  - geometry role;
  - shape index;
  - mesh URI when available.

The build result carries both maps:

```python
source_to_instance_id: dict[OpticalSourceKey, str]
instance_to_source: dict[str, OpticalSourceKey]
```

`OpticalInstanceSpec.source_key` also stores the source key so the registry
remains explainable after the build result is no longer in scope.

## 5. Transform Semantics

`ShapeInstance.origin_xyz/origin_rpy` becomes
`OpticalInstanceSpec.X_body_geometry`. Per-frame body pose composition remains
in `OpticalSceneCache`:

```text
X_world_geometry = frame.X_world[body_index] @ X_body_geometry
```

The builder does not read frames.

## 6. Tests

Added `tests/unit/optics/test_registry_builder.py`:

- collision-only box registry build;
- provenance map checks;
- builder output consumed by `OpticalSceneCache` and
  `CpuReferenceOpticalExecutor`;
- unsupported smooth shape diagnostic.

Verification:

```text
PYTHONPATH=. pytest tests/unit/optics -q
17 passed

ruff check optics tests/unit/optics
All checks passed
```

## 7. Deferred

- visual-preferred builder policy;
- explicit optical asset policy;
- static `Scene.static_geometries` builder;
- sphere/capsule triangulation policy;
- material policy beyond one default material;
- `binding_kind` / `dynamic_source_id` for non-rigid producers.
