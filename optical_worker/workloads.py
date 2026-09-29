"""Materializers for the exact Q54 P12.4 logical workloads.

Every result-capable native input starts here.  The generic scene/replay
constructors remain useful for diagnostics, but their output is accepted by a
result lane only when it matches the identities frozen in
``workload_contract.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from optics.registry import (
    OpticalInstanceSpec,
    OpticalLightSpec,
    OpticalMaterialSpec,
    OpticalWorldRegistry,
)
from optics.scene_contract import PackedOpticalSceneHost
from physics.spatial import SpatialTransform

from .assets import AssetManifestEvidence
from .cases import get_case
from .native_input import CanonicalFixedRenderFixtureHost, CanonicalTransformReplayHost
from .workload_contract import frozen_workload_content

ASYMMETRIC_FIXED_TRANSFORM_ID = "asymmetric_fixed_world_v1"
GO2_FIXED_TRANSFORM_ID = "go2_static_asset_world_v1"


def build_canonical_packed_scene(
    case_id: str,
    *,
    go2_model_dir: Path | None = None,
    asset_evidence: AssetManifestEvidence | None = None,
) -> PackedOpticalSceneHost:
    """Build and verify the one packed logical scene frozen for ``case_id``."""

    case = get_case(case_id)
    if case_id == "asymmetric_synthetic_7x3_v1":
        registry = _asymmetric_registry()
    elif case_id in {
        "synthetic_dynamic_rigid_960x640_v1",
        "physics_body_triangle_960x640_v1",
    }:
        registry = _body_triangle_registry()
    elif case_id in {"go2_static_960x640_v1", "go2_static_1920x1080_v1"}:
        if go2_model_dir is None or asset_evidence is None:
            raise ValueError("canonical Go2 scene requires preflighted model_dir and asset evidence")
        expected = frozen_workload_content(case_id)
        if (
            asset_evidence.source_revision != expected.asset_source_revision
            or asset_evidence.aggregate_sha256 != expected.asset_aggregate_sha256
        ):
            raise ValueError("Go2 asset evidence does not match the frozen workload source")
        from optics.mjcf_visual import import_mjcf_visual_scene

        registry = import_mjcf_visual_scene(
            Path(go2_model_dir), model_xml="go2.xml"
        ).registry
    else:  # pragma: no cover - get_case owns the public case set
        raise ValueError(f"no canonical scene materializer for {case_id!r}")

    packed = PackedOpticalSceneHost.from_registry(
        registry,
        source_contract_id=case.case_id,
        source_contract_sha256=case.sha256,
        scene_id=case.scene_id,
    )
    expected = frozen_workload_content(case_id)
    if packed.scene_contract_sha256 != expected.scene_contract_sha256:
        raise ValueError("materialized scene does not match the frozen workload digest")
    return packed


def build_canonical_transform_replay(case_id: str) -> CanonicalTransformReplayHost:
    """Build and verify the exact replay sequence for an R1/R2 case."""

    case = get_case(case_id)
    frame_count = int(case.transform_sequence.get("frames", 0))
    if case_id == "synthetic_dynamic_rigid_960x640_v1":
        rotations = np.broadcast_to(
            np.eye(3, dtype=np.float32), (frame_count, 1, 3, 3)
        ).copy()
        translations = np.zeros((frame_count, 1, 3), dtype=np.float32)
        offsets = tuple(case.transform_sequence["z_offset_m_by_frame_mod_4"])
        translations[:, 0, 2] = np.asarray(
            [offsets[index % len(offsets)] for index in range(frame_count)],
            dtype=np.float32,
        )
    elif case_id == "go2_static_1920x1080_v1":
        rotations = np.empty((frame_count, 0, 3, 3), dtype=np.float32)
        translations = np.empty((frame_count, 0, 3), dtype=np.float32)
    else:
        raise ValueError(f"case {case_id!r} has no frozen native replay")
    replay = CanonicalTransformReplayHost.from_arrays(
        case_id=case_id,
        x_world_R=rotations,
        x_world_r=translations,
    )
    if replay.transform_replay_sha256 != frozen_workload_content(
        case_id
    ).transform_replay_sha256:
        raise ValueError("materialized replay does not match the frozen workload digest")
    return replay


def build_canonical_fixed_fixture(
    packed_scene: PackedOpticalSceneHost,
) -> CanonicalFixedRenderFixtureHost:
    """Build and verify the exact fixed world/BVH fixture for an R0 case."""

    case_id = packed_scene.source_contract_id
    if case_id == "asymmetric_synthetic_7x3_v1":
        transform_id = ASYMMETRIC_FIXED_TRANSFORM_ID
    elif case_id in {"go2_static_960x640_v1", "go2_static_1920x1080_v1"}:
        transform_id = GO2_FIXED_TRANSFORM_ID
    else:
        raise ValueError(f"case {case_id!r} has no frozen fixed render fixture")
    fixture = CanonicalFixedRenderFixtureHost.from_scene_contract(
        packed_scene,
        fixed_transform_id=transform_id,
        x_world_R=np.empty((0, 3, 3), dtype=np.float32),
        x_world_r=np.empty((0, 3), dtype=np.float32),
    )
    if fixture.fixed_render_fixture_sha256 != frozen_workload_content(
        case_id
    ).fixed_render_fixture_sha256:
        raise ValueError("materialized fixed fixture does not match the frozen workload digest")
    return fixture


def _body_triangle_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_body", albedo_rgb=(0.9, 0.35, 0.2)))
    registry.add_light(
        OpticalLightSpec(
            "dynamic_smoke_sun",
            "directional",
            position_or_direction_world=[0.1, -0.2, 1.0],
            intensity=1.5,
        )
    )
    registry.add_triangle_mesh_geometry(
        "body_tri",
        vertices_local=[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
        triangles=[[0, 1, 2]],
    )
    registry.add_instance(
        OpticalInstanceSpec(
            "body_tri",
            "body_tri",
            "mat_body",
            body_index=0,
            X_body_geometry=SpatialTransform.from_translation(
                np.array([0.0, 0.0, 0.25], dtype=np.float64)
            ),
        )
    )
    return registry


def _asymmetric_registry() -> OpticalWorldRegistry:
    registry = OpticalWorldRegistry()
    registry.add_light(
        OpticalLightSpec(
            "directional_oblique_v1",
            "directional",
            position_or_direction_world=[0.35, -0.2, 1.0],
            intensity=1.4,
            color_rgb=(0.95, 0.9, 0.8),
        )
    )
    registry.add_light(
        OpticalLightSpec(
            "point_offset_v1",
            "point",
            position_or_direction_world=[1.7, -0.9, 1.2],
            intensity=4.0,
            color_rgb=(0.6, 0.75, 1.0),
        )
    )
    for py in range(3):
        for px in range(7):
            index = py * 7 + px
            material_id = f"mat_{index:02d}"
            geometry_id = f"tri_{index:02d}"
            x0 = (px - 3) * 0.31 + py * 0.017
            y0 = (py - 1) * 0.29 + px * 0.011
            z0 = 2.4 + 0.037 * index
            registry.add_material(
                OpticalMaterialSpec(
                    material_id,
                    albedo_rgb=(
                        0.08 + 0.035 * px,
                        0.11 + 0.09 * py,
                        0.17 + 0.019 * index,
                    ),
                )
            )
            registry.add_triangle_mesh_geometry(
                geometry_id,
                vertices_local=[
                    [x0, y0, z0],
                    [x0 + 0.23 + px * 0.003, y0 + 0.01, z0 + 0.006],
                    [x0 + 0.02, y0 + 0.21 + py * 0.007, z0 - 0.004],
                ],
                triangles=[[0, 1, 2]],
            )
            registry.add_instance(
                OpticalInstanceSpec(
                    f"instance_{index:02d}",
                    geometry_id,
                    material_id,
                    roles=frozenset({"rgb", "depth", "segmentation"}),
                )
            )
    return registry
