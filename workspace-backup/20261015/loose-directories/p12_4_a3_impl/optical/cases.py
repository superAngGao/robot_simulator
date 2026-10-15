"""Canonical workload definitions for Q54 P12.4 optical benchmarks.

The definitions in this module are dependency-light on purpose.  Benchmark
lanes independently materialize these logical cases as Warp or Torch CUDA
inputs; neither lane is allowed to derive its input from the other backend.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

CASE_SCHEMA_VERSION = "q54.p12.4.case.v1"


@dataclass(frozen=True)
class CanonicalOpticalCase:
    """Backend-independent logical input for one optical benchmark family."""

    case_id: str
    family: str
    scene_id: str
    frame_source: str
    geometry_mode: str
    width: int
    height: int
    camera: dict[str, Any]
    transform_sequence: dict[str, Any]
    lighting: dict[str, Any]
    render_contract: dict[str, Any]
    seed: int = 0
    required_asset_manifest: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = CASE_SCHEMA_VERSION
        return payload

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_payload())).hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    """Serialize JSON deterministically for case/config content hashes."""

    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_cases() -> tuple[CanonicalOpticalCase, ...]:
    """Return the frozen P12.4 canonical workload families."""

    common_render = {
        "ray_layout": "row_major_py_times_width_plus_px",
        "ray_dtype": "float32",
        "scene_dtype": "float32",
        "miss_distance": "positive_infinity",
        "tie_break": "source_order_key",
        "sensor_role": "rgb",
    }
    go2_camera = {
        "builder": "tools.optical_pipeline_lab.camera_presets.build_lab_video_camera",
        "mode": "camera_orbit",
        "frames": 120,
        "fps": 30.0,
        "focal_pixels_per_width": 0.72,
        "principal_point": "pixel_center",
        "max_distance_m": 20.0,
    }
    return (
        CanonicalOpticalCase(
            case_id="asymmetric_synthetic_7x3_v1",
            family="asymmetric_synthetic_correctness",
            scene_id="asymmetric_triangle_material_grid_v1",
            frame_source="canonical_host_definition",
            geometry_mode="static",
            width=7,
            height=3,
            camera={
                "builder": "explicit_pinhole",
                "fx": 5.25,
                "fy": 7.75,
                "cx": 1.3,
                "cy": 0.8,
                "translation_world": [0.31, -0.27, 0.43],
                "rotation": "non_identity_compound_xyz_v1",
                "max_distance_m": 50.0,
            },
            transform_sequence={"kind": "single_static_frame", "frames": 1},
            lighting={
                "ambient_rgb": [0.03, 0.05, 0.07],
                "lights": ["directional_oblique_v1", "point_offset_v1"],
            },
            render_contract={
                **common_render,
                "scene_asymmetry": "per_pixel_geometry_and_material",
                "required_profiles": ["direct_light_full", "rgb_preview", "render_only"],
                "shadow_modes": [False, True],
            },
            seed=54001,
        ),
        CanonicalOpticalCase(
            case_id="go2_static_960x640_v1",
            family="go2_static_historical_anchor",
            scene_id="mujoco_menagerie_unitree_go2_visual",
            frame_source="static_asset_builder",
            geometry_mode="static",
            width=960,
            height=640,
            camera=dict(go2_camera),
            transform_sequence={"kind": "static_asset", "frames": 120, "fps": 30.0},
            lighting={"source": "go2_preview_direct_light_v1"},
            render_contract={
                **common_render,
                "required_profiles": ["direct_light_full", "rgb_preview", "render_only"],
                "shadow_modes": [False, True],
            },
            seed=54002,
            required_asset_manifest="mujoco_menagerie/unitree_go2@run_manifest",
        ),
        CanonicalOpticalCase(
            case_id="go2_static_1920x1080_v1",
            family="go2_static_product_target",
            scene_id="mujoco_menagerie_unitree_go2_visual",
            frame_source="static_asset_builder",
            geometry_mode="static",
            width=1920,
            height=1080,
            camera=dict(go2_camera),
            transform_sequence={"kind": "static_asset", "frames": 120, "fps": 30.0},
            lighting={"source": "go2_preview_direct_light_v1"},
            render_contract={
                **common_render,
                "required_profiles": ["direct_light_full", "rgb_preview", "render_only"],
                "shadow_modes": [False, True],
            },
            seed=54003,
            required_asset_manifest="mujoco_menagerie/unitree_go2@run_manifest",
        ),
        CanonicalOpticalCase(
            case_id="synthetic_dynamic_rigid_960x640_v1",
            family="deterministic_dynamic_rigid_replay",
            scene_id="synthetic_body_triangle",
            frame_source="independent_native_replay",
            geometry_mode="dynamic_rigid",
            width=960,
            height=640,
            camera={
                "builder": "tools.optical_pipeline_lab.camera_presets.build_model_bounds_camera",
                "mode": "fixed_view",
                "view": "front",
                "focal_pixels_per_width": 0.72,
                "max_distance_m": 20.0,
            },
            transform_sequence={
                "kind": "body_triangle_z_cycle_v1",
                "frames": 120,
                "fps": 30.0,
                "z_offset_m_by_frame_mod_4": [0.0, 0.04, 0.08, 0.12],
            },
            lighting={"source": "dynamic_smoke_sun"},
            render_contract={
                **common_render,
                "required_profiles": ["direct_light_full", "rgb_preview", "render_only"],
                "shadow_modes": [False, True],
            },
            seed=54004,
        ),
        CanonicalOpticalCase(
            case_id="physics_body_triangle_960x640_v1",
            family="gpu_engine_physics_to_image",
            scene_id="physics_body_triangle_video_smoke",
            frame_source="physics_published_frame",
            geometry_mode="dynamic_rigid",
            width=960,
            height=640,
            camera={
                "builder": "tools.optical_pipeline_lab.camera_presets.build_model_bounds_camera",
                "mode": "fixed_view",
                "view": "front",
                "focal_pixels_per_width": 0.72,
                "max_distance_m": 20.0,
            },
            transform_sequence={
                "kind": "gpu_engine_scripted_height_v1",
                "frames": 120,
                "physics_dt_s": 0.0001,
                "height_m_by_frame_mod_4": [0.50, 0.54, 0.58, 0.62],
                "reset_between_backend_lanes": True,
            },
            lighting={"source": "dynamic_smoke_sun"},
            render_contract={
                **common_render,
                "required_profiles": ["direct_light_full", "rgb_preview", "render_only"],
                "shadow_modes": [False, True],
                "publication_qos": "lossless_device_borrow",
            },
            seed=54005,
        ),
    )


def cases_by_id() -> dict[str, CanonicalOpticalCase]:
    cases = canonical_cases()
    return {case.case_id: case for case in cases}


def case_hashes() -> dict[str, str]:
    return {case.case_id: case.sha256 for case in canonical_cases()}


def get_case(case_id: str) -> CanonicalOpticalCase:
    try:
        return cases_by_id()[case_id]
    except KeyError as exc:
        expected = ", ".join(sorted(cases_by_id()))
        raise KeyError(f"Unknown canonical optical case {case_id!r}; expected one of: {expected}") from exc
