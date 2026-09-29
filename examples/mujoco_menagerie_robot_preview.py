"""Render an open-source MuJoCo Menagerie robot model with the optical pipeline.

This example imports visual mesh geoms from a Menagerie MJCF model and writes
RGB, depth, segmentation, and panel PNGs using the in-repo CPU optical
renderer.

Example:

    python examples/mujoco_menagerie_robot_preview.py \
      --model-dir out/external/mujoco_menagerie/unitree_go2 \
      --model-xml go2.xml \
      --out out/menagerie_go2_preview

The script intentionally keeps third-party assets outside the repository. Use
MuJoCo Menagerie's model-specific LICENSE file for attribution and reuse terms.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.optical_direct_light_preview import write_preview_images
from optics import (
    CpuDirectLightOpticalExecutor,
    OpticalFrameInputs,
    OpticalSceneCache,
)
from optics.mjcf_visual import (
    ImportedMjcfVisualScene as ImportedMjcfVisualScene,
)
from optics.mjcf_visual import (
    _add_lights as _add_lights,
)
from optics.mjcf_visual import (
    _add_materials as _add_materials,
)
from optics.mjcf_visual import (
    _add_visual_geom as _add_visual_geom,
)
from optics.mjcf_visual import (
    _axis_angle_R as _axis_angle_R,
)
from optics.mjcf_visual import (
    _body_transform as _body_transform,
)
from optics.mjcf_visual import (
    _geom_material_id as _geom_material_id,
)
from optics.mjcf_visual import (
    _home_joint_values as _home_joint_values,
)
from optics.mjcf_visual import (
    _MeshAsset as _MeshAsset,
)
from optics.mjcf_visual import (
    _parse_vec as _parse_vec,
)
from optics.mjcf_visual import (
    _quat_to_R as _quat_to_R,
)
from optics.mjcf_visual import (
    _read_mesh_assets as _read_mesh_assets,
)
from optics.mjcf_visual import (
    _walk_body as _walk_body,
)
from optics.mjcf_visual import (
    import_mjcf_visual_scene as import_mjcf_visual_scene,
)
from physics.spatial import SpatialTransform
from sensing import (
    OpticalPinholeCameraSpec,
    build_pinhole_camera_image_result,
    build_pinhole_camera_rays,
)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = import_mjcf_visual_scene(Path(args.model_dir), model_xml=args.model_xml)
    camera = make_model_camera(
        bounds_min=scene.bounds_min,
        bounds_max=scene.bounds_max,
        width=args.width,
        height=args.height,
        frame_id=scene.frame.frame_id,
        sim_time=scene.frame.sim_time,
        view=args.view,
    )
    image = render_imported_scene(scene, camera)
    outputs = write_preview_images(image, out_dir)

    license_path = Path(args.model_dir) / "LICENSE"
    print(
        "Wrote MuJoCo Menagerie optical preview "
        f"({scene.num_visual_geoms} visual geoms, {scene.num_triangles} triangles):"
    )
    if license_path.exists():
        print(f"  license: {license_path}")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


def render_imported_scene(scene: ImportedMjcfVisualScene, camera: OpticalPinholeCameraSpec):
    snapshot = OpticalSceneCache(scene.registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(scene.frame),
        acceleration="cpu_bvh",
    )
    rays = build_pinhole_camera_rays(camera)
    result = CpuDirectLightOpticalExecutor(
        ambient_rgb=(0.08, 0.085, 0.09),
        background_rgb=(0.025, 0.03, 0.038),
    ).execute(snapshot, rays)
    return build_pinhole_camera_image_result(result, camera, rays=rays)


def make_model_camera(
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    width: int,
    height: int,
    frame_id: int,
    sim_time: float,
    view: str,
) -> OpticalPinholeCameraSpec:
    center = (bounds_min + bounds_max) * 0.5
    extent = float(np.linalg.norm(bounds_max - bounds_min))
    if extent <= 1.0e-9:
        extent = 1.0
    if view == "top":
        eye = center + np.array([0.0, -0.15 * extent, 1.7 * extent], dtype=np.float64)
    elif view == "side":
        eye = center + np.array([1.6 * extent, -0.08 * extent, 0.35 * extent], dtype=np.float64)
    else:
        eye = center + np.array([1.25 * extent, -1.85 * extent, 0.75 * extent], dtype=np.float64)
    focal = 0.72 * float(width)
    return OpticalPinholeCameraSpec(
        frame_id=frame_id,
        sim_time=sim_time,
        env_idx=0,
        sensor_id="menagerie_preview_camera",
        width=int(width),
        height=int(height),
        fx=focal,
        fy=focal,
        cx=(int(width) - 1) / 2.0,
        cy=(int(height) - 1) / 2.0,
        X_world_camera=SpatialTransform(_look_at_camera_R(eye, center), eye),
        max_distance=20.0,
        sensor_role="rgb",
    )


def _look_at_camera_R(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    up_world = np.asarray(up, dtype=np.float64)
    x_axis = np.cross(z_axis, up_world)
    if np.linalg.norm(x_axis) < 1.0e-9:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", default="out/external/mujoco_menagerie/unitree_go2")
    parser.add_argument("--model-xml", default="go2.xml")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=420)
    parser.add_argument("--view", choices=("front", "side", "top"), default="front")
    parser.add_argument("--out", default="out/menagerie_go2_preview")
    return parser.parse_args()


if __name__ == "__main__":
    main()
