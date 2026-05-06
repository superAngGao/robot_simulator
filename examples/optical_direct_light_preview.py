"""CPU optical direct-light preview.

This example renders a tiny optical scene with the in-repo CPU ray tracer:

    python examples/optical_direct_light_preview.py
    python examples/optical_direct_light_preview.py --width 640 --height 400 --out out/optical

It does not require Rerun. The script writes PNG previews for RGB, projected
depth, numeric instance segmentation, and a three-panel contact sheet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from optics import (
    CpuDirectLightOpticalExecutor,
    OpticalFrameInputs,
    OpticalInstanceSpec,
    OpticalLightSpec,
    OpticalMaterialSpec,
    OpticalSceneCache,
    OpticalWorldRegistry,
)
from physics.publish import CpuPublishedFrame
from physics.spatial import SpatialTransform
from sensing import OpticalPinholeCameraSpec, build_pinhole_camera_image_result, build_pinhole_camera_rays


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = render_preview(width=args.width, height=args.height)
    outputs = write_preview_images(image, out_dir)

    print("Wrote optical preview:")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


def render_preview(*, width: int = 480, height: int = 320):
    """Render a small CPU optical scene and return an OpticalCameraImageResult."""
    registry = build_preview_registry()
    frame = make_empty_frame(frame_id=1, sim_time=0.016)
    snapshot = OpticalSceneCache(registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(frame),
        acceleration="cpu_bvh",
    )
    camera = make_preview_camera(width=width, height=height, frame_id=frame.frame_id, sim_time=frame.sim_time)
    rays = build_pinhole_camera_rays(camera)
    result = CpuDirectLightOpticalExecutor(
        ambient_rgb=(0.06, 0.065, 0.075),
        background_rgb=(0.02, 0.025, 0.035),
    ).execute(snapshot, rays)
    return build_pinhole_camera_image_result(result, camera, rays=rays)


def build_preview_registry() -> OpticalWorldRegistry:
    """Create floor, wall, two cube meshes, and simple lights."""
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_floor", albedo_rgb=(0.72, 0.72, 0.68)))
    registry.add_material(OpticalMaterialSpec("mat_red", albedo_rgb=(0.9, 0.18, 0.12)))
    registry.add_material(OpticalMaterialSpec("mat_teal", albedo_rgb=(0.08, 0.65, 0.75)))
    registry.add_material(OpticalMaterialSpec("mat_wall", albedo_rgb=(0.52, 0.55, 0.62)))

    registry.add_plane_geometry("floor", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_plane_geometry("back_wall", normal_local=[0.0, -1.0, 0.0], point_local=[0.0, 2.2, 0.0])
    vertices, triangles = cube_mesh(size=1.0)
    registry.add_triangle_mesh_geometry("cube", vertices_local=vertices, triangles=triangles)

    visible_roles = frozenset({"rgb", "depth", "segmentation"})
    registry.add_instance(OpticalInstanceSpec("floor", "floor", "mat_floor", roles=visible_roles))
    registry.add_instance(OpticalInstanceSpec("back_wall", "back_wall", "mat_wall", roles=visible_roles))
    registry.add_instance(
        OpticalInstanceSpec(
            "red_cube",
            "cube",
            "mat_red",
            X_body_geometry=SpatialTransform(np.eye(3), np.array([-0.7, 0.05, 0.55])),
            roles=visible_roles,
        )
    )
    registry.add_instance(
        OpticalInstanceSpec(
            "teal_cube",
            "cube",
            "mat_teal",
            X_body_geometry=SpatialTransform(np.eye(3), np.array([0.75, 0.45, 0.38])),
            roles=visible_roles,
        )
    )

    registry.add_light(
        OpticalLightSpec(
            "sun",
            kind="directional",
            position_or_direction_world=np.array([-0.45, -0.65, 1.0]),
            intensity=1.7,
            color_rgb=(1.0, 0.96, 0.9),
        )
    )
    registry.add_light(
        OpticalLightSpec(
            "fill",
            kind="point",
            position_or_direction_world=np.array([2.5, -3.0, 2.2]),
            intensity=4.5,
            color_rgb=(0.55, 0.7, 1.0),
        )
    )
    return registry


def make_preview_camera(
    *,
    width: int,
    height: int,
    frame_id: int,
    sim_time: float,
) -> OpticalPinholeCameraSpec:
    eye = np.array([3.2, -5.0, 2.2], dtype=np.float64)
    target = np.array([0.0, 0.25, 0.55], dtype=np.float64)
    focal = 0.75 * float(width)
    return OpticalPinholeCameraSpec(
        frame_id=frame_id,
        sim_time=sim_time,
        env_idx=0,
        sensor_id="preview_cam",
        width=width,
        height=height,
        fx=focal,
        fy=focal,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        X_world_camera=SpatialTransform(look_at_camera_R(eye, target), eye),
        max_distance=20.0,
        sensor_role="rgb",
    )


def write_preview_images(
    image,
    out_dir: Path,
    *,
    rgb_title: str = "CPU direct-light RGB",
) -> dict[str, Path]:
    rgb = np.asarray(image.channel("rgb"), dtype=np.float64)
    depth = np.asarray(image.channel("depth_m"), dtype=np.float64)
    segmentation = np.asarray(image.channel("numeric_instance_id"), dtype=np.int64)

    rgb_preview = linear_rgb_to_preview_uint8(rgb)
    depth_preview = metric_depth_to_preview_uint8(depth)
    segmentation_preview = segmentation_to_preview_uint8(segmentation)

    paths = {
        "rgb": out_dir / "rgb.png",
        "depth": out_dir / "depth_m.png",
        "segmentation": out_dir / "numeric_instance_id.png",
        "panel": out_dir / "panel.png",
    }
    Image.fromarray(rgb_preview).save(paths["rgb"])
    Image.fromarray(depth_preview).save(paths["depth"])
    Image.fromarray(segmentation_preview).save(paths["segmentation"])
    write_panel(rgb_preview, depth_preview, segmentation_preview, paths["panel"], rgb_title=rgb_title)
    return paths


def linear_rgb_to_preview_uint8(rgb: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    display = np.clip(finite, 0.0, 1.0) ** (1.0 / 2.2)
    return np.rint(display * 255.0).astype(np.uint8)


def metric_depth_to_preview_uint8(depth: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(depth)
    preview = np.zeros(depth.shape, dtype=np.float64)
    if np.any(finite_mask):
        lo, hi = np.percentile(depth[finite_mask], [2.0, 98.0])
        preview[finite_mask] = 1.0 - np.clip((depth[finite_mask] - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return np.rint(preview * 255.0).astype(np.uint8)


def segmentation_to_preview_uint8(segmentation: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [25, 25, 30],
            [245, 143, 51],
            [62, 162, 84],
            [147, 110, 196],
            [150, 91, 76],
            [65, 156, 210],
        ],
        dtype=np.uint8,
    )
    return palette[np.mod(np.maximum(segmentation, 0), len(palette))]


def write_panel(
    rgb: np.ndarray,
    depth: np.ndarray,
    segmentation: np.ndarray,
    path: Path,
    *,
    rgb_title: str = "CPU direct-light RGB",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=140)
    panels = (
        (rgb, rgb_title, None),
        (depth, "projected depth_m", "magma"),
        (segmentation, "numeric_instance_id", None),
    )
    for ax, (data, title, cmap) in zip(axes, panels, strict=True):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout(pad=0.6)
    fig.savefig(path)
    plt.close(fig)


def cube_mesh(*, size: float) -> tuple[np.ndarray, np.ndarray]:
    h = float(size) / 2.0
    vertices = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, triangles


def look_at_camera_R(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Return R_world_camera for OpenCV-style +Z optical-axis cameras."""
    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    up_world = np.asarray(up, dtype=np.float64)
    x_axis = np.cross(z_axis, up_world)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def make_empty_frame(*, frame_id: int, sim_time: float) -> CpuPublishedFrame:
    return CpuPublishedFrame(
        frame_id=frame_id,
        sim_time=sim_time,
        step_index=frame_id,
        env_mask=None,
        q=np.zeros(0, dtype=np.float64),
        qdot=np.zeros(0, dtype=np.float64),
        X_world=[],
        v_bodies=[],
        contact_count=0,
        contacts=[],
        telemetry=None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=480, help="camera image width in pixels")
    parser.add_argument("--height", type=int, default=320, help="camera image height in pixels")
    parser.add_argument("--out", type=Path, default=Path("out/optical_direct_light"), help="output directory")
    return parser.parse_args()


if __name__ == "__main__":
    main()
