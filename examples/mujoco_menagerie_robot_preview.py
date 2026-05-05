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
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.optical_direct_light_preview import write_preview_images
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
from sensing import (
    OpticalPinholeCameraSpec,
    build_pinhole_camera_image_result,
    build_pinhole_camera_rays,
)


@dataclass(frozen=True)
class ImportedMjcfVisualScene:
    registry: OpticalWorldRegistry
    frame: CpuPublishedFrame
    num_meshes: int
    num_visual_geoms: int
    num_triangles: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray


@dataclass(frozen=True)
class _MeshAsset:
    name: str
    path: Path
    scale: np.ndarray


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


def import_mjcf_visual_scene(model_dir: Path, *, model_xml: str) -> ImportedMjcfVisualScene:
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - optional dependency guard.
        raise SystemExit("mujoco_menagerie_robot_preview.py requires trimesh") from exc

    xml_path = model_dir / model_xml
    root = ET.parse(xml_path).getroot()
    mesh_dir = (
        model_dir / root.find("compiler").get("meshdir", "assets")
        if root.find("compiler") is not None
        else model_dir
    )

    registry = OpticalWorldRegistry()
    _add_materials(registry, root)
    _add_lights(registry)
    registry.add_plane_geometry("floor", normal_local=[0.0, 0.0, 1.0], point_local=[0.0, 0.0, 0.0])
    registry.add_instance(
        OpticalInstanceSpec("floor", "floor", "mat_floor", roles=frozenset({"rgb", "depth", "segmentation"}))
    )

    mesh_assets = _read_mesh_assets(root, mesh_dir)
    for mesh in mesh_assets.values():
        loaded = trimesh.load_mesh(mesh.path, process=False)
        if hasattr(loaded, "dump"):
            parts = loaded.dump()
            loaded = parts[0] if parts else loaded
        vertices = np.asarray(loaded.vertices, dtype=np.float64) * mesh.scale[None, :]
        faces = np.asarray(loaded.faces, dtype=np.int64)
        registry.add_triangle_mesh_geometry(mesh.name, vertices_local=vertices, triangles=faces)

    joint_values = _home_joint_values(root)
    state = {
        "mesh_assets": mesh_assets,
        "registry": registry,
        "joint_values": joint_values,
        "joint_cursor": 0,
        "visual_count": 0,
        "triangle_count": 0,
        "bounds_min": np.array([np.inf, np.inf, np.inf], dtype=np.float64),
        "bounds_max": np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64),
    }

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF model has no worldbody")
    for body in worldbody.findall("body"):
        _walk_body(body, SpatialTransform.identity(), state)

    if state["visual_count"] == 0:
        raise ValueError("No visual mesh geoms were imported")

    frame = CpuPublishedFrame(
        frame_id=1,
        sim_time=0.016,
        step_index=1,
        env_mask=None,
        q=np.zeros(0, dtype=np.float64),
        qdot=np.zeros(0, dtype=np.float64),
        X_world=[],
        v_bodies=[],
        contact_count=0,
        contacts=[],
        telemetry=None,
    )
    return ImportedMjcfVisualScene(
        registry=registry,
        frame=frame,
        num_meshes=len(mesh_assets),
        num_visual_geoms=int(state["visual_count"]),
        num_triangles=int(state["triangle_count"]),
        bounds_min=state["bounds_min"],
        bounds_max=state["bounds_max"],
    )


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


def _read_mesh_assets(root: ET.Element, mesh_dir: Path) -> dict[str, _MeshAsset]:
    assets: dict[str, _MeshAsset] = {}
    asset_elem = root.find("asset")
    if asset_elem is None:
        return assets
    for mesh_elem in asset_elem.findall("mesh"):
        filename = mesh_elem.get("file")
        if filename is None:
            continue
        name = mesh_elem.get("name") or Path(filename).stem
        scale = _parse_vec(mesh_elem.get("scale"), default=(1.0, 1.0, 1.0), length=3)
        assets[name] = _MeshAsset(name=name, path=mesh_dir / filename, scale=scale)
    return assets


def _walk_body(body: ET.Element, parent_X: SpatialTransform, state: dict[str, object]) -> None:
    body_X = parent_X @ _body_transform(body, state)
    for geom in body.findall("geom"):
        _add_visual_geom(geom, body_X, state)
    for child in body.findall("body"):
        _walk_body(child, body_X, state)


def _body_transform(body: ET.Element, state: dict[str, object]) -> SpatialTransform:
    joints = body.findall("joint")
    freejoint = body.find("freejoint")
    qpos = state["joint_values"]
    if freejoint is not None and len(qpos) >= 7:
        body_R = _quat_to_R(qpos[3:7])
        body_r = qpos[0:3]
        return SpatialTransform(body_R, body_r)

    R = _quat_to_R(_parse_vec(body.get("quat"), default=(1.0, 0.0, 0.0, 0.0), length=4))
    r = _parse_vec(body.get("pos"), default=(0.0, 0.0, 0.0), length=3)
    cursor = int(state["joint_cursor"])
    for joint in joints:
        if joint.get("type", "hinge") != "hinge":
            cursor += 1
            continue
        angle = float(qpos[7 + cursor]) if len(qpos) > 7 + cursor else 0.0
        axis = _parse_vec(joint.get("axis"), default=(0.0, 1.0, 0.0), length=3)
        R = R @ _axis_angle_R(axis, angle)
        cursor += 1
    state["joint_cursor"] = cursor
    return SpatialTransform(R, r)


def _add_visual_geom(geom: ET.Element, body_X: SpatialTransform, state: dict[str, object]) -> None:
    mesh_name = geom.get("mesh")
    if mesh_name is None or mesh_name not in state["mesh_assets"]:
        return
    geom_class = geom.get("class", "")
    group = geom.get("group")
    if geom_class and geom_class != "visual":
        return
    if group is not None and group != "2":
        return

    geom_X = body_X @ SpatialTransform(
        _quat_to_R(_parse_vec(geom.get("quat"), default=(1.0, 0.0, 0.0, 0.0), length=4)),
        _parse_vec(geom.get("pos"), default=(0.0, 0.0, 0.0), length=3),
    )
    registry: OpticalWorldRegistry = state["registry"]
    visual_index = int(state["visual_count"])
    material_id = _geom_material_id(geom)
    instance_id = f"visual_{visual_index:04d}_{mesh_name}"
    registry.add_instance(
        OpticalInstanceSpec(
            instance_id,
            mesh_name,
            material_id,
            X_body_geometry=geom_X,
            roles=frozenset({"rgb", "depth", "segmentation"}),
        )
    )
    mesh_asset: _MeshAsset = state["mesh_assets"][mesh_name]
    geom_record = registry.geometry[mesh_name]
    vertices = np.asarray(geom_record.vertices_local, dtype=np.float64)
    world_vertices = vertices @ geom_X.R.T + geom_X.r
    state["bounds_min"] = np.minimum(state["bounds_min"], np.min(world_vertices, axis=0))
    state["bounds_max"] = np.maximum(state["bounds_max"], np.max(world_vertices, axis=0))
    state["visual_count"] = visual_index + 1
    state["triangle_count"] = int(state["triangle_count"]) + int(np.asarray(geom_record.triangles).shape[0])
    _ = mesh_asset


def _add_materials(registry: OpticalWorldRegistry, root: ET.Element) -> None:
    registry.add_material(OpticalMaterialSpec("mat_floor", albedo_rgb=(0.55, 0.57, 0.54)))
    registry.add_material(OpticalMaterialSpec("mat_default", albedo_rgb=(0.8, 0.82, 0.84)))
    asset_elem = root.find("asset")
    if asset_elem is None:
        return
    for mat in asset_elem.findall("material"):
        name = mat.get("name")
        if not name:
            continue
        rgba = _parse_vec(mat.get("rgba"), default=(0.8, 0.82, 0.84, 1.0), length=4)
        registry.add_material(
            OpticalMaterialSpec(f"mat_{name}", albedo_rgb=tuple(float(v) for v in rgba[:3]))
        )


def _add_lights(registry: OpticalWorldRegistry) -> None:
    registry.add_light(
        OpticalLightSpec(
            "key",
            kind="directional",
            position_or_direction_world=np.array([-0.4, -0.55, 1.0], dtype=np.float64),
            intensity=1.8,
            color_rgb=(1.0, 0.96, 0.9),
        )
    )
    registry.add_light(
        OpticalLightSpec(
            "fill",
            kind="point",
            position_or_direction_world=np.array([2.0, -3.0, 2.0], dtype=np.float64),
            intensity=3.0,
            color_rgb=(0.55, 0.68, 1.0),
        )
    )


def _geom_material_id(geom: ET.Element) -> str:
    material = geom.get("material")
    if material:
        return f"mat_{material}"
    return "mat_default"


def _home_joint_values(root: ET.Element) -> np.ndarray:
    keyframe = root.find("keyframe")
    if keyframe is None:
        return np.empty(0, dtype=np.float64)
    key = keyframe.find("key")
    if key is None or key.get("qpos") is None:
        return np.empty(0, dtype=np.float64)
    return np.fromstring(key.get("qpos", ""), sep=" ", dtype=np.float64)


def _parse_vec(value: str | None, *, default: tuple[float, ...], length: int) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    parsed = np.fromstring(value, sep=" ", dtype=np.float64)
    if parsed.shape != (length,):
        raise ValueError(f"Expected {length} floats, got {value!r}")
    return parsed


def _quat_to_R(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 1.0e-12:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle_R(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm <= 1.0e-12:
        return np.eye(3)
    x, y, z = a / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
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
