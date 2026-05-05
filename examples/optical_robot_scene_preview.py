"""Render and save a robot-like optical scene preview.

This example uses the same scene generator as the L5C benchmark harness:

    python examples/optical_robot_scene_preview.py
    python examples/optical_robot_scene_preview.py --detail dense --width 640 --height 420

The default `preview` detail is intentionally moderate so CPU BVH/direct-light
rendering finishes quickly. Benchmark-scale `dense`/`xlarge` scenes should be
measured with `benchmarks/bench_optical_device_scene.py`.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.robot_optical_scene import build_robot_optical_scene, make_robot_camera
from examples.optical_direct_light_preview import write_preview_images
from optics import CpuDirectLightOpticalExecutor, OpticalFrameInputs, OpticalSceneCache
from sensing import build_pinhole_camera_image_result, build_pinhole_camera_rays


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image, scene_triangles = render_robot_preview(
        width=args.width,
        height=args.height,
        detail=args.detail,
        num_robots=args.num_robots,
        view=args.view,
    )
    outputs = write_preview_images(image, out_dir)

    print(f"Wrote robot optical preview ({scene_triangles} triangles):")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


def render_robot_preview(
    *,
    width: int = 480,
    height: int = 320,
    detail: str = "preview",
    num_robots: int = 1,
    view: str = "front",
):
    scene = build_robot_optical_scene(detail=detail, num_robots=num_robots)
    snapshot = OpticalSceneCache(scene.registry).snapshot_from_frame_inputs(
        OpticalFrameInputs.from_published_frame(scene.cpu_frame),
        acceleration="cpu_bvh",
    )
    camera = make_robot_camera(
        width=width,
        height=height,
        frame_id=scene.cpu_frame.frame_id,
        sim_time=scene.cpu_frame.sim_time,
        num_robots=num_robots,
        view=view,
        sensor_role="rgb",
    )
    rays = build_pinhole_camera_rays(camera)
    result = CpuDirectLightOpticalExecutor(
        ambient_rgb=(0.08, 0.085, 0.09),
        background_rgb=(0.025, 0.03, 0.038),
    ).execute(snapshot, rays)
    return build_pinhole_camera_image_result(result, camera, rays=rays), scene.num_triangles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--detail", choices=("proxy", "preview", "dense", "xlarge"), default="preview")
    parser.add_argument("--num-robots", type=int, default=1)
    parser.add_argument("--view", choices=("front", "top", "ego"), default="front")
    parser.add_argument("--out", default="out/optical_robot_scene")
    return parser.parse_args()


if __name__ == "__main__":
    main()
