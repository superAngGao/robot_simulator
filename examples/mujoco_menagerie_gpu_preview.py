"""Render a MuJoCo Menagerie robot model with the GPU optical pipeline.

This example imports the same visual mesh geoms as
`mujoco_menagerie_robot_preview.py`, then renders RGB/depth/segmentation PNGs
with `GpuDeviceBvhDirectLightOpticalExecutor` and hard shadows.

Example:

    conda run -n env_tilelang_20260119 python examples/mujoco_menagerie_gpu_preview.py \
      --model-dir out/external/mujoco_menagerie/unitree_go2 \
      --model-xml go2.xml \
      --out out/menagerie_go2_gpu_preview
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.mujoco_menagerie_robot_preview import import_mjcf_visual_scene, make_model_camera
from examples.optical_direct_light_preview import write_preview_images
from optics import (
    DeviceOpticalSceneCache,
    GpuDeviceBvhDirectLightOpticalExecutor,
    build_cuda_lbvh_from_snapshot,
    build_device_bvh_from_snapshot,
    refit_device_bvh_from_snapshot,
    stage_optical_compute_result_to_host,
)
from physics.publish import GpuPublishedFrame
from sensing import build_pinhole_camera_image_result, build_pinhole_camera_rays

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - example-only guard.
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


def main() -> None:
    args = _parse_args()
    if wp is None:
        raise SystemExit(
            "mujoco_menagerie_gpu_preview.py requires warp with CUDA support"
        ) from _WARP_IMPORT_ERROR

    timings = _TimingRecorder()
    total_start = time.perf_counter()
    with timings.measure("warp_init"):
        if not args.verbose_warp:
            wp.config.quiet = True
        wp.init()
        device = wp.get_device(args.device)

    out_dir = Path(args.out)
    with timings.measure("output_dir"):
        out_dir.mkdir(parents=True, exist_ok=True)

    with timings.measure("import_scene"):
        scene = import_mjcf_visual_scene(Path(args.model_dir), model_xml=args.model_xml)
    with timings.measure("camera_rays"):
        camera = make_model_camera(
            bounds_min=scene.bounds_min,
            bounds_max=scene.bounds_max,
            width=args.width,
            height=args.height,
            frame_id=scene.frame.frame_id,
            sim_time=scene.frame.sim_time,
            view=args.view,
        )
        rays = build_pinhole_camera_rays(camera)
    with timings.measure("gpu_frame_stream"):
        gpu_frame = _static_gpu_frame(
            frame_id=scene.frame.frame_id,
            sim_time=scene.frame.sim_time,
            device=device,
        )
        stream = wp.Stream(device=device)

    with timings.measure("device_scene_snapshot"):
        cache = DeviceOpticalSceneCache(scene.registry, device=device, stream=stream)
        snapshot = cache.snapshot_from_gpu_frame(gpu_frame, env_idx=0, stream=stream, include_aabb=True)
    with timings.measure("bvh_build"):
        if args.bvh_backend == "cuda_lbvh":
            bvh = build_cuda_lbvh_from_snapshot(snapshot, device=device, stream=stream)
        else:
            bvh = build_device_bvh_from_snapshot(
                snapshot,
                device=device,
                stream=stream,
                split_strategy=args.bvh_split_strategy,
            )
    with timings.measure("executor_init"):
        executor = GpuDeviceBvhDirectLightOpticalExecutor(
            device=device,
            stream=stream,
            shadows=not args.no_shadows,
            ambient_rgb=(0.08, 0.085, 0.09),
            background_rgb=(0.025, 0.03, 0.038),
        )

    setup_benchmark_ms = []
    if bvh.stats.supports_refit:
        setup_benchmark_ms = _run_snapshot_refit_benchmark(
            cache,
            gpu_frame,
            bvh,
            stream=stream,
            warmup=args.setup_warmup,
            repeat=args.setup_repeat,
        )
    for snapshot_ms, refit_ms in setup_benchmark_ms:
        timings.add("snapshot_benchmark", snapshot_ms)
        timings.add("refit_benchmark", refit_ms)

    render_benchmark_ms = _run_render_benchmark(
        executor,
        snapshot,
        bvh,
        rays,
        warmup=args.render_warmup,
        repeat=args.render_repeat,
    )
    for elapsed_ms in render_benchmark_ms:
        timings.add("render_benchmark_execute_wait", elapsed_ms)

    with timings.measure("render_execute_wait"):
        result = executor.execute(snapshot, bvh, rays)
        wp.synchronize_event(result.ready_event)
    with timings.measure("stage_host"):
        host_result = stage_optical_compute_result_to_host(result)
    with timings.measure("image_build"):
        image = build_pinhole_camera_image_result(
            _per_ray_result(host_result, rays.num_rays),
            camera,
            rays=rays,
        )
    with timings.measure("write_images"):
        outputs = write_preview_images(image, out_dir, rgb_title="GPU direct-light RGB")
    total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
    timings.add("total", total_elapsed_ms)
    if args.timing_csv:
        timings.write_csv(Path(args.timing_csv))

    shadow_overflow = int(host_result.channel("shadow_stack_overflow_count")[0])
    bvh_overflow = int(host_result.channel("bvh_stack_overflow_count")[0])
    if args.fail_on_overflow and (shadow_overflow or bvh_overflow):
        raise SystemExit(f"BVH stack overflow detected: primary={bvh_overflow}, shadow={shadow_overflow}")

    print(
        "Wrote GPU MuJoCo Menagerie optical preview "
        f"({scene.num_visual_geoms} visual geoms, {scene.num_triangles} triangles, "
        f"{camera.width}x{camera.height}, shadows={not args.no_shadows}):"
    )
    print(
        "  diagnostics: "
        f"primary_overflow={bvh_overflow}, "
        f"primary_max_stack={int(host_result.channel('bvh_max_stack_depth')[0])}, "
        f"shadow_overflow={shadow_overflow}, "
        f"shadow_max_stack={int(host_result.channel('shadow_max_stack_depth')[0])}, "
        f"elapsed_ms={total_elapsed_ms:.1f}"
    )
    print("  timings_ms:")
    for row in timings.summary_rows():
        if row["phase"] in {"render_benchmark_execute_wait", "snapshot_benchmark", "refit_benchmark"}:
            print(
                "    "
                f"{row['phase']}: repeat={int(row['count'])}, "
                f"p50={row['p50_ms']:.3f}, p90={row['p90_ms']:.3f}, "
                f"mean={row['mean_ms']:.3f}"
            )
        else:
            print(f"    {row['phase']}: {row['mean_ms']:.3f}")
    print(
        "  bvh_build_detail_ms: "
        f"strategy={bvh.stats.split_strategy}, "
        f"supports_refit={bvh.stats.supports_refit}, "
        f"device_to_host={bvh.stats.device_to_host_ms:.3f}, "
        f"host_build={bvh.stats.host_build_ms:.3f}, "
        f"upload={bvh.stats.upload_ms:.3f}, "
        f"total={bvh.stats.total_ms:.3f}"
    )
    for label, path in outputs.items():
        print(f"  {label}: {path}")


def _static_gpu_frame(*, frame_id: int, sim_time: float, device) -> GpuPublishedFrame:
    x_world_R = wp.zeros((1, 0, 3, 3), dtype=wp.float32, device=device)
    x_world_r = wp.zeros((1, 0, 3), dtype=wp.float32, device=device)
    return GpuPublishedFrame(
        slot_id=0,
        frame_id=frame_id,
        sim_time=sim_time,
        step_index=frame_id,
        env_mask_wp=None,
        q_wp=None,
        qdot_wp=None,
        x_world_R_wp=x_world_R,
        x_world_r_wp=x_world_r,
        v_bodies_wp=None,
        contact_count_wp=None,
        contact_cache_ref=None,
        telemetry_ref=None,
        ready_event=None,
        slot_meta=None,
    )


def _per_ray_result(result, num_rays: int):
    channels = {
        name: value
        for name, value in result.channels.items()
        if getattr(value, "shape", ())[:1] == (num_rays,)
    }
    return replace(result, channels=channels)


def _run_render_benchmark(
    executor,
    snapshot,
    bvh,
    rays,
    *,
    warmup: int,
    repeat: int,
) -> list[float]:
    for _ in range(max(warmup, 0)):
        result = executor.execute(snapshot, bvh, rays)
        wp.synchronize_event(result.ready_event)

    elapsed_ms: list[float] = []
    for _ in range(max(repeat, 0)):
        start = time.perf_counter()
        result = executor.execute(snapshot, bvh, rays)
        wp.synchronize_event(result.ready_event)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
    return elapsed_ms


def _run_snapshot_refit_benchmark(
    cache: DeviceOpticalSceneCache,
    gpu_frame: GpuPublishedFrame,
    bvh,
    *,
    stream,
    warmup: int,
    repeat: int,
) -> list[tuple[float, float]]:
    for _ in range(max(warmup, 0)):
        snapshot = cache.snapshot_from_gpu_frame(gpu_frame, env_idx=0, stream=stream, include_aabb=True)
        bvh = refit_device_bvh_from_snapshot(snapshot, bvh, stream=stream)
        wp.synchronize_event(bvh.ready_event)

    elapsed_ms: list[tuple[float, float]] = []
    for _ in range(max(repeat, 0)):
        snapshot_start = time.perf_counter()
        snapshot = cache.snapshot_from_gpu_frame(gpu_frame, env_idx=0, stream=stream, include_aabb=True)
        wp.synchronize_event(snapshot.ready_event)
        snapshot_ms = (time.perf_counter() - snapshot_start) * 1000.0

        refit_start = time.perf_counter()
        bvh = refit_device_bvh_from_snapshot(snapshot, bvh, stream=stream)
        wp.synchronize_event(bvh.ready_event)
        refit_ms = (time.perf_counter() - refit_start) * 1000.0
        elapsed_ms.append((snapshot_ms, refit_ms))
    return elapsed_ms


class _TimingRecorder:
    def __init__(self) -> None:
        self._rows: list[tuple[str, float]] = []

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(phase, (time.perf_counter() - start) * 1000.0)

    def add(self, phase: str, elapsed_ms: float) -> None:
        self._rows.append((phase, float(elapsed_ms)))

    def summary_rows(self) -> list[dict[str, float | str]]:
        phases: list[str] = []
        for phase, _ in self._rows:
            if phase not in phases:
                phases.append(phase)

        rows: list[dict[str, float | str]] = []
        for phase in phases:
            samples = [elapsed for row_phase, elapsed in self._rows if row_phase == phase]
            sorted_samples = sorted(samples)
            rows.append(
                {
                    "phase": phase,
                    "count": float(len(samples)),
                    "mean_ms": statistics.fmean(samples),
                    "p50_ms": statistics.median(sorted_samples),
                    "p90_ms": _percentile(sorted_samples, 0.90),
                    "min_ms": sorted_samples[0],
                    "max_ms": sorted_samples[-1],
                }
            )
        return rows

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ("phase", "count", "mean_ms", "p50_ms", "p90_ms", "min_ms", "max_ms")
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.summary_rows())


def _percentile(sorted_samples: list[float], q: float) -> float:
    if not sorted_samples:
        return float("nan")
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    pos = q * (len(sorted_samples) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_samples) - 1)
    weight = pos - lo
    return sorted_samples[lo] * (1.0 - weight) + sorted_samples[hi] * weight


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", default="out/external/mujoco_menagerie/unitree_go2")
    parser.add_argument("--model-xml", default="go2.xml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--view", choices=("front", "side", "top"), default="front")
    parser.add_argument("--out", default="out/menagerie_go2_gpu_preview")
    parser.add_argument("--no-shadows", action="store_true", help="Disable shadow rays.")
    parser.add_argument(
        "--bvh-backend",
        choices=("cpu", "cuda_lbvh"),
        default="cpu",
        help="BVH build backend for the initial tree.",
    )
    parser.add_argument(
        "--bvh-split-strategy",
        choices=("sort", "partition"),
        default="sort",
        help="CPU BVH median split implementation used for the initial build.",
    )
    parser.add_argument(
        "--fail-on-overflow",
        action="store_true",
        help="Exit non-zero on BVH stack overflow.",
    )
    parser.add_argument(
        "--timing-csv",
        help="Optional path for structured timing CSV output.",
    )
    parser.add_argument(
        "--render-warmup",
        type=int,
        default=0,
        help="Render-only warmup iterations after setup and before the final image render.",
    )
    parser.add_argument(
        "--render-repeat",
        type=int,
        default=0,
        help="Render-only benchmark iterations after warmup and before the final image render.",
    )
    parser.add_argument(
        "--setup-warmup",
        type=int,
        default=0,
        help="Snapshot+BVH refit warmup iterations after the initial BVH build.",
    )
    parser.add_argument(
        "--setup-repeat",
        type=int,
        default=0,
        help="Snapshot+BVH refit benchmark iterations after setup warmup.",
    )
    parser.add_argument("--verbose-warp", action="store_true", help="Allow Warp logs on stdout.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
