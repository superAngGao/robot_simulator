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
import math
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.mujoco_menagerie_robot_preview import (
    _look_at_camera_R,
    import_mjcf_visual_scene,
    make_model_camera,
)
from examples.optical_direct_light_preview import linear_rgb_to_preview_uint8, write_preview_images
from optics import (
    DeviceOpticalSceneCache,
    GpuDeviceBvhDirectLightOpticalExecutor,
    OpticalOutputProfile,
    build_cuda_lbvh_from_snapshot,
    build_device_bvh_from_snapshot,
    refit_device_bvh_from_snapshot,
    stage_optical_channels,
    stage_optical_compute_result_to_host,
)
from physics.publish import GpuPublishedFrame
from physics.spatial import SpatialTransform
from sensing import OpticalPinholeCameraSpec, build_pinhole_camera_image_result, build_pinhole_camera_rays

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - example-only guard.
    Image = None
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - example-only guard.
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None

_VIDEO_DIAGNOSTIC_CHANNELS = (
    "bvh_stack_overflow_count",
    "shadow_stack_overflow_count",
    "bvh_max_stack_depth",
    "shadow_max_stack_depth",
)
_VIDEO_RGB_CHANNELS = ("rgb",) + _VIDEO_DIAGNOSTIC_CHANNELS
_RENDER_PROFILE_PHASES = (
    "host_ray_upload",
    "raygen_camera_params",
    "raygen_buffer_alloc",
    "raygen_kernel",
    "first_hit_wait_inputs",
    "first_hit_output_alloc",
    "first_hit_init_outputs",
    "first_hit_alloc_hit_mask",
    "first_hit_alloc_range_m",
    "first_hit_alloc_position_world",
    "first_hit_alloc_normal_world",
    "first_hit_alloc_ids",
    "first_hit_alloc_diagnostics",
    "first_hit_kernel",
    "shade_wait_geometry",
    "shade_output_alloc",
    "shade_alloc_rgb",
    "shade_alloc_intensity",
    "shade_alloc_diagnostics",
    "shade_kernel",
)
_NAN = float("nan")


def main() -> None:
    args = _parse_args()
    if wp is None:
        raise SystemExit(
            "mujoco_menagerie_gpu_preview.py requires warp with CUDA support"
        ) from _WARP_IMPORT_ERROR
    _render_many_views(args)


def _render_many_views(args: argparse.Namespace) -> None:
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
    for detail_name, detail_elapsed_ms in bvh.stats.detail_ms:
        timings.add(f"bvh_build_{detail_name}", detail_elapsed_ms)
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

    for _ in range(max(args.warmup_renders, 0)):
        if args.video_frames > 0 and args.video_raygen == "gpu":
            with timings.measure("warmup_camera_spec"):
                warmup_camera = make_model_camera(
                    bounds_min=scene.bounds_min,
                    bounds_max=scene.bounds_max,
                    width=args.width,
                    height=args.height,
                    frame_id=scene.frame.frame_id,
                    sim_time=scene.frame.sim_time,
                    view=args.views[0],
                )
            _execute_warmup_render(
                executor,
                snapshot,
                bvh,
                (warmup_camera, None),
                timings,
                use_gpu_raygen=True,
            )
            continue
        _execute_warmup_render(
            executor,
            snapshot,
            bvh,
            _build_camera_rays_for_view(scene, args, args.views[0], timings, phase="warmup"),
            timings,
            use_gpu_raygen=False,
        )

    if args.video_frames > 0:
        video_rows = _run_video_benchmark(
            scene,
            args,
            executor,
            snapshot,
            bvh,
            out_dir,
        )
        for row in video_rows.summary_rows():
            timings.add(f"video_{row['phase']}_mean", float(row["mean_ms"]))
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        timings.add("total", total_elapsed_ms)
        if args.timing_csv:
            timings.write_csv(Path(args.timing_csv))
        _print_setup_summary(scene, args, timings, bvh, total_elapsed_ms)
        _print_video_summary(args, video_rows)
        _print_timing_summary(timings, bvh)
        return

    rendered_views = {}
    for view in args.views:
        camera, rays = _build_camera_rays_for_view(scene, args, view, timings, phase=f"view_{view}")
        for elapsed_ms in _run_render_benchmark(
            executor,
            snapshot,
            bvh,
            rays,
            warmup=args.render_warmup,
            repeat=args.render_repeat,
        ):
            timings.add(f"view_{view}_render_benchmark_execute_wait", elapsed_ms)

        with timings.measure(f"view_{view}_render_execute_wait"):
            result = executor.execute(snapshot, bvh, rays)
            wp.synchronize_event(result.ready_event)
        with timings.measure(f"view_{view}_readback_host"):
            host_result = stage_optical_compute_result_to_host(result)
        with timings.measure(f"view_{view}_image_build"):
            image = build_pinhole_camera_image_result(
                _per_ray_result(host_result, rays.num_rays),
                camera,
                rays=rays,
            )
        view_out_dir = out_dir / view if len(args.views) > 1 else out_dir
        with timings.measure(f"view_{view}_write_images"):
            view_out_dir.mkdir(parents=True, exist_ok=True)
            outputs = write_preview_images(image, view_out_dir, rgb_title="GPU direct-light RGB")
        rendered_views[view] = {
            "camera": camera,
            "host_result": host_result,
            "outputs": outputs,
        }

    total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
    timings.add("total", total_elapsed_ms)
    if args.timing_csv:
        timings.write_csv(Path(args.timing_csv))

    _print_setup_summary(scene, args, timings, bvh, total_elapsed_ms)
    _print_rendered_views(args, rendered_views)
    _print_timing_summary(timings, bvh)


def _print_setup_summary(
    scene,
    args: argparse.Namespace,
    timings: "_TimingRecorder",
    bvh,
    total_elapsed_ms: float,
) -> None:
    print(
        "Wrote GPU MuJoCo Menagerie optical preview "
        f"({scene.num_visual_geoms} visual geoms, {scene.num_triangles} triangles, "
        f"{args.width}x{args.height}, views={','.join(args.views)}, "
        f"shadows={not args.no_shadows}, elapsed_ms={total_elapsed_ms:.1f}):"
    )


def _print_rendered_views(args: argparse.Namespace, rendered_views: dict[str, dict[str, object]]) -> None:
    for view, rendered in rendered_views.items():
        host_result = rendered["host_result"]
        shadow_overflow = int(host_result.channel("shadow_stack_overflow_count")[0])
        bvh_overflow = int(host_result.channel("bvh_stack_overflow_count")[0])
        if args.fail_on_overflow and (shadow_overflow or bvh_overflow):
            raise SystemExit(f"BVH stack overflow detected: primary={bvh_overflow}, shadow={shadow_overflow}")
        print(
            f"  {view}: "
            f"primary_overflow={bvh_overflow}, "
            f"primary_max_stack={int(host_result.channel('bvh_max_stack_depth')[0])}, "
            f"shadow_overflow={shadow_overflow}, "
            f"shadow_max_stack={int(host_result.channel('shadow_max_stack_depth')[0])}"
        )
        for label, path in rendered["outputs"].items():
            print(f"    {label}: {path}")


def _print_timing_summary(timings: "_TimingRecorder", bvh) -> None:
    print("  timings_ms:")
    for row in timings.summary_rows():
        if row["count"] > 1:
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
    if bvh.stats.detail_ms:
        print("  bvh_build_phase_ms:")
        for detail_name, detail_elapsed_ms in bvh.stats.detail_ms:
            print(f"    {detail_name}: {detail_elapsed_ms:.3f}")


def _print_video_summary(args: argparse.Namespace, video_rows: "_FrameTimingRecorder") -> None:
    summary = video_rows.video_summary()
    print(
        "  video_benchmark: "
        f"frames={args.video_frames}, mode={args.video_mode}, "
        "geometry=static, "
        f"raygen={args.video_raygen}, "
        f"ray_cache={args.video_ray_cache}, "
        f"readback={args.video_readback}, "
        f"render_profile={args.render_profile}, "
        f"write_frames={args.write_frames}, "
        f"fps_mean={summary['fps_mean']:.2f}, "
        f"frame_p50_ms={summary['frame_p50_ms']:.3f}, "
        f"frame_p90_ms={summary['frame_p90_ms']:.3f}"
    )
    if video_rows.csv_path is not None:
        print(f"    frame_timing_csv: {video_rows.csv_path}")


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


def _build_camera_rays_for_view(
    scene,
    args: argparse.Namespace,
    view: str,
    timings: "_TimingRecorder",
    *,
    phase: str,
):
    with timings.measure(f"{phase}_camera_rays"):
        camera = make_model_camera(
            bounds_min=scene.bounds_min,
            bounds_max=scene.bounds_max,
            width=args.width,
            height=args.height,
            frame_id=scene.frame.frame_id,
            sim_time=scene.frame.sim_time,
            view=view,
        )
        rays = build_pinhole_camera_rays(camera)
    return camera, rays


def _build_video_camera(scene, args: argparse.Namespace, frame_index: int) -> OpticalPinholeCameraSpec:
    if args.video_mode == "fixed_view":
        return make_model_camera(
            bounds_min=scene.bounds_min,
            bounds_max=scene.bounds_max,
            width=args.width,
            height=args.height,
            frame_id=scene.frame.frame_id,
            sim_time=scene.frame.sim_time,
            view=args.view,
        )
    if args.video_mode == "pose_sequence":
        raise SystemExit("video-mode=pose_sequence is reserved for future changing-geometry benchmarks")
    center = (scene.bounds_min + scene.bounds_max) * 0.5
    extent = float(np.linalg.norm(scene.bounds_max - scene.bounds_min))
    if extent <= 1.0e-9:
        extent = 1.0
    angle = 2.0 * math.pi * float(frame_index) / max(float(args.video_frames), 1.0)
    radius = 1.85 * extent
    eye = center + np.array(
        [
            radius * math.cos(angle - math.pi * 0.32),
            radius * math.sin(angle - math.pi * 0.32),
            0.72 * extent,
        ],
        dtype=np.float64,
    )
    focal = 0.72 * float(args.width)
    return OpticalPinholeCameraSpec(
        frame_id=scene.frame.frame_id,
        sim_time=scene.frame.sim_time,
        env_idx=0,
        sensor_id=f"menagerie_video_camera_{frame_index:06d}",
        width=int(args.width),
        height=int(args.height),
        fx=focal,
        fy=focal,
        cx=(int(args.width) - 1) / 2.0,
        cy=(int(args.height) - 1) / 2.0,
        X_world_camera=SpatialTransform(_look_at_camera_R(eye, center), eye),
        max_distance=20.0,
        sensor_role="rgb",
    )


def _execute_warmup_render(
    executor,
    snapshot,
    bvh,
    camera_and_rays,
    timings: "_TimingRecorder",
    *,
    use_gpu_raygen: bool,
) -> None:
    camera, rays = camera_and_rays
    with timings.measure("warmup_render_execute_wait"):
        result = (
            executor.execute_camera(snapshot, bvh, camera)
            if use_gpu_raygen
            else executor.execute(snapshot, bvh, rays)
        )
        wp.synchronize_event(result.ready_event)


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


def _run_video_benchmark(
    scene,
    args: argparse.Namespace,
    executor,
    snapshot,
    bvh,
    out_dir: Path,
) -> "_FrameTimingRecorder":
    if args.write_frames and Image is None:
        raise SystemExit("Writing video benchmark frames requires Pillow") from _PIL_IMPORT_ERROR

    frame_dir = out_dir / "frames"
    if args.write_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)
    if args.video_raygen == "gpu" and args.video_ray_cache != "off":
        raise SystemExit("--video-raygen gpu computes camera rays on device; use --video-ray-cache off")
    csv_path = Path(args.frame_timing_csv) if args.frame_timing_csv else out_dir / "frame_timing.csv"
    rows = _FrameTimingRecorder(csv_path=csv_path)
    rolling_window_ms: list[float] = []
    ray_cache = (
        _precompute_video_camera_rays(scene, args)
        if args.video_raygen == "host" and args.video_ray_cache == "precompute"
        else None
    )

    for frame_index in range(int(args.video_frames)):
        frame_start = time.perf_counter()

        if args.video_raygen == "gpu":
            camera = _build_video_camera(scene, args, frame_index)
            rays = None
            camera_rays_ms = _NAN
        elif ray_cache is None:
            camera_start = time.perf_counter()
            camera = _build_video_camera(scene, args, frame_index)
            rays = build_pinhole_camera_rays(camera)
            camera_rays_ms = (time.perf_counter() - camera_start) * 1000.0
        else:
            camera, rays = ray_cache[frame_index]
            camera_rays_ms = _NAN

        render_start = time.perf_counter()
        output_profile = _video_output_profile(args.video_readback)
        render_profile = [] if args.render_profile else None
        result = (
            executor.execute_camera(
                snapshot,
                bvh,
                camera,
                output_profile=output_profile,
                render_profile=render_profile,
            )
            if args.video_raygen == "gpu"
            else executor.execute(
                snapshot,
                bvh,
                rays,
                output_profile=output_profile,
                render_profile=render_profile,
            )
        )
        wp.synchronize_event(result.ready_event)
        render_execute_ms = (time.perf_counter() - render_start) * 1000.0
        render_profile_row = _render_profile_row(render_profile)

        readback_start = time.perf_counter()
        host_channels = _readback_video_result(result, args.video_readback)
        readback_host_ms = (
            (time.perf_counter() - readback_start) * 1000.0 if args.video_readback != "none" else _NAN
        )

        image_build_ms = _NAN
        encode_or_write_ms = _NAN
        frame_path = ""
        if args.write_frames:
            image_start = time.perf_counter()
            rgb_preview = _rgb_array_to_preview_uint8(host_channels["rgb"])
            image_build_ms = (time.perf_counter() - image_start) * 1000.0

            write_start = time.perf_counter()
            path = frame_dir / f"rgb_{frame_index:06d}.png"
            Image.fromarray(rgb_preview).save(path)
            encode_or_write_ms = (time.perf_counter() - write_start) * 1000.0
            frame_path = str(path)

        frame_total_ms = (time.perf_counter() - frame_start) * 1000.0
        rolling_window_ms.append(frame_total_ms)
        if len(rolling_window_ms) > 30:
            rolling_window_ms.pop(0)
        instant_fps = 1000.0 / frame_total_ms if frame_total_ms > 0.0 else float("inf")
        rolling_fps = (
            1000.0 * float(len(rolling_window_ms)) / sum(rolling_window_ms)
            if sum(rolling_window_ms) > 0.0
            else float("inf")
        )
        primary_overflow = _staged_scalar(host_channels, "bvh_stack_overflow_count")
        shadow_overflow = _staged_scalar(host_channels, "shadow_stack_overflow_count")
        primary_max_stack = _staged_scalar(host_channels, "bvh_max_stack_depth")
        shadow_max_stack = _staged_scalar(host_channels, "shadow_max_stack_depth")
        if args.fail_on_overflow and (primary_overflow or shadow_overflow):
            raise SystemExit(
                f"BVH stack overflow detected at frame {frame_index}: "
                f"primary={primary_overflow}, shadow={shadow_overflow}"
            )

        rows.add(
            {
                "frame_index": frame_index,
                "sim_time": camera.sim_time,
                "video_time": float(frame_index) / float(args.video_fps),
                "geometry_mode": "static",
                "raygen_mode": args.video_raygen,
                "ray_cache_mode": args.video_ray_cache,
                "readback_mode": args.video_readback,
                "write_mode": "rgb_png" if args.write_frames else "none",
                "camera_rays_ms": camera_rays_ms,
                "snapshot_ms": _NAN,
                "refit_ms": _NAN,
                "rebuild_ms": _NAN,
                "render_execute_ms": render_execute_ms,
                **render_profile_row,
                "readback_host_ms": readback_host_ms,
                "image_build_ms": image_build_ms,
                "encode_or_write_ms": encode_or_write_ms,
                "frame_total_ms": frame_total_ms,
                "instant_fps": instant_fps,
                "rolling_fps": rolling_fps,
                "primary_overflow": primary_overflow,
                "shadow_overflow": shadow_overflow,
                "primary_max_stack": primary_max_stack,
                "shadow_max_stack": shadow_max_stack,
                "frame_path": frame_path,
            }
        )
        if args.progress_every > 0 and (
            frame_index == 0
            or (frame_index + 1) % int(args.progress_every) == 0
            or frame_index + 1 == int(args.video_frames)
        ):
            print(
                "  video_frame "
                f"{frame_index + 1}/{args.video_frames}: "
                f"total={frame_total_ms:.3f}ms, "
                f"render={render_execute_ms:.3f}ms, "
                f"{_format_render_profile(render_profile_row)}"
                f"readback={_format_ms(readback_host_ms)}, "
                f"fps={instant_fps:.2f}, rolling_fps={rolling_fps:.2f}, "
                f"overflow=({_format_scalar(primary_overflow)},{_format_scalar(shadow_overflow)})"
            )

    rows.write_csv()
    return rows


def _precompute_video_camera_rays(
    scene,
    args: argparse.Namespace,
) -> list[tuple[OpticalPinholeCameraSpec, object]]:
    start = time.perf_counter()
    ray_cache = [
        (camera := _build_video_camera(scene, args, frame_index), build_pinhole_camera_rays(camera))
        for frame_index in range(int(args.video_frames))
    ]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(
        "  video_ray_cache: "
        f"mode=precompute, frames={args.video_frames}, elapsed_ms={elapsed_ms:.3f}, "
        f"per_frame_ms={elapsed_ms / max(int(args.video_frames), 1):.3f}"
    )
    return ray_cache


def _readback_video_result(result, readback_mode: str) -> dict[str, np.ndarray]:
    if readback_mode == "none":
        return {}
    if readback_mode == "rgb":
        return stage_optical_channels(result, _VIDEO_RGB_CHANNELS, canonical_dtypes=False)
    return stage_optical_compute_result_to_host(result).channels


def _video_output_profile(readback_mode: str) -> OpticalOutputProfile:
    if readback_mode == "rgb":
        return OpticalOutputProfile.RGB_PREVIEW
    if readback_mode == "none":
        return OpticalOutputProfile.RENDER_ONLY
    return OpticalOutputProfile.DIRECT_LIGHT_FULL


def _render_profile_row(render_profile: list[tuple[str, float]] | None) -> dict[str, float]:
    row = {f"render_{phase}_ms": _NAN for phase in _RENDER_PROFILE_PHASES}
    if render_profile is None:
        return row
    for name, elapsed_ms in render_profile:
        phase = name.removesuffix("_ms")
        key = f"render_{phase}_ms"
        if key in row:
            previous = row[key]
            row[key] = float(elapsed_ms) if math.isnan(previous) else previous + float(elapsed_ms)
    return row


def _format_render_profile(row: dict[str, float]) -> str:
    first_hit = float(row["render_first_hit_kernel_ms"])
    shade = float(row["render_shade_kernel_ms"])
    raygen = float(row["render_raygen_kernel_ms"])
    if math.isnan(first_hit) and math.isnan(shade) and math.isnan(raygen):
        return ""
    parts = []
    if not math.isnan(raygen):
        parts.append(f"raygen={raygen:.3f}ms")
    if not math.isnan(first_hit):
        parts.append(f"first_hit={first_hit:.3f}ms")
    if not math.isnan(shade):
        parts.append(f"shade={shade:.3f}ms")
    return f"profile=({', '.join(parts)}), "


def _rgb_array_to_preview_uint8(rgb: np.ndarray) -> np.ndarray:
    # Benchmark-only RGB writer: keep display conversion identical to preview PNGs
    # without constructing the full camera image result.
    return linear_rgb_to_preview_uint8(np.asarray(rgb, dtype=np.float64))


def _staged_scalar(channels: dict[str, object], name: str) -> float | int:
    if name not in channels:
        return _NAN
    return int(np.asarray(channels[name]).reshape(-1)[0])


def _format_ms(value: float) -> str:
    return "NaN" if math.isnan(float(value)) else f"{float(value):.3f}ms"


def _format_scalar(value: float | int) -> str:
    return "NaN" if isinstance(value, float) and math.isnan(value) else str(int(value))


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


class _FrameTimingRecorder:
    _FIELDNAMES = (
        "frame_index",
        "sim_time",
        "video_time",
        "geometry_mode",
        "raygen_mode",
        "ray_cache_mode",
        "readback_mode",
        "write_mode",
        "camera_rays_ms",
        "snapshot_ms",
        "refit_ms",
        "rebuild_ms",
        "render_execute_ms",
        *(f"render_{phase}_ms" for phase in _RENDER_PROFILE_PHASES),
        "readback_host_ms",
        "image_build_ms",
        "encode_or_write_ms",
        "frame_total_ms",
        "instant_fps",
        "rolling_fps",
        "primary_overflow",
        "shadow_overflow",
        "primary_max_stack",
        "shadow_max_stack",
        "frame_path",
    )

    def __init__(self, *, csv_path: Path | None) -> None:
        self.csv_path = csv_path
        self._rows: list[dict[str, float | int | str]] = []

    def add(self, row: dict[str, float | int | str]) -> None:
        self._rows.append(row)

    def summary_rows(self) -> list[dict[str, float | str]]:
        phases = (
            "camera_rays",
            "snapshot",
            "refit",
            "rebuild",
            "render_execute",
            *(f"render_{phase}" for phase in _RENDER_PROFILE_PHASES),
            "readback_host",
            "image_build",
            "encode_or_write",
            "frame_total",
        )
        rows: list[dict[str, float | str]] = []
        for phase in phases:
            key = f"{phase}_ms"
            samples = [float(row[key]) for row in self._rows if not math.isnan(float(row[key]))]
            if not samples:
                continue
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

    def video_summary(self) -> dict[str, float]:
        frame_ms = [float(row["frame_total_ms"]) for row in self._rows]
        if not frame_ms:
            return {"fps_mean": 0.0, "frame_p50_ms": 0.0, "frame_p90_ms": 0.0}
        sorted_frame_ms = sorted(frame_ms)
        total_ms = sum(frame_ms)
        return {
            "fps_mean": 1000.0 * float(len(frame_ms)) / total_ms if total_ms > 0.0 else 0.0,
            "frame_p50_ms": statistics.median(sorted_frame_ms),
            "frame_p90_ms": _percentile(sorted_frame_ms, 0.90),
        }

    def write_csv(self) -> None:
        if self.csv_path is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._rows)


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
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("front", "side", "top"),
        help="Render multiple views in one warmed session. Defaults to --view.",
    )
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
        "--warmup-renders",
        type=int,
        default=0,
        help="Full render warmup passes before writing any requested views.",
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
    parser.add_argument(
        "--video-frames",
        type=int,
        default=0,
        help="Run a warmed per-frame video benchmark instead of writing the normal preview images.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="Playback timestamp rate recorded as video_time in the per-frame CSV.",
    )
    parser.add_argument(
        "--video-mode",
        choices=("camera_orbit", "fixed_view", "pose_sequence"),
        default="camera_orbit",
        help="Video benchmark mode. pose_sequence is reserved for future changing-geometry tests.",
    )
    parser.add_argument(
        "--video-ray-cache",
        choices=("off", "precompute"),
        default="off",
        help="Cache video camera rays outside the timed frame loop.",
    )
    parser.add_argument(
        "--video-raygen",
        choices=("host", "gpu"),
        default="host",
        help="Generate video camera rays as host ray batches or inside the GPU camera path.",
    )
    parser.add_argument(
        "--video-readback",
        dest="video_readback",
        choices=("full", "rgb", "none"),
        default="full",
        help="Host readback mode used by the video benchmark.",
    )
    parser.add_argument(
        "--frame-timing-csv",
        help="Per-frame video timing CSV path. Defaults to <out>/frame_timing.csv in video mode.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print video benchmark progress every N frames; use 0 to disable.",
    )
    parser.add_argument(
        "--render-profile",
        action="store_true",
        help=(
            "Record diagnostic sub-step render timings in video mode. This inserts "
            "extra GPU synchronizations and should not be used as a throughput number."
        ),
    )
    parser.add_argument(
        "--write-frames",
        action="store_true",
        help="Write RGB PNG frames during video benchmark.",
    )
    parser.add_argument(
        "--no-write-frames",
        dest="write_frames",
        action="store_false",
        help="Do not write PNG frames during video benchmark.",
    )
    parser.set_defaults(write_frames=False)
    parser.add_argument("--verbose-warp", action="store_true", help="Allow Warp logs on stdout.")
    args = parser.parse_args()
    if args.views is None:
        args.views = [args.view]
    if args.video_frames < 0:
        parser.error("--video-frames must be >= 0")
    if args.video_fps <= 0.0:
        parser.error("--video-fps must be > 0")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if args.video_readback == "none" and args.write_frames:
        parser.error("--video-readback=none cannot be combined with --write-frames")
    if args.video_readback == "none" and args.fail_on_overflow:
        parser.error("--video-readback=none cannot honor --fail-on-overflow")
    return args


if __name__ == "__main__":
    main()
