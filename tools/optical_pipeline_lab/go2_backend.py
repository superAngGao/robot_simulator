"""Go2 MuJoCo Menagerie backend for the Optical Pipeline Lab.

This backend imports the same visual mesh geoms as
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
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.mujoco_menagerie_robot_preview import (
    _look_at_camera_R,
    import_mjcf_visual_scene,
    make_model_camera,
)
from examples.optical_direct_light_preview import write_preview_images
from optics import (
    DeviceOpticalSceneCache,
    refit_device_bvh_from_snapshot,
    stage_optical_compute_result_to_host,
)
from physics.publish import GpuPublishedFrame
from physics.spatial import SpatialTransform
from sensing import OpticalPinholeCameraSpec, build_pinhole_camera_image_result, build_pinhole_camera_rays
from tools.optical_pipeline_lab import dynamic_frames, video_loop
from tools.optical_pipeline_lab import render_session as _render_session
from tools.optical_pipeline_lab.scenarios import DEFAULT_RENDER_HEIGHT, DEFAULT_RENDER_WIDTH
from tools.optical_pipeline_lab.timing import (
    FrameTimingRecorder,
    TimingRecorder,
)

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - example-only guard.
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


OpticalLabRenderFrameContext = _render_session.OpticalLabRenderFrameContext
OpticalLabRenderOptions = _render_session.OpticalLabRenderOptions
OpticalLabRenderPipeline = _render_session.OpticalLabRenderPipeline
OpticalLabRenderSession = _render_session.OpticalLabRenderSession
OpticalLabRenderSource = _render_session.OpticalLabRenderSource
OpticalLabRenderWorkspace = _render_session.OpticalLabRenderWorkspace

# Generic video helpers live in video_loop; Go2 keeps private aliases for lab compatibility.
_include_shadow_traversal_stats = video_loop.include_shadow_traversal_stats
_pack_video_rgb8 = video_loop.pack_video_rgb8
_render_profile_buffer_for_request = video_loop.render_profile_buffer_for_request
_render_profile_row = video_loop.render_profile_row
_render_profile_row_from_timing = video_loop.render_profile_row_from_timing
_video_camera_for_frame_inputs = video_loop.video_camera_for_frame_inputs
_video_delivery_request = video_loop.video_delivery_request_from_options
_video_frame_inputs = video_loop.video_frame_inputs
_video_geometry_mode = video_loop.video_geometry_mode
_video_rays_for_camera = video_loop.video_rays_for_camera
_video_readback_channels = video_loop.video_readback_channels
_video_render_request = video_loop.video_render_request


def main() -> None:
    args = _parse_args()
    if wp is None:
        raise SystemExit(
            "mujoco_menagerie_gpu_preview.py requires warp with CUDA support"
        ) from _WARP_IMPORT_ERROR
    render_many_views(args)


def render_many_views(args: argparse.Namespace) -> None:
    if wp is None:
        raise SystemExit(
            "mujoco_menagerie_gpu_preview.py requires warp with CUDA support"
        ) from _WARP_IMPORT_ERROR
    timings = TimingRecorder()
    total_start = time.perf_counter()
    options = _render_options_from_args(args)
    pipeline = OpticalLabRenderPipeline.create_from_source_factory(
        lambda workspace: build_go2_render_source(args, workspace=workspace),
        options,
        timings,
        scene_for_source=_scene_from_render_source,
        pack_rgb8=_pack_video_rgb8,
        render_profile_buffer_for_request=_render_profile_buffer_for_request,
        render_profile_row=_render_profile_row,
    )
    session = pipeline.session
    _configure_dynamic_video_frame_inputs(args, session)
    scene = session.scene

    out_dir = Path(args.out)
    with timings.measure("output_dir"):
        out_dir.mkdir(parents=True, exist_ok=True)

    setup_benchmark_ms = []
    if session.bvh.stats.supports_refit:
        setup_benchmark_ms = _run_snapshot_refit_benchmark(
            session.cache,
            session.gpu_frame,
            session.bvh,
            stream=session.stream,
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
                session.executor,
                session.snapshot,
                session.bvh,
                (warmup_camera, None),
                timings,
                use_gpu_raygen=True,
            )
            continue
        _execute_warmup_render(
            session.executor,
            session.snapshot,
            session.bvh,
            _build_camera_rays_for_view(scene, args, args.views[0], timings, phase="warmup"),
            timings,
            use_gpu_raygen=False,
        )

    if args.video_frames > 0:
        video_rows = _run_video_benchmark(
            pipeline,
            args,
            out_dir,
        )
        for row in video_rows.summary_rows():
            timings.add(f"video_{row['phase']}_mean", float(row["mean_ms"]))
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        timings.add("total", total_elapsed_ms)
        if args.timing_csv:
            timings.write_csv(Path(args.timing_csv))
        _print_setup_summary(scene, args, timings, session.bvh, total_elapsed_ms)
        _print_video_summary(args, video_rows)
        _print_timing_summary(timings, session.bvh)
        return

    rendered_views = {}
    for view in args.views:
        camera, rays = _build_camera_rays_for_view(scene, args, view, timings, phase=f"view_{view}")
        for elapsed_ms in _run_render_benchmark(
            session.executor,
            session.snapshot,
            session.bvh,
            rays,
            warmup=args.render_warmup,
            repeat=args.render_repeat,
        ):
            timings.add(f"view_{view}_render_benchmark_execute_wait", elapsed_ms)

        with timings.measure(f"view_{view}_render_execute_wait"):
            result = session.executor.execute(session.snapshot, session.bvh, rays)
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

    _print_setup_summary(scene, args, timings, session.bvh, total_elapsed_ms)
    _print_rendered_views(args, rendered_views)
    _print_timing_summary(timings, session.bvh)


def _print_setup_summary(
    scene,
    args: argparse.Namespace,
    timings: TimingRecorder,
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


def _print_timing_summary(timings: TimingRecorder, bvh) -> None:
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


def _print_video_summary(args: argparse.Namespace, video_rows: FrameTimingRecorder) -> None:
    summary = video_rows.video_summary()
    frame_inputs = getattr(args, "video_frame_inputs", None)
    summary_frame_inputs = frame_inputs[0] if frame_inputs else None
    print(
        "  video_benchmark: "
        f"frames={args.video_frames}, mode={args.video_mode}, "
        f"geometry={_video_geometry_mode(args, frame_inputs=summary_frame_inputs)}, "
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


def _build_scene_for_preset(scene_preset: str, args: argparse.Namespace):
    if scene_preset == "go2_menagerie_static":
        return import_mjcf_visual_scene(Path(args.model_dir), model_xml=args.model_xml)
    if scene_preset == "synthetic_body_triangle":
        return _synthetic_body_triangle_scene()
    raise NotImplementedError(
        f"scene_preset={scene_preset!r} is reserved; use go2_menagerie_static/synthetic_body_triangle for now"
    )


def _render_options_from_args(args: argparse.Namespace) -> OpticalLabRenderOptions:
    return OpticalLabRenderOptions(
        device=args.device,
        bvh_backend=str(args.bvh_backend),
        bvh_split_strategy=str(args.bvh_split_strategy),
        shadows=not args.no_shadows,
        verbose_warp=bool(args.verbose_warp),
    )


def build_go2_render_source(
    args: argparse.Namespace,
    *,
    workspace: OpticalLabRenderWorkspace,
) -> OpticalLabRenderSource:
    scene_preset = getattr(args, "scene_preset", "go2_menagerie_static")
    scene = _build_scene_for_preset(scene_preset, args)
    base_frame = _base_gpu_frame_for_scene(
        scene_preset,
        frame_id=scene.frame.frame_id,
        sim_time=scene.frame.sim_time,
        device=workspace.device,
    )
    return OpticalLabRenderSource(
        registry=scene.registry,
        base_frame=base_frame,
        bounds_min=scene.bounds_min,
        bounds_max=scene.bounds_max,
        metadata={
            "scene": scene,
            "scene_preset": scene_preset,
        },
    )


def _scene_from_render_source(source: OpticalLabRenderSource):
    return source.metadata["scene"]


def _synthetic_body_triangle_scene():
    registry = dynamic_frames.make_body_bound_triangle_registry()
    return SimpleNamespace(
        registry=registry,
        frame=SimpleNamespace(frame_id=0, sim_time=0.0),
        bounds_min=np.array([-0.35, -0.35, -0.05], dtype=np.float64),
        bounds_max=np.array([0.45, 0.45, 0.85], dtype=np.float64),
        num_visual_geoms=1,
        num_triangles=1,
    )


def _base_gpu_frame_for_scene(
    scene_preset: str,
    *,
    frame_id: int,
    sim_time: float,
    device,
) -> GpuPublishedFrame:
    if scene_preset == "synthetic_body_triangle":
        return dynamic_frames.make_gpu_pose_frame(
            wp_module=wp,
            translations=np.zeros((1, 1, 3), dtype=np.float32),
            frame_id=frame_id,
            sim_time=sim_time,
            device=device,
        )
    return _static_gpu_frame(frame_id=frame_id, sim_time=sim_time, device=device)


def _configure_dynamic_video_frame_inputs(args: argparse.Namespace, session: OpticalLabRenderSession) -> None:
    if getattr(args, "video_frame_inputs", None) is not None:
        return
    if getattr(args, "scene_preset", "go2_menagerie_static") != "synthetic_body_triangle":
        return
    args.video_frame_inputs = _synthetic_body_triangle_video_frame_inputs(
        session.gpu_frame,
        frames=int(args.video_frames),
        fps=float(args.video_fps),
    )
    args.video_geometry_mode = "dynamic_rigid"


def _synthetic_body_triangle_video_frame_inputs(
    base_frame: GpuPublishedFrame,
    *,
    frames: int,
    fps: float,
) -> list[GpuPublishedFrame]:
    frame_inputs: list[GpuPublishedFrame] = []
    sim_dt = 1.0 / fps if fps > 0.0 else 0.0
    for frame_index in range(max(frames, 0)):
        z_offset = 0.04 * float(frame_index % 4)
        frame_inputs.append(
            dynamic_frames.clone_and_perturb_gpu_published_pose_frame(
                base_frame,
                wp_module=wp,
                translation_offsets={(0, 0): [0.0, 0.0, z_offset]},
                frame_id=base_frame.frame_id + frame_index,
                sim_time=base_frame.sim_time + sim_dt * float(frame_index),
                step_index=base_frame.step_index + frame_index,
                slot_id=frame_index,
            )
        )
    return frame_inputs


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
    timings: TimingRecorder,
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
    timings: TimingRecorder,
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


def _render_video_frame(
    pipeline: OpticalLabRenderPipeline,
    args: argparse.Namespace,
    frame_index: int,
    ray_cache: list[tuple[OpticalPinholeCameraSpec, object]] | None,
) -> video_loop.RenderedVideoFrame:
    return video_loop.render_video_frame(
        pipeline,
        args,
        frame_index,
        ray_cache,
        build_video_camera=_build_video_camera,
    )


def _run_video_benchmark(
    pipeline: OpticalLabRenderPipeline,
    args: argparse.Namespace,
    out_dir: Path,
) -> FrameTimingRecorder:
    return video_loop.run_video_benchmark(
        pipeline,
        args,
        out_dir,
        build_video_camera=_build_video_camera,
        pack_rgb8=_pack_video_rgb8,
        synchronize_event=getattr(wp, "synchronize_event", lambda event: None),
        render_frame=_render_video_frame,
    )


def _build_torch_async_warmup_result(
    pipeline: OpticalLabRenderPipeline,
    args: argparse.Namespace,
    delivery_request,
) -> tuple[object, bool]:
    return video_loop.build_torch_async_warmup_result(
        pipeline,
        args,
        delivery_request,
        build_video_camera=_build_video_camera,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", default="out/external/mujoco_menagerie/unitree_go2")
    parser.add_argument("--model-xml", default="go2.xml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=DEFAULT_RENDER_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_RENDER_HEIGHT)
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
        choices=("full", "rgb", "rgb8", "none"),
        default="full",
        help="Host readback mode used by the video benchmark.",
    )
    parser.add_argument(
        "--video-readback-delivery",
        choices=("sync", "torch_async"),
        default="sync",
        help="Readback delivery implementation for video benchmark experiments.",
    )
    parser.add_argument(
        "--video-readback-ring-depth",
        type=int,
        default=2,
        help="Pinned host slot count for async video readback experiments.",
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
    if args.video_readback_delivery == "torch_async" and args.video_readback not in ("rgb", "rgb8"):
        parser.error("--video-readback-delivery=torch_async currently requires --video-readback=rgb/rgb8")
    if args.video_readback_ring_depth <= 0:
        parser.error("--video-readback-ring-depth must be > 0")
    return args


if __name__ == "__main__":
    main()
