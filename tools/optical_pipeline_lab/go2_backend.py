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
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
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
from optics.render_api import (
    DeliveryPolicy as RuntimeDeliveryPolicy,
)
from optics.render_api import (
    DeliveryRequest,
    DeliveryResult,
    FramePrepareTiming,
    RenderBackend,
    RenderDiagnosticsRequest,
    RenderRequest,
    RenderResult,
    RenderTimingSummary,
)
from optics.render_api import (
    ReadbackPayload as RuntimeReadbackPayload,
)
from optics.render_api import (
    WritePolicy as RuntimeWritePolicy,
)
from physics.publish import GpuPublishedFrame
from physics.spatial import SpatialTransform
from sensing import OpticalPinholeCameraSpec, build_pinhole_camera_image_result, build_pinhole_camera_rays
from tools.optical_pipeline_lab.async_readback import (
    TorchAsyncReadbackJob,
    TorchAsyncReadbackRing,
    torch_async_readback_available,
    torch_async_readback_import_error,
)
from tools.optical_pipeline_lab.rgb_pack import (
    pack_linear_rgb_to_preview_uint8,
    rgb_pack_available,
    rgb_pack_import_error,
)
from tools.optical_pipeline_lab.scenarios import DEFAULT_RENDER_HEIGHT, DEFAULT_RENDER_WIDTH
from tools.optical_pipeline_lab.timing import (
    NAN as _NAN,
)
from tools.optical_pipeline_lab.timing import (
    RENDER_PROFILE_PHASES as _RENDER_PROFILE_PHASES,
)
from tools.optical_pipeline_lab.timing import (
    SHADOW_TRAVERSAL_COUNTER_FIELDS,
    FrameTimingRecorder,
    TimingRecorder,
)

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
_VIDEO_SHADOW_TRAVERSAL_CHANNELS = SHADOW_TRAVERSAL_COUNTER_FIELDS
_VIDEO_RGB_CHANNELS = ("rgb",) + _VIDEO_DIAGNOSTIC_CHANNELS
_VIDEO_RGB8_CHANNELS = ("rgb8",) + _VIDEO_DIAGNOSTIC_CHANNELS


@dataclass
class _AsyncVideoReadbackJob:
    frame_index: int
    frame_start: float
    camera: OpticalPinholeCameraSpec
    render_execute_ms: float
    pack_rgb8_ms: float
    render_profile_row: dict[str, float]
    camera_rays_ms: float
    readback_job: TorchAsyncReadbackJob
    prepare_timing: Mapping[str, float] = field(default_factory=dict)


@dataclass
class _RenderedVideoFrame:
    frame_index: int
    camera: OpticalPinholeCameraSpec
    result: object
    camera_rays_ms: float
    render_execute_ms: float
    pack_rgb8_ms: float
    render_profile_row: dict[str, float]
    include_shadow_traversal_stats: bool
    prepare_timing: Mapping[str, float] = field(default_factory=dict)


@dataclass
class Go2RenderSession:
    """Go2 render resource bundle owned by one lab/example run."""

    scene: object
    device: object
    stream: object
    gpu_frame: GpuPublishedFrame
    cache: DeviceOpticalSceneCache
    snapshot: object
    bvh: object
    executor: GpuDeviceBvhDirectLightOpticalExecutor
    bvh_backend: str = "cuda_lbvh"
    bvh_split_strategy: str = "sort"

    @classmethod
    def create(cls, args: argparse.Namespace, timings: TimingRecorder) -> "Go2RenderSession":
        with timings.measure("warp_init"):
            if not args.verbose_warp:
                wp.config.quiet = True
            wp.init()
            device = wp.get_device(args.device)

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

        return cls(
            scene=scene,
            device=device,
            stream=stream,
            gpu_frame=gpu_frame,
            cache=cache,
            snapshot=snapshot,
            bvh=bvh,
            executor=executor,
            bvh_backend=str(args.bvh_backend),
            bvh_split_strategy=str(args.bvh_split_strategy),
        )

    def execute_frame(
        self,
        *,
        camera: OpticalPinholeCameraSpec,
        rays,
        use_gpu_raygen: bool,
        output_profile: OpticalOutputProfile,
        render_profile: list[tuple[str, float]] | None,
    ):
        if use_gpu_raygen:
            return self.executor.execute_camera(
                self.snapshot,
                self.bvh,
                camera,
                output_profile=output_profile,
                render_profile=render_profile,
            )
        return self.executor.execute(
            self.snapshot,
            self.bvh,
            rays,
            output_profile=output_profile,
            render_profile=render_profile,
        )

    def execute_request(
        self,
        request: RenderRequest,
        *,
        render_profile: list[tuple[str, float]] | None,
        snapshot=None,
        bvh=None,
    ):
        if request.backend is not RenderBackend.DIRECT_LIGHT:
            raise NotImplementedError("Go2RenderSession currently supports only direct-light rendering")
        snapshot = self.snapshot if snapshot is None else snapshot
        bvh = self.bvh if bvh is None else bvh
        if request.camera is not None:
            return self.executor.execute_camera(
                snapshot,
                bvh,
                request.camera,
                output_profile=request.output_profile,
                render_profile=render_profile,
            )
        return self.executor.execute(
            snapshot,
            bvh,
            request.rays,
            output_profile=request.output_profile,
            render_profile=render_profile,
        )

    def pack_rgb8(self, result):
        return _pack_video_rgb8(result)


@dataclass
class Go2RenderFrameContext:
    """Frame-scoped Go2 render context backed by one lab session snapshot."""

    session: Go2RenderSession
    env_idx: int = 0
    snapshot: object | None = None
    bvh: object | None = None
    prepare_timing: Mapping[str, float] = field(default_factory=dict)

    @property
    def frame_id(self) -> int:
        return int(self.session.scene.frame.frame_id)

    @property
    def sim_time(self) -> float:
        return float(self.session.scene.frame.sim_time)

    def render(
        self,
        request: RenderRequest,
    ) -> RenderResult:
        """Render a request and return after the compute result is ready."""

        render_profile = _render_profile_buffer_for_request(request)
        render_start = time.perf_counter()
        compute_result = self.session.execute_request(
            request,
            render_profile=render_profile,
            snapshot=self.snapshot,
            bvh=self.bvh,
        )
        wp.synchronize_event(compute_result.ready_event)
        render_execute_ms = (time.perf_counter() - render_start) * 1000.0
        profile_row = _render_profile_row(
            render_profile if request.diagnostics.profile_timing else None,
            render_execute_ms=render_execute_ms,
        )
        return RenderResult(
            compute=compute_result,
            timing={
                "render_execute_ms": render_execute_ms,
                **profile_row,
            },
            render_timing=RenderTimingSummary.from_flat_mapping(
                {
                    "render_execute_ms": render_execute_ms,
                    **profile_row,
                }
            ),
        )


@dataclass
class Go2RenderPipeline:
    """Go2-specific internal pipeline facade for lab experiments."""

    session: Go2RenderSession
    default_delivery: DeliveryRequest | None = None

    @classmethod
    def create(cls, args: argparse.Namespace, timings: TimingRecorder) -> "Go2RenderPipeline":
        return cls(session=Go2RenderSession.create(args, timings))

    def begin_frame(
        self,
        frame_inputs: GpuPublishedFrame | None = None,
        *,
        env_idx: int = 0,
    ) -> Go2RenderFrameContext:
        if frame_inputs is not None and frame_inputs is not self.session.gpu_frame:
            return self._begin_dynamic_frame(frame_inputs, env_idx=env_idx)
        return Go2RenderFrameContext(
            self.session,
            env_idx=env_idx,
            prepare_timing=FramePrepareTiming().to_flat_mapping(),
        )

    def _begin_dynamic_frame(
        self,
        frame_inputs: GpuPublishedFrame,
        *,
        env_idx: int,
    ) -> Go2RenderFrameContext:
        snapshot_start = time.perf_counter()
        snapshot = self.session.cache.snapshot_from_gpu_frame(
            frame_inputs,
            env_idx=env_idx,
            stream=self.session.stream,
            include_aabb=True,
        )
        wp.synchronize_event(snapshot.ready_event)
        snapshot_ms = (time.perf_counter() - snapshot_start) * 1000.0

        accel_refit_ms = _NAN
        accel_rebuild_ms = _NAN
        if bool(getattr(self.session.bvh.stats, "supports_refit", False)):
            refit_start = time.perf_counter()
            bvh = refit_device_bvh_from_snapshot(
                snapshot,
                self.session.bvh,
                stream=self.session.stream,
            )
            wp.synchronize_event(bvh.ready_event)
            accel_refit_ms = (time.perf_counter() - refit_start) * 1000.0
        else:
            rebuild_start = time.perf_counter()
            bvh = self._rebuild_dynamic_bvh(snapshot)
            wp.synchronize_event(bvh.ready_event)
            accel_rebuild_ms = (time.perf_counter() - rebuild_start) * 1000.0

        return Go2RenderFrameContext(
            self.session,
            env_idx=env_idx,
            snapshot=snapshot,
            bvh=bvh,
            prepare_timing=FramePrepareTiming(
                snapshot_ms=snapshot_ms,
                accel_refit_ms=accel_refit_ms,
                accel_rebuild_ms=accel_rebuild_ms,
            ).to_flat_mapping(),
        )

    def _rebuild_dynamic_bvh(self, snapshot):
        if self.session.bvh_backend == "cuda_lbvh":
            return build_cuda_lbvh_from_snapshot(
                snapshot,
                device=self.session.device,
                stream=self.session.stream,
            )
        return build_device_bvh_from_snapshot(
            snapshot,
            device=self.session.device,
            stream=self.session.stream,
            split_strategy=self.session.bvh_split_strategy,
        )

    def deliver(
        self,
        rendered: RenderResult,
        request: DeliveryRequest | None = None,
    ) -> DeliveryResult:
        raise NotImplementedError("Go2RenderPipeline delivery remains owned by the lab benchmark loops")


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
    pipeline = Go2RenderPipeline.create(args, timings)
    session = pipeline.session
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
    pipeline: Go2RenderPipeline,
    args: argparse.Namespace,
    frame_index: int,
    ray_cache: list[tuple[OpticalPinholeCameraSpec, object]] | None,
) -> _RenderedVideoFrame:
    session = pipeline.session
    if args.video_raygen == "gpu":
        camera = _build_video_camera(session.scene, args, frame_index)
        rays = None
        camera_rays_ms = _NAN
    elif ray_cache is None:
        camera_start = time.perf_counter()
        camera = _build_video_camera(session.scene, args, frame_index)
        rays = build_pinhole_camera_rays(camera)
        camera_rays_ms = (time.perf_counter() - camera_start) * 1000.0
    else:
        camera, rays = ray_cache[frame_index]
        camera_rays_ms = _NAN

    render_request = _video_render_request(
        camera=camera,
        rays=rays,
        use_gpu_raygen=args.video_raygen == "gpu",
        readback_mode=args.video_readback,
        profile_timing=bool(args.render_profile),
        fail_on_overflow=bool(args.fail_on_overflow),
    )
    frame_context = pipeline.begin_frame(env_idx=camera.env_idx)
    render_result = frame_context.render(render_request)
    result = render_result.compute
    render_execute_ms = float(render_result.timing["render_execute_ms"])
    render_profile_row = _render_profile_row_from_timing(render_result.timing)
    prepare_timing = dict(frame_context.prepare_timing)

    pack_rgb8_ms = _NAN
    if args.video_readback == "rgb8":
        pack_start = time.perf_counter()
        result = session.pack_rgb8(result)
        wp.synchronize_event(result.ready_event)
        pack_rgb8_ms = (time.perf_counter() - pack_start) * 1000.0

    return _RenderedVideoFrame(
        frame_index=int(frame_index),
        camera=camera,
        result=result,
        camera_rays_ms=camera_rays_ms,
        render_execute_ms=render_execute_ms,
        pack_rgb8_ms=pack_rgb8_ms,
        render_profile_row=render_profile_row,
        include_shadow_traversal_stats=_include_shadow_traversal_stats(render_request),
        prepare_timing=prepare_timing,
    )


def _video_render_request(
    *,
    camera: OpticalPinholeCameraSpec,
    rays,
    use_gpu_raygen: bool,
    readback_mode: str,
    profile_timing: bool,
    fail_on_overflow: bool,
    traversal_counters: bool | None = None,
) -> RenderRequest:
    traversal_counters = profile_timing if traversal_counters is None else traversal_counters
    return RenderRequest(
        frame_id=camera.frame_id,
        sim_time=camera.sim_time,
        env_idx=camera.env_idx,
        camera=camera if use_gpu_raygen else None,
        rays=None if use_gpu_raygen else rays,
        use_gpu_raygen=use_gpu_raygen,
        backend=RenderBackend.DIRECT_LIGHT,
        output_profile=_video_output_profile(readback_mode),
        diagnostics=RenderDiagnosticsRequest(
            profile_timing=profile_timing,
            traversal_counters=traversal_counters,
            fail_on_overflow=fail_on_overflow,
        ),
    )


def _render_profile_buffer_for_request(request: RenderRequest) -> list[tuple[str, float]] | None:
    if request.diagnostics.profile_timing or request.diagnostics.traversal_counters:
        return []
    return None


def _include_shadow_traversal_stats(request: RenderRequest) -> bool:
    return bool(request.diagnostics.traversal_counters)


def _prepare_timing_value(timing: Mapping[str, float], key: str) -> float:
    return float(timing.get(key, _NAN))


def _render_profile_row_from_timing(timing: Mapping[str, float]) -> dict[str, float]:
    row = {
        f"render_{phase}_ms": float(timing.get(f"render_{phase}_ms", _NAN))
        for phase in _RENDER_PROFILE_PHASES
    }
    row["render_overhead_ms"] = float(timing.get("render_overhead_ms", _NAN))
    return row


def _video_delivery_request(
    *,
    readback_mode: str,
    delivery_mode: str,
    ring_depth: int,
    write_frames: bool,
) -> DeliveryRequest:
    payload = RuntimeReadbackPayload(readback_mode)
    if payload is RuntimeReadbackPayload.NONE:
        policy = RuntimeDeliveryPolicy.DEVICE_ONLY
    elif delivery_mode == "torch_async":
        policy = RuntimeDeliveryPolicy.TORCH_ASYNC_ORDERED
    else:
        policy = RuntimeDeliveryPolicy.SYNC_HOST
    return DeliveryRequest(
        payload=payload,
        policy=policy,
        ring_depth=ring_depth,
        write_policy=(RuntimeWritePolicy.PNG_SEQUENCE if write_frames else RuntimeWritePolicy.NONE),
    )


def _run_video_benchmark(
    pipeline: Go2RenderPipeline,
    args: argparse.Namespace,
    out_dir: Path,
) -> FrameTimingRecorder:
    if args.video_readback_delivery == "torch_async":
        return _run_video_benchmark_torch_async(
            pipeline,
            args,
            out_dir,
        )
    session = pipeline.session

    if args.write_frames and Image is None:
        raise SystemExit("Writing video benchmark frames requires Pillow") from _PIL_IMPORT_ERROR
    delivery_request = _video_delivery_request(
        readback_mode=args.video_readback,
        delivery_mode=args.video_readback_delivery,
        ring_depth=int(args.video_readback_ring_depth),
        write_frames=bool(args.write_frames),
    )

    frame_dir = out_dir / "frames"
    if delivery_request.write_policy is RuntimeWritePolicy.PNG_SEQUENCE:
        frame_dir.mkdir(parents=True, exist_ok=True)
    if args.video_raygen == "gpu" and args.video_ray_cache != "off":
        raise SystemExit("--video-raygen gpu computes camera rays on device; use --video-ray-cache off")
    csv_path = Path(args.frame_timing_csv) if args.frame_timing_csv else out_dir / "frame_timing.csv"
    rows = FrameTimingRecorder(
        csv_path=csv_path,
        default_fields=getattr(args, "lab_frame_defaults", None),
    )
    rolling_window_ms: list[float] = []
    ray_cache = (
        _precompute_video_camera_rays(session.scene, args)
        if args.video_raygen == "host" and args.video_ray_cache == "precompute"
        else None
    )

    for frame_index in range(int(args.video_frames)):
        frame_start = time.perf_counter()
        rendered = _render_video_frame(
            pipeline,
            args,
            frame_index,
            ray_cache,
        )

        readback_start = time.perf_counter()
        host_channels = _readback_video_result(
            rendered.result,
            delivery_request.payload.value,
            include_shadow_traversal_stats=rendered.include_shadow_traversal_stats,
        )
        readback_host_ms = (
            (time.perf_counter() - readback_start) * 1000.0
            if delivery_request.payload is not RuntimeReadbackPayload.NONE
            else _NAN
        )

        image_build_ms = _NAN
        encode_or_write_ms = _NAN
        frame_path = ""
        if delivery_request.write_policy is RuntimeWritePolicy.PNG_SEQUENCE:
            image_start = time.perf_counter()
            rgb_preview = _host_rgb_preview_for_readback(host_channels, delivery_request.payload.value)
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
        shadow_traversal_fields = _staged_shadow_traversal_fields(host_channels)
        if args.fail_on_overflow and (primary_overflow or shadow_overflow):
            raise SystemExit(
                f"BVH stack overflow detected at frame {frame_index}: "
                f"primary={primary_overflow}, shadow={shadow_overflow}"
            )

        rows.add(
            {
                "frame_index": frame_index,
                "sim_time": rendered.camera.sim_time,
                "video_time": float(frame_index) / float(args.video_fps),
                "geometry_mode": "static",
                "raygen_mode": args.video_raygen,
                "ray_cache_mode": args.video_ray_cache,
                "readback_mode": delivery_request.payload.value,
                "write_mode": (
                    "rgb_png" if delivery_request.write_policy is RuntimeWritePolicy.PNG_SEQUENCE else "none"
                ),
                "camera_rays_ms": rendered.camera_rays_ms,
                "snapshot_ms": _prepare_timing_value(rendered.prepare_timing, "snapshot_ms"),
                "accel_refit_ms": _prepare_timing_value(rendered.prepare_timing, "accel_refit_ms"),
                "accel_rebuild_ms": _prepare_timing_value(rendered.prepare_timing, "accel_rebuild_ms"),
                "render_execute_ms": rendered.render_execute_ms,
                "pack_rgb8_ms": rendered.pack_rgb8_ms,
                **rendered.render_profile_row,
                "readback_submit_ms": _NAN,
                "readback_wait_ms": _NAN,
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
                **shadow_traversal_fields,
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
                f"render={rendered.render_execute_ms:.3f}ms, "
                f"{_format_pack_rgb8(rendered.pack_rgb8_ms)}"
                f"{_format_render_profile(rendered.render_profile_row)}"
                f"readback={_format_ms(readback_host_ms)}, "
                f"fps={instant_fps:.2f}, rolling_fps={rolling_fps:.2f}, "
                f"overflow=({_format_scalar(primary_overflow)},{_format_scalar(shadow_overflow)})"
            )

    rows.write_csv()
    return rows


def _run_video_benchmark_torch_async(
    pipeline: Go2RenderPipeline,
    args: argparse.Namespace,
    out_dir: Path,
) -> FrameTimingRecorder:
    if not torch_async_readback_available():
        raise SystemExit(
            "--video-readback-delivery=torch_async requires torch with CUDA support"
        ) from torch_async_readback_import_error()
    if args.video_readback not in ("rgb", "rgb8"):
        raise SystemExit(
            "--video-readback-delivery=torch_async currently supports --video-readback=rgb/rgb8 only"
        )
    if args.write_frames and Image is None:
        raise SystemExit("Writing video benchmark frames requires Pillow") from _PIL_IMPORT_ERROR
    if args.video_raygen == "gpu" and args.video_ray_cache != "off":
        raise SystemExit("--video-raygen gpu computes camera rays on device; use --video-ray-cache off")
    if args.video_readback_ring_depth <= 0:
        raise SystemExit("--video-readback-ring-depth must be > 0")
    session = pipeline.session
    delivery_request = _video_delivery_request(
        readback_mode=args.video_readback,
        delivery_mode=args.video_readback_delivery,
        ring_depth=int(args.video_readback_ring_depth),
        write_frames=bool(args.write_frames),
    )

    frame_dir = out_dir / "frames"
    if delivery_request.write_policy is RuntimeWritePolicy.PNG_SEQUENCE:
        frame_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.frame_timing_csv) if args.frame_timing_csv else out_dir / "frame_timing.csv"
    rows = FrameTimingRecorder(
        csv_path=csv_path,
        default_fields=getattr(args, "lab_frame_defaults", None),
    )
    rolling_window_ms: list[float] = []
    ray_cache = (
        _precompute_video_camera_rays(session.scene, args)
        if args.video_raygen == "host" and args.video_ray_cache == "precompute"
        else None
    )
    readback_ring = _prepare_torch_async_readback_ring(
        pipeline,
        args,
        delivery_request,
    )
    pending: _AsyncVideoReadbackJob | None = None
    first_frame_start: float | None = None
    last_completion_time: float | None = None

    for frame_index in range(int(args.video_frames)):
        frame_start = time.perf_counter()
        if first_frame_start is None:
            first_frame_start = frame_start
        rendered = _render_video_frame(
            pipeline,
            args,
            frame_index,
            ray_cache,
        )

        if pending is not None and readback_ring.ring_depth <= 1:
            last_completion_time = _complete_torch_async_video_readback(
                pending,
                args,
                rows,
                rolling_window_ms,
                frame_dir,
                delivery_request=delivery_request,
                first_frame_start=first_frame_start,
                last_completion_time=last_completion_time,
                latest_rendered_frame_index=frame_index,
                ring_block_count=1,
            )
            pending = None

        job = _submit_torch_async_video_readback(
            readback_ring,
            rendered.result,
            frame_index=frame_index,
            frame_start=frame_start,
            camera=rendered.camera,
            render_execute_ms=rendered.render_execute_ms,
            pack_rgb8_ms=rendered.pack_rgb8_ms,
            render_profile_row=rendered.render_profile_row,
            camera_rays_ms=rendered.camera_rays_ms,
            prepare_timing=rendered.prepare_timing,
        )
        if pending is not None:
            last_completion_time = _complete_torch_async_video_readback(
                pending,
                args,
                rows,
                rolling_window_ms,
                frame_dir,
                delivery_request=delivery_request,
                first_frame_start=first_frame_start,
                last_completion_time=last_completion_time,
                latest_rendered_frame_index=frame_index,
                ring_block_count=0,
            )
        pending = job

    if pending is not None:
        last_completion_time = _complete_torch_async_video_readback(
            pending,
            args,
            rows,
            rolling_window_ms,
            frame_dir,
            delivery_request=delivery_request,
            first_frame_start=first_frame_start if first_frame_start is not None else time.perf_counter(),
            last_completion_time=last_completion_time,
            latest_rendered_frame_index=max(int(args.video_frames) - 1, 0),
            ring_block_count=0,
        )

    rows.write_csv()
    return rows


def _prepare_torch_async_readback_ring(
    pipeline: Go2RenderPipeline,
    args: argparse.Namespace,
    delivery_request: DeliveryRequest,
) -> TorchAsyncReadbackRing:
    session = pipeline.session
    warmup_camera = _build_video_camera(session.scene, args, 0)
    warmup_request = _video_render_request(
        camera=warmup_camera,
        rays=None,
        use_gpu_raygen=True,
        readback_mode=delivery_request.payload.value,
        profile_timing=bool(args.render_profile),
        fail_on_overflow=bool(args.fail_on_overflow),
    )
    warmup_frame = pipeline.begin_frame(env_idx=warmup_camera.env_idx)
    warmup_result = warmup_frame.render(warmup_request).compute
    if delivery_request.payload is RuntimeReadbackPayload.RGB8:
        warmup_result = session.pack_rgb8(warmup_result)
        wp.synchronize_event(warmup_result.ready_event)
    return TorchAsyncReadbackRing.from_warmup_result(
        warmup_result,
        channels=_video_readback_channels(
            delivery_request.payload.value,
            include_shadow_traversal_stats=_include_shadow_traversal_stats(warmup_request),
        ),
        ring_depth=delivery_request.ring_depth,
    )


def _submit_torch_async_video_readback(
    readback_ring: TorchAsyncReadbackRing,
    result,
    *,
    frame_index: int,
    frame_start: float,
    camera: OpticalPinholeCameraSpec,
    render_execute_ms: float,
    pack_rgb8_ms: float,
    render_profile_row: dict[str, float],
    camera_rays_ms: float,
    prepare_timing: Mapping[str, float] | None = None,
) -> _AsyncVideoReadbackJob:
    readback_job = readback_ring.submit(result, frame_index=frame_index)
    return _AsyncVideoReadbackJob(
        frame_index=frame_index,
        frame_start=frame_start,
        camera=camera,
        render_execute_ms=render_execute_ms,
        pack_rgb8_ms=pack_rgb8_ms,
        render_profile_row=render_profile_row,
        camera_rays_ms=camera_rays_ms,
        readback_job=readback_job,
        prepare_timing={} if prepare_timing is None else dict(prepare_timing),
    )


def _complete_torch_async_video_readback(
    job: _AsyncVideoReadbackJob,
    args: argparse.Namespace,
    rows: FrameTimingRecorder,
    rolling_window_ms: list[float],
    frame_dir: Path,
    *,
    delivery_request: DeliveryRequest,
    first_frame_start: float,
    last_completion_time: float | None,
    latest_rendered_frame_index: int,
    ring_block_count: int,
) -> float:
    readback_wait_ms = job.readback_job.synchronize()
    completion_time = time.perf_counter()
    readback_copy_ms = job.readback_job.copy_elapsed_ms()
    host_channels = job.readback_job.host_channels()

    image_build_ms = _NAN
    encode_or_write_ms = _NAN
    frame_path = ""
    if delivery_request.write_policy is RuntimeWritePolicy.PNG_SEQUENCE:
        image_start = time.perf_counter()
        rgb_preview = _host_rgb_preview_for_readback(host_channels, delivery_request.payload.value)
        image_build_ms = (time.perf_counter() - image_start) * 1000.0

        write_start = time.perf_counter()
        path = frame_dir / f"rgb_{job.frame_index:06d}.png"
        Image.fromarray(rgb_preview).save(path)
        encode_or_write_ms = (time.perf_counter() - write_start) * 1000.0
        frame_path = str(path)

    previous_completion = last_completion_time if last_completion_time is not None else first_frame_start
    frame_total_ms = (completion_time - previous_completion) * 1000.0
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
    shadow_traversal_fields = _staged_shadow_traversal_fields(host_channels)
    if args.fail_on_overflow and (primary_overflow or shadow_overflow):
        raise SystemExit(
            f"BVH stack overflow detected at frame {job.frame_index}: "
            f"primary={primary_overflow}, shadow={shadow_overflow}"
        )

    render_delivery_ms = _render_delivery_ms(
        render_execute_ms=job.render_execute_ms,
        pack_rgb8_ms=job.pack_rgb8_ms,
    )
    overlap_ratio = _overlap_ratio(
        observed_ms=frame_total_ms,
        render_ms=render_delivery_ms,
        readback_ms=readback_copy_ms,
    )
    rows.add(
        {
            "frame_index": job.frame_index,
            "sim_time": job.camera.sim_time,
            "video_time": float(job.frame_index) / float(args.video_fps),
            "geometry_mode": "static",
            "delivery_policy": "torch_async",
            "raygen_mode": args.video_raygen,
            "ray_cache_mode": args.video_ray_cache,
            "readback_mode": f"torch_async_{delivery_request.payload.value}",
            "write_mode": (
                "rgb_png" if delivery_request.write_policy is RuntimeWritePolicy.PNG_SEQUENCE else "none"
            ),
            "camera_rays_ms": job.camera_rays_ms,
            "snapshot_ms": _prepare_timing_value(job.prepare_timing, "snapshot_ms"),
            "accel_refit_ms": _prepare_timing_value(job.prepare_timing, "accel_refit_ms"),
            "accel_rebuild_ms": _prepare_timing_value(job.prepare_timing, "accel_rebuild_ms"),
            "render_execute_ms": job.render_execute_ms,
            "pack_rgb8_ms": job.pack_rgb8_ms,
            **job.render_profile_row,
            "readback_submit_ms": job.readback_job.submit_ms,
            "readback_wait_ms": readback_wait_ms,
            "readback_host_ms": readback_copy_ms,
            "image_build_ms": image_build_ms,
            "encode_or_write_ms": encode_or_write_ms,
            "frame_total_ms": frame_total_ms,
            "instant_fps": instant_fps,
            "rolling_fps": rolling_fps,
            "primary_overflow": primary_overflow,
            "shadow_overflow": shadow_overflow,
            "primary_max_stack": primary_max_stack,
            "shadow_max_stack": shadow_max_stack,
            **shadow_traversal_fields,
            "readback_lag_frames": max(int(latest_rendered_frame_index) - int(job.frame_index), 0),
            "readback_ring_depth": delivery_request.ring_depth,
            "readback_ring_block_count": int(ring_block_count),
            "completed_frame_index": job.frame_index,
            "overlap_ratio": overlap_ratio,
            "frame_path": frame_path,
        }
    )
    if args.progress_every > 0 and (
        job.frame_index == 0
        or (job.frame_index + 1) % int(args.progress_every) == 0
        or job.frame_index + 1 == int(args.video_frames)
    ):
        print(
            "  video_frame "
            f"{job.frame_index + 1}/{args.video_frames}: "
            f"total={frame_total_ms:.3f}ms, "
            f"render={job.render_execute_ms:.3f}ms, "
            f"{_format_pack_rgb8(job.pack_rgb8_ms)}"
            f"{_format_render_profile(job.render_profile_row)}"
            f"readback={readback_copy_ms:.3f}ms, "
            f"submit={job.readback_job.submit_ms:.3f}ms, wait={readback_wait_ms:.3f}ms, "
            f"lag={max(int(latest_rendered_frame_index) - int(job.frame_index), 0)}, "
            f"overlap={_format_ratio(overlap_ratio)}, "
            f"fps={instant_fps:.2f}, rolling_fps={rolling_fps:.2f}, "
            f"overflow=({_format_scalar(primary_overflow)},{_format_scalar(shadow_overflow)})"
        )
    return completion_time


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


def _readback_video_result(
    result,
    readback_mode: str,
    *,
    include_shadow_traversal_stats: bool = False,
) -> dict[str, np.ndarray]:
    if readback_mode == "none":
        return {}
    channels = _video_readback_channels(
        readback_mode,
        include_shadow_traversal_stats=include_shadow_traversal_stats,
    )
    if readback_mode == "rgb":
        return stage_optical_channels(result, channels, canonical_dtypes=False)
    if readback_mode == "rgb8":
        return stage_optical_channels(result, channels, canonical_dtypes=False)
    return stage_optical_compute_result_to_host(result).channels


def _video_output_profile(readback_mode: str) -> OpticalOutputProfile:
    if readback_mode in ("rgb", "rgb8"):
        return OpticalOutputProfile.RGB_PREVIEW
    if readback_mode == "none":
        return OpticalOutputProfile.RENDER_ONLY
    return OpticalOutputProfile.DIRECT_LIGHT_FULL


def _video_readback_channels(
    readback_mode: str,
    *,
    include_shadow_traversal_stats: bool = False,
) -> tuple[str, ...]:
    if readback_mode == "rgb8":
        channels = _VIDEO_RGB8_CHANNELS
    else:
        channels = _VIDEO_RGB_CHANNELS
    if include_shadow_traversal_stats:
        channels = channels + _VIDEO_SHADOW_TRAVERSAL_CHANNELS
    return channels


def _pack_video_rgb8(result):
    if not rgb_pack_available():
        raise SystemExit("--video-readback=rgb8 requires warp RGB8 packing") from rgb_pack_import_error()
    return pack_linear_rgb_to_preview_uint8(result)


def _render_profile_row(
    render_profile: list[tuple[str, float]] | None,
    *,
    render_execute_ms: float | None = None,
) -> dict[str, float]:
    row = {f"render_{phase}_ms": _NAN for phase in _RENDER_PROFILE_PHASES}
    if render_profile is None:
        return row
    profile_total_ms = 0.0
    matched_phase_count = 0
    for name, elapsed_ms in render_profile:
        phase = name.removesuffix("_ms")
        key = f"render_{phase}_ms"
        if key in row:
            elapsed = float(elapsed_ms)
            previous = row[key]
            row[key] = elapsed if math.isnan(previous) else previous + elapsed
            profile_total_ms += elapsed
            matched_phase_count += 1
    if render_execute_ms is not None and matched_phase_count > 0:
        row["render_overhead_ms"] = float(render_execute_ms) - profile_total_ms
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


def _host_rgb_preview_for_readback(host_channels: dict[str, object], readback_mode: str) -> np.ndarray:
    if readback_mode == "rgb8":
        return np.asarray(host_channels["rgb8"], dtype=np.uint8)
    return _rgb_array_to_preview_uint8(host_channels["rgb"])


def _staged_scalar(channels: dict[str, object], name: str) -> float | int:
    if name not in channels:
        return _NAN
    return int(np.asarray(channels[name]).reshape(-1)[0])


def _staged_shadow_traversal_fields(channels: dict[str, object]) -> dict[str, float | int]:
    return {name: _staged_scalar(channels, name) for name in _VIDEO_SHADOW_TRAVERSAL_CHANNELS}


def _format_ms(value: float) -> str:
    return "NaN" if math.isnan(float(value)) else f"{float(value):.3f}ms"


def _format_pack_rgb8(value: float) -> str:
    return "" if math.isnan(float(value)) else f"pack_rgb8={float(value):.3f}ms, "


def _render_delivery_ms(*, render_execute_ms: float, pack_rgb8_ms: float) -> float:
    if math.isnan(float(pack_rgb8_ms)):
        return float(render_execute_ms)
    return float(render_execute_ms) + float(pack_rgb8_ms)


def _overlap_ratio(*, observed_ms: float, render_ms: float, readback_ms: float) -> float:
    denominator = float(render_ms) + float(readback_ms)
    if denominator <= 0.0 or math.isnan(denominator):
        return _NAN
    return 1.0 - float(observed_ms) / denominator


def _format_ratio(value: float) -> str:
    return "NaN" if math.isnan(float(value)) else f"{float(value):.3f}"


def _format_scalar(value: float | int) -> str:
    return "NaN" if isinstance(value, float) and math.isnan(value) else str(int(value))


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
