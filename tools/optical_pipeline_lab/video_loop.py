"""Generic video render loop helpers for Optical Pipeline Lab experiments."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from optics import OpticalOutputProfile
from optics.render_api import (
    DeliveryRequest,
    RenderBackend,
    RenderDiagnosticsRequest,
    RenderRequest,
)
from sensing import OpticalPinholeCameraSpec, build_pinhole_camera_rays
from tools.optical_pipeline_lab.async_readback import (
    torch_async_readback_available,
    torch_async_readback_import_error,
)
from tools.optical_pipeline_lab.delivery import (
    RenderedVideoFrame,
    VideoDeliveryFacade,
    VideoDeliveryRunConfig,
    VideoFrameTimingRowBuilder,
    video_delivery_request,
)
from tools.optical_pipeline_lab.delivery import (
    video_readback_channels as _delivery_video_readback_channels,
)
from tools.optical_pipeline_lab.rgb_pack import (
    pack_linear_rgb_to_preview_uint8,
    rgb_pack_available,
    rgb_pack_import_error,
)
from tools.optical_pipeline_lab.timing import NAN as _NAN
from tools.optical_pipeline_lab.timing import RENDER_PROFILE_PHASES as _RENDER_PROFILE_PHASES
from tools.optical_pipeline_lab.timing import FrameTimingRecorder

VideoCameraBuilder = Callable[[object, object, int], OpticalPinholeCameraSpec]
VideoRenderFrameFn = Callable[
    [object, object, int, list[tuple[OpticalPinholeCameraSpec, object]] | None],
    RenderedVideoFrame,
]


@dataclass(frozen=True)
class FrameIdentity:
    """Minimal per-frame identity used to align cameras with render frames."""

    frame_id: int
    sim_time: float
    env_idx: int | None = None


FrameIdentityForIndex = Callable[[int], FrameIdentity | None]
VideoGeometryModeForIndex = Callable[[int, FrameIdentity | None], str]


@dataclass(frozen=True)
class VideoRenderPlan:
    """Camera, request, and metadata needed to render one video frame."""

    camera: OpticalPinholeCameraSpec
    rays: object | None
    request: RenderRequest
    camera_rays_ms: float
    geometry_mode: str
    include_shadow_traversal_stats: bool


def render_video_frame(
    pipeline,
    args,
    frame_index: int,
    ray_cache: list[tuple[OpticalPinholeCameraSpec, object]] | None,
    *,
    build_video_camera: VideoCameraBuilder,
) -> RenderedVideoFrame:
    session = pipeline.session
    frame_inputs = video_frame_inputs(args, frame_index)
    frame_identity = frame_identity_from_inputs(frame_inputs)
    geometry_mode = video_geometry_mode(args, frame_inputs=frame_inputs)
    plan = build_video_render_plan(
        session.scene,
        args,
        frame_index,
        ray_cache,
        build_video_camera=build_video_camera,
        frame_identity=frame_identity,
        geometry_mode=geometry_mode,
    )
    frame_context = pipeline.begin_frame(frame_inputs=frame_inputs, env_idx=plan.camera.env_idx)
    return render_video_frame_from_context(frame_context, plan, frame_index=frame_index)


def build_video_render_plan(
    scene,
    args,
    frame_index: int,
    ray_cache: list[tuple[OpticalPinholeCameraSpec, object]] | None,
    *,
    build_video_camera: VideoCameraBuilder,
    frame_identity: FrameIdentity | None = None,
    geometry_mode: str | None = None,
) -> VideoRenderPlan:
    """Build camera, rays, and render request without acquiring a frame context."""

    if args.video_raygen == "gpu":
        camera = build_video_camera(scene, args, frame_index)
        camera = video_camera_for_frame_identity(camera, frame_identity=frame_identity)
        rays = None
        camera_rays_ms = _NAN
    elif ray_cache is None:
        camera_start = time.perf_counter()
        camera = build_video_camera(scene, args, frame_index)
        camera = video_camera_for_frame_identity(camera, frame_identity=frame_identity)
        rays = build_pinhole_camera_rays(camera)
        camera_rays_ms = (time.perf_counter() - camera_start) * 1000.0
    else:
        camera, rays = ray_cache[frame_index]
        camera = video_camera_for_frame_identity(camera, frame_identity=frame_identity)
        rays = video_rays_for_camera(rays, camera)
        camera_rays_ms = _NAN

    render_request = video_render_request(
        camera=camera,
        rays=rays,
        use_gpu_raygen=args.video_raygen == "gpu",
        readback_mode=args.video_readback,
        profile_timing=bool(args.render_profile),
        fail_on_overflow=bool(args.fail_on_overflow),
    )

    return VideoRenderPlan(
        camera=camera,
        rays=rays,
        request=render_request,
        camera_rays_ms=camera_rays_ms,
        geometry_mode=(
            str(geometry_mode)
            if geometry_mode is not None
            else video_geometry_mode(args, frame_inputs=frame_identity)
        ),
        include_shadow_traversal_stats=include_shadow_traversal_stats(render_request),
    )


def render_video_frame_from_context(
    frame_context,
    plan: VideoRenderPlan,
    *,
    frame_index: int,
) -> RenderedVideoFrame:
    """Render one planned video frame from an already-acquired frame context."""

    render_result = frame_context.render(plan.request)
    result = render_result.compute
    render_execute_ms = float(render_result.timing["render_execute_ms"])
    render_profile = render_profile_row_from_timing(render_result.timing)
    prepare_timing = dict(frame_context.prepare_timing)

    return RenderedVideoFrame(
        frame_index=int(frame_index),
        camera=plan.camera,
        result=result,
        camera_rays_ms=plan.camera_rays_ms,
        render_execute_ms=render_execute_ms,
        render_profile_row=render_profile,
        include_shadow_traversal_stats=plan.include_shadow_traversal_stats,
        geometry_mode=plan.geometry_mode,
        prepare_timing=prepare_timing,
        render=render_result,
    )


def video_render_request(
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
        output_profile=video_output_profile(readback_mode),
        diagnostics=RenderDiagnosticsRequest(
            profile_timing=profile_timing,
            traversal_counters=traversal_counters,
            fail_on_overflow=fail_on_overflow,
        ),
    )


def render_profile_buffer_for_request(request: RenderRequest) -> list[tuple[str, float]] | None:
    if request.diagnostics.profile_timing or request.diagnostics.traversal_counters:
        return []
    return None


def include_shadow_traversal_stats(request: RenderRequest) -> bool:
    return bool(request.diagnostics.traversal_counters)


def video_frame_inputs(args, frame_index: int):
    frame_inputs = getattr(args, "video_frame_inputs", None)
    if frame_inputs is None:
        return None
    if frame_index >= len(frame_inputs):
        raise IndexError("video_frame_inputs must contain at least video_frames entries")
    return frame_inputs[frame_index]


def frame_identity_from_inputs(frame_inputs) -> FrameIdentity | None:
    if frame_inputs is None:
        return None
    return FrameIdentity(
        frame_id=int(getattr(frame_inputs, "frame_id")),
        sim_time=float(getattr(frame_inputs, "sim_time")),
        env_idx=(
            int(getattr(frame_inputs, "env_idx"))
            if getattr(frame_inputs, "env_idx", None) is not None
            else None
        ),
    )


def video_camera_for_frame_inputs(
    camera: OpticalPinholeCameraSpec,
    *,
    frame_inputs,
) -> OpticalPinholeCameraSpec:
    return video_camera_for_frame_identity(
        camera,
        frame_identity=frame_identity_from_inputs(frame_inputs),
    )


def video_camera_for_frame_identity(
    camera: OpticalPinholeCameraSpec,
    *,
    frame_identity: FrameIdentity | None,
) -> OpticalPinholeCameraSpec:
    if frame_identity is None:
        return camera
    changes = {
        "frame_id": int(frame_identity.frame_id),
        "sim_time": float(frame_identity.sim_time),
    }
    if frame_identity.env_idx is not None:
        changes["env_idx"] = int(frame_identity.env_idx)
    return replace(camera, **changes)


def video_rays_for_camera(rays, camera: OpticalPinholeCameraSpec):
    return replace(rays, frame_id=camera.frame_id, sim_time=camera.sim_time, env_idx=camera.env_idx)


def video_geometry_mode(args, *, frame_inputs) -> str:
    mode = getattr(args, "video_geometry_mode", None)
    if mode is not None:
        return str(mode)
    return "dynamic_rigid" if frame_inputs is not None else "static"


def render_profile_row_from_timing(timing: Mapping[str, float]) -> dict[str, float]:
    row = {
        f"render_{phase}_ms": float(timing.get(f"render_{phase}_ms", _NAN))
        for phase in _RENDER_PROFILE_PHASES
    }
    row["render_overhead_ms"] = float(timing.get("render_overhead_ms", _NAN))
    return row


def video_delivery_request_from_options(
    *,
    readback_mode: str,
    delivery_mode: str,
    ring_depth: int,
    write_frames: bool,
) -> DeliveryRequest:
    return video_delivery_request(
        readback_mode=readback_mode,
        delivery_mode=delivery_mode,
        ring_depth=ring_depth,
        write_frames=write_frames,
    )


def run_video_benchmark(
    pipeline,
    args,
    out_dir: Path,
    *,
    build_video_camera: VideoCameraBuilder,
    pack_rgb8: Callable[[object], object],
    synchronize_event: Callable[[object], None],
    render_frame: VideoRenderFrameFn | None = None,
) -> FrameTimingRecorder:
    session = pipeline.session

    _validate_video_benchmark_args(args)
    delivery_request = _video_delivery_request_for_args(args)
    rows = _video_frame_timing_recorder(args, out_dir)
    ray_cache = _video_ray_cache_for_scene(session.scene, args, build_video_camera=build_video_camera)
    delivery = VideoDeliveryFacade.create(
        request=delivery_request,
        delivery_policy_label=args.video_readback_delivery,
        frame_dir=out_dir / "frames",
        pack_rgb8=getattr(session, "pack_rgb8", pack_rgb8),
        synchronize_event=synchronize_event,
        warmup_result_factory=(
            lambda: build_torch_async_warmup_result(
                pipeline,
                args,
                delivery_request,
                build_video_camera=build_video_camera,
            )
        ),
    )
    row_builder = _video_frame_timing_row_builder(args, delivery_request)
    if render_frame is None:

        def render_frame(
            pipeline,
            args,
            frame_index: int,
            ray_cache: list[tuple[OpticalPinholeCameraSpec, object]] | None,
        ) -> RenderedVideoFrame:
            return render_video_frame(
                pipeline,
                args,
                frame_index,
                ray_cache,
                build_video_camera=build_video_camera,
            )

    def render_frame_at_index(frame_index: int) -> RenderedVideoFrame:
        return render_frame(pipeline, args, frame_index, ray_cache)

    _run_video_delivery_loop(
        args,
        delivery=delivery,
        row_builder=row_builder,
        rows=rows,
        render_frame_at_index=render_frame_at_index,
    )
    return rows


def run_video_benchmark_with_frame_contexts(
    scene,
    args,
    out_dir: Path,
    *,
    frame_provider,
    build_video_camera: VideoCameraBuilder,
    pack_rgb8: Callable[[object], object],
    synchronize_event: Callable[[object], None],
    frame_identity_for_index: FrameIdentityForIndex | None = None,
    geometry_mode_for_index: VideoGeometryModeForIndex | None = None,
) -> FrameTimingRecorder:
    """Run video delivery from an explicit frame-context provider.

    This provider-backed path intentionally does not accept the legacy
    ``render_frame(pipeline, args, ...)`` callback because that callback owns
    ``pipeline.begin_frame(...)``.
    """

    _validate_video_benchmark_args(args)
    delivery_request = _video_delivery_request_for_args(args)
    rows = _video_frame_timing_recorder(args, out_dir)
    ray_cache = _video_ray_cache_for_scene(scene, args, build_video_camera=build_video_camera)
    delivery = VideoDeliveryFacade.create(
        request=delivery_request,
        delivery_policy_label=args.video_readback_delivery,
        frame_dir=out_dir / "frames",
        pack_rgb8=pack_rgb8,
        synchronize_event=synchronize_event,
        warmup_result_factory=(
            lambda: build_provider_backed_torch_async_warmup_result(
                scene,
                args,
                frame_provider=frame_provider,
                build_video_camera=build_video_camera,
                frame_identity_for_index=frame_identity_for_index,
                geometry_mode_for_index=geometry_mode_for_index,
            )
        ),
    )
    row_builder = _video_frame_timing_row_builder(args, delivery_request)

    def render_frame_at_index(frame_index: int) -> RenderedVideoFrame:
        frame_identity = _frame_identity_for_index(frame_identity_for_index, frame_index)
        plan = build_video_render_plan(
            scene,
            args,
            frame_index,
            ray_cache,
            build_video_camera=build_video_camera,
            frame_identity=frame_identity,
            geometry_mode=_geometry_mode_for_index(
                args,
                geometry_mode_for_index,
                frame_index,
                frame_identity,
            ),
        )
        with frame_provider.begin_frame(frame_index, env_idx=plan.camera.env_idx) as frame_context:
            return render_video_frame_from_context(
                frame_context,
                plan,
                frame_index=frame_index,
            )

    _run_video_delivery_loop(
        args,
        delivery=delivery,
        row_builder=row_builder,
        rows=rows,
        render_frame_at_index=render_frame_at_index,
    )
    return rows


def _run_video_delivery_loop(
    args,
    *,
    delivery: VideoDeliveryFacade,
    row_builder: VideoFrameTimingRowBuilder,
    rows: FrameTimingRecorder,
    render_frame_at_index: Callable[[int], RenderedVideoFrame],
) -> None:
    for frame_index in range(int(args.video_frames)):
        frame_start = time.perf_counter()
        rendered = render_frame_at_index(frame_index)

        for completed in delivery.complete_available(latest_rendered_frame_index=rendered.frame_index):
            record_delivered_video_frame(rows, row_builder, completed, args)
        completed = delivery.submit(rendered, frame_start=frame_start)
        if completed is not None:
            record_delivered_video_frame(rows, row_builder, completed, args)
        for completed in delivery.complete_available(latest_rendered_frame_index=rendered.frame_index):
            record_delivered_video_frame(rows, row_builder, completed, args)

    for completed in delivery.flush():
        record_delivered_video_frame(rows, row_builder, completed, args)

    rows.write_csv()


def build_torch_async_warmup_result(
    pipeline,
    args,
    delivery_request: DeliveryRequest,
    *,
    build_video_camera: VideoCameraBuilder,
) -> tuple[object, bool]:
    session = pipeline.session
    warmup_camera = build_video_camera(session.scene, args, 0)
    warmup_request = video_render_request(
        camera=warmup_camera,
        rays=None,
        use_gpu_raygen=True,
        readback_mode=delivery_request.payload.value,
        profile_timing=bool(args.render_profile),
        fail_on_overflow=bool(args.fail_on_overflow),
    )
    warmup_frame = pipeline.begin_frame(env_idx=warmup_camera.env_idx)
    warmup_result = warmup_frame.render(warmup_request).compute
    return warmup_result, include_shadow_traversal_stats(warmup_request)


def build_provider_backed_torch_async_warmup_result(
    scene,
    args,
    *,
    frame_provider,
    build_video_camera: VideoCameraBuilder,
    frame_identity_for_index: FrameIdentityForIndex | None = None,
    geometry_mode_for_index: VideoGeometryModeForIndex | None = None,
) -> tuple[object, bool]:
    """Build an async-readback warmup result through provider lifecycle."""

    frame_index = 0
    frame_identity = _frame_identity_for_index(frame_identity_for_index, frame_index)
    plan = build_video_render_plan(
        scene,
        args,
        frame_index,
        None,
        build_video_camera=build_video_camera,
        frame_identity=frame_identity,
        geometry_mode=_geometry_mode_for_index(
            args,
            geometry_mode_for_index,
            frame_index,
            frame_identity,
        ),
    )
    with frame_provider.begin_frame(frame_index, env_idx=plan.camera.env_idx) as frame_context:
        rendered = render_video_frame_from_context(frame_context, plan, frame_index=frame_index)
    return rendered.result, rendered.include_shadow_traversal_stats


def _validate_video_benchmark_args(args) -> None:
    if args.video_readback_delivery == "torch_async" and not torch_async_readback_available():
        raise SystemExit(
            "--video-readback-delivery=torch_async requires torch with CUDA support"
        ) from torch_async_readback_import_error()
    if args.video_readback_delivery == "torch_async" and args.video_readback not in ("rgb", "rgb8"):
        raise SystemExit(
            "--video-readback-delivery=torch_async currently supports --video-readback=rgb/rgb8 only"
        )
    if args.video_raygen == "gpu" and args.video_ray_cache != "off":
        raise SystemExit("--video-raygen gpu computes camera rays on device; use --video-ray-cache off")
    if args.video_readback_ring_depth <= 0:
        raise SystemExit("--video-readback-ring-depth must be > 0")


def _video_delivery_request_for_args(args) -> DeliveryRequest:
    return video_delivery_request_from_options(
        readback_mode=args.video_readback,
        delivery_mode=args.video_readback_delivery,
        ring_depth=int(args.video_readback_ring_depth),
        write_frames=bool(args.write_frames),
    )


def _video_frame_timing_recorder(args, out_dir: Path) -> FrameTimingRecorder:
    csv_path = Path(args.frame_timing_csv) if args.frame_timing_csv else out_dir / "frame_timing.csv"
    return FrameTimingRecorder(
        csv_path=csv_path,
        default_fields=getattr(args, "lab_frame_defaults", None),
    )


def _video_ray_cache_for_scene(
    scene,
    args,
    *,
    build_video_camera: VideoCameraBuilder,
) -> list[tuple[OpticalPinholeCameraSpec, object]] | None:
    if args.video_raygen == "host" and args.video_ray_cache == "precompute":
        return precompute_video_camera_rays(scene, args, build_video_camera=build_video_camera)
    return None


def _video_frame_timing_row_builder(
    args,
    delivery_request: DeliveryRequest,
) -> VideoFrameTimingRowBuilder:
    return VideoFrameTimingRowBuilder(
        VideoDeliveryRunConfig(
            video_fps=float(args.video_fps),
            video_frames=int(args.video_frames),
            video_raygen=args.video_raygen,
            video_ray_cache=args.video_ray_cache,
            delivery_policy_label=args.video_readback_delivery,
            fail_on_overflow=bool(args.fail_on_overflow),
        )
    ).bind_request(delivery_request)


def _frame_identity_for_index(
    frame_identity_for_index: FrameIdentityForIndex | None,
    frame_index: int,
) -> FrameIdentity | None:
    return None if frame_identity_for_index is None else frame_identity_for_index(frame_index)


def _geometry_mode_for_index(
    args,
    geometry_mode_for_index: VideoGeometryModeForIndex | None,
    frame_index: int,
    frame_identity: FrameIdentity | None,
) -> str:
    if geometry_mode_for_index is not None:
        return geometry_mode_for_index(frame_index, frame_identity)
    return video_geometry_mode(args, frame_inputs=frame_identity)


def record_delivered_video_frame(
    rows: FrameTimingRecorder,
    row_builder: VideoFrameTimingRowBuilder,
    delivered,
    args,
) -> None:
    rows.add(row_builder.build_row(delivered))
    if args.progress_every > 0 and (
        delivered.completed_frame_index == 0
        or (delivered.completed_frame_index + 1) % int(args.progress_every) == 0
        or delivered.completed_frame_index + 1 == int(args.video_frames)
    ):
        print(row_builder.progress_line(delivered))


def precompute_video_camera_rays(
    scene,
    args,
    *,
    build_video_camera: VideoCameraBuilder,
) -> list[tuple[OpticalPinholeCameraSpec, object]]:
    start = time.perf_counter()
    ray_cache = [
        (camera := build_video_camera(scene, args, frame_index), build_pinhole_camera_rays(camera))
        for frame_index in range(int(args.video_frames))
    ]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(
        "  video_ray_cache: "
        f"mode=precompute, frames={args.video_frames}, elapsed_ms={elapsed_ms:.3f}, "
        f"per_frame_ms={elapsed_ms / max(int(args.video_frames), 1):.3f}"
    )
    return ray_cache


def video_output_profile(readback_mode: str) -> OpticalOutputProfile:
    if readback_mode in ("rgb", "rgb8"):
        return OpticalOutputProfile.RGB_PREVIEW
    if readback_mode == "none":
        return OpticalOutputProfile.RENDER_ONLY
    return OpticalOutputProfile.DIRECT_LIGHT_FULL


def video_readback_channels(
    readback_mode: str,
    *,
    include_shadow_traversal_stats: bool = False,
) -> tuple[str, ...]:
    return _delivery_video_readback_channels(
        readback_mode,
        include_shadow_traversal_stats=include_shadow_traversal_stats,
    )


def pack_video_rgb8(result):
    if not rgb_pack_available():
        raise SystemExit("--video-readback=rgb8 requires warp RGB8 packing") from rgb_pack_import_error()
    return pack_linear_rgb_to_preview_uint8(result)


def render_profile_row(
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
