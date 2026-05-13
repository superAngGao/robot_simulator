"""Go2 render session, frame context, and pipeline types for the lab backend."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from optics import (
    DeviceOpticalSceneCache,
    GpuDeviceBvhDirectLightOpticalExecutor,
    OpticalOutputProfile,
    build_cuda_lbvh_from_snapshot,
    build_device_bvh_from_snapshot,
    refit_device_bvh_from_snapshot,
)
from optics.render_api import (
    DeliveryRequest,
    DeliveryResult,
    FramePrepareTiming,
    RenderBackend,
    RenderRequest,
    RenderResult,
    RenderTimingSummary,
)
from physics.publish import GpuPublishedFrame
from sensing import OpticalPinholeCameraSpec
from tools.optical_pipeline_lab.rgb_pack import (
    pack_linear_rgb_to_preview_uint8,
    rgb_pack_available,
    rgb_pack_import_error,
)
from tools.optical_pipeline_lab.timing import NAN as _NAN
from tools.optical_pipeline_lab.timing import RENDER_PROFILE_PHASES as _RENDER_PROFILE_PHASES
from tools.optical_pipeline_lab.timing import TimingRecorder

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - example-only guard.
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


RenderProfileBufferFactory = Callable[[RenderRequest], list[tuple[str, float]] | None]
RenderProfileRowFactory = Callable[..., dict[str, float]]


def _default_render_profile_buffer_for_request(request: RenderRequest) -> list[tuple[str, float]] | None:
    if request.diagnostics.profile_timing or request.diagnostics.traversal_counters:
        return []
    return None


def _default_render_profile_row(
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


def _default_pack_rgb8(result):
    if not rgb_pack_available():
        raise SystemExit("--video-readback=rgb8 requires warp RGB8 packing") from rgb_pack_import_error()
    return pack_linear_rgb_to_preview_uint8(result)


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
    bvh_backend: str = "cpu"
    bvh_split_strategy: str = "sort"
    pack_rgb8_fn: Callable[[object], object] = _default_pack_rgb8
    render_profile_buffer_for_request: RenderProfileBufferFactory = _default_render_profile_buffer_for_request
    render_profile_row: RenderProfileRowFactory = _default_render_profile_row

    @classmethod
    def create(
        cls,
        args: argparse.Namespace,
        timings: TimingRecorder,
        *,
        scene_factory: Callable[[str, argparse.Namespace], object],
        base_gpu_frame_factory: Callable[..., GpuPublishedFrame],
        pack_rgb8: Callable[[object], object],
        render_profile_buffer_for_request: RenderProfileBufferFactory,
        render_profile_row: RenderProfileRowFactory,
    ) -> "Go2RenderSession":
        with timings.measure("warp_init"):
            if not args.verbose_warp:
                wp.config.quiet = True
            wp.init()
            device = wp.get_device(args.device)

        scene_preset = getattr(args, "scene_preset", "go2_menagerie_static")
        with timings.measure("import_scene"):
            scene = scene_factory(scene_preset, args)
        with timings.measure("gpu_frame_stream"):
            stream = wp.Stream(device=device)
            gpu_frame = base_gpu_frame_factory(
                scene_preset,
                frame_id=scene.frame.frame_id,
                sim_time=scene.frame.sim_time,
                device=device,
            )

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
            pack_rgb8_fn=pack_rgb8,
            render_profile_buffer_for_request=render_profile_buffer_for_request,
            render_profile_row=render_profile_row,
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
        return self.pack_rgb8_fn(result)


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

        render_profile_factory = getattr(
            self.session,
            "render_profile_buffer_for_request",
            _default_render_profile_buffer_for_request,
        )
        render_profile_row = getattr(self.session, "render_profile_row", _default_render_profile_row)
        render_profile = render_profile_factory(request)
        render_start = time.perf_counter()
        compute_result = self.session.execute_request(
            request,
            render_profile=render_profile,
            snapshot=self.snapshot,
            bvh=self.bvh,
        )
        wp.synchronize_event(compute_result.ready_event)
        render_execute_ms = (time.perf_counter() - render_start) * 1000.0
        profile_row = render_profile_row(
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
    def create(
        cls,
        args: argparse.Namespace,
        timings: TimingRecorder,
        *,
        scene_factory: Callable[[str, argparse.Namespace], object],
        base_gpu_frame_factory: Callable[..., GpuPublishedFrame],
        pack_rgb8: Callable[[object], object],
        render_profile_buffer_for_request: RenderProfileBufferFactory,
        render_profile_row: RenderProfileRowFactory,
    ) -> "Go2RenderPipeline":
        return cls(
            session=Go2RenderSession.create(
                args,
                timings,
                scene_factory=scene_factory,
                base_gpu_frame_factory=base_gpu_frame_factory,
                pack_rgb8=pack_rgb8,
                render_profile_buffer_for_request=render_profile_buffer_for_request,
                render_profile_row=render_profile_row,
            )
        )

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

    def create_delivery_runtime(self, request: DeliveryRequest):
        raise NotImplementedError("Go2RenderPipeline delivery remains owned by the lab benchmark loops")

    def deliver(
        self,
        rendered: RenderResult,
        request: DeliveryRequest | None = None,
    ) -> DeliveryResult:
        raise NotImplementedError("Go2RenderPipeline delivery remains owned by the lab benchmark loops")
