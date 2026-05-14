"""Optical Lab render session, frame context, and pipeline types."""

from __future__ import annotations

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
class OpticalLabRenderSource:
    """Scene/source adapter output consumed by the lab render pipeline."""

    registry: object
    base_frame: GpuPublishedFrame
    # Scene world-space AABB hints for setup; not camera/frustum hints.
    bounds_min: object | None = None
    bounds_max: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def frame(self) -> GpuPublishedFrame:
        return self.base_frame


@dataclass(frozen=True)
class OpticalLabRenderOptions:
    """Configuration intent for building one lab render session."""

    device: object = "cuda:0"
    bvh_backend: str = "cpu"
    bvh_split_strategy: str = "sort"
    shadows: bool = True
    verbose_warp: bool = False


@dataclass
class OpticalLabRenderWorkspace:
    """Lab-local render workspace for device and stream ownership."""

    device: object
    stream: object


RenderSourceFactory = Callable[[OpticalLabRenderWorkspace], OpticalLabRenderSource]
RenderSceneFactory = Callable[[OpticalLabRenderSource], object]


@dataclass(init=False)
class OpticalLabRenderSession:
    """Optical Lab render resource bundle owned by one lab run."""

    scene: object
    source: OpticalLabRenderSource
    workspace: OpticalLabRenderWorkspace
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

    def __init__(
        self,
        *,
        scene: object,
        source: OpticalLabRenderSource | None = None,
        gpu_frame: GpuPublishedFrame,
        cache: DeviceOpticalSceneCache,
        snapshot: object,
        bvh: object,
        executor: GpuDeviceBvhDirectLightOpticalExecutor,
        workspace: OpticalLabRenderWorkspace | None = None,
        device: object | None = None,
        stream: object | None = None,
        bvh_backend: str = "cpu",
        bvh_split_strategy: str = "sort",
        pack_rgb8_fn: Callable[[object], object] = _default_pack_rgb8,
        render_profile_buffer_for_request: RenderProfileBufferFactory = (
            _default_render_profile_buffer_for_request
        ),
        render_profile_row: RenderProfileRowFactory = _default_render_profile_row,
    ) -> None:
        if workspace is None:
            if device is None or stream is None:
                raise TypeError("OpticalLabRenderSession requires workspace or both device and stream")
            workspace = OpticalLabRenderWorkspace(device=device, stream=stream)
        elif device is not None or stream is not None:
            raise TypeError("OpticalLabRenderSession accepts either workspace or device/stream, not both")
        self.scene = scene
        self.source = (
            source
            if source is not None
            else OpticalLabRenderSource(
                registry=getattr(scene, "registry", None),
                base_frame=gpu_frame,
                bounds_min=getattr(scene, "bounds_min", None),
                bounds_max=getattr(scene, "bounds_max", None),
            )
        )
        self.workspace = workspace
        self.gpu_frame = gpu_frame
        self.cache = cache
        self.snapshot = snapshot
        self.bvh = bvh
        self.executor = executor
        self.bvh_backend = str(bvh_backend)
        self.bvh_split_strategy = str(bvh_split_strategy)
        self.pack_rgb8_fn = pack_rgb8_fn
        self.render_profile_buffer_for_request = render_profile_buffer_for_request
        self.render_profile_row = render_profile_row

    @property
    def device(self):
        return self.workspace.device

    @property
    def stream(self):
        return self.workspace.stream

    @classmethod
    def create_from_source(
        cls,
        source: OpticalLabRenderSource,
        options: OpticalLabRenderOptions,
        timings: TimingRecorder,
        *,
        pack_rgb8: Callable[[object], object] = _default_pack_rgb8,
        render_profile_buffer_for_request: RenderProfileBufferFactory = (
            _default_render_profile_buffer_for_request
        ),
        render_profile_row: RenderProfileRowFactory = _default_render_profile_row,
    ) -> "OpticalLabRenderSession":
        with timings.measure("warp_init"):
            if not options.verbose_warp:
                wp.config.quiet = True
            wp.init()
            device = wp.get_device(options.device)

        with timings.measure("gpu_frame_stream"):
            workspace = OpticalLabRenderWorkspace(device=device, stream=wp.Stream(device=device))

        return cls._create_from_initialized_source(
            source,
            options,
            timings,
            workspace=workspace,
            scene=source,
            pack_rgb8=pack_rgb8,
            render_profile_buffer_for_request=render_profile_buffer_for_request,
            render_profile_row=render_profile_row,
        )

    @classmethod
    def create_from_source_factory(
        cls,
        source_factory: RenderSourceFactory,
        options: OpticalLabRenderOptions,
        timings: TimingRecorder,
        *,
        scene_for_source: RenderSceneFactory | None = None,
        pack_rgb8: Callable[[object], object] = _default_pack_rgb8,
        render_profile_buffer_for_request: RenderProfileBufferFactory = (
            _default_render_profile_buffer_for_request
        ),
        render_profile_row: RenderProfileRowFactory = _default_render_profile_row,
    ) -> "OpticalLabRenderSession":
        with timings.measure("warp_init"):
            if not options.verbose_warp:
                wp.config.quiet = True
            wp.init()
            device = wp.get_device(options.device)

        with timings.measure("gpu_frame_stream"):
            workspace = OpticalLabRenderWorkspace(device=device, stream=wp.Stream(device=device))

        with timings.measure("import_scene"):
            source = source_factory(workspace)
        scene = source if scene_for_source is None else scene_for_source(source)

        return cls._create_from_initialized_source(
            source,
            options,
            timings,
            workspace=workspace,
            scene=scene,
            pack_rgb8=pack_rgb8,
            render_profile_buffer_for_request=render_profile_buffer_for_request,
            render_profile_row=render_profile_row,
        )

    @classmethod
    def _create_from_initialized_source(
        cls,
        source: OpticalLabRenderSource,
        options: OpticalLabRenderOptions,
        timings: TimingRecorder,
        *,
        workspace: OpticalLabRenderWorkspace,
        scene: object,
        pack_rgb8: Callable[[object], object],
        render_profile_buffer_for_request: RenderProfileBufferFactory,
        render_profile_row: RenderProfileRowFactory,
    ) -> "OpticalLabRenderSession":
        with timings.measure("device_scene_snapshot"):
            cache = DeviceOpticalSceneCache(
                source.registry,
                device=workspace.device,
                stream=workspace.stream,
            )
            snapshot = cache.snapshot_from_gpu_frame(
                source.base_frame,
                env_idx=0,
                stream=workspace.stream,
                include_aabb=True,
            )
        with timings.measure("bvh_build"):
            if options.bvh_backend == "cuda_lbvh":
                bvh = build_cuda_lbvh_from_snapshot(
                    snapshot,
                    device=workspace.device,
                    stream=workspace.stream,
                )
            else:
                bvh = build_device_bvh_from_snapshot(
                    snapshot,
                    device=workspace.device,
                    stream=workspace.stream,
                    split_strategy=options.bvh_split_strategy,
                )
        for detail_name, detail_elapsed_ms in bvh.stats.detail_ms:
            timings.add(f"bvh_build_{detail_name}", detail_elapsed_ms)

        with timings.measure("executor_init"):
            executor = GpuDeviceBvhDirectLightOpticalExecutor(
                device=workspace.device,
                stream=workspace.stream,
                shadows=options.shadows,
                ambient_rgb=(0.08, 0.085, 0.09),
                background_rgb=(0.025, 0.03, 0.038),
            )

        return cls(
            scene=scene,
            source=source,
            workspace=workspace,
            gpu_frame=source.base_frame,
            cache=cache,
            snapshot=snapshot,
            bvh=bvh,
            executor=executor,
            bvh_backend=str(options.bvh_backend),
            bvh_split_strategy=str(options.bvh_split_strategy),
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
            raise NotImplementedError(
                "OpticalLabRenderSession currently supports only direct-light rendering"
            )
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
class OpticalLabRenderFrameContext:
    """Frame-scoped lab render context backed by one session snapshot."""

    session: OpticalLabRenderSession
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
class OpticalLabRenderPipeline:
    """Optical Lab internal pipeline facade for render experiments."""

    session: OpticalLabRenderSession
    default_delivery: DeliveryRequest | None = None

    @classmethod
    def create_from_source(
        cls,
        source: OpticalLabRenderSource,
        options: OpticalLabRenderOptions,
        timings: TimingRecorder,
        *,
        pack_rgb8: Callable[[object], object] = _default_pack_rgb8,
        render_profile_buffer_for_request: RenderProfileBufferFactory = (
            _default_render_profile_buffer_for_request
        ),
        render_profile_row: RenderProfileRowFactory = _default_render_profile_row,
    ) -> "OpticalLabRenderPipeline":
        return cls(
            session=OpticalLabRenderSession.create_from_source(
                source,
                options,
                timings,
                pack_rgb8=pack_rgb8,
                render_profile_buffer_for_request=render_profile_buffer_for_request,
                render_profile_row=render_profile_row,
            )
        )

    @classmethod
    def create_from_source_factory(
        cls,
        source_factory: RenderSourceFactory,
        options: OpticalLabRenderOptions,
        timings: TimingRecorder,
        *,
        scene_for_source: RenderSceneFactory | None = None,
        pack_rgb8: Callable[[object], object] = _default_pack_rgb8,
        render_profile_buffer_for_request: RenderProfileBufferFactory = (
            _default_render_profile_buffer_for_request
        ),
        render_profile_row: RenderProfileRowFactory = _default_render_profile_row,
    ) -> "OpticalLabRenderPipeline":
        return cls(
            session=OpticalLabRenderSession.create_from_source_factory(
                source_factory,
                options,
                timings,
                scene_for_source=scene_for_source,
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
    ) -> OpticalLabRenderFrameContext:
        if frame_inputs is not None and frame_inputs is not self.session.gpu_frame:
            return self._begin_dynamic_frame(frame_inputs, env_idx=env_idx)
        return OpticalLabRenderFrameContext(
            self.session,
            env_idx=env_idx,
            prepare_timing=FramePrepareTiming().to_flat_mapping(),
        )

    def _begin_dynamic_frame(
        self,
        frame_inputs: GpuPublishedFrame,
        *,
        env_idx: int,
    ) -> OpticalLabRenderFrameContext:
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

        return OpticalLabRenderFrameContext(
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
        raise NotImplementedError(
            "OpticalLabRenderPipeline delivery remains owned by the lab benchmark loops"
        )

    def deliver(
        self,
        rendered: RenderResult,
        request: DeliveryRequest | None = None,
    ) -> DeliveryResult:
        raise NotImplementedError(
            "OpticalLabRenderPipeline delivery remains owned by the lab benchmark loops"
        )
