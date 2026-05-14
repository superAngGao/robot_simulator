import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.optical_pipeline_lab.__main__ as lab_main
import tools.optical_pipeline_lab.async_readback as async_readback
import tools.optical_pipeline_lab.delivery as delivery
import tools.optical_pipeline_lab.dynamic_frames as dynamic_frames
import tools.optical_pipeline_lab.go2_backend as go2_backend
import tools.optical_pipeline_lab.go2_session as go2_session
import tools.optical_pipeline_lab.rgb_pack as rgb_pack
from optics.render_api import DeliveryPolicy as RuntimeDeliveryPolicy
from optics.render_api import DeliveryResult as RuntimeDeliveryResult
from optics.render_api import DeliveryTimingSummary, RenderTimingSummary
from optics.render_api import OpticalRenderPipeline as RuntimeOpticalRenderPipeline
from optics.render_api import ReadbackPayload as RuntimeReadbackPayload
from optics.render_api import RenderBackend as RuntimeRenderBackend
from optics.render_api import RenderFrameContext as RuntimeRenderFrameContext
from optics.render_api import RenderResult as RuntimeRenderResult
from optics.render_api import WritePolicy as RuntimeWritePolicy
from physics.publish import GpuPublishedFrame
from tools.optical_pipeline_lab import (
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_WIDTH,
    AccelPolicy,
    DeliveryPolicy,
    FrameTimingRecorder,
    GeometryMode,
    OpticalLabScenarioConfig,
    OpticalLabScenarioFamily,
    ReadbackPayload,
    TimingRecorder,
    percentile,
)
from tools.optical_pipeline_lab.matrix import (
    MatrixCase,
    MatrixRunOptions,
    MatrixSuite,
    get_suite,
    run_matrix_suite,
    run_options_for_case,
)
from tools.optical_pipeline_lab.presets import get_preset
from tools.optical_pipeline_lab.reports import format_summary_rows
from tools.optical_pipeline_lab.runner import (
    DEFAULT_LAB_WARMUP_RENDERS,
    LabRunOptions,
    apply_run_overrides,
    build_menagerie_example_args,
    run_scenario,
    validate_run,
    write_scenario_config,
)


def test_percentile_interpolates_sorted_samples():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([10.0, 20.0], 0.9) == pytest.approx(19.0)
    assert percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert math.isnan(percentile([], 0.9))


def test_timing_recorder_writes_summary_csv(tmp_path: Path):
    recorder = TimingRecorder()
    recorder.add("render", 1.0)
    recorder.add("render", 3.0)

    rows = recorder.summary_rows()
    assert rows == [
        {
            "phase": "render",
            "count": 2.0,
            "mean_ms": 2.0,
            "p50_ms": 2.0,
            "p90_ms": pytest.approx(2.8),
            "min_ms": 1.0,
            "max_ms": 3.0,
        }
    ]

    path = tmp_path / "timing.csv"
    recorder.write_csv(path)
    with path.open(newline="") as f:
        written = list(csv.DictReader(f))
    assert written[0]["phase"] == "render"
    assert float(written[0]["mean_ms"]) == 2.0


def test_frame_timing_recorder_normalizes_lab_schema(tmp_path: Path):
    path = tmp_path / "frame_timing.csv"
    recorder = FrameTimingRecorder(csv_path=path)
    recorder.add(
        {
            "frame_index": 0,
            "scenario_name": "smoke",
            "render_execute_ms": 2.0,
            "pack_rgb8_ms": 0.25,
            "readback_host_ms": float("nan"),
            "frame_total_ms": 4.0,
        }
    )
    recorder.add(
        {
            "frame_index": 1,
            "scenario_name": "smoke",
            "render_execute_ms": 4.0,
            "frame_total_ms": 8.0,
        }
    )

    summary = {row["phase"]: row for row in recorder.summary_rows()}
    assert summary["render_execute"]["mean_ms"] == 3.0
    assert summary["pack_rgb8"]["mean_ms"] == 0.25
    assert summary["frame_total"]["p90_ms"] == pytest.approx(7.6)
    assert "readback_host" not in summary

    video = recorder.video_summary()
    assert video["fps_mean"] == pytest.approx(1000.0 * 2.0 / 12.0)

    recorder.write_csv()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        written = list(reader)
    assert "delivery_policy" in reader.fieldnames
    assert "overlap_ratio" in reader.fieldnames
    assert "pack_rgb8_ms" in reader.fieldnames
    assert "shadow_traversal_ray_count" in reader.fieldnames
    assert "accel_refit_ms" in reader.fieldnames
    assert "refit_ms" not in reader.fieldnames
    assert written[0]["scenario_name"] == "smoke"


def test_frame_timing_recorder_applies_default_lab_fields(tmp_path: Path):
    path = tmp_path / "frame_timing.csv"
    recorder = FrameTimingRecorder(
        csv_path=path,
        default_fields={
            "scenario_name": "go2",
            "device": "cuda:1",
            "width": 160,
            "height": 120,
        },
    )

    recorder.add({"frame_index": 0, "frame_total_ms": 1.5})
    recorder.write_csv()

    with path.open(newline="") as f:
        written = list(csv.DictReader(f))
    assert written[0]["scenario_name"] == "go2"
    assert written[0]["device"] == "cuda:1"
    assert written[0]["width"] == "160"
    assert written[0]["height"] == "120"


def test_render_profile_row_computes_unclamped_overhead():
    row = go2_backend._render_profile_row(
        [
            ("raygen_kernel", 1.0),
            ("first_hit_kernel_ms", 2.0),
            ("shade_kernel", 3.0),
            ("unknown_phase", 100.0),
        ],
        render_execute_ms=5.5,
    )

    assert row["render_raygen_kernel_ms"] == 1.0
    assert row["render_first_hit_kernel_ms"] == 2.0
    assert row["render_shade_kernel_ms"] == 3.0
    assert row["render_overhead_ms"] == -0.5
    assert "render_unknown_phase_ms" not in row


def test_render_profile_row_from_timing_preserves_overhead():
    row = go2_backend._render_profile_row_from_timing(
        {
            "render_execute_ms": 5.5,
            "render_raygen_kernel_ms": 1.0,
            "render_first_hit_kernel_ms": 2.0,
            "render_shade_kernel_ms": 3.0,
            "render_overhead_ms": -0.5,
        }
    )

    assert row["render_raygen_kernel_ms"] == 1.0
    assert row["render_first_hit_kernel_ms"] == 2.0
    assert row["render_shade_kernel_ms"] == 3.0
    assert row["render_overhead_ms"] == -0.5


def test_video_render_request_maps_lab_options_to_runtime_api():
    camera = SimpleNamespace(frame_id=3, sim_time=0.1, env_idx=0)

    request = go2_backend._video_render_request(
        camera=camera,
        rays=None,
        use_gpu_raygen=True,
        readback_mode="rgb8",
        profile_timing=True,
        fail_on_overflow=False,
    )

    assert request.backend is RuntimeRenderBackend.DIRECT_LIGHT
    assert request.camera is camera
    assert request.rays is None
    assert request.output_profile.value == "rgb_preview"
    assert request.diagnostics.profile_timing is True
    assert request.diagnostics.traversal_counters is True
    assert request.diagnostics.fail_on_overflow is False


def test_render_request_diagnostics_drive_profile_buffer_and_traversal_readback():
    camera = SimpleNamespace(frame_id=4, sim_time=0.2, env_idx=0)

    request = go2_backend._video_render_request(
        camera=camera,
        rays=None,
        use_gpu_raygen=True,
        readback_mode="rgb",
        profile_timing=False,
        fail_on_overflow=True,
        traversal_counters=True,
    )

    assert request.diagnostics.profile_timing is False
    assert request.diagnostics.traversal_counters is True
    assert go2_backend._render_profile_buffer_for_request(request) == []
    assert go2_backend._include_shadow_traversal_stats(request) is True

    request = go2_backend._video_render_request(
        camera=camera,
        rays=None,
        use_gpu_raygen=True,
        readback_mode="rgb",
        profile_timing=False,
        fail_on_overflow=True,
        traversal_counters=False,
    )

    assert go2_backend._render_profile_buffer_for_request(request) is None
    assert go2_backend._include_shadow_traversal_stats(request) is False


def test_go2_pipeline_create_uses_backend_callback_boundary(monkeypatch: pytest.MonkeyPatch):
    scene = SimpleNamespace(
        registry=object(),
        frame=SimpleNamespace(frame_id=7, sim_time=0.7),
    )
    gpu_frame = object()
    snapshot = object()
    bvh = SimpleNamespace(stats=SimpleNamespace(detail_ms=[]))

    class FakeWp:
        config = SimpleNamespace(quiet=False)

        class Stream:
            def __init__(self, *, device):
                self.device = device

        @staticmethod
        def init():
            return None

        @staticmethod
        def get_device(device):
            return f"device:{device}"

    class FakeCache:
        def __init__(self, registry, *, device, stream):
            self.registry = registry
            self.device = device
            self.stream = stream

        def snapshot_from_gpu_frame(self, frame, *, env_idx, stream, include_aabb):
            assert frame is gpu_frame
            assert env_idx == 0
            assert stream is self.stream
            assert include_aabb is True
            return snapshot

    class FakeExecutor:
        def __init__(self, *, device, stream, shadows, ambient_rgb, background_rgb):
            self.device = device
            self.stream = stream
            self.shadows = shadows
            self.ambient_rgb = ambient_rgb
            self.background_rgb = background_rgb

    def fake_scene_factory(scene_preset, args):
        assert scene_preset == "synthetic"
        assert args.scene_preset == "synthetic"
        return scene

    def fake_gpu_frame_factory(scene_preset, *, frame_id, sim_time, device):
        assert scene_preset == "synthetic"
        assert frame_id == 7
        assert sim_time == 0.7
        assert device == "device:cuda:fake"
        return gpu_frame

    def fake_build_bvh(snapshot_arg, *, device, stream, split_strategy):
        assert snapshot_arg is snapshot
        assert device == "device:cuda:fake"
        assert split_strategy == "partition"
        assert stream.device == "device:cuda:fake"
        return bvh

    monkeypatch.setattr(go2_session, "wp", FakeWp)
    monkeypatch.setattr(go2_session, "DeviceOpticalSceneCache", FakeCache)
    monkeypatch.setattr(go2_session, "build_device_bvh_from_snapshot", fake_build_bvh)
    monkeypatch.setattr(go2_session, "GpuDeviceBvhDirectLightOpticalExecutor", FakeExecutor)

    def profile_buffer(request):
        return []

    def profile_row(render_profile, *, render_execute_ms=None):
        return {"render_overhead_ms": 0.0}

    def pack_rgb8(result):
        return ("packed", result)

    pipeline = go2_backend.Go2RenderPipeline.create(
        SimpleNamespace(
            verbose_warp=False,
            device="cuda:fake",
            scene_preset="synthetic",
            bvh_backend="cpu",
            bvh_split_strategy="partition",
            no_shadows=True,
        ),
        go2_backend.TimingRecorder(),
        scene_factory=fake_scene_factory,
        base_gpu_frame_factory=fake_gpu_frame_factory,
        pack_rgb8=pack_rgb8,
        render_profile_buffer_for_request=profile_buffer,
        render_profile_row=profile_row,
    )

    assert pipeline.session.scene is scene
    assert isinstance(pipeline.session.workspace, go2_backend.Go2RenderWorkspace)
    assert pipeline.session.workspace.device == "device:cuda:fake"
    assert pipeline.session.workspace.stream.device == "device:cuda:fake"
    assert pipeline.session.device is pipeline.session.workspace.device
    assert pipeline.session.stream is pipeline.session.workspace.stream
    assert pipeline.session.gpu_frame is gpu_frame
    assert pipeline.session.snapshot is snapshot
    assert pipeline.session.bvh is bvh
    assert pipeline.session.executor.shadows is False
    assert pipeline.session.pack_rgb8("rgb") == ("packed", "rgb")
    assert pipeline.session.render_profile_buffer_for_request is profile_buffer
    assert pipeline.session.render_profile_row is profile_row


def test_go2_render_session_accepts_workspace_with_device_stream_compatibility():
    workspace = go2_backend.Go2RenderWorkspace(device="device", stream="stream")
    session = go2_backend.Go2RenderSession(
        scene=object(),
        workspace=workspace,
        gpu_frame=object(),
        cache=object(),
        snapshot=object(),
        bvh=object(),
        executor=object(),
    )

    assert session.workspace is workspace
    assert session.device == "device"
    assert session.stream == "stream"


def test_go2_pipeline_frame_context_wraps_render_result(monkeypatch: pytest.MonkeyPatch):
    compute = SimpleNamespace(ready_event=object())
    monkeypatch.setattr(go2_session, "wp", SimpleNamespace(synchronize_event=lambda event: None))

    class FakeSession:
        scene = SimpleNamespace(frame=SimpleNamespace(frame_id=4, sim_time=0.2))
        gpu_frame = object()

        def __init__(self):
            self.calls = []

        def execute_request(self, request, *, render_profile, snapshot=None, bvh=None):
            self.calls.append((request, render_profile, snapshot, bvh))
            render_profile.append(("shade_kernel_ms", 2.0))
            return compute

    session = FakeSession()
    pipeline = go2_backend.Go2RenderPipeline(session=session)
    frame = pipeline.begin_frame(env_idx=0)
    request = go2_backend._video_render_request(
        camera=SimpleNamespace(frame_id=4, sim_time=0.2, env_idx=0),
        rays=None,
        use_gpu_raygen=True,
        readback_mode="rgb",
        profile_timing=True,
        fail_on_overflow=True,
    )

    assert isinstance(pipeline, RuntimeOpticalRenderPipeline)
    assert isinstance(frame, RuntimeRenderFrameContext)
    assert frame.frame_id == 4
    assert frame.sim_time == 0.2
    assert frame.env_idx == 0

    rendered = frame.render(request)

    assert isinstance(rendered, RuntimeRenderResult)
    assert rendered.compute is compute
    assert rendered.render_timing is not None
    assert rendered.render_timing.execute_ms >= 0.0
    assert rendered.timing["render_shade_kernel_ms"] == 2.0
    assert rendered.timing["render_execute_ms"] >= 0.0
    assert math.isnan(float(frame.prepare_timing["snapshot_ms"]))
    assert len(session.calls) == 1
    assert session.calls[0][0] is request
    assert session.calls[0][1] == [("shade_kernel_ms", 2.0)]
    assert session.calls[0][2] is None
    assert session.calls[0][3] is None


def test_render_video_frame_passes_dynamic_frame_inputs(monkeypatch: pytest.MonkeyPatch):
    frame_inputs = SimpleNamespace(frame_id=9, sim_time=0.9)
    compute = SimpleNamespace(ready_event=object())
    camera = go2_backend.OpticalPinholeCameraSpec(
        frame_id=8,
        sim_time=0.8,
        env_idx=1,
        sensor_id="camera",
        width=16,
        height=8,
        fx=10.0,
        fy=10.0,
        cx=7.5,
        cy=3.5,
    )
    captured: dict[str, object] = {}

    class FakeFrameContext:
        prepare_timing = {
            "snapshot_ms": 1.0,
            "accel_refit_ms": 2.0,
            "accel_rebuild_ms": float("nan"),
        }

        def render(self, request):
            captured["request"] = request
            captured["render_result"] = RuntimeRenderResult(
                compute=compute,
                timing={
                    "render_execute_ms": 3.0,
                    **go2_backend._render_profile_row(None),
                },
            )
            return captured["render_result"]

    class FakePipeline:
        session = SimpleNamespace(scene=object())

        def begin_frame(self, frame_inputs=None, *, env_idx=0):
            captured["frame_inputs"] = frame_inputs
            captured["env_idx"] = env_idx
            return FakeFrameContext()

    monkeypatch.setattr(go2_backend, "_build_video_camera", lambda scene, args, frame_index: camera)

    rendered = go2_backend._render_video_frame(
        FakePipeline(),
        SimpleNamespace(
            video_raygen="gpu",
            video_readback="none",
            render_profile=False,
            fail_on_overflow=False,
            video_frame_inputs=[frame_inputs],
            video_geometry_mode="dynamic_rigid",
        ),
        0,
        None,
    )

    assert captured["frame_inputs"] is frame_inputs
    assert captured["env_idx"] == 1
    request = captured["request"]
    assert request.frame_id == 9
    assert request.sim_time == 0.9
    assert request.camera.frame_id == 9
    assert rendered.geometry_mode == "dynamic_rigid"
    assert rendered.prepare_timing["snapshot_ms"] == 1.0
    assert rendered.render_execute_ms == 3.0
    assert rendered.render is captured["render_result"]
    assert rendered.result is compute


def test_go2_pipeline_static_begin_frame_accepts_session_frame_inputs():
    session = SimpleNamespace(
        scene=SimpleNamespace(frame=SimpleNamespace(frame_id=1, sim_time=0.0)),
        gpu_frame=object(),
    )
    pipeline = go2_backend.Go2RenderPipeline(session=session)

    frame = pipeline.begin_frame(frame_inputs=session.gpu_frame, env_idx=3)

    assert frame.snapshot is None
    assert frame.bvh is None
    assert frame.env_idx == 3
    assert math.isnan(float(frame.prepare_timing["snapshot_ms"]))


def test_go2_pipeline_dynamic_begin_frame_refits_frame_specific_snapshot(monkeypatch: pytest.MonkeyPatch):
    sync_events = []
    frame_inputs = object()
    snapshot = SimpleNamespace(ready_event="snapshot_ready")
    refit_bvh = SimpleNamespace(ready_event="refit_ready")

    class FakeCache:
        def __init__(self):
            self.calls = []

        def snapshot_from_gpu_frame(self, frame, *, env_idx, stream, include_aabb):
            self.calls.append((frame, env_idx, stream, include_aabb))
            return snapshot

    def fake_refit(snapshot_arg, bvh_arg, *, stream):
        assert snapshot_arg is snapshot
        assert bvh_arg is session.bvh
        assert stream == "stream"
        return refit_bvh

    session = SimpleNamespace(
        scene=SimpleNamespace(frame=SimpleNamespace(frame_id=1, sim_time=0.0)),
        gpu_frame=object(),
        cache=FakeCache(),
        stream="stream",
        device="cuda:fake",
        bvh=SimpleNamespace(stats=SimpleNamespace(supports_refit=True)),
        bvh_backend="cpu",
        bvh_split_strategy="sort",
    )
    monkeypatch.setattr(
        go2_session,
        "wp",
        SimpleNamespace(synchronize_event=lambda event: sync_events.append(event)),
    )
    monkeypatch.setattr(go2_session, "refit_device_bvh_from_snapshot", fake_refit)

    frame = go2_backend.Go2RenderPipeline(session=session).begin_frame(
        frame_inputs=frame_inputs,
        env_idx=2,
    )

    assert frame.snapshot is snapshot
    assert frame.bvh is refit_bvh
    assert frame.prepare_timing["snapshot_ms"] >= 0.0
    assert frame.prepare_timing["accel_refit_ms"] >= 0.0
    assert math.isnan(float(frame.prepare_timing["accel_rebuild_ms"]))
    assert session.cache.calls == [(frame_inputs, 2, "stream", True)]
    assert sync_events == ["snapshot_ready", "refit_ready"]


def test_go2_pipeline_dynamic_begin_frame_rebuilds_when_refit_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    frame_inputs = object()
    snapshot = SimpleNamespace(ready_event="snapshot_ready")
    rebuilt_bvh = SimpleNamespace(ready_event="rebuild_ready")

    class FakeCache:
        def snapshot_from_gpu_frame(self, frame, *, env_idx, stream, include_aabb):
            return snapshot

    def fake_build(snapshot_arg, *, device, stream, split_strategy):
        assert snapshot_arg is snapshot
        assert device == "cuda:fake"
        assert stream == "stream"
        assert split_strategy == "partition"
        return rebuilt_bvh

    session = SimpleNamespace(
        scene=SimpleNamespace(frame=SimpleNamespace(frame_id=1, sim_time=0.0)),
        gpu_frame=object(),
        cache=FakeCache(),
        stream="stream",
        device="cuda:fake",
        bvh=SimpleNamespace(stats=SimpleNamespace(supports_refit=False)),
        bvh_backend="cpu",
        bvh_split_strategy="partition",
    )
    monkeypatch.setattr(go2_session, "wp", SimpleNamespace(synchronize_event=lambda event: None))
    monkeypatch.setattr(go2_session, "build_device_bvh_from_snapshot", fake_build)

    frame = go2_backend.Go2RenderPipeline(session=session).begin_frame(
        frame_inputs=frame_inputs,
        env_idx=0,
    )

    assert frame.snapshot is snapshot
    assert frame.bvh is rebuilt_bvh
    assert math.isnan(float(frame.prepare_timing["accel_refit_ms"]))
    assert frame.prepare_timing["accel_rebuild_ms"] >= 0.0


def test_go2_pipeline_dynamic_begin_frame_rebuilds_cuda_lbvh_when_configured(
    monkeypatch: pytest.MonkeyPatch,
):
    snapshot = SimpleNamespace(ready_event="snapshot_ready")
    rebuilt_bvh = SimpleNamespace(ready_event="cuda_rebuild_ready")

    class FakeCache:
        def snapshot_from_gpu_frame(self, frame, *, env_idx, stream, include_aabb):
            return snapshot

    def fake_cuda_build(snapshot_arg, *, device, stream):
        assert snapshot_arg is snapshot
        assert device == "cuda:fake"
        assert stream == "stream"
        return rebuilt_bvh

    session = SimpleNamespace(
        scene=SimpleNamespace(frame=SimpleNamespace(frame_id=1, sim_time=0.0)),
        gpu_frame=object(),
        cache=FakeCache(),
        stream="stream",
        device="cuda:fake",
        bvh=SimpleNamespace(stats=SimpleNamespace(supports_refit=False)),
        bvh_backend="cuda_lbvh",
        bvh_split_strategy="partition",
    )
    monkeypatch.setattr(go2_session, "wp", SimpleNamespace(synchronize_event=lambda event: None))
    monkeypatch.setattr(go2_session, "build_cuda_lbvh_from_snapshot", fake_cuda_build)

    frame = go2_backend.Go2RenderPipeline(session=session).begin_frame(
        frame_inputs=object(),
        env_idx=0,
    )

    assert frame.bvh is rebuilt_bvh
    assert math.isnan(float(frame.prepare_timing["accel_refit_ms"]))
    assert frame.prepare_timing["accel_rebuild_ms"] >= 0.0


def test_torch_async_readback_warmup_uses_pipeline_frame_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    compute = SimpleNamespace(ready_event=object())
    camera = SimpleNamespace(frame_id=5, sim_time=0.3, env_idx=2)
    captured: dict[str, object] = {}

    class FakeFrame:
        def __init__(self):
            self.requests = []

        def render(self, request):
            self.requests.append(request)
            return RuntimeRenderResult(compute=compute)

    class FakePipeline:
        def __init__(self):
            self.session = SimpleNamespace(scene=object())
            self.frame = FakeFrame()
            self.begin_calls = []

        def begin_frame(self, *, env_idx=0):
            self.begin_calls.append(env_idx)
            return self.frame

    def fake_from_warmup_result(warmup_result, *, channels, ring_depth):
        captured["warmup_result"] = warmup_result
        captured["channels"] = channels
        captured["ring_depth"] = ring_depth
        return "ring"

    monkeypatch.setattr(go2_backend, "_build_video_camera", lambda scene, args, frame_index: camera)
    monkeypatch.setattr(
        delivery.TorchAsyncReadbackRing,
        "from_warmup_result",
        staticmethod(fake_from_warmup_result),
    )

    pipeline = FakePipeline()
    delivery_request = go2_backend._video_delivery_request(
        readback_mode="rgb",
        delivery_mode="torch_async",
        ring_depth=4,
        write_frames=False,
    )

    facade = delivery.VideoDeliveryFacade.create(
        request=delivery_request,
        delivery_policy_label="torch_async",
        frame_dir=tmp_path,
        pack_rgb8=lambda result: result,
        synchronize_event=lambda event: None,
        warmup_result_factory=lambda: go2_backend._build_torch_async_warmup_result(
            pipeline=pipeline,
            args=SimpleNamespace(render_profile=True, fail_on_overflow=False),
            delivery_request=delivery_request,
        ),
    )

    assert facade._readback_ring == "ring"
    assert pipeline.begin_calls == [2]
    assert len(pipeline.frame.requests) == 1
    request = pipeline.frame.requests[0]
    assert request.camera is camera
    assert request.diagnostics.profile_timing is True
    assert request.diagnostics.traversal_counters is True
    assert captured["warmup_result"] is compute
    assert captured["ring_depth"] == 4
    assert "rgb" in captured["channels"]
    assert "shadow_traversal_ray_count" in captured["channels"]


def test_video_delivery_request_maps_lab_options_to_runtime_api():
    request = go2_backend._video_delivery_request(
        readback_mode="none",
        delivery_mode="sync",
        ring_depth=2,
        write_frames=False,
    )

    assert request.payload is RuntimeReadbackPayload.NONE
    assert request.policy is RuntimeDeliveryPolicy.DEVICE_ONLY
    assert request.write_policy is RuntimeWritePolicy.NONE

    request = go2_backend._video_delivery_request(
        readback_mode="rgb8",
        delivery_mode="torch_async",
        ring_depth=3,
        write_frames=True,
    )

    assert request.payload is RuntimeReadbackPayload.RGB8
    assert request.policy is RuntimeDeliveryPolicy.TORCH_ASYNC_ORDERED
    assert request.ring_depth == 3
    assert request.write_policy is RuntimeWritePolicy.PNG_SEQUENCE

    request = go2_backend._video_delivery_request(
        readback_mode="full",
        delivery_mode="sync",
        ring_depth=2,
        write_frames=False,
    )

    assert request.payload is RuntimeReadbackPayload.FULL
    assert request.policy is RuntimeDeliveryPolicy.SYNC_HOST

    with pytest.raises(ValueError, match="RGB or RGB8"):
        go2_backend._video_delivery_request(
            readback_mode="full",
            delivery_mode="torch_async",
            ring_depth=2,
            write_frames=False,
        )


def test_sync_rgb8_delivery_packs_after_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_result = object()
    packed_result = SimpleNamespace(ready_event=object())
    calls: list[object] = []
    request = delivery.video_delivery_request(
        readback_mode="rgb8",
        delivery_mode="sync",
        ring_depth=2,
        write_frames=False,
    )

    def fake_pack(result):
        calls.append(result)
        return packed_result

    def fake_stage(result, channels, *, canonical_dtypes):
        assert result is packed_result
        assert "rgb8" in channels
        assert canonical_dtypes is False
        return {
            "rgb8": np.zeros((1, 3), dtype=np.uint8),
            "bvh_stack_overflow_count": np.array([0], dtype=np.int32),
            "shadow_stack_overflow_count": np.array([0], dtype=np.int32),
            "bvh_max_stack_depth": np.array([1], dtype=np.int32),
            "shadow_max_stack_depth": np.array([1], dtype=np.int32),
        }

    monkeypatch.setattr(delivery, "rgb_pack_available", lambda: True)
    monkeypatch.setattr(delivery, "stage_optical_channels", fake_stage)
    facade = delivery.VideoDeliveryFacade(
        request=request,
        delivery_policy_label="sync",
        frame_dir=tmp_path,
        pack_rgb8=fake_pack,
        synchronize_event=lambda event: None,
    )
    rendered = delivery.RenderedVideoFrame(
        frame_index=0,
        camera=SimpleNamespace(sim_time=0.0),
        result=raw_result,
        camera_rays_ms=float("nan"),
        render_execute_ms=1.0,
        render_profile_row=go2_backend._render_profile_row(None),
        include_shadow_traversal_stats=False,
    )

    delivered = facade.submit(rendered, frame_start=0.0)

    assert delivered is not None
    assert calls == [raw_result]
    assert delivered.delivery_timing.pack_rgb8_ms >= 0.0
    assert delivered.delivery_timing.readback_host_ms >= 0.0
    assert not hasattr(rendered, "pack_rgb8_ms")


def test_sync_video_readback_none_row_does_not_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_render_video_frame(pipeline, args, frame_index, ray_cache):
        return delivery.RenderedVideoFrame(
            frame_index=frame_index,
            camera=SimpleNamespace(sim_time=0.0),
            result=object(),
            camera_rays_ms=float("nan"),
            render_execute_ms=1.25,
            render_profile_row=go2_backend._render_profile_row(None),
            include_shadow_traversal_stats=False,
            geometry_mode="dynamic_rigid",
            prepare_timing={
                "snapshot_ms": 0.5,
                "accel_refit_ms": 0.25,
                "accel_rebuild_ms": float("nan"),
            },
        )

    def fail_if_staged(*args, **kwargs):
        raise AssertionError("readback=none should not stage host channels")

    monkeypatch.setattr(go2_backend, "_render_video_frame", fake_render_video_frame)
    monkeypatch.setattr(delivery, "stage_optical_channels", fail_if_staged)
    monkeypatch.setattr(delivery, "stage_optical_compute_result_to_host", fail_if_staged)

    frame_timing_csv = tmp_path / "frame_timing.csv"
    rows = go2_backend._run_video_benchmark(
        pipeline=SimpleNamespace(session=SimpleNamespace(scene=object())),
        args=SimpleNamespace(
            video_readback_delivery="sync",
            write_frames=False,
            video_readback="none",
            video_readback_ring_depth=2,
            video_raygen="gpu",
            video_ray_cache="off",
            video_frames=1,
            video_fps=30.0,
            render_profile=False,
            fail_on_overflow=False,
            progress_every=0,
            frame_timing_csv=str(frame_timing_csv),
            lab_frame_defaults={
                "readback_payload": "none",
                "delivery_policy": "sync",
            },
        ),
        out_dir=tmp_path,
    )

    row = rows._rows[0]
    assert row["readback_mode"] == "none"
    assert row["write_mode"] == "none"
    assert row["delivery_policy"] == "sync"
    assert row["geometry_mode"] == "dynamic_rigid"
    assert row["snapshot_ms"] == 0.5
    assert row["accel_refit_ms"] == 0.25
    assert math.isnan(float(row["pack_rgb8_ms"]))
    assert math.isnan(float(row["accel_rebuild_ms"]))
    assert math.isnan(float(row["readback_host_ms"]))
    assert math.isnan(float(row["image_build_ms"]))
    assert math.isnan(float(row["encode_or_write_ms"]))
    assert row["frame_path"] == ""
    assert frame_timing_csv.exists()


def test_torch_async_delivery_facade_reports_ring_depth_blocking_modes(tmp_path: Path):
    request = delivery.video_delivery_request(
        readback_mode="rgb",
        delivery_mode="torch_async",
        ring_depth=1,
        write_frames=False,
    )

    class FakeJob:
        submit_ms = 0.25

        def __init__(self, frame_index: int):
            self.frame_index = frame_index
            self.sync_count = 0

        def synchronize(self):
            self.sync_count += 1
            return 0.5

        def copy_elapsed_ms(self):
            return 0.75

        def host_channels(self):
            return {
                "rgb": np.zeros((1, 3), dtype=np.float32),
                "bvh_stack_overflow_count": np.array([0], dtype=np.int32),
                "shadow_stack_overflow_count": np.array([0], dtype=np.int32),
                "bvh_max_stack_depth": np.array([1], dtype=np.int32),
                "shadow_max_stack_depth": np.array([1], dtype=np.int32),
            }

    class FakeRing:
        def __init__(self, ring_depth: int):
            self.ring_depth = ring_depth

        def submit(self, result, *, frame_index: int):
            return FakeJob(frame_index)

    def rendered(frame_index: int):
        return delivery.RenderedVideoFrame(
            frame_index=frame_index,
            camera=SimpleNamespace(sim_time=float(frame_index)),
            result=object(),
            camera_rays_ms=float("nan"),
            render_execute_ms=1.0,
            render_profile_row=go2_backend._render_profile_row(None),
            include_shadow_traversal_stats=False,
        )

    facade = delivery.VideoDeliveryFacade(
        request=request,
        delivery_policy_label="torch_async",
        frame_dir=tmp_path,
        pack_rgb8=lambda result: result,
        synchronize_event=lambda event: None,
        readback_ring=FakeRing(1),
    )
    assert facade.submit(rendered(0), frame_start=1.0) is None
    assert facade.complete_available() == []
    completed = facade.complete_available(latest_rendered_frame_index=1)
    assert [frame.completed_frame_index for frame in completed] == [0]
    assert facade.submit(rendered(1), frame_start=2.0) is None
    assert completed[0].readback_ring_depth == 1
    assert completed[0].readback_ring_block_count == 1

    request = delivery.video_delivery_request(
        readback_mode="rgb",
        delivery_mode="torch_async",
        ring_depth=2,
        write_frames=False,
    )
    facade = delivery.VideoDeliveryFacade(
        request=request,
        delivery_policy_label="torch_async",
        frame_dir=tmp_path,
        pack_rgb8=lambda result: result,
        synchronize_event=lambda event: None,
        readback_ring=FakeRing(2),
    )
    assert facade.submit(rendered(0), frame_start=1.0) is None
    assert facade.complete_available() == []
    assert facade.complete_available(latest_rendered_frame_index=1) == []
    assert facade.submit(rendered(1), frame_start=2.0) is None
    completed = facade.complete_available(latest_rendered_frame_index=1)
    assert [frame.completed_frame_index for frame in completed] == [0]
    assert completed[0].readback_ring_depth == 2
    assert completed[0].readback_ring_block_count == 0


def test_torch_async_delivery_facade_flush_completes_pending_frame(tmp_path: Path):
    request = delivery.video_delivery_request(
        readback_mode="rgb",
        delivery_mode="torch_async",
        ring_depth=2,
        write_frames=False,
    )

    class FakeJob:
        submit_ms = 0.1

        def synchronize(self):
            return 0.2

        def copy_elapsed_ms(self):
            return 0.3

        def host_channels(self):
            return {
                "rgb": np.zeros((1, 3), dtype=np.float32),
                "bvh_stack_overflow_count": np.array([0], dtype=np.int32),
                "shadow_stack_overflow_count": np.array([0], dtype=np.int32),
                "bvh_max_stack_depth": np.array([1], dtype=np.int32),
                "shadow_max_stack_depth": np.array([1], dtype=np.int32),
            }

    class FakeRing:
        ring_depth = 2

        def submit(self, result, *, frame_index: int):
            return FakeJob()

    facade = delivery.VideoDeliveryFacade(
        request=request,
        delivery_policy_label="torch_async",
        frame_dir=tmp_path,
        pack_rgb8=lambda result: result,
        synchronize_event=lambda event: None,
        readback_ring=FakeRing(),
    )
    rendered = delivery.RenderedVideoFrame(
        frame_index=0,
        camera=SimpleNamespace(sim_time=0.0),
        result=object(),
        camera_rays_ms=float("nan"),
        render_execute_ms=1.0,
        render_profile_row=go2_backend._render_profile_row(None),
        include_shadow_traversal_stats=False,
    )

    assert facade.submit(rendered, frame_start=1.0) is None
    completed = facade.flush()

    assert [frame.completed_frame_index for frame in completed] == [0]
    assert completed[0].readback_ring_depth == 2
    assert completed[0].readback_ring_block_count == 0
    assert completed[0].delivery_timing.readback_submit_ms == 0.1
    assert completed[0].delivery_timing.readback_wait_ms == 0.2
    assert completed[0].delivery_timing.readback_host_ms == 0.3


def test_video_frame_timing_row_builder_requires_bound_request():
    builder = delivery.VideoFrameTimingRowBuilder(
        delivery.VideoDeliveryRunConfig(
            video_fps=30.0,
            video_frames=1,
            video_raygen="gpu",
            video_ray_cache="off",
            delivery_policy_label="sync",
            fail_on_overflow=False,
        )
    )
    rendered = delivery.RenderedVideoFrame(
        frame_index=0,
        camera=SimpleNamespace(sim_time=0.0),
        result=object(),
        camera_rays_ms=float("nan"),
        render_execute_ms=1.0,
        render_profile_row=go2_backend._render_profile_row(None),
        include_shadow_traversal_stats=False,
    )
    delivered = delivery.DeliveredVideoFrame(
        rendered=rendered,
        completed_frame_index=0,
        host_channels={},
        delivery_timing=DeliveryTimingSummary(),
        observed_frame_ms=1.0,
    )

    with pytest.raises(RuntimeError, match="bind_request"):
        builder.build_row(delivered)


def test_delivered_video_frame_bridges_to_runtime_delivery_result():
    rendered = delivery.RenderedVideoFrame(
        frame_index=1,
        camera=SimpleNamespace(sim_time=0.0),
        result=object(),
        camera_rays_ms=float("nan"),
        render_execute_ms=1.0,
        render_profile_row=go2_backend._render_profile_row(None),
        include_shadow_traversal_stats=False,
    )
    host_channels = {"rgb": np.zeros((1, 3), dtype=np.float32)}
    timing = DeliveryTimingSummary(
        pack_rgb8_ms=0.1,
        readback_submit_ms=0.2,
        readback_wait_ms=0.3,
        readback_host_ms=0.4,
        image_build_ms=0.5,
        encode_write_ms=0.6,
    )
    delivered = delivery.DeliveredVideoFrame(
        rendered=rendered,
        completed_frame_index=1,
        host_channels=host_channels,
        delivery_timing=timing,
        observed_frame_ms=2.0,
        frame_path="frames/rgb_000001.png",
        readback_lag_frames=1,
        readback_ring_depth=2,
        readback_ring_block_count=3,
        overlap_ratio=0.25,
    )

    runtime = delivered.to_runtime_delivery_result()

    assert isinstance(runtime, RuntimeDeliveryResult)
    assert runtime.completed_frame_index == 1
    assert runtime.frame_index == 1
    assert runtime.host_channels is host_channels
    assert runtime.delivery is timing
    assert runtime.lag_frames == 1
    assert runtime.ring_depth == 2
    assert runtime.ring_block_count == 3
    assert not hasattr(runtime, "observed_frame_ms")
    assert not hasattr(runtime, "frame_path")
    assert not hasattr(runtime, "overlap_ratio")


def test_rendered_video_frame_render_execute_ms_prefers_runtime_timing():
    compute = SimpleNamespace(ready_event=object())
    rendered = delivery.RenderedVideoFrame(
        frame_index=1,
        camera=SimpleNamespace(sim_time=0.0),
        result=object(),
        camera_rays_ms=float("nan"),
        render_execute_ms=1.0,
        render_profile_row=go2_backend._render_profile_row(None),
        include_shadow_traversal_stats=False,
        render=RuntimeRenderResult(
            compute=compute,
            timing={"render_execute_ms": 2.0},
            render_timing=RenderTimingSummary(execute_ms=3.0),
        ),
    )

    assert rendered.render_execute_ms_value() == 3.0


def test_rendered_video_frame_render_execute_ms_falls_back_to_runtime_mapping():
    compute = SimpleNamespace(ready_event=object())
    rendered = delivery.RenderedVideoFrame(
        frame_index=1,
        camera=SimpleNamespace(sim_time=0.0),
        result=object(),
        camera_rays_ms=float("nan"),
        render_execute_ms=1.0,
        render_profile_row=go2_backend._render_profile_row(None),
        include_shadow_traversal_stats=False,
        render=RuntimeRenderResult(
            compute=compute,
            timing={"render_execute_ms": 2.0},
            render_timing=None,
        ),
    )

    assert rendered.render_execute_ms_value() == 2.0


def test_rendered_video_frame_render_execute_ms_preserves_stored_fallback():
    rendered = delivery.RenderedVideoFrame(
        frame_index=1,
        camera=SimpleNamespace(sim_time=0.0),
        result=object(),
        camera_rays_ms=float("nan"),
        render_execute_ms=1.0,
        render_profile_row=go2_backend._render_profile_row(None),
        include_shadow_traversal_stats=False,
    )

    assert rendered.render_execute_ms_value() == 1.0


def test_video_frame_timing_row_builder_torch_async_row_and_progress():
    request = delivery.video_delivery_request(
        readback_mode="rgb",
        delivery_mode="torch_async",
        ring_depth=2,
        write_frames=False,
    )
    builder = delivery.VideoFrameTimingRowBuilder(
        delivery.VideoDeliveryRunConfig(
            video_fps=20.0,
            video_frames=4,
            video_raygen="gpu",
            video_ray_cache="off",
            delivery_policy_label="torch_async",
            fail_on_overflow=False,
        )
    ).bind_request(request)
    render_profile = go2_backend._render_profile_row(
        [("raygen_kernel", 0.1), ("first_hit_kernel_ms", 0.2), ("shade_kernel", 0.3)],
        render_execute_ms=1.0,
    )
    rendered = delivery.RenderedVideoFrame(
        frame_index=1,
        camera=SimpleNamespace(sim_time=0.05),
        result=object(),
        camera_rays_ms=float("nan"),
        render_execute_ms=99.0,
        render_profile_row=render_profile,
        include_shadow_traversal_stats=True,
        geometry_mode="dynamic_rigid",
        prepare_timing={
            "snapshot_ms": 0.4,
            "accel_refit_ms": 0.5,
            "accel_rebuild_ms": float("nan"),
        },
        render=RuntimeRenderResult(
            compute=object(),
            timing={"render_execute_ms": 1.0},
            render_timing=None,
        ),
    )
    host_channels = {
        "rgb": np.zeros((1, 3), dtype=np.float32),
        "bvh_stack_overflow_count": np.array([0], dtype=np.int32),
        "shadow_stack_overflow_count": np.array([0], dtype=np.int32),
        "bvh_max_stack_depth": np.array([2], dtype=np.int32),
        "shadow_max_stack_depth": np.array([3], dtype=np.int32),
        "shadow_traversal_ray_count": np.array([4], dtype=np.int32),
        "shadow_traversal_triangle_test_count": np.array([5], dtype=np.int32),
    }
    delivered = delivery.DeliveredVideoFrame(
        rendered=rendered,
        completed_frame_index=1,
        host_channels=host_channels,
        delivery_timing=DeliveryTimingSummary(
            pack_rgb8_ms=0.6,
            readback_submit_ms=0.7,
            readback_wait_ms=0.8,
            readback_host_ms=0.9,
            image_build_ms=float("nan"),
            encode_write_ms=float("nan"),
        ),
        observed_frame_ms=2.0,
        readback_lag_frames=1,
        readback_ring_depth=2,
        readback_ring_block_count=0,
        overlap_ratio=0.25,
    )

    row = builder.build_row(delivered)
    progress = builder.progress_line(delivered)

    assert row["frame_index"] == 1
    assert row["completed_frame_index"] == 1
    assert row["delivery_policy"] == "torch_async"
    assert row["readback_mode"] == "torch_async_rgb"
    assert row["geometry_mode"] == "dynamic_rigid"
    assert row["snapshot_ms"] == 0.4
    assert row["accel_refit_ms"] == 0.5
    assert row["pack_rgb8_ms"] == 0.6
    assert row["readback_lag_frames"] == 1
    assert row["readback_ring_depth"] == 2
    assert row["readback_ring_block_count"] == 0
    assert row["overlap_ratio"] == 0.25
    assert row["shadow_traversal_ray_count"] == 4
    assert row["shadow_traversal_triangle_test_count"] == 5
    assert "pack_rgb8=0.600ms" in progress
    assert "overlap=0.250" in progress
    assert "lag=1" in progress


def test_video_readback_channels_include_shadow_traversal_stats_only_when_requested():
    assert "shadow_traversal_ray_count" not in go2_backend._video_readback_channels("rgb8")

    channels = go2_backend._video_readback_channels("rgb8", include_shadow_traversal_stats=True)

    assert "rgb8" in channels
    assert "shadow_stack_overflow_count" in channels
    assert "shadow_traversal_ray_count" in channels
    assert "shadow_traversal_triangle_test_count" in channels


def test_go2_video_ordered_static_preset_is_currently_implemented():
    config = get_preset("go2_video_ordered_static")

    assert config.scenario_family is OpticalLabScenarioFamily.VIDEO_ORDERED_EXPORT
    assert config.geometry_mode is GeometryMode.STATIC
    assert config.delivery_policy is DeliveryPolicy.SYNC
    assert config.width == DEFAULT_RENDER_WIDTH
    assert config.height == DEFAULT_RENDER_HEIGHT
    config.validate_implemented()


def test_synthetic_dynamic_smoke_preset_is_currently_implemented():
    config = get_preset("synthetic_body_triangle_dynamic_smoke")

    assert config.scenario_family is OpticalLabScenarioFamily.VIDEO_ORDERED_EXPORT
    assert config.scene_preset == "synthetic_body_triangle"
    assert config.geometry_mode is GeometryMode.DYNAMIC_RIGID
    assert config.accel_policy is AccelPolicy.REFIT_EACH_FRAME
    assert config.readback_payload is ReadbackPayload.RGB
    config.validate_implemented()


def test_default_render_resolution_is_1080p():
    config = OpticalLabScenarioConfig(
        scenario_name="default_resolution",
        scenario_family=OpticalLabScenarioFamily.RENDER_BENCH,
    )

    assert DEFAULT_RENDER_WIDTH == 1920
    assert DEFAULT_RENDER_HEIGHT == 1080
    assert config.width == 1920
    assert config.height == 1080


def test_lab_default_warmup_covers_readback_startup_spikes(tmp_path: Path):
    assert DEFAULT_LAB_WARMUP_RENDERS == 5
    assert LabRunOptions(out=tmp_path / "run").warmup_renders == 5
    assert MatrixRunOptions(out=tmp_path / "matrix").warmup_renders == 5


def test_async_readback_dependency_probe_is_import_safe():
    assert isinstance(async_readback.torch_async_readback_available(), bool)
    if async_readback.torch_async_readback_available():
        assert async_readback.torch_async_readback_import_error() is None


def test_async_readback_ring_rejects_invalid_depth():
    with pytest.raises(ValueError, match="ring_depth"):
        async_readback.TorchAsyncReadbackRing(
            channels=("rgb",),
            ring_depth=0,
            copy_stream=object(),
            slots=[],
        )


def test_async_readback_job_uses_start_to_done_event_order():
    class FakeStartEvent:
        def __init__(self):
            self.elapsed_to = None

        def elapsed_time(self, done_event):
            self.elapsed_to = done_event
            return 12.5

    start_event = FakeStartEvent()
    done_event = object()
    slot = async_readback.TorchAsyncReadbackSlot(
        index=0,
        host_tensors={},
        copy_start_event=start_event,
        copy_done_event=done_event,
    )
    job = async_readback.TorchAsyncReadbackJob(
        frame_index=0,
        slot=slot,
        submit_ms=0.1,
        result=object(),
    )

    assert job.copy_elapsed_ms() == 12.5
    assert start_event.elapsed_to is done_event


def test_rgb_pack_dependency_probe_is_import_safe():
    assert isinstance(rgb_pack.rgb_pack_available(), bool)
    if rgb_pack.rgb_pack_available():
        assert rgb_pack.rgb_pack_import_error() is None


def test_rgb_pack_raises_import_error_when_warp_is_unavailable(monkeypatch):
    error = RuntimeError("warp unavailable for test")
    monkeypatch.setattr(rgb_pack, "wp", None)
    monkeypatch.setattr(rgb_pack, "_WARP_IMPORT_ERROR", error)

    with pytest.raises(ImportError, match="RGB8 packing requires warp") as exc_info:
        rgb_pack.pack_linear_rgb_to_preview_uint8(object())

    assert exc_info.value.__cause__ is error


class _FakeWpArray:
    def __init__(self, values, *, dtype=np.float32, device="cuda:fake"):
        self.values = np.asarray(values, dtype=dtype).copy()
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.device = device

    def numpy(self):
        return self.values.copy()


class _FakeWpModule:
    @staticmethod
    def zeros(shape, *, dtype=None, device=None):
        resolved_dtype = dtype or np.float32
        return _FakeWpArray(
            np.zeros(shape, dtype=resolved_dtype),
            dtype=resolved_dtype,
            device=device,
        )

    @staticmethod
    def array(values, *, dtype=None, device=None):
        return _FakeWpArray(values, dtype=dtype or np.float32, device=device)

    @staticmethod
    def copy(dst, src):
        dst.values[...] = src.values


def _fake_gpu_pose_frame() -> GpuPublishedFrame:
    translations = np.array([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]], dtype=np.float32)
    frame = dynamic_frames.make_gpu_pose_frame(
        wp_module=_FakeWpModule,
        translations=translations,
        slot_id=7,
        frame_id=11,
        sim_time=0.11,
        step_index=11,
    )
    frame.q_wp = object()
    frame.qdot_wp = object()
    frame.v_bodies_wp = object()
    frame.contact_count_wp = object()
    frame.contact_cache_ref = object()
    frame.telemetry_ref = object()
    frame.ready_event = object()
    return frame


def test_dynamic_frame_clone_is_pose_only_and_independent():
    frame = _fake_gpu_pose_frame()

    cloned = dynamic_frames.clone_gpu_published_pose_frame(
        frame,
        wp_module=_FakeWpModule,
        frame_id=12,
        sim_time=0.12,
        step_index=12,
    )

    assert dynamic_frames.gpu_pose_shape(cloned) == (1, 2)
    assert cloned.frame_id == 12
    assert cloned.sim_time == 0.12
    assert cloned.q_wp is None
    assert cloned.slot_meta is None
    assert cloned.x_world_R_wp is not frame.x_world_R_wp
    assert cloned.x_world_r_wp is not frame.x_world_r_wp
    cloned.x_world_r_wp.values[0, 1, 0] = 99.0
    assert frame.x_world_r_wp.numpy()[0, 1, 0] == 1.0


def test_dynamic_frame_perturb_applies_translation_offsets_without_mutating_source():
    frame = _fake_gpu_pose_frame()

    moved = dynamic_frames.clone_and_perturb_gpu_published_pose_frame(
        frame,
        wp_module=_FakeWpModule,
        translation_offsets={(0, 1): [0.5, -1.0, 2.0]},
        frame_id=13,
    )

    assert moved.frame_id == 13
    assert moved.x_world_r_wp.numpy()[0, 1].tolist() == pytest.approx([1.5, 1.0, 5.0])
    assert frame.x_world_r_wp.numpy()[0, 1].tolist() == pytest.approx([1.0, 2.0, 3.0])

    with pytest.raises(IndexError, match="body_idx"):
        dynamic_frames.clone_and_perturb_gpu_published_pose_frame(
            frame,
            wp_module=_FakeWpModule,
            translation_offsets={(0, 2): [0.0, 0.0, 0.0]},
        )
    with pytest.raises(IndexError, match="env_idx"):
        dynamic_frames.clone_and_perturb_gpu_published_pose_frame(
            frame,
            wp_module=_FakeWpModule,
            translation_offsets={(1, 0): [0.0, 0.0, 0.0]},
        )


def test_dynamic_frame_tiny_body_bound_scene_builder_is_import_safe():
    registry = dynamic_frames.make_body_bound_triangle_registry()

    assert len(registry.instances) == 1
    assert registry.instances[0].body_index == 0
    assert dynamic_frames.gpu_pose_shape(_fake_gpu_pose_frame()) == (1, 2)


def test_reserved_lab_modes_fail_loudly():
    config = OpticalLabScenarioConfig(
        scenario_name="future_dynamic",
        scenario_family=OpticalLabScenarioFamily.VIDEO_ORDERED_EXPORT,
        geometry_mode=GeometryMode.DYNAMIC_RIGID,
    )
    with pytest.raises(NotImplementedError, match="dynamic_rigid"):
        config.validate_implemented()

    config = OpticalLabScenarioConfig(
        scenario_name="future_async",
        scenario_family=OpticalLabScenarioFamily.VIDEO_ORDERED_EXPORT,
        accel_policy=AccelPolicy.REFIT_EACH_FRAME,
    )
    with pytest.raises(NotImplementedError, match="refit_each_frame"):
        config.validate_implemented()


def test_run_overrides_update_config_without_changing_preset_source():
    preset = get_preset("go2_video_ordered_static")
    updated = apply_run_overrides(
        preset,
        device="cuda:1",
        width=320,
        height=240,
        readback="none",
        shadows=False,
        write_frames=True,
    )

    assert preset.device == "cuda:0"
    assert preset.readback_payload.value == "rgb"
    assert updated.device == "cuda:1"
    assert updated.width == 320
    assert updated.height == 240
    assert updated.readback_payload.value == "none"
    assert updated.output_profile == "render_only"
    assert updated.shadows is False
    assert updated.write_policy.value == "png_sequence"


def test_run_overrides_support_rgb8_delivery_payload():
    updated = apply_run_overrides(
        get_preset("go2_video_ordered_static"),
        readback="rgb8",
    )

    assert updated.readback_payload is ReadbackPayload.RGB8
    assert updated.output_profile == "rgb_preview"
    updated.validate_implemented()
    validate_run(
        updated,
        LabRunOptions(
            out=Path("out"),
            video_readback_delivery="torch_async",
        ),
    )


def test_lab_runner_translates_go2_preset_to_menagerie_example_args(tmp_path: Path):
    config = apply_run_overrides(
        get_preset("go2_video_ordered_static"),
        device="cuda:1",
        readback="rgb",
        write_frames=False,
    )
    options = LabRunOptions(
        out=tmp_path / "run",
        frames=3,
        warmup_renders=1,
        progress_every=0,
        video_raygen="gpu",
        video_readback_delivery="torch_async",
        video_readback_ring_depth=3,
    )

    args = build_menagerie_example_args(config, options)

    assert args.device == "cuda:1"
    assert args.out == str(tmp_path / "run")
    assert args.bvh_backend == "cuda_lbvh"
    assert args.video_frames == 3
    assert args.video_mode == "camera_orbit"
    assert args.video_raygen == "gpu"
    assert args.video_ray_cache == "off"
    assert args.video_readback == "rgb"
    assert args.video_readback_delivery == "torch_async"
    assert args.video_readback_ring_depth == 3
    assert args.frame_timing_csv == str(tmp_path / "run" / "frame_timing.csv")
    assert args.timing_csv == str(tmp_path / "run" / "timing.csv")
    assert args.write_frames is False
    assert args.no_shadows is False
    assert args.lab_frame_defaults["scenario_name"] == "go2_video_ordered_static"
    assert args.lab_frame_defaults["device"] == "cuda:1"
    assert args.lab_frame_defaults["readback_payload"] == "rgb"


def test_lab_runner_translates_dynamic_smoke_preset_to_video_args(tmp_path: Path):
    config = get_preset("synthetic_body_triangle_dynamic_smoke")
    options = LabRunOptions(out=tmp_path / "dynamic", frames=2, progress_every=0)

    args = build_menagerie_example_args(config, options)

    assert args.scene_preset == "synthetic_body_triangle"
    assert args.bvh_backend == "cpu"
    assert args.video_mode == "fixed_view"
    assert args.video_geometry_mode == "dynamic_rigid"
    assert args.video_readback == "rgb"
    assert args.no_shadows is True
    assert args.lab_frame_defaults["scenario_name"] == "synthetic_body_triangle_dynamic_smoke"
    assert args.lab_frame_defaults["geometry_mode"] == "dynamic_rigid"
    assert args.lab_frame_defaults["accel_policy"] == "refit_each_frame"


def test_go2_backend_configures_synthetic_dynamic_video_frames(monkeypatch: pytest.MonkeyPatch):
    base_frame = dynamic_frames.make_gpu_pose_frame(
        wp_module=_FakeWpModule,
        translations=np.zeros((1, 1, 3), dtype=np.float32),
        frame_id=20,
        sim_time=2.0,
        step_index=20,
    )
    args = SimpleNamespace(
        scene_preset="synthetic_body_triangle",
        video_frames=3,
        video_fps=10.0,
    )
    monkeypatch.setattr(go2_backend, "wp", _FakeWpModule)

    go2_backend._configure_dynamic_video_frame_inputs(
        args,
        SimpleNamespace(gpu_frame=base_frame),
    )

    assert args.video_geometry_mode == "dynamic_rigid"
    assert [frame.frame_id for frame in args.video_frame_inputs] == [20, 21, 22]
    assert [frame.sim_time for frame in args.video_frame_inputs] == pytest.approx([2.0, 2.1, 2.2])
    assert args.video_frame_inputs[2].x_world_r_wp.numpy()[0, 0].tolist() == pytest.approx([0.0, 0.0, 0.08])
    assert base_frame.x_world_r_wp.numpy()[0, 0].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_lab_runner_writes_serialized_scenario_config(tmp_path: Path):
    config = get_preset("go2_video_ordered_static")
    options = LabRunOptions(out=tmp_path / "run", frames=2)
    path = tmp_path / "run" / "scenario_config.json"

    write_scenario_config(path, config, options)

    payload = json.loads(path.read_text())
    assert payload["scenario"]["scenario_name"] == "go2_video_ordered_static"
    assert payload["scenario"]["accel_backend"] == "cuda_lbvh"
    assert payload["scenario"]["readback_payload"] == "rgb"
    assert payload["run_options"]["out"] == str(tmp_path / "run")
    assert payload["run_options"]["frames"] == 2


def test_lab_runner_rejects_unsatisfiable_readback_combinations(tmp_path: Path):
    config = apply_run_overrides(
        get_preset("go2_video_ordered_static"),
        readback="none",
        write_frames=True,
    )
    with pytest.raises(ValueError, match="write_policy"):
        validate_run(config, LabRunOptions(out=tmp_path / "run", fail_on_overflow=False))

    config = apply_run_overrides(get_preset("go2_video_ordered_static"), readback="none")
    with pytest.raises(ValueError, match="fail_on_overflow"):
        validate_run(config, LabRunOptions(out=tmp_path / "run"))

    validate_run(config, LabRunOptions(out=tmp_path / "run", fail_on_overflow=False))


def test_lab_runner_rejects_gpu_raygen_with_ray_cache(tmp_path: Path):
    config = get_preset("go2_video_ordered_static")

    with pytest.raises(ValueError, match="video_ray_cache"):
        validate_run(
            config,
            LabRunOptions(
                out=tmp_path / "run",
                video_raygen="gpu",
                video_ray_cache="precompute",
            ),
        )


def test_lab_runner_rejects_async_readback_for_non_rgb_payload(tmp_path: Path):
    config = apply_run_overrides(get_preset("go2_video_ordered_static"), readback="full")

    with pytest.raises(ValueError, match="torch_async"):
        validate_run(
            config,
            LabRunOptions(
                out=tmp_path / "run",
                video_readback_delivery="torch_async",
            ),
        )


def test_lab_runner_rejects_invalid_async_ring_depth(tmp_path: Path):
    config = get_preset("go2_video_ordered_static")

    with pytest.raises(ValueError, match="video_readback_ring_depth"):
        validate_run(
            config,
            LabRunOptions(
                out=tmp_path / "run",
                video_readback_ring_depth=0,
            ),
        )


def test_run_scenario_smoke_delegates_to_go2_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[object] = []

    def fake_render_many_views(args):
        calls.append(args)

    monkeypatch.setattr(go2_backend, "render_many_views", fake_render_many_views)

    config = apply_run_overrides(
        get_preset("go2_video_ordered_static"),
        width=80,
        height=60,
        readback="none",
    )
    options = LabRunOptions(out=tmp_path / "run", frames=1, fail_on_overflow=False)

    run_scenario(config, options)

    assert len(calls) == 1
    args = calls[0]
    assert args.width == 80
    assert args.height == 60
    assert args.video_readback == "none"
    assert (tmp_path / "run" / "scenario_config.json").exists()


def test_reports_format_summary_rows():
    lines = format_summary_rows(
        [
            {
                "phase": "render",
                "count": 2.0,
                "p50_ms": 1.25,
                "p90_ms": 1.75,
                "mean_ms": 1.5,
            },
            {
                "phase": "setup",
                "count": 1.0,
                "mean_ms": 10.0,
            },
        ]
    )

    assert lines == [
        "render: repeat=2, p50=1.250, p90=1.750, mean=1.500",
        "setup: 10.000",
    ]


def test_cli_describe_prints_preset(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["optical_pipeline_lab", "describe", "--preset", "go2_video_ordered_static"],
    )

    lab_main.main()

    captured = capsys.readouterr()
    assert "scenario_name: go2_video_ordered_static" in captured.out
    assert "accel_backend: cuda_lbvh" in captured.out


def test_cli_run_dispatches_to_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[object, object]] = []

    def fake_run_scenario(config, options):
        calls.append((config, options))

    monkeypatch.setattr(lab_main, "run_scenario", fake_run_scenario)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optical_pipeline_lab",
            "run",
            "--preset",
            "go2_video_ordered_static",
            "--out",
            str(tmp_path / "run"),
            "--device",
            "cuda:1",
            "--width",
            "80",
            "--height",
            "60",
            "--frames",
            "2",
            "--readback",
            "rgb",
            "--video-readback-delivery",
            "torch_async",
            "--video-readback-ring-depth",
            "3",
            "--no-shadows",
        ],
    )

    lab_main.main()

    assert len(calls) == 1
    config, options = calls[0]
    assert config.device == "cuda:1"
    assert config.width == 80
    assert config.height == 60
    assert config.readback_payload.value == "rgb"
    assert config.shadows is False
    assert options.out == tmp_path / "run"
    assert options.frames == 2
    assert options.video_readback_delivery == "torch_async"
    assert options.video_readback_ring_depth == 3


def test_go2_baseline_suite_cases_are_ordered_and_1080p():
    suite = get_suite("go2_video_ordered_baseline")

    assert suite.name == "go2_video_ordered_baseline"
    assert suite.preset == "go2_video_ordered_static"
    assert [case.name for case in suite.cases] == [
        "smoke_160x120_shadow_readback_none",
        "1080p_shadow_readback_none",
        "1080p_no_shadow_readback_none",
        "1080p_shadow_readback_rgb",
    ]
    assert suite.cases[1].width == DEFAULT_RENDER_WIDTH
    assert suite.cases[1].height == DEFAULT_RENDER_HEIGHT
    assert suite.cases[2].shadows is False

    debug_suite = get_suite("go2_video_ordered_baseline", include_full_debug=True)
    assert debug_suite.cases[-1].name == "1080p_shadow_readback_full"
    assert debug_suite.cases[-1].readback_payload.value == "full"


def test_go2_legacy_960_suite_matches_plan_comparison_cases():
    suite = get_suite("go2_video_ordered_legacy_960")

    assert suite.name == "go2_video_ordered_legacy_960"
    assert suite.preset == "go2_video_ordered_static"
    assert [case.name for case in suite.cases] == [
        "legacy_960x640_shadow_readback_none",
        "legacy_960x640_no_shadow_readback_none",
        "legacy_960x640_shadow_readback_rgb",
    ]
    assert all(case.width == 960 for case in suite.cases)
    assert all(case.height == 640 for case in suite.cases)
    assert suite.cases[0].readback_payload is ReadbackPayload.NONE
    assert suite.cases[1].shadows is False
    assert suite.cases[2].readback_payload is ReadbackPayload.RGB

    debug_suite = get_suite("go2_video_ordered_legacy_960", include_full_debug=True)
    assert debug_suite.cases[-1].name == "legacy_960x640_shadow_readback_full"
    assert debug_suite.cases[-1].readback_payload is ReadbackPayload.FULL


def test_go2_delivery_smoke_suite_covers_sync_and_async_facade_modes():
    suite = get_suite("go2_video_delivery_smoke")

    assert suite.name == "go2_video_delivery_smoke"
    assert suite.preset == "go2_video_ordered_static"
    assert [case.name for case in suite.cases] == [
        "smoke_160x120_shadow_readback_none_sync",
        "smoke_160x120_shadow_readback_rgb_sync",
        "smoke_160x120_shadow_readback_rgb8_torch_async_ring2",
    ]
    assert [case.readback_payload for case in suite.cases] == [
        ReadbackPayload.NONE,
        ReadbackPayload.RGB,
        ReadbackPayload.RGB8,
    ]
    assert [case.video_readback_delivery for case in suite.cases] == [
        "sync",
        "sync",
        "torch_async",
    ]
    assert suite.cases[-1].video_readback_ring_depth == 2

    debug_suite = get_suite("go2_video_delivery_smoke", include_full_debug=True)
    assert debug_suite.cases[-1].name == "smoke_160x120_shadow_readback_full_sync"
    assert debug_suite.cases[-1].readback_payload is ReadbackPayload.FULL


def test_matrix_case_delivery_options_flow_to_run_options(tmp_path: Path):
    case = MatrixCase(
        name="rgb8_async",
        width=160,
        height=120,
        readback_payload=ReadbackPayload.RGB8,
        video_readback_delivery="torch_async",
        video_readback_ring_depth=3,
    )

    options = run_options_for_case(case, MatrixRunOptions(out=tmp_path / "matrix"))

    assert options.video_readback_delivery == "torch_async"
    assert options.video_readback_ring_depth == 3
    assert options.fail_on_overflow is True


def test_matrix_suite_runs_cases_and_writes_summary(tmp_path: Path):
    suite = MatrixSuite(
        name="tiny_suite",
        preset="go2_video_ordered_static",
        cases=(
            MatrixCase(
                name="render_only",
                width=160,
                height=120,
                readback_payload=ReadbackPayload.NONE,
            ),
            MatrixCase(
                name="rgb",
                width=320,
                height=240,
                readback_payload=ReadbackPayload.RGB,
            ),
        ),
    )
    calls: list[tuple[object, object]] = []

    def fake_run(config, options):
        calls.append((config, options))
        _write_fake_frame_timing(options.out / "frame_timing.csv")

    rows = run_matrix_suite(
        suite,
        MatrixRunOptions(out=tmp_path / "matrix", frames=2, progress_every=0),
        run_one=fake_run,
    )

    assert len(calls) == 2
    assert calls[0][1].fail_on_overflow is False
    assert calls[1][1].fail_on_overflow is True
    assert rows[0]["status"] == "passed"
    assert rows[0]["fps_mean"] == pytest.approx(1000.0 * 2.0 / 12.0)
    assert rows[0]["frame_p90_ms"] == pytest.approx(7.6)
    assert rows[0]["render_execute_mean_ms"] == 3.0
    assert rows[0]["readback_host_mean_ms"] == pytest.approx(1.5)

    with (tmp_path / "matrix" / "matrix_summary.csv").open(newline="") as f:
        written = list(csv.DictReader(f))
    assert written[0]["case_name"] == "render_only"
    assert written[0]["video_readback_delivery"] == "sync"
    assert written[1]["case_name"] == "rgb"

    suite_config = json.loads((tmp_path / "matrix" / "suite_config.json").read_text())
    assert suite_config["suite"]["name"] == "tiny_suite"


def test_matrix_suite_records_failed_case_and_continues(tmp_path: Path):
    suite = MatrixSuite(
        name="failure_suite",
        preset="go2_video_ordered_static",
        cases=(
            MatrixCase(
                name="fails",
                width=160,
                height=120,
                readback_payload=ReadbackPayload.NONE,
            ),
            MatrixCase(
                name="passes",
                width=160,
                height=120,
                readback_payload=ReadbackPayload.NONE,
            ),
        ),
    )

    def fake_run(config, options):
        if options.out.name == "fails":
            raise RuntimeError("boom")
        _write_fake_frame_timing(options.out / "frame_timing.csv")

    rows = run_matrix_suite(
        suite,
        MatrixRunOptions(out=tmp_path / "matrix", frames=2),
        run_one=fake_run,
    )

    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "boom"
    assert rows[1]["status"] == "passed"


def test_cli_matrix_dispatches_to_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[object, object]] = []

    def fake_run_matrix_suite(suite, options):
        calls.append((suite, options))

    monkeypatch.setattr(lab_main, "run_matrix_suite", fake_run_matrix_suite)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optical_pipeline_lab",
            "matrix",
            "--suite",
            "go2_video_delivery_smoke",
            "--out",
            str(tmp_path / "matrix"),
            "--device",
            "cuda:1",
            "--frames",
            "2",
            "--include-full-debug",
        ],
    )

    lab_main.main()

    assert len(calls) == 1
    suite, options = calls[0]
    assert suite.name == "go2_video_delivery_smoke"
    assert suite.cases[-2].video_readback_delivery == "torch_async"
    assert suite.cases[-1].readback_payload.value == "full"
    assert options.out == tmp_path / "matrix"
    assert options.device == "cuda:1"
    assert options.frames == 2


def _write_fake_frame_timing(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "frame_total_ms",
        "render_execute_ms",
        "readback_host_ms",
        "image_build_ms",
        "encode_write_ms",
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "frame_total_ms": 4.0,
                "render_execute_ms": 2.0,
                "readback_host_ms": 1.0,
                "image_build_ms": "nan",
                "encode_write_ms": "nan",
            }
        )
        writer.writerow(
            {
                "frame_total_ms": 8.0,
                "render_execute_ms": 4.0,
                "readback_host_ms": 2.0,
                "image_build_ms": "nan",
                "encode_write_ms": "nan",
            }
        )
