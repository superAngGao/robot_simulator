import csv
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.optical_pipeline_lab as optical_pipeline_lab
import tools.optical_pipeline_lab.__main__ as lab_main
import tools.optical_pipeline_lab.async_readback as async_readback
import tools.optical_pipeline_lab.camera_presets as camera_presets
import tools.optical_pipeline_lab.delivery as delivery
import tools.optical_pipeline_lab.dynamic_frames as dynamic_frames
import tools.optical_pipeline_lab.frame_contexts as frame_contexts
import tools.optical_pipeline_lab.frame_products as frame_products
import tools.optical_pipeline_lab.frame_runtime as frame_runtime
import tools.optical_pipeline_lab.frame_tick as frame_tick
import tools.optical_pipeline_lab.go2_backend as go2_backend
import tools.optical_pipeline_lab.observation_products as observation_products
import tools.optical_pipeline_lab.physics_runtime as physics_runtime
import tools.optical_pipeline_lab.physics_source as physics_source
import tools.optical_pipeline_lab.preset_products as preset_products
import tools.optical_pipeline_lab.preset_runtime as preset_runtime
import tools.optical_pipeline_lab.preset_workflows as preset_workflows
import tools.optical_pipeline_lab.product_specs as product_specs
import tools.optical_pipeline_lab.product_workflow as product_workflow
import tools.optical_pipeline_lab.render_session as render_session
import tools.optical_pipeline_lab.rgb_pack as rgb_pack
import tools.optical_pipeline_lab.runner as lab_runner
import tools.optical_pipeline_lab.static_asset_source as static_asset_source
import tools.optical_pipeline_lab.video_loop as video_loop
import tools.optical_pipeline_lab.video_products as video_products
from optics.render_api import DeliveryPolicy as RuntimeDeliveryPolicy
from optics.render_api import DeliveryResult as RuntimeDeliveryResult
from optics.render_api import DeliveryTimingSummary, RenderTimingSummary
from optics.render_api import OpticalRenderPipeline as RuntimeOpticalRenderPipeline
from optics.render_api import ReadbackPayload as RuntimeReadbackPayload
from optics.render_api import RenderBackend as RuntimeRenderBackend
from optics.render_api import RenderFrameContext as RuntimeRenderFrameContext
from optics.render_api import RenderResult as RuntimeRenderResult
from optics.render_api import WritePolicy as RuntimeWritePolicy
from physics.publish import CpuPublishedFrame, GpuPublishedFrame
from physics.spatial import SpatialTransform
from rl_env.obs import locomotion_obs_schema
from tools.optical_pipeline_lab import (
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_WIDTH,
    AccelPolicy,
    ClockOwnerKind,
    DeliveryPolicy,
    FrameSourceKind,
    FrameTimingRecorder,
    GeometryMode,
    OpticalLabScenarioConfig,
    OpticalLabScenarioFamily,
    ReadbackPayload,
    RenderBackend,
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
    ArtifactOutput,
    LabRunOptions,
    RunScenarioUnsupportedError,
    apply_run_overrides,
    build_menagerie_example_args,
    build_physics_video_args,
    can_run_scenario,
    create_physics_render_runtime_for_config,
    render_options_for_config,
    run_physics_stepped_video_product_scenario,
    run_physics_stepped_video_scenario,
    run_scenario,
    validate_physics_video_product_run,
    validate_physics_video_run,
    validate_run,
    validate_run_scenario_supported,
    write_scenario_config,
)


def test_percentile_interpolates_sorted_samples():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([10.0, 20.0], 0.9) == pytest.approx(19.0)
    assert percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert math.isnan(percentile([], 0.9))


def test_optical_pipeline_lab_exports_p9_product_contracts():
    assert optical_pipeline_lab.ArtifactOutput is lab_runner.ArtifactOutput
    assert optical_pipeline_lab.SimulationFrameTick is frame_tick.SimulationFrameTick
    assert (
        optical_pipeline_lab.simulation_frame_tick_from_published_frame
        is frame_tick.simulation_frame_tick_from_published_frame
    )
    assert optical_pipeline_lab.FrameProductResult is frame_products.FrameProductResult
    assert optical_pipeline_lab.FrameProduct is frame_products.FrameProduct
    assert optical_pipeline_lab.MultiProductFrameRunner is frame_products.MultiProductFrameRunner
    assert optical_pipeline_lab.DebugFrameProduct is frame_products.DebugFrameProduct
    assert (
        optical_pipeline_lab.PublishedStateObservationProduct
        is observation_products.PublishedStateObservationProduct
    )
    assert optical_pipeline_lab.PhysicsOwnedProductWorkflow is product_workflow.PhysicsOwnedProductWorkflow
    assert optical_pipeline_lab.PhysicsProductRunResult is product_workflow.PhysicsProductRunResult
    assert optical_pipeline_lab.ProductBuildContext is product_specs.ProductBuildContext
    assert optical_pipeline_lab.ProductSpec is product_specs.ProductSpec
    assert optical_pipeline_lab.DebugProductSpec is product_specs.DebugProductSpec
    assert optical_pipeline_lab.ObservationProductSpec is product_specs.ObservationProductSpec
    assert optical_pipeline_lab.VideoProductSpec is product_specs.VideoProductSpec
    assert optical_pipeline_lab.create_runtime_for_lab_preset is preset_runtime.create_runtime_for_lab_preset
    assert optical_pipeline_lab.resolve_lab_product_specs is preset_products.resolve_lab_product_specs
    assert optical_pipeline_lab.supported_lab_product_strings is preset_products.supported_lab_product_strings
    assert optical_pipeline_lab.supported_runtime_presets is preset_runtime.supported_runtime_presets
    assert optical_pipeline_lab.run_optical_lab_preset is preset_workflows.run_optical_lab_preset
    assert optical_pipeline_lab.run_optical_lab_workflow is product_workflow.run_optical_lab_workflow
    assert optical_pipeline_lab.run_optical_lab_products is product_workflow.run_optical_lab_products
    assert optical_pipeline_lab.run_physics_products is product_workflow.run_physics_products
    assert optical_pipeline_lab.run_physics_product_scenario is product_workflow.run_physics_product_scenario
    assert optical_pipeline_lab.run_physics_product_preset is product_workflow.run_physics_product_preset


def test_optical_pipeline_lab_observation_product_export_is_lazy():
    script = """
import sys
import tools.optical_pipeline_lab

assert "tools.optical_pipeline_lab.observation_products" not in sys.modules
assert "rl_env.managers" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_optical_pipeline_lab_debug_product_spec_export_stays_lightweight():
    script = """
import sys
import tools.optical_pipeline_lab as lab

_ = lab.DebugProductSpec
_ = lab.VideoProductSpec
assert "tools.optical_pipeline_lab.observation_products" not in sys.modules
assert "rl_env.managers" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_optical_pipeline_lab_getattr_unknown_raises_attribute_error():
    with pytest.raises(AttributeError, match="not_a_lab_export"):
        optical_pipeline_lab.__getattr__("not_a_lab_export")


def test_artifact_output_uses_root_and_keeps_lab_run_options_compatibility(tmp_path: Path):
    output = ArtifactOutput(root=tmp_path / "root", frames=3)
    legacy = LabRunOptions(out=tmp_path / "legacy", frames=4)
    replaced = legacy.replace_frames(5)

    assert output.root == tmp_path / "root"
    assert output.out == output.root
    assert legacy.root == tmp_path / "legacy"
    assert legacy.out == legacy.root
    assert replaced.root == legacy.root
    assert replaced.frames == 5
    assert replaced._frames_explicit is True

    with pytest.raises(ValueError, match="conflicting root and out"):
        ArtifactOutput(root=tmp_path / "a", out=tmp_path / "b")


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
    assert "clock_owner" in reader.fieldnames
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


def test_video_render_plan_consumes_frame_identity_without_retaining_inputs():
    camera = go2_backend.OpticalPinholeCameraSpec(
        frame_id=8,
        sim_time=0.8,
        env_idx=4,
        sensor_id="camera",
        width=16,
        height=8,
        fx=10.0,
        fy=10.0,
        cx=7.5,
        cy=3.5,
    )
    args = SimpleNamespace(
        video_raygen="gpu",
        video_readback="rgb8",
        render_profile=True,
        fail_on_overflow=True,
        video_geometry_mode="dynamic_rigid",
    )

    plan = video_loop.build_video_render_plan(
        object(),
        args,
        0,
        None,
        build_video_camera=lambda scene, args_arg, frame_index: camera,
        frame_identity=video_loop.FrameIdentity(frame_id=9, sim_time=0.9),
    )

    assert isinstance(plan, video_loop.VideoRenderPlan)
    assert plan.camera.frame_id == 9
    assert plan.camera.sim_time == 0.9
    assert plan.camera.env_idx == 4
    assert plan.rays is None
    assert plan.request.frame_id == 9
    assert plan.request.sim_time == 0.9
    assert plan.request.camera is plan.camera
    assert plan.geometry_mode == "dynamic_rigid"
    assert plan.include_shadow_traversal_stats is True
    assert not hasattr(plan, "frame_inputs")

    plan = video_loop.build_video_render_plan(
        object(),
        args,
        0,
        None,
        build_video_camera=lambda scene, args_arg, frame_index: camera,
        frame_identity=video_loop.FrameIdentity(frame_id=10, sim_time=1.0, env_idx=2),
    )

    assert plan.camera.env_idx == 2


def test_render_video_frame_from_context_preserves_plan_and_prepare_timing():
    compute = SimpleNamespace(ready_event=object())
    camera = go2_backend.OpticalPinholeCameraSpec(
        frame_id=11,
        sim_time=1.1,
        env_idx=3,
        sensor_id="camera",
        width=16,
        height=8,
        fx=10.0,
        fy=10.0,
        cx=7.5,
        cy=3.5,
    )
    request = video_loop.video_render_request(
        camera=camera,
        rays=None,
        use_gpu_raygen=True,
        readback_mode="none",
        profile_timing=False,
        fail_on_overflow=False,
    )
    plan = video_loop.VideoRenderPlan(
        camera=camera,
        rays=None,
        request=request,
        camera_rays_ms=float("nan"),
        geometry_mode="dynamic_rigid",
        include_shadow_traversal_stats=False,
    )
    render_result = RuntimeRenderResult(
        compute=compute,
        timing={
            "render_execute_ms": 3.5,
            **go2_backend._render_profile_row(None),
        },
    )
    captured: dict[str, object] = {}

    class FakeFrameContext:
        prepare_timing = {
            "snapshot_ms": 1.0,
            "accel_refit_ms": 2.0,
            "accel_rebuild_ms": float("nan"),
        }

        def render(self, request_arg):
            captured["request"] = request_arg
            return render_result

    rendered = video_loop.render_video_frame_from_context(
        FakeFrameContext(),
        plan,
        frame_index=7,
    )

    assert captured["request"] is request
    assert rendered.frame_index == 7
    assert rendered.camera is camera
    assert rendered.result is compute
    assert rendered.render_execute_ms == 3.5
    assert rendered.geometry_mode == "dynamic_rigid"
    assert rendered.prepare_timing["snapshot_ms"] == 1.0
    assert rendered.prepare_timing["accel_refit_ms"] == 2.0
    assert math.isnan(float(rendered.prepare_timing["accel_rebuild_ms"]))
    assert rendered.render is render_result


def test_go2_video_helper_aliases_delegate_to_generic_video_loop():
    assert go2_backend._video_render_request is video_loop.video_render_request
    assert go2_backend._video_delivery_request is video_loop.video_delivery_request_from_options
    assert go2_backend._render_profile_row is video_loop.render_profile_row
    assert go2_backend._video_readback_channels is video_loop.video_readback_channels


def test_lab_render_source_naming_does_not_use_adapter_language():
    doc = render_session.OpticalLabRenderSource.__doc__ or ""

    assert "adapter" not in doc.lower()
    assert "registry" in doc
    assert "base frame reference" in doc


def test_lab_render_source_exposes_base_frame_as_scene_frame():
    base_frame = SimpleNamespace(frame_id=11, sim_time=1.1)
    source = render_session.OpticalLabRenderSource(
        registry=object(),
        base_frame=base_frame,
        bounds_min=(-1.0, -2.0, -3.0),
        bounds_max=(1.0, 2.0, 3.0),
        metadata={"source_kind": "unit-test"},
    )

    assert source.frame is base_frame
    assert source.frame.frame_id == 11
    assert source.bounds_min == (-1.0, -2.0, -3.0)
    assert source.bounds_max == (1.0, 2.0, 3.0)
    assert source.metadata["source_kind"] == "unit-test"


def test_physics_render_source_wraps_published_frame_scene_view():
    registry = object()
    base_frame = SimpleNamespace(frame_id=21, sim_time=2.1)

    source = physics_source.build_physics_render_source(
        registry=registry,
        base_frame=base_frame,
        bounds_min=(-0.1, -0.2, -0.3),
        bounds_max=(0.4, 0.5, 0.6),
        metadata={"producer": "gpu_engine"},
    )
    scene = physics_source.scene_from_physics_render_source(source)

    assert source.registry is registry
    assert source.frame is base_frame
    assert source.metadata["source_kind"] == "physics"
    assert source.metadata["producer"] == "gpu_engine"
    assert scene.registry is registry
    assert scene.frame is base_frame
    assert scene.bounds_min == (-0.1, -0.2, -0.3)
    assert scene.bounds_max == (0.4, 0.5, 0.6)


def test_create_physics_render_pipeline_uses_canonical_source_factory(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = object()
    base_frame = SimpleNamespace(frame_id=31, sim_time=0.31)
    options = render_session.OpticalLabRenderOptions(device="cuda:physics")
    timings = TimingRecorder()
    captured: dict[str, object] = {}

    def fake_create_from_source_factory(source_factory, options_arg, timings_arg, *, scene_for_source):
        source = source_factory(SimpleNamespace(device="fake-device", stream="fake-stream"))
        captured["source"] = source
        captured["options"] = options_arg
        captured["timings"] = timings_arg
        captured["scene"] = scene_for_source(source)
        return "physics-pipeline"

    monkeypatch.setattr(
        physics_source.OpticalLabRenderPipeline,
        "create_from_source_factory",
        staticmethod(fake_create_from_source_factory),
    )

    pipeline = physics_source.create_physics_render_pipeline(
        registry=registry,
        base_frame=base_frame,
        options=options,
        timings=timings,
        bounds_min=(-0.1, -0.2, -0.3),
        bounds_max=(0.4, 0.5, 0.6),
        metadata={"producer": "gpu_engine"},
    )

    source = captured["source"]
    assert pipeline == "physics-pipeline"
    assert source.registry is registry
    assert source.base_frame is base_frame
    assert source.bounds_min == (-0.1, -0.2, -0.3)
    assert source.bounds_max == (0.4, 0.5, 0.6)
    assert source.metadata["producer"] == "gpu_engine"
    assert source.metadata["source_kind"] == "physics"
    assert captured["scene"] is source.metadata["scene"]
    assert captured["options"] is options
    assert captured["timings"] is timings


def test_physics_render_consumer_defaults_to_device_borrow_lifecycle():
    consumer = physics_source.physics_render_consumer(
        "optical_lab_camera",
        max_lag_frames=2,
    )

    assert consumer.consumer_id == "optical_lab_camera"
    assert consumer.consumer_kind == "render_backed_sensing"
    assert consumer.qos_mode == "lossless"
    assert consumer.access_mode == "borrow"
    assert consumer.consumer_location == "device"
    assert consumer.max_lag_frames == 2


def test_register_physics_render_consumer_builds_and_registers_default_consumer():
    calls: list[object] = []

    class FakeEngine:
        def register_consumer(self, consumer):
            calls.append(consumer)

    consumer = physics_source.register_physics_render_consumer(
        FakeEngine(),
        "optical_lab_camera",
        max_lag_frames=3,
    )

    assert calls == [consumer]
    assert consumer.consumer_id == "optical_lab_camera"
    assert consumer.consumer_kind == "render_backed_sensing"
    assert consumer.access_mode == "borrow"
    assert consumer.consumer_location == "device"
    assert consumer.max_lag_frames == 3


def test_register_physics_render_consumer_accepts_existing_consumer():
    calls: list[object] = []
    existing = physics_source.physics_render_consumer("already_built")

    class FakeEngine:
        def register_consumer(self, consumer):
            calls.append(consumer)

    consumer = physics_source.register_physics_render_consumer(
        FakeEngine(),
        "ignored_id",
        consumer=existing,
    )

    assert consumer is existing
    assert calls == [existing]


def test_physics_render_runtime_begin_frame_uses_registered_consumer():
    published_frame = SimpleNamespace(frame_id=51, sim_time=0.51)
    borrowed_frame = SimpleNamespace(frame_id=51, sim_time=0.51)
    frame_context = object()
    consumer = physics_source.physics_render_consumer("runtime_consumer")
    calls: list[tuple[object, ...]] = []

    class FakeEngine:
        def borrow_device_frame(self, consumer_id, frame_id, *, stream):
            calls.append(("borrow_device_frame", consumer_id, frame_id, stream))
            return borrowed_frame

        def complete_device_consumer(self, consumer_id, frame_id, *, stream):
            calls.append(("complete_device_consumer", consumer_id, frame_id, stream))
            return "done-event"

    class FakePipeline:
        session = SimpleNamespace(stream="render-stream")

        def begin_frame(self, frame_inputs=None, *, env_idx=0):
            calls.append(("begin_frame", frame_inputs, env_idx))
            return frame_context

    runtime = physics_source.PhysicsLabRenderRuntime(
        engine=FakeEngine(),
        pipeline=FakePipeline(),
        consumer=consumer,
    )

    with runtime.begin_frame(published_frame=published_frame, env_idx=6) as lease:
        assert runtime.consumer_id == "runtime_consumer"
        assert lease.frame is borrowed_frame
        assert lease.frame_context is frame_context

    assert lease.done_event == "done-event"
    assert calls == [
        ("borrow_device_frame", "runtime_consumer", 51, "render-stream"),
        ("begin_frame", borrowed_frame, 6),
        ("complete_device_consumer", "runtime_consumer", 51, "render-stream"),
    ]


def test_create_physics_render_runtime_builds_pipeline_and_registers_consumer(
    monkeypatch: pytest.MonkeyPatch,
):
    engine = object()
    registry = object()
    base_frame = SimpleNamespace(frame_id=52, sim_time=0.52)
    options = render_session.OpticalLabRenderOptions(device="cuda:runtime")
    timings = TimingRecorder()
    captured: dict[str, object] = {}

    def fake_create_physics_render_pipeline(**kwargs):
        captured["pipeline_kwargs"] = kwargs
        return "runtime-pipeline"

    def fake_register_physics_render_consumer(engine_arg, consumer_id, **kwargs):
        captured["register_engine"] = engine_arg
        captured["register_consumer_id"] = consumer_id
        captured["register_kwargs"] = kwargs
        return physics_source.physics_render_consumer(consumer_id)

    monkeypatch.setattr(
        physics_source,
        "create_physics_render_pipeline",
        fake_create_physics_render_pipeline,
    )
    monkeypatch.setattr(
        physics_source,
        "register_physics_render_consumer",
        fake_register_physics_render_consumer,
    )

    runtime = physics_source.create_physics_render_runtime(
        engine=engine,
        registry=registry,
        base_frame=base_frame,
        options=options,
        timings=timings,
        consumer_id="runtime_consumer",
        max_lag_frames=4,
        bounds_min=(-0.1, -0.2, -0.3),
        bounds_max=(0.4, 0.5, 0.6),
        metadata={"producer": "gpu_engine"},
    )

    pipeline_kwargs = captured["pipeline_kwargs"]
    assert runtime.engine is engine
    assert runtime.pipeline == "runtime-pipeline"
    assert runtime.consumer.consumer_id == "runtime_consumer"
    assert pipeline_kwargs["registry"] is registry
    assert pipeline_kwargs["base_frame"] is base_frame
    assert pipeline_kwargs["options"] is options
    assert pipeline_kwargs["timings"] is timings
    assert pipeline_kwargs["bounds_min"] == (-0.1, -0.2, -0.3)
    assert pipeline_kwargs["bounds_max"] == (0.4, 0.5, 0.6)
    assert pipeline_kwargs["metadata"] == {"producer": "gpu_engine"}
    assert captured["register_engine"] is engine
    assert captured["register_consumer_id"] == "runtime_consumer"
    assert captured["register_kwargs"]["max_lag_frames"] == 4


def test_physics_lab_scenario_runtime_steps_and_closes_once():
    frames = {
        0: SimpleNamespace(frame_id=80, sim_time=8.0),
        1: SimpleNamespace(frame_id=81, sim_time=8.1),
    }
    calls: list[tuple[object, ...]] = []

    def step_frame(frame_index: int):
        calls.append(("step_frame", frame_index))
        return frames[frame_index]

    def close():
        calls.append(("close",))

    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=79, sim_time=7.9),
        step_frame_fn=step_frame,
        bounds_min=(-0.2, -0.2, 0.0),
        bounds_max=(0.4, 0.4, 1.2),
        metadata={"producer": "fake"},
        close_fn=close,
    )

    assert runtime.step_frame(0) is frames[0]
    with pytest.raises(RuntimeError, match="body failed"):
        with runtime as entered:
            assert entered is runtime
            assert entered.step_frame(1) is frames[1]
            raise RuntimeError("body failed")

    assert runtime.closed is True
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.step_frame(0)
    assert calls == [
        ("step_frame", 0),
        ("step_frame", 1),
        ("close",),
    ]


def test_physics_lab_scenario_runtime_step_tick_wraps_published_frame_metadata():
    published_frame = SimpleNamespace(frame_id=82, sim_time=8.2)
    calls: list[tuple[object, ...]] = []

    def step_frame(frame_index: int):
        calls.append(("step_frame", frame_index))
        return published_frame

    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=81, sim_time=8.1),
        step_frame_fn=step_frame,
        metadata={"producer": "fake", "runtime_owner": "physics_lab"},
    )

    tick = runtime.step_tick(2, env_idx=3, metadata={"product_set": "debug"})

    assert isinstance(tick, frame_tick.SimulationFrameTick)
    assert tick.frame_index == 2
    assert tick.env_idx == 3
    assert tick.frame_id == 82
    assert tick.sim_time == 8.2
    assert tick.published_frame is published_frame
    assert tick.metadata == {
        "producer": "fake",
        "runtime_owner": "physics_lab",
        "product_set": "debug",
    }
    assert calls == [("step_frame", 2)]


def test_debug_frame_product_records_tick_identity_and_selected_metadata():
    tick = frame_tick.SimulationFrameTick(
        frame_index=4,
        env_idx=2,
        frame_id=84,
        sim_time=8.4,
        published_frame=object(),
        metadata={
            "producer": "fake",
            "runtime_owner": "physics_lab",
            "ignore": "not-selected",
        },
    )
    product = frame_products.DebugFrameProduct(
        product_name="debug_identity",
        metadata_keys=("producer", "missing"),
    )

    assert product.begin_run() is None
    result = product.consume(tick)

    assert result == frame_products.FrameProductResult(
        product_name="debug_identity",
        frame_index=4,
        frame_id=84,
        sim_time=8.4,
        env_idx=2,
        payload={
            "frame_index": 4,
            "frame_id": 84,
            "sim_time": 8.4,
            "env_idx": 2,
            "metadata": {"producer": "fake"},
        },
        metadata={"producer": "fake"},
    )
    assert product.end_run() == (result,)


def test_debug_frame_product_records_all_metadata_and_zero_frame_run():
    tick = frame_tick.SimulationFrameTick(
        frame_index=7,
        env_idx=1,
        frame_id=87,
        sim_time=8.7,
        published_frame=object(),
        metadata={
            "producer": "fake",
            "runtime_owner": "physics_lab",
        },
    )
    product = frame_products.DebugFrameProduct()

    assert product.begin_run() is None
    assert product.end_run() == ()

    result = product.consume(tick)
    assert result.metadata == {
        "producer": "fake",
        "runtime_owner": "physics_lab",
    }
    assert result.payload["metadata"] == {
        "producer": "fake",
        "runtime_owner": "physics_lab",
    }


def test_multi_product_frame_runner_preserves_order_and_none_results():
    tick = frame_tick.SimulationFrameTick(
        frame_index=5,
        env_idx=3,
        frame_id=85,
        sim_time=8.5,
        published_frame=object(),
        metadata={"producer": "fake"},
    )
    calls: list[tuple[object, ...]] = []

    class RecordingProduct:
        product_name = "recording"

        def begin_run(self):
            calls.append(("begin", self.product_name))
            return "recording-ready"

        def consume(self, tick_arg):
            calls.append(("consume", self.product_name, tick_arg.frame_index))
            return frame_products.FrameProductResult.from_tick(
                product_name=self.product_name,
                tick=tick_arg,
                payload={"kind": "recording"},
            )

        def end_run(self):
            calls.append(("end", self.product_name))
            return "recording-done"

    class ObservingProduct:
        product_name = "observer"

        def begin_run(self):
            calls.append(("begin", self.product_name))
            return None

        def consume(self, tick_arg):
            calls.append(("consume", self.product_name, tick_arg.frame_index))
            return None

        def end_run(self):
            calls.append(("end", self.product_name))
            return {"observed": True}

    runner = frame_products.MultiProductFrameRunner(
        products=(RecordingProduct(), ObservingProduct()),
    )

    assert runner.begin_run() == {
        "recording": "recording-ready",
        "observer": None,
    }
    results = runner.step(tick)
    assert len(results) == 2
    assert results[0] == frame_products.FrameProductResult(
        product_name="recording",
        frame_index=5,
        frame_id=85,
        sim_time=8.5,
        env_idx=3,
        payload={"kind": "recording"},
    )
    assert results[1] is None
    assert runner.end_run() == {
        "recording": "recording-done",
        "observer": {"observed": True},
    }
    assert calls == [
        ("begin", "recording"),
        ("begin", "observer"),
        ("consume", "recording", 5),
        ("consume", "observer", 5),
        ("end", "recording"),
        ("end", "observer"),
    ]


def test_multi_product_frame_runner_stops_on_product_exception():
    tick = frame_tick.SimulationFrameTick(
        frame_index=6,
        env_idx=0,
        frame_id=86,
        sim_time=8.6,
        published_frame=object(),
    )
    calls: list[tuple[object, ...]] = []

    class FailingProduct:
        product_name = "failing"

        def begin_run(self):
            return None

        def consume(self, tick_arg):
            calls.append(("consume", self.product_name, tick_arg.frame_index))
            raise RuntimeError("product failed")

        def end_run(self):
            return None

    class UnreachedProduct:
        product_name = "unreached"

        def begin_run(self):
            return None

        def consume(self, tick_arg):
            calls.append(("consume", self.product_name, tick_arg.frame_index))
            return None

        def end_run(self):
            return None

    runner = frame_products.MultiProductFrameRunner(
        products=(FailingProduct(), UnreachedProduct()),
    )

    with pytest.raises(RuntimeError, match="product failed"):
        runner.step(tick)
    assert calls == [("consume", "failing", 6)]


def test_multi_product_frame_runner_begin_run_is_fail_fast():
    calls: list[tuple[object, ...]] = []

    class FailingBeginProduct:
        product_name = "failing_begin"

        def begin_run(self):
            calls.append(("begin", self.product_name))
            raise RuntimeError("begin failed")

        def consume(self, tick_arg):
            return None

        def end_run(self):
            return None

    class UnreachedProduct:
        product_name = "unreached"

        def begin_run(self):
            calls.append(("begin", self.product_name))
            return None

        def consume(self, tick_arg):
            return None

        def end_run(self):
            return None

    runner = frame_products.MultiProductFrameRunner(
        products=(FailingBeginProduct(), UnreachedProduct()),
    )

    with pytest.raises(RuntimeError, match="begin failed"):
        runner.begin_run()
    assert calls == [("begin", "failing_begin")]


def test_multi_product_frame_runner_end_run_is_fail_fast():
    calls: list[tuple[object, ...]] = []

    class FailingEndProduct:
        product_name = "failing_end"

        def begin_run(self):
            return None

        def consume(self, tick_arg):
            return None

        def end_run(self):
            calls.append(("end", self.product_name))
            raise RuntimeError("end failed")

    class UnreachedProduct:
        product_name = "unreached"

        def begin_run(self):
            return None

        def consume(self, tick_arg):
            return None

        def end_run(self):
            calls.append(("end", self.product_name))
            return None

    runner = frame_products.MultiProductFrameRunner(
        products=(FailingEndProduct(), UnreachedProduct()),
    )

    with pytest.raises(RuntimeError, match="end failed"):
        runner.end_run()
    assert calls == [("end", "failing_end")]


def test_multi_product_frame_runner_rejects_duplicate_product_names():
    product = frame_products.DebugFrameProduct(product_name="duplicate")

    with pytest.raises(ValueError, match="product_name values must be unique"):
        frame_products.MultiProductFrameRunner(products=(product, product))


def test_physics_owned_product_workflow_runs_video_debug_observation_on_one_tick_stream(tmp_path: Path):
    schema = locomotion_obs_schema(
        num_actuated_joints=2,
        num_contact_bodies=2,
        include_contact_mask=True,
    )
    published_frames = [
        CpuPublishedFrame(
            frame_id=140 + frame_index,
            sim_time=14.0 + frame_index,
            step_index=140 + frame_index,
            env_mask=None,
            q=np.array([1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.5, -0.5]),
            qdot=np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 1.5, -1.5]),
            X_world=[SpatialTransform.identity()],
            v_bodies=np.array([[1.0, 2.0, 3.0, 0.4, 0.5, 0.6]]),
            contact_count=1,
            contacts=object(),
            telemetry=None,
            contact_mask=np.array([1, 0], dtype=np.int32),
        )
        for frame_index in range(2)
    ]
    calls: list[tuple[object, ...]] = []

    def step_frame(frame_index: int):
        calls.append(("step", frame_index))
        return published_frames[frame_index]

    class FakeVideoProduct:
        product_name = "video"

        def begin_run(self):
            calls.append(("begin", self.product_name))
            return None

        def consume(self, tick):
            calls.append(("video", tick.frame_index, tick.frame_id, tick.sim_time))
            return frame_products.FrameProductResult.from_tick(
                product_name=self.product_name,
                tick=tick,
                payload={"published_frame": tick.published_frame},
            )

        def end_run(self):
            calls.append(("end", self.product_name))
            return {"video_timing_csv": tmp_path / "workflow" / "frame_timing.csv"}

    class RecordingObservationProduct(observation_products.PublishedStateObservationProduct):
        def consume(self, tick):
            calls.append(
                ("observation", tick.frame_index, tick.frame_id, tick.sim_time, tick.published_frame)
            )
            return super().consume(tick)

    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=published_frames[0],
        step_frame_fn=step_frame,
        metadata={"runtime_owner": "p10_test"},
    )
    debug_product = frame_products.DebugFrameProduct(product_name="debug", metadata_keys=None)
    observation_product = RecordingObservationProduct(
        engine=object(),
        schema=schema,
        actuated_q_indices=np.array([7, 8], dtype=np.intp),
        actuated_v_indices=np.array([6, 7], dtype=np.intp),
        contact_body_names=("left_foot", "right_foot"),
    )
    workflow = product_workflow.PhysicsOwnedProductWorkflow(
        runtime=runtime,
        products=(FakeVideoProduct(), debug_product, observation_product),
        output=ArtifactOutput(root=tmp_path / "workflow"),
    )

    result = workflow.run(frames=2)

    assert result.begin_outputs == {
        "video": None,
        "debug": None,
        "observation": None,
    }
    assert result.end_outputs["video"] == {"video_timing_csv": tmp_path / "workflow" / "frame_timing.csv"}
    assert result.end_outputs["debug"] == tuple(debug_product.records)
    assert result.end_outputs["observation"] == tuple(observation_product.records)
    assert result.artifacts == {"root": tmp_path / "workflow"}
    assert [len(frame) for frame in result.frame_results] == [3, 3]
    assert [record.frame_id for record in result.product_results["video"]] == [140, 141]
    assert [record.frame_id for record in result.product_results["debug"]] == [140, 141]
    assert [record.frame_id for record in result.product_results["observation"]] == [140, 141]
    np.testing.assert_allclose(
        result.product_results["observation"][0].payload["observation"].numpy(),
        [
            1.0,
            2.0,
            3.0,
            0.4,
            0.5,
            0.6,
            1.0,
            0.0,
            0.0,
            0.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            1.0,
            0.0,
        ],
    )
    assert calls == [
        ("begin", "video"),
        ("step", 0),
        ("video", 0, 140, 14.0),
        ("observation", 0, 140, 14.0, published_frames[0]),
        ("step", 1),
        ("video", 1, 141, 15.0),
        ("observation", 1, 141, 15.0, published_frames[1]),
        ("end", "video"),
    ]

    with pytest.raises(RuntimeError, match="already run"):
        workflow.run(frames=1)


def test_physics_owned_product_workflow_runtime_ownership():
    calls: list[tuple[str]] = []

    def close_runtime():
        calls.append(("close",))

    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=149, sim_time=14.9),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=150 + frame_index,
            sim_time=15.0 + frame_index,
        ),
        close_fn=close_runtime,
    )

    product_workflow.PhysicsOwnedProductWorkflow(
        runtime=runtime,
        products=(frame_products.DebugFrameProduct(),),
        owns_runtime=False,
    ).close()
    assert runtime.closed is False

    product_workflow.PhysicsOwnedProductWorkflow(
        runtime=runtime,
        products=(frame_products.DebugFrameProduct(),),
        owns_runtime=True,
    ).close()
    assert runtime.closed is True
    assert calls == [("close",)]


def test_physics_owned_product_workflow_rejects_explicit_output_frame_conflict(tmp_path: Path):
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=19, sim_time=1.9),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=20 + frame_index,
            sim_time=2.0 + frame_index,
        ),
    )
    workflow = product_workflow.PhysicsOwnedProductWorkflow(
        runtime=runtime,
        products=(frame_products.DebugFrameProduct(),),
        output=ArtifactOutput(root=tmp_path / "workflow", frames=2),
    )

    with pytest.raises(ValueError, match="ArtifactOutput.frames conflicts"):
        workflow.run(frames=1)


def test_run_physics_products_wraps_existing_runtime_and_closes_when_owned(tmp_path: Path):
    calls: list[tuple[object, ...]] = []

    def close_runtime():
        calls.append(("close",))

    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=29, sim_time=2.9),
        step_frame_fn=lambda frame_index: calls.append(("step", frame_index))
        or SimpleNamespace(frame_id=30 + frame_index, sim_time=3.0 + frame_index),
        close_fn=close_runtime,
    )
    debug_product = frame_products.DebugFrameProduct(product_name="debug", metadata_keys=None)

    result = product_workflow.run_physics_products(
        runtime=runtime,
        products=(debug_product,),
        frames=2,
        output=ArtifactOutput(root=tmp_path / "workflow"),
        owns_runtime=True,
    )

    assert runtime.closed is True
    assert calls == [("step", 0), ("step", 1), ("close",)]
    assert result.artifacts == {"root": tmp_path / "workflow"}
    assert [record.frame_id for record in result.product_results["debug"]] == [30, 31]


def test_run_physics_product_scenario_writes_config_and_runs_products(tmp_path: Path):
    calls: list[tuple[object, ...]] = []
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=39, sim_time=3.9),
        step_frame_fn=lambda frame_index: calls.append(("step", frame_index))
        or SimpleNamespace(frame_id=40 + frame_index, sim_time=4.0 + frame_index),
        metadata={"runtime_owner": "p10_helper_test"},
    )
    output = ArtifactOutput(root=tmp_path / "scenario", frames=2)

    result = product_workflow.run_physics_product_scenario(
        get_preset("physics_body_triangle_video_smoke"),
        output,
        runtime=runtime,
        products=(frame_products.DebugFrameProduct(metadata_keys=None),),
    )

    payload = json.loads((tmp_path / "scenario" / "scenario_config.json").read_text())
    assert payload["scenario"]["frame_source"] == "physics_published_frame"
    assert payload["scenario"]["clock_owner"] == "external_physics_runtime"
    assert payload["run_options"]["root"] == str(tmp_path / "scenario")
    assert calls == [("step", 0), ("step", 1)]
    assert [record.frame_id for record in result.product_results["debug"]] == [40, 41]


def test_run_physics_product_preset_delegates_to_named_preset(tmp_path: Path):
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=49, sim_time=4.9),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=50 + frame_index,
            sim_time=5.0 + frame_index,
        ),
    )

    result = product_workflow.run_physics_product_preset(
        "physics_body_triangle_video_smoke",
        ArtifactOutput(root=tmp_path / "preset"),
        runtime=runtime,
        products=(frame_products.DebugFrameProduct(),),
        frames=1,
    )

    payload = json.loads((tmp_path / "preset" / "scenario_config.json").read_text())
    assert payload["scenario"]["scenario_name"] == "physics_body_triangle_video_smoke"
    assert [record.frame_id for record in result.product_results["debug"]] == [50]


def test_run_optical_lab_workflow_runs_preset_products_with_out_path(tmp_path: Path):
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=52, sim_time=5.2),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=53 + frame_index,
            sim_time=5.3 + frame_index,
        ),
        metadata={"runtime_owner": "workflow_api_test"},
    )

    result = product_workflow.run_optical_lab_workflow(
        preset="physics_body_triangle_video_smoke",
        out=tmp_path / "workflow",
        runtime=runtime,
        products=(product_specs.DebugProductSpec(metadata_keys=("runtime_owner",)),),
        frames=2,
    )

    payload = json.loads((tmp_path / "workflow" / "scenario_config.json").read_text())
    assert payload["scenario"]["scenario_name"] == "physics_body_triangle_video_smoke"
    assert payload["run_options"]["root"] == str(tmp_path / "workflow")
    assert payload["run_options"]["frames"] == 2
    assert result.artifacts == {"root": tmp_path / "workflow"}
    assert [record.frame_id for record in result.product_results["debug"]] == [53, 54]
    assert result.product_results["debug"][0].metadata == {"runtime_owner": "workflow_api_test"}


def test_run_optical_lab_products_accepts_explicit_config_alias(tmp_path: Path):
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=54, sim_time=5.4),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=55 + frame_index,
            sim_time=5.5 + frame_index,
        ),
    )

    result = product_workflow.run_optical_lab_products(
        config=get_preset("physics_body_triangle_video_smoke"),
        output=ArtifactOutput(root=tmp_path / "products", frames=1),
        runtime=runtime,
        products=(product_specs.DebugProductSpec(),),
    )

    assert [record.frame_id for record in result.product_results["debug"]] == [55]


def test_run_optical_lab_workflow_merges_frames_into_output_artifact(tmp_path: Path):
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=56, sim_time=5.6),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=57 + frame_index,
            sim_time=5.7 + frame_index,
        ),
    )

    product_workflow.run_optical_lab_workflow(
        preset="physics_body_triangle_video_smoke",
        output=ArtifactOutput(root=tmp_path / "merged"),
        runtime=runtime,
        products=(product_specs.DebugProductSpec(),),
        frames=2,
    )

    payload = json.loads((tmp_path / "merged" / "scenario_config.json").read_text())
    assert payload["run_options"]["frames"] == 2


def test_run_optical_lab_workflow_rejects_explicit_output_frame_conflict(tmp_path: Path):
    with pytest.raises(ValueError, match="ArtifactOutput.frames conflicts"):
        product_workflow.run_optical_lab_workflow(
            preset="physics_body_triangle_video_smoke",
            output=ArtifactOutput(root=tmp_path / "frame_conflict", frames=2),
            runtime=object(),
            products=(),
            frames=1,
        )

    assert not (tmp_path / "frame_conflict").exists()


def test_run_optical_lab_workflow_requires_exactly_one_config_source(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one of preset or config"):
        product_workflow.run_optical_lab_workflow(
            runtime=object(),
            products=(),
            out=tmp_path / "missing",
            frames=1,
        )

    with pytest.raises(ValueError, match="exactly one of preset or config"):
        product_workflow.run_optical_lab_workflow(
            preset="physics_body_triangle_video_smoke",
            config=get_preset("physics_body_triangle_video_smoke"),
            runtime=object(),
            products=(),
            out=tmp_path / "both",
            frames=1,
        )


def test_run_optical_lab_workflow_closes_owned_runtime_on_setup_error(tmp_path: Path):
    calls: list[tuple[str]] = []
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=58, sim_time=5.8),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=59 + frame_index,
            sim_time=5.9 + frame_index,
        ),
        close_fn=lambda: calls.append(("close",)),
    )

    with pytest.raises(ValueError, match="conflicting output root and out"):
        product_workflow.run_optical_lab_workflow(
            preset="physics_body_triangle_video_smoke",
            output=ArtifactOutput(root=tmp_path / "a"),
            out=tmp_path / "b",
            runtime=runtime,
            products=(),
            frames=1,
            owns_runtime=True,
        )

    assert runtime.closed is True
    assert calls == [("close",)]


def test_run_optical_lab_workflow_requires_output_or_out():
    with pytest.raises(TypeError, match="requires output or out"):
        product_workflow.run_optical_lab_workflow(
            preset="physics_body_triangle_video_smoke",
            runtime=object(),
            products=(),
            frames=1,
        )


def test_run_optical_lab_workflow_rejects_conflicting_output_paths(tmp_path: Path):
    with pytest.raises(ValueError, match="conflicting output root and out"):
        product_workflow.run_optical_lab_workflow(
            preset="physics_body_triangle_video_smoke",
            output=ArtifactOutput(root=tmp_path / "a"),
            out=tmp_path / "b",
            runtime=object(),
            products=(),
            frames=1,
        )


def test_create_runtime_for_lab_preset_builds_reviewed_physics_runtime(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, object]] = []
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=61, sim_time=6.1),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=62 + frame_index,
            sim_time=6.2 + frame_index,
        ),
    )

    def fake_create_runtime(**kwargs):
        calls.append(kwargs)
        return runtime

    monkeypatch.setattr(
        preset_runtime,
        "create_physics_body_triangle_lab_runtime",
        fake_create_runtime,
    )

    created = preset_runtime.create_runtime_for_lab_preset(
        "physics_body_triangle_video_smoke",
        device="cuda:test",
        initial_height=1.25,
        metadata={"owner": "p11_test"},
    )

    assert created is runtime
    assert calls == [
        {
            "device": "cuda:test",
            "initial_height": 1.25,
            "metadata": {"owner": "p11_test"},
        }
    ]


def test_create_runtime_for_lab_preset_uses_factory_default_device(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, object]] = []
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=63, sim_time=6.3),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=64 + frame_index,
            sim_time=6.4 + frame_index,
        ),
    )

    monkeypatch.setattr(
        preset_runtime,
        "create_physics_body_triangle_lab_runtime",
        lambda **kwargs: calls.append(kwargs) or runtime,
    )

    created = preset_runtime.create_runtime_for_lab_preset(
        "physics_body_triangle_video_smoke",
        dt=2.0e-4,
    )

    assert created is runtime
    assert calls == [{"dt": 2.0e-4}]


def test_create_runtime_for_lab_preset_rejects_unregistered_preset():
    assert preset_runtime.supported_runtime_presets() == ("physics_body_triangle_video_smoke",)

    with pytest.raises(NotImplementedError, match="not registered.*go2_video_ordered_static"):
        preset_runtime.create_runtime_for_lab_preset("go2_video_ordered_static")


def test_resolve_lab_product_specs_builds_reviewed_video_and_debug_specs():
    resolved = preset_products.resolve_lab_product_specs(
        preset="physics_body_triangle_video_smoke",
        products=("video", "debug"),
    )

    video, debug = resolved
    assert isinstance(video, product_specs.VideoProductSpec)
    assert video.product_name == "video"
    assert video.build_video_camera is camera_presets.build_lab_video_camera
    assert video.pack_rgb8 is video_loop.pack_video_rgb8
    assert video.synchronize_event is video_products.synchronize_ready_event
    assert isinstance(debug, product_specs.DebugProductSpec)
    assert debug.product_name == "debug"
    assert preset_products.supported_lab_product_strings(preset="physics_body_triangle_video_smoke") == (
        "debug",
        "video",
    )


def test_resolve_lab_product_specs_does_not_import_go2_backend():
    script = """
import sys

from tools.optical_pipeline_lab.preset_products import resolve_lab_product_specs

resolve_lab_product_specs(
    preset="physics_body_triangle_video_smoke",
    products=("video",),
)
assert "tools.optical_pipeline_lab.go2_backend" not in sys.modules
assert "examples.mujoco_menagerie_robot_preview" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_resolve_lab_product_specs_passes_through_explicit_products():
    class FakeFrameProduct:
        product_name = "fake"

        def begin_run(self):
            return None

        def consume(self, tick):
            return None

        def end_run(self):
            return None

    debug = product_specs.DebugProductSpec(product_name="explicit_debug")
    frame_product = FakeFrameProduct()

    resolved = preset_products.resolve_lab_product_specs(
        preset="physics_body_triangle_video_smoke",
        products=(debug, frame_product),
    )

    assert resolved == (debug, frame_product)


def test_resolve_lab_product_specs_requires_explicit_observation_spec():
    with pytest.raises(ValueError, match="ObservationProductSpec.from_scenario"):
        preset_products.resolve_lab_product_specs(
            preset="physics_body_triangle_video_smoke",
            products=("observation",),
        )


def test_resolve_lab_product_specs_rejects_unknown_product_string():
    with pytest.raises(ValueError, match="Unsupported Optical Lab product string 'depth'"):
        preset_products.resolve_lab_product_specs(
            preset="physics_body_triangle_video_smoke",
            products=("depth",),
        )


def test_resolve_lab_product_specs_rejects_unregistered_video_preset():
    assert preset_products.supported_lab_product_strings(preset="go2_video_ordered_static") == ("debug",)

    with pytest.raises(
        NotImplementedError,
        match="video product is not registered.*go2_video_ordered_static",
    ):
        preset_products.resolve_lab_product_specs(
            preset="go2_video_ordered_static",
            products=("video",),
        )


def test_run_optical_lab_preset_delegates_to_p10_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    closed: list[bool] = []
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=200, sim_time=20.0),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=201 + frame_index,
            sim_time=20.1 + frame_index,
        ),
        close_fn=lambda: closed.append(True),
    )
    calls: list[dict[str, object]] = []

    def fake_factory(preset: str, *, device: str | None = None, **kwargs: object):
        calls.append({"preset": preset, "device": device, "kwargs": kwargs})
        return runtime

    monkeypatch.setattr(preset_workflows, "create_runtime_for_lab_preset", fake_factory)

    result = preset_workflows.run_optical_lab_preset(
        "physics_body_triangle_video_smoke",
        frames=2,
        products=("debug",),
        out=tmp_path / "preset",
        device="cpu",
        runtime_kwargs={"dt": 2.0e-4},
        initial_height=1.25,
    )

    assert calls == [
        {
            "preset": "physics_body_triangle_video_smoke",
            "device": "cpu",
            "kwargs": {"dt": 2.0e-4, "initial_height": 1.25},
        }
    ]
    assert [record.frame_id for record in result.product_results["debug"]] == [201, 202]
    assert result.artifacts == {"root": tmp_path / "preset"}
    assert closed == [True]
    payload = json.loads((tmp_path / "preset" / "scenario_config.json").read_text())
    assert payload["scenario"]["scenario_name"] == "physics_body_triangle_video_smoke"
    assert payload["run_options"]["frames"] == 2


def test_run_optical_lab_preset_accepts_frame_product_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class RecordingProduct:
        product_name = "recording"

        def begin_run(self):
            return None

        def consume(self, tick):
            return frame_products.FrameProductResult.from_tick(
                product_name=self.product_name,
                tick=tick,
                payload={"seen": tick.frame_index},
            )

        def end_run(self):
            return None

    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=210, sim_time=21.0),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=211 + frame_index,
            sim_time=21.1 + frame_index,
        ),
    )
    monkeypatch.setattr(
        preset_workflows,
        "create_runtime_for_lab_preset",
        lambda preset, **kwargs: runtime,
    )

    result = preset_workflows.run_optical_lab_preset(
        "physics_body_triangle_video_smoke",
        frames=2,
        products=(RecordingProduct(),),
        out=tmp_path / "frame_product",
    )

    assert [record.frame_id for record in result.product_results["recording"]] == [211, 212]


def test_run_optical_lab_preset_rejects_products_before_creating_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[object] = []
    monkeypatch.setattr(
        preset_workflows,
        "create_runtime_for_lab_preset",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="ObservationProductSpec.from_scenario"):
        preset_workflows.run_optical_lab_preset(
            "physics_body_triangle_video_smoke",
            frames=1,
            products=("observation",),
            out=tmp_path / "invalid_product",
        )

    assert calls == []
    assert not (tmp_path / "invalid_product").exists()


def test_run_optical_lab_preset_closes_runtime_on_p10_setup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    closed: list[bool] = []
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=220, sim_time=22.0),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=221 + frame_index,
            sim_time=22.1 + frame_index,
        ),
        close_fn=lambda: closed.append(True),
    )
    monkeypatch.setattr(
        preset_workflows,
        "create_runtime_for_lab_preset",
        lambda preset, **kwargs: runtime,
    )

    with pytest.raises(ValueError, match="conflicting output root and out"):
        preset_workflows.run_optical_lab_preset(
            "physics_body_triangle_video_smoke",
            frames=1,
            products=("debug",),
            output=ArtifactOutput(root=tmp_path / "a"),
            out=tmp_path / "b",
        )

    assert closed == [True]


def test_run_optical_lab_preset_does_not_import_go2_backend():
    script = """
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from tools.optical_pipeline_lab.physics_runtime import PhysicsLabScenarioRuntime
import tools.optical_pipeline_lab.preset_workflows as workflows

runtime = PhysicsLabScenarioRuntime(
    engine=object(),
    registry=object(),
    base_frame=SimpleNamespace(frame_id=230, sim_time=23.0),
    step_frame_fn=lambda frame_index: SimpleNamespace(
        frame_id=231 + frame_index,
        sim_time=23.1 + frame_index,
    ),
)
workflows.create_runtime_for_lab_preset = lambda preset, **kwargs: runtime
with tempfile.TemporaryDirectory() as tmp:
    workflows.run_optical_lab_preset(
        "physics_body_triangle_video_smoke",
        frames=0,
        products=("debug",),
        out=Path(tmp) / "preset",
    )
assert "tools.optical_pipeline_lab.go2_backend" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_optical_lab_examples_dry_run():
    for script in (
        "examples/optical_lab/physics_body_triangle_video_debug.py",
        "examples/optical_lab/physics_body_triangle_observation.py",
    ):
        subprocess.run(
            [
                sys.executable,
                script,
                "--dry-run",
            ],
            check=True,
        )


def test_run_physics_product_scenario_requires_physics_owned_clock(tmp_path: Path):
    config = replace(
        get_preset("physics_body_triangle_video_smoke"),
        frame_source=FrameSourceKind.STATIC_ASSET_BUILDER,
    )

    with pytest.raises(ValueError, match="frame_source='physics_published_frame'"):
        product_workflow.run_physics_product_scenario(
            config,
            ArtifactOutput(root=tmp_path / "invalid"),
            runtime=object(),
            products=(),
            frames=1,
        )

    assert not (tmp_path / "invalid").exists()


def test_run_physics_product_scenario_rejects_frame_conflict_before_writing(tmp_path: Path):
    with pytest.raises(ValueError, match="ArtifactOutput.frames conflicts"):
        product_workflow.run_physics_product_scenario(
            get_preset("physics_body_triangle_video_smoke"),
            ArtifactOutput(root=tmp_path / "conflict", frames=2),
            runtime=object(),
            products=(),
            frames=1,
        )

    assert not (tmp_path / "conflict").exists()


def test_run_physics_product_scenario_builds_debug_product_spec(tmp_path: Path):
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=59, sim_time=5.9),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=60 + frame_index,
            sim_time=6.0 + frame_index,
        ),
        metadata={"runtime_owner": "p10_spec_test", "ignored": "value"},
    )

    result = product_workflow.run_physics_product_scenario(
        get_preset("physics_body_triangle_video_smoke"),
        ArtifactOutput(root=tmp_path / "debug_spec"),
        runtime=runtime,
        products=(
            product_specs.DebugProductSpec(
                product_name="debug_spec",
                metadata_keys=("runtime_owner",),
            ),
        ),
        frames=1,
    )

    debug_result = result.product_results["debug_spec"][0]
    assert debug_result.frame_id == 60
    assert debug_result.metadata == {"runtime_owner": "p10_spec_test"}
    assert result.end_outputs["debug_spec"][0].frame_id == 60


def test_run_physics_product_scenario_builds_observation_product_spec(tmp_path: Path):
    schema = locomotion_obs_schema(
        num_actuated_joints=2,
        num_contact_bodies=2,
        include_contact_mask=True,
    )
    published_frame = CpuPublishedFrame(
        frame_id=70,
        sim_time=7.0,
        step_index=70,
        env_mask=None,
        q=np.array([1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.5, -0.5]),
        qdot=np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 1.5, -1.5]),
        X_world=[SpatialTransform.identity()],
        v_bodies=np.array([[1.0, 2.0, 3.0, 0.4, 0.5, 0.6]]),
        contact_count=1,
        contacts=object(),
        telemetry=None,
        contact_mask=np.array([0, 1], dtype=np.int32),
    )
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=published_frame,
        step_frame_fn=lambda frame_index: published_frame,
    )
    spec = product_specs.ObservationProductSpec.from_scenario(
        get_preset("physics_body_triangle_video_smoke"),
        schema=schema,
        actuated_q_indices=np.array([7, 8], dtype=np.intp),
        actuated_v_indices=np.array([6, 7], dtype=np.intp),
        contact_body_names=("left_foot", "right_foot"),
    )

    result = product_workflow.run_physics_product_scenario(
        get_preset("physics_body_triangle_video_smoke"),
        ArtifactOutput(root=tmp_path / "obs_spec"),
        runtime=runtime,
        products=(spec,),
        frames=1,
    )

    observation_result = result.product_results["observation"][0]
    np.testing.assert_allclose(
        observation_result.payload["observation"].numpy(),
        [
            1.0,
            2.0,
            3.0,
            0.4,
            0.5,
            0.6,
            1.0,
            0.0,
            0.0,
            0.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            0.0,
            1.0,
        ],
    )


def test_run_physics_product_scenario_builds_video_product_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []
    render_runtime = SimpleNamespace(pipeline=SimpleNamespace(session=SimpleNamespace(scene=object())))

    def fake_create_runtime(*args, **kwargs):
        calls.append(("create_runtime", kwargs["consumer_id"], kwargs["metadata"]["runtime_owner"]))
        return render_runtime

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, published_frame=None, env_idx: int = 0):
            calls.append(("begin_frame", frame_index, published_frame.frame_id, env_idx))

            class Scope:
                def __enter__(self_inner):
                    calls.append(("enter", frame_index))
                    return SimpleNamespace(
                        frame_id=published_frame.frame_id,
                        sim_time=published_frame.sim_time,
                        env_idx=env_idx,
                    )

                def __exit__(self_inner, exc_type, exc, tb):
                    calls.append(("exit", frame_index, exc_type))

            return Scope()

    class FakeDelivery:
        def complete_available(self, *, latest_rendered_frame_index=None):
            calls.append(("complete_available", latest_rendered_frame_index))
            return []

        def submit(self, rendered, *, frame_start):
            calls.append(("submit", rendered.frame_index, frame_start >= 0.0))
            return None

        def flush(self):
            calls.append(("flush",))
            return []

    def fake_provider(runtime_arg, *, delivery_mode):
        calls.append(("provider_factory", runtime_arg is render_runtime, delivery_mode))
        return FakeProvider()

    def fake_delivery_create(**kwargs):
        calls.append(("delivery_create", kwargs["delivery_policy_label"]))
        return FakeDelivery()

    def fake_build_plan(
        scene, args, frame_index, ray_cache, *, build_video_camera, frame_identity, geometry_mode
    ):
        calls.append(("plan", frame_index, frame_identity.frame_id, frame_identity.sim_time, geometry_mode))
        return SimpleNamespace(request=object(), camera=object())

    def fake_render_from_context(frame_context, plan, *, frame_index):
        calls.append(("render", frame_context.frame_id, frame_index))
        return SimpleNamespace(frame_index=frame_index)

    def step_physics_frame(frame_index: int):
        calls.append(("step", frame_index))
        return SimpleNamespace(frame_id=90 + frame_index, sim_time=9.0 + frame_index)

    scenario_runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=89, sim_time=8.9),
        step_frame_fn=step_physics_frame,
        metadata={"runtime_owner": "video_spec_test"},
    )

    monkeypatch.setattr(lab_runner, "create_physics_render_runtime_for_config", fake_create_runtime)
    monkeypatch.setattr(frame_contexts, "physics_frame_context_provider", fake_provider)
    monkeypatch.setattr(lab_runner.VideoDeliveryFacade, "create", staticmethod(fake_delivery_create))
    monkeypatch.setattr(lab_runner, "build_video_render_plan", fake_build_plan)
    monkeypatch.setattr(lab_runner, "render_video_frame_from_context", fake_render_from_context)

    result = product_workflow.run_physics_product_scenario(
        get_preset("physics_body_triangle_video_smoke"),
        ArtifactOutput(root=tmp_path / "video_spec", frames=2, progress_every=0),
        runtime=scenario_runtime,
        products=(
            product_specs.VideoProductSpec(
                build_video_camera=lambda scene, args, frame_index: object(),
                synchronize_event=lambda event: None,
                pack_rgb8=lambda result: result,
                consumer_id="video_spec_consumer",
                product_name="video_spec",
            ),
            product_specs.DebugProductSpec(metadata_keys=None),
        ),
    )

    assert [record.frame_id for record in result.product_results["video_spec"]] == [90, 91]
    assert [record.frame_id for record in result.product_results["debug"]] == [90, 91]
    assert isinstance(result.end_outputs["video_spec"]["rows"], FrameTimingRecorder)
    assert calls == [
        ("create_runtime", "video_spec_consumer", "video_spec_test"),
        ("provider_factory", True, "sync"),
        ("delivery_create", "sync"),
        ("step", 0),
        ("begin_frame", 0, 90, 0),
        ("enter", 0),
        ("plan", 0, 90, 9.0, "dynamic_rigid"),
        ("render", 90, 0),
        ("exit", 0, None),
        ("complete_available", 0),
        ("submit", 0, True),
        ("complete_available", 0),
        ("step", 1),
        ("begin_frame", 1, 91, 0),
        ("enter", 1),
        ("plan", 1, 91, 10.0, "dynamic_rigid"),
        ("render", 91, 1),
        ("exit", 1, None),
        ("complete_available", 1),
        ("submit", 1, True),
        ("complete_available", 1),
        ("flush",),
    ]


def test_observation_product_spec_from_scenario_requires_physics_source():
    with pytest.raises(ValueError, match="frame_source='physics_published_frame'"):
        product_specs.ObservationProductSpec.from_scenario(
            replace(
                get_preset("physics_body_triangle_video_smoke"),
                frame_source=FrameSourceKind.STATIC_ASSET_BUILDER,
            ),
            schema=locomotion_obs_schema(num_actuated_joints=0),
        )


def test_run_physics_product_scenario_rejects_invalid_product_before_writing(tmp_path: Path):
    calls: list[tuple[str]] = []
    runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=79, sim_time=7.9),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=80 + frame_index,
            sim_time=8.0 + frame_index,
        ),
        close_fn=lambda: calls.append(("close",)),
    )

    with pytest.raises(TypeError, match="FrameProduct instances or ProductSpec"):
        product_workflow.run_physics_product_scenario(
            get_preset("physics_body_triangle_video_smoke"),
            ArtifactOutput(root=tmp_path / "invalid_product"),
            runtime=runtime,
            products=(object(),),
            frames=1,
            owns_runtime=True,
        )

    assert runtime.closed is True
    assert calls == [("close",)]
    assert not (tmp_path / "invalid_product").exists()


def test_published_state_observation_product_builds_obs_schema_vector_from_tick():
    schema = locomotion_obs_schema(
        num_actuated_joints=2,
        num_contact_bodies=2,
        include_contact_mask=True,
    )
    product = observation_products.PublishedStateObservationProduct(
        engine=object(),
        schema=schema,
        root_body_idx=0,
        actuated_q_indices=np.array([7, 8], dtype=np.intp),
        actuated_v_indices=np.array([6, 7], dtype=np.intp),
        contact_body_names=("left_foot", "right_foot"),
    )
    published_frame = CpuPublishedFrame(
        frame_id=130,
        sim_time=13.0,
        step_index=130,
        env_mask=None,
        q=np.array([1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.5, -0.5]),
        qdot=np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 1.5, -1.5]),
        X_world=[SpatialTransform.identity()],
        v_bodies=np.array([[1.0, 2.0, 3.0, 0.4, 0.5, 0.6]]),
        contact_count=1,
        contacts=object(),
        telemetry=None,
        contact_mask=np.array([1, 0], dtype=np.int32),
    )
    tick = frame_tick.SimulationFrameTick(
        frame_index=3,
        env_idx=0,
        frame_id=130,
        sim_time=13.0,
        published_frame=published_frame,
        metadata={"runtime_owner": "physics_lab"},
    )

    assert product.begin_run() is None
    result = product.consume(tick)
    observation = result.payload["observation"]

    assert result.product_name == "observation"
    assert result.metadata == {
        "schema_names": schema.names,
        "schema_dim": schema.dim,
    }
    assert observation.shape == (schema.dim,)
    np.testing.assert_allclose(
        observation.numpy(),
        [
            1.0,
            2.0,
            3.0,
            0.4,
            0.5,
            0.6,
            1.0,
            0.0,
            0.0,
            0.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            1.0,
            0.0,
        ],
    )
    assert product.end_run() == (result,)


def test_published_state_observation_product_requires_published_contact_mask():
    schema = locomotion_obs_schema(
        num_actuated_joints=0,
        num_contact_bodies=1,
        include_contact_mask=True,
    )
    product = observation_products.PublishedStateObservationProduct(
        engine=object(),
        schema=schema,
        actuated_q_indices=np.array([], dtype=np.intp),
        actuated_v_indices=np.array([], dtype=np.intp),
        contact_body_names=("foot",),
    )
    published_frame = CpuPublishedFrame(
        frame_id=131,
        sim_time=13.1,
        step_index=131,
        env_mask=None,
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        qdot=np.array([], dtype=np.float64),
        X_world=[SpatialTransform.identity()],
        v_bodies=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        contact_count=1,
        contacts=[("foot", object())],
        telemetry=None,
        contact_mask=None,
    )
    tick = frame_tick.SimulationFrameTick(
        frame_index=4,
        env_idx=0,
        frame_id=131,
        sim_time=13.1,
        published_frame=published_frame,
    )

    with pytest.raises(ValueError, match="missing contact_mask"):
        product.consume(tick)


def test_published_state_observation_product_requires_actuated_q_indices():
    schema = locomotion_obs_schema(num_actuated_joints=1)

    with pytest.raises(ValueError, match="actuated_q_indices"):
        observation_products.PublishedStateObservationProduct(
            engine=object(),
            schema=schema,
            actuated_v_indices=np.array([6], dtype=np.intp),
        )


def test_published_state_observation_product_requires_actuated_v_indices():
    schema = locomotion_obs_schema(num_actuated_joints=1)

    with pytest.raises(ValueError, match="actuated_v_indices"):
        observation_products.PublishedStateObservationProduct(
            engine=object(),
            schema=schema,
            actuated_q_indices=np.array([7], dtype=np.intp),
        )


def test_published_state_observation_product_requires_contact_body_names_for_mask():
    schema = locomotion_obs_schema(
        num_actuated_joints=0,
        num_contact_bodies=2,
        include_contact_mask=True,
    )

    with pytest.raises(ValueError, match="contact_body_names length"):
        observation_products.PublishedStateObservationProduct(
            engine=object(),
            schema=schema,
            actuated_q_indices=np.array([], dtype=np.intp),
            actuated_v_indices=np.array([], dtype=np.intp),
            contact_body_names=("left_foot",),
        )


def test_create_physics_body_triangle_lab_runtime_owns_reset_and_step(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []
    frames = [
        SimpleNamespace(frame_id=90, sim_time=9.0, ready_event="ready-90"),
        SimpleNamespace(frame_id=91, sim_time=9.1, ready_event="ready-91"),
    ]

    class FakeTree:
        def default_state(self):
            calls.append(("default_state",))
            return np.zeros(7, dtype=np.float64), np.zeros(6, dtype=np.float64)

    class FakeMerged:
        tree = FakeTree()
        nv = 6

    class FakeEngine:
        def __init__(self):
            self._frames = list(frames)
            self._latest = None

        def step(self, *, q, qdot, dt):
            calls.append(("step", float(q[6]), tuple(qdot.tolist()), float(dt)))
            self._latest = self._frames.pop(0)

        def latest_published_frame(self):
            calls.append(("latest_published_frame",))
            return self._latest

        def close(self):
            calls.append(("engine_close",))

    fake_engine = FakeEngine()

    monkeypatch.setattr(physics_runtime, "_build_ball_model", lambda: "ball-model")

    def fake_merge(model):
        calls.append(("merge", model))
        return FakeMerged()

    def fake_create_engine(merged, *, device):
        calls.append(("create_engine", merged is not None, device))
        return fake_engine

    monkeypatch.setattr(physics_runtime, "_merge_single_ball_model", fake_merge)
    monkeypatch.setattr(physics_runtime, "_create_gpu_engine", fake_create_engine)

    runtime = physics_runtime.create_physics_body_triangle_lab_runtime(
        device="cuda:fake",
        initial_height=0.5,
        height_for_frame=lambda frame_index: 0.7 + 0.2 * frame_index,
        dt=1.0e-4,
        synchronize_event=lambda event: calls.append(("sync", event)),
        metadata={"test": "runtime"},
    )

    assert runtime.engine is fake_engine
    assert runtime.base_frame is frames[0]
    assert runtime.bounds_min == (-0.2, -0.2, 0.0)
    assert runtime.bounds_max == (0.4, 0.4, 1.2)
    assert runtime.metadata["producer"] == "gpu_engine"
    assert runtime.metadata["runtime_owner"] == "physics_body_triangle_lab"
    assert runtime.metadata["test"] == "runtime"

    assert runtime.step_frame(0) is frames[1]
    runtime.close()
    runtime.close()

    assert calls == [
        ("merge", "ball-model"),
        ("create_engine", True, "cuda:fake"),
        ("default_state",),
        ("step", 0.5, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1.0e-4),
        ("latest_published_frame",),
        ("sync", "ready-90"),
        ("default_state",),
        ("step", 0.7, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1.0e-4),
        ("latest_published_frame",),
        ("sync", "ready-91"),
        ("engine_close",),
    ]


def test_lab_runner_creates_physics_render_runtime_from_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    engine = object()
    registry = object()
    base_frame = SimpleNamespace(frame_id=61, sim_time=0.61)
    timings = TimingRecorder()
    scene = object()
    consumer = physics_source.physics_render_consumer("existing_consumer")
    captured: dict[str, object] = {}

    def fake_create_physics_render_runtime(**kwargs):
        captured.update(kwargs)
        return "physics-runtime"

    monkeypatch.setattr(
        physics_source,
        "create_physics_render_runtime",
        fake_create_physics_render_runtime,
    )
    config = OpticalLabScenarioConfig(
        scenario_name="physics_published_frame_smoke",
        scenario_family=OpticalLabScenarioFamily.SENSOR_ORDERED,
        frame_source=FrameSourceKind.PHYSICS_PUBLISHED_FRAME,
        clock_owner=ClockOwnerKind.EXTERNAL_PHYSICS_RUNTIME,
        geometry_mode=GeometryMode.DYNAMIC_RIGID,
        device="cuda:physics",
        shadows=False,
    )
    options = LabRunOptions(out=tmp_path / "physics", verbose_warp=True)

    runtime = create_physics_render_runtime_for_config(
        config,
        options,
        engine=engine,
        registry=registry,
        base_frame=base_frame,
        timings=timings,
        consumer_id="runtime_consumer",
        consumer=consumer,
        qos_mode="latest",
        max_lag_frames=2,
        bounds_min=(-0.1, -0.2, -0.3),
        bounds_max=(0.4, 0.5, 0.6),
        scene=scene,
        metadata={"producer": "gpu_engine"},
    )

    assert runtime == "physics-runtime"
    assert captured["engine"] is engine
    assert captured["registry"] is registry
    assert captured["base_frame"] is base_frame
    assert captured["timings"] is timings
    assert captured["consumer_id"] == "runtime_consumer"
    assert captured["consumer"] is consumer
    assert captured["qos_mode"] == "latest"
    assert captured["max_lag_frames"] == 2
    assert captured["bounds_min"] == (-0.1, -0.2, -0.3)
    assert captured["bounds_max"] == (0.4, 0.5, 0.6)
    assert captured["scene"] is scene
    assert captured["metadata"] == {"producer": "gpu_engine"}
    render_options = captured["options"]
    assert isinstance(render_options, render_session.OpticalLabRenderOptions)
    assert render_options.device == "cuda:physics"
    assert render_options.bvh_backend == "cuda_lbvh"
    assert render_options.shadows is False
    assert render_options.verbose_warp is True


def test_lab_runner_physics_runtime_helper_rejects_non_physics_frame_source(tmp_path: Path):
    config = get_preset("go2_video_ordered_static")

    with pytest.raises(ValueError, match="physics_published_frame"):
        create_physics_render_runtime_for_config(
            config,
            LabRunOptions(out=tmp_path / "go2"),
            engine=object(),
            registry=object(),
            base_frame=SimpleNamespace(frame_id=62, sim_time=0.62),
            timings=TimingRecorder(),
        )


def test_static_frame_context_provider_wraps_pipeline_begin_frame():
    frame_context = object()
    calls: list[tuple[object, ...]] = []

    class FakePipeline:
        def begin_frame(self, frame_inputs=None, *, env_idx=0):
            calls.append(("begin_frame", frame_inputs, env_idx))
            return frame_context

    provider = frame_contexts.static_frame_context_provider(FakePipeline())

    with provider.begin_frame(7, env_idx=3) as context:
        assert context is frame_context

    assert calls == [("begin_frame", None, 3)]


def test_synthetic_frame_sequence_context_provider_passes_frame_inputs():
    frames = [
        SimpleNamespace(frame_id=10, sim_time=1.0),
        SimpleNamespace(frame_id=11, sim_time=1.1),
    ]
    frame_context = object()
    calls: list[tuple[object, ...]] = []

    class FakePipeline:
        def begin_frame(self, frame_inputs=None, *, env_idx=0):
            calls.append(("begin_frame", frame_inputs, env_idx))
            return frame_context

    provider = frame_contexts.synthetic_frame_sequence_context_provider(
        FakePipeline(),
        frames,
    )

    with provider.begin_frame(1, env_idx=4) as context:
        assert context is frame_context

    assert calls == [("begin_frame", frames[1], 4)]


def test_physics_frame_context_provider_yields_context_and_completes_lease():
    published_frame = SimpleNamespace(frame_id=71, sim_time=0.71)
    borrowed_frame = SimpleNamespace(frame_id=71, sim_time=0.71)
    frame_context = object()
    consumer = physics_source.physics_render_consumer("provider_consumer")
    calls: list[tuple[object, ...]] = []

    class FakeEngine:
        def borrow_device_frame(self, consumer_id, frame_id, *, stream):
            calls.append(("borrow_device_frame", consumer_id, frame_id, stream))
            return borrowed_frame

        def complete_device_consumer(self, consumer_id, frame_id, *, stream):
            calls.append(("complete_device_consumer", consumer_id, frame_id, stream))
            return "done-event"

    class FakePipeline:
        session = SimpleNamespace(stream="render-stream")

        def begin_frame(self, frame_inputs=None, *, env_idx=0):
            calls.append(("begin_frame", frame_inputs, env_idx))
            return frame_context

    runtime = physics_source.PhysicsLabRenderRuntime(
        engine=FakeEngine(),
        pipeline=FakePipeline(),
        consumer=consumer,
    )
    provider = frame_contexts.physics_frame_context_provider(runtime)

    with provider.begin_frame(0, published_frame=published_frame, env_idx=5) as context:
        assert context is frame_context
        assert calls == [
            ("borrow_device_frame", "provider_consumer", 71, "render-stream"),
            ("begin_frame", borrowed_frame, 5),
        ]

    assert calls == [
        ("borrow_device_frame", "provider_consumer", 71, "render-stream"),
        ("begin_frame", borrowed_frame, 5),
        ("complete_device_consumer", "provider_consumer", 71, "render-stream"),
    ]


def test_physics_frame_context_provider_completes_lease_when_body_raises():
    published_frame = SimpleNamespace(frame_id=72, sim_time=0.72)
    borrowed_frame = SimpleNamespace(frame_id=72, sim_time=0.72)
    frame_context = object()
    consumer = physics_source.physics_render_consumer("provider_consumer")
    calls: list[tuple[object, ...]] = []

    class FakeEngine:
        def borrow_device_frame(self, consumer_id, frame_id, *, stream):
            calls.append(("borrow_device_frame", consumer_id, frame_id, stream))
            return borrowed_frame

        def complete_device_consumer(self, consumer_id, frame_id, *, stream):
            calls.append(("complete_device_consumer", consumer_id, frame_id, stream))
            return "done-event"

    class FakePipeline:
        session = SimpleNamespace(stream="render-stream")

        def begin_frame(self, frame_inputs=None, *, env_idx=0):
            calls.append(("begin_frame", frame_inputs, env_idx))
            return frame_context

    runtime = physics_source.PhysicsLabRenderRuntime(
        engine=FakeEngine(),
        pipeline=FakePipeline(),
        consumer=consumer,
    )
    provider = frame_contexts.physics_frame_context_provider(runtime)

    with pytest.raises(RuntimeError, match="consumer failed"):
        with provider.begin_frame(0, published_frame=published_frame, env_idx=6) as context:
            assert context is frame_context
            raise RuntimeError("consumer failed")

    assert calls == [
        ("borrow_device_frame", "provider_consumer", 72, "render-stream"),
        ("begin_frame", borrowed_frame, 6),
        ("complete_device_consumer", "provider_consumer", 72, "render-stream"),
    ]


def test_physics_frame_context_provider_rejects_torch_async_until_provider_warmup():
    with pytest.raises(ValueError, match="provider-backed warmup"):
        frame_contexts.physics_frame_context_provider(
            object(),
            delivery_mode="torch_async",
        )


def test_frame_workflow_runner_releases_provider_before_video_delivery_submit():
    calls: list[tuple[object, ...]] = []
    frame_context = object()
    rendered = delivery.RenderedVideoFrame(
        frame_index=3,
        camera=SimpleNamespace(sim_time=0.3),
        result=object(),
        camera_rays_ms=float("nan"),
        render_execute_ms=1.0,
        render_profile_row=go2_backend._render_profile_row(None),
        include_shadow_traversal_stats=False,
    )
    delivered = SimpleNamespace(completed_frame_index=3)
    recorded: list[object] = []
    published_frame = object()

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, env_idx: int = 0, published_frame=None):
            calls.append(("begin_frame", frame_index, env_idx, published_frame))

            class Scope:
                def __enter__(self_inner):
                    calls.append(("enter", frame_index))
                    return frame_context

                def __exit__(self_inner, exc_type, exc, tb):
                    calls.append(("exit", frame_index))

            return Scope()

    class FakeDelivery:
        def complete_available(self, *, latest_rendered_frame_index=None):
            calls.append(("complete_available", latest_rendered_frame_index))
            return []

        def submit(self, rendered_arg, *, frame_start):
            calls.append(("submit", rendered_arg.frame_index, frame_start >= 0.0))
            return delivered

    def consume(context, frame_index: int):
        calls.append(("consume", context, frame_index))
        return rendered

    runner = frame_runtime.FrameWorkflowRunner(
        frame_provider=FakeProvider(),
        video_consumer=consume,
        delivery=FakeDelivery(),
        delivered_video_recorder=recorded.append,
    )

    result = runner.step(
        3,
        env_idx=8,
        provider_kwargs={"published_frame": published_frame},
    )

    assert isinstance(result, frame_runtime.FrameWorkflowResult)
    assert result.frame_index == 3
    assert result.video is rendered
    assert result.delivered_video == (delivered,)
    assert recorded == [delivered]
    assert calls == [
        ("begin_frame", 3, 8, published_frame),
        ("enter", 3),
        ("consume", frame_context, 3),
        ("exit", 3),
        ("complete_available", 3),
        ("submit", 3, True),
        ("complete_available", 3),
    ]


def test_frame_workflow_runner_keeps_typed_result_when_video_consumer_is_disabled():
    calls: list[tuple[object, ...]] = []

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, env_idx: int = 0):
            calls.append(("begin_frame", frame_index, env_idx))

            class Scope:
                def __enter__(self_inner):
                    calls.append(("enter", frame_index))
                    return object()

                def __exit__(self_inner, exc_type, exc, tb):
                    calls.append(("exit", frame_index))

            return Scope()

    class FakeDelivery:
        def submit(self, rendered, *, frame_start):
            raise AssertionError("disabled video consumer should not submit delivery")

        def complete_available(self, *, latest_rendered_frame_index=None):
            raise AssertionError("disabled video consumer should not complete delivery")

    def consume(context, frame_index: int):
        calls.append(("consume", frame_index))
        return None

    runner = frame_runtime.FrameWorkflowRunner(
        frame_provider=FakeProvider(),
        video_consumer=consume,
        delivery=FakeDelivery(),
    )

    result = runner.step(4, env_idx=9)

    assert result == frame_runtime.FrameWorkflowResult(
        frame_index=4,
        video=None,
        delivered_video=(),
    )
    assert calls == [
        ("begin_frame", 4, 9),
        ("enter", 4),
        ("consume", 4),
        ("exit", 4),
    ]


def test_frame_workflow_runner_exits_provider_when_video_consumer_raises():
    calls: list[tuple[object, ...]] = []
    frame_context = object()

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, env_idx: int = 0):
            calls.append(("begin_frame", frame_index, env_idx))

            class Scope:
                def __enter__(self_inner):
                    calls.append(("enter", frame_index))
                    return frame_context

                def __exit__(self_inner, exc_type, exc, tb):
                    calls.append(("exit", frame_index, exc_type))

            return Scope()

    class FakeDelivery:
        def submit(self, rendered, *, frame_start):
            raise AssertionError("consumer failure should not submit delivery")

        def complete_available(self, *, latest_rendered_frame_index=None):
            raise AssertionError("consumer failure should not complete delivery")

    def consume(context, frame_index: int):
        calls.append(("consume", context, frame_index))
        raise RuntimeError("consumer failed")

    runner = frame_runtime.FrameWorkflowRunner(
        frame_provider=FakeProvider(),
        video_consumer=consume,
        delivery=FakeDelivery(),
    )

    with pytest.raises(RuntimeError, match="consumer failed"):
        runner.step(5, env_idx=10)

    assert calls == [
        ("begin_frame", 5, 10),
        ("enter", 5),
        ("consume", frame_context, 5),
        ("exit", 5, RuntimeError),
    ]


def test_frame_workflow_runner_flush_records_pending_video_delivery():
    delivered = (SimpleNamespace(completed_frame_index=1), SimpleNamespace(completed_frame_index=2))
    recorded: list[object] = []

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, env_idx: int = 0):
            raise AssertionError("flush should not acquire a provider frame")

    class FakeDelivery:
        def flush(self):
            return list(delivered)

    runner = frame_runtime.FrameWorkflowRunner(
        frame_provider=FakeProvider(),
        video_consumer=lambda context, frame_index: None,
        delivery=FakeDelivery(),
        delivered_video_recorder=recorded.append,
    )

    assert runner.flush() == delivered
    assert recorded == list(delivered)


def test_begin_physics_render_frame_borrows_prepares_and_completes_once():
    published_frame = SimpleNamespace(frame_id=41, sim_time=0.41)
    borrowed_frame = SimpleNamespace(frame_id=41, sim_time=0.41)
    frame_context = object()
    calls: list[tuple[object, ...]] = []

    class FakeEngine:
        def latest_published_frame(self):
            calls.append(("latest_published_frame",))
            return published_frame

        def borrow_device_frame(self, consumer_id, frame_id, *, stream):
            calls.append(("borrow_device_frame", consumer_id, frame_id, stream))
            return borrowed_frame

        def complete_device_consumer(self, consumer_id, frame_id, *, stream):
            calls.append(("complete_device_consumer", consumer_id, frame_id, stream))
            return "done-event"

    class FakePipeline:
        session = SimpleNamespace(stream="render-stream")

        def begin_frame(self, frame_inputs=None, *, env_idx=0):
            calls.append(("begin_frame", frame_inputs, env_idx))
            return frame_context

    lease = physics_source.begin_physics_render_frame(
        FakeEngine(),
        FakePipeline(),
        consumer_id="optical_lab_camera",
        env_idx=3,
    )

    assert lease.frame is borrowed_frame
    assert lease.frame_context is frame_context
    assert lease.completed is False
    assert calls == [
        ("latest_published_frame",),
        ("borrow_device_frame", "optical_lab_camera", 41, "render-stream"),
        ("begin_frame", borrowed_frame, 3),
    ]

    assert lease.complete() == "done-event"
    assert lease.complete() == "done-event"
    assert lease.completed is True
    assert calls == [
        ("latest_published_frame",),
        ("borrow_device_frame", "optical_lab_camera", 41, "render-stream"),
        ("begin_frame", borrowed_frame, 3),
        ("complete_device_consumer", "optical_lab_camera", 41, "render-stream"),
    ]


def test_begin_physics_render_frame_context_manager_completes_on_exit():
    published_frame = SimpleNamespace(frame_id=42, sim_time=0.42)
    borrowed_frame = SimpleNamespace(frame_id=42, sim_time=0.42)
    calls: list[tuple[object, ...]] = []

    class FakeEngine:
        def borrow_device_frame(self, consumer_id, frame_id, *, stream):
            calls.append(("borrow_device_frame", consumer_id, frame_id, stream))
            return borrowed_frame

        def complete_device_consumer(self, consumer_id, frame_id, *, stream):
            calls.append(("complete_device_consumer", consumer_id, frame_id, stream))
            return "done-event"

    class FakePipeline:
        session = SimpleNamespace(stream="render-stream")

        def begin_frame(self, frame_inputs=None, *, env_idx=0):
            calls.append(("begin_frame", frame_inputs, env_idx))
            return object()

    with physics_source.begin_physics_render_frame(
        FakeEngine(),
        FakePipeline(),
        consumer_id="optical_lab_camera",
        published_frame=published_frame,
        env_idx=4,
    ) as lease:
        assert lease.frame is borrowed_frame
        assert lease.completed is False

    assert lease.completed is True
    assert calls == [
        ("borrow_device_frame", "optical_lab_camera", 42, "render-stream"),
        ("begin_frame", borrowed_frame, 4),
        ("complete_device_consumer", "optical_lab_camera", 42, "render-stream"),
    ]


def test_begin_physics_render_frame_completes_borrow_when_prepare_fails():
    published_frame = SimpleNamespace(frame_id=43, sim_time=0.43)
    borrowed_frame = SimpleNamespace(frame_id=43, sim_time=0.43)
    calls: list[tuple[object, ...]] = []

    class FakeEngine:
        def borrow_device_frame(self, consumer_id, frame_id, *, stream):
            calls.append(("borrow_device_frame", consumer_id, frame_id, stream))
            return borrowed_frame

        def complete_device_consumer(self, consumer_id, frame_id, *, stream):
            calls.append(("complete_device_consumer", consumer_id, frame_id, stream))
            return "done-event"

    class FakePipeline:
        session = SimpleNamespace(stream="render-stream")

        def begin_frame(self, frame_inputs=None, *, env_idx=0):
            calls.append(("begin_frame", frame_inputs, env_idx))
            raise RuntimeError("prepare failed")

    with pytest.raises(RuntimeError, match="prepare failed"):
        physics_source.begin_physics_render_frame(
            FakeEngine(),
            FakePipeline(),
            consumer_id="optical_lab_camera",
            published_frame=published_frame,
            env_idx=5,
        )

    assert calls == [
        ("borrow_device_frame", "optical_lab_camera", 43, "render-stream"),
        ("begin_frame", borrowed_frame, 5),
        ("complete_device_consumer", "optical_lab_camera", 43, "render-stream"),
    ]


def test_begin_physics_render_frame_completes_borrow_when_prepare_raises_base_exception():
    published_frame = SimpleNamespace(frame_id=44, sim_time=0.44)
    borrowed_frame = SimpleNamespace(frame_id=44, sim_time=0.44)
    calls: list[tuple[object, ...]] = []

    class FakeEngine:
        def borrow_device_frame(self, consumer_id, frame_id, *, stream):
            calls.append(("borrow_device_frame", consumer_id, frame_id, stream))
            return borrowed_frame

        def complete_device_consumer(self, consumer_id, frame_id, *, stream):
            calls.append(("complete_device_consumer", consumer_id, frame_id, stream))
            return "done-event"

    class FakePipeline:
        session = SimpleNamespace(stream="render-stream")

        def begin_frame(self, frame_inputs=None, *, env_idx=0):
            calls.append(("begin_frame", frame_inputs, env_idx))
            raise KeyboardInterrupt("interrupted during prepare")

    with pytest.raises(KeyboardInterrupt, match="interrupted during prepare"):
        physics_source.begin_physics_render_frame(
            FakeEngine(),
            FakePipeline(),
            consumer_id="optical_lab_camera",
            published_frame=published_frame,
            env_idx=6,
        )

    assert calls == [
        ("borrow_device_frame", "optical_lab_camera", 44, "render-stream"),
        ("begin_frame", borrowed_frame, 6),
        ("complete_device_consumer", "optical_lab_camera", 44, "render-stream"),
    ]


def test_lab_render_pipeline_create_from_source_builds_canonical_session(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = object()
    base_frame = SimpleNamespace(frame_id=12, sim_time=1.2)
    source = render_session.OpticalLabRenderSource(
        registry=registry,
        base_frame=base_frame,
        bounds_min=(-1.0, -1.0, -1.0),
        bounds_max=(1.0, 1.0, 1.0),
    )
    snapshot = object()
    bvh = SimpleNamespace(stats=SimpleNamespace(detail_ms=[("partition", 0.25)]))

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
        def __init__(self, registry_arg, *, device, stream):
            assert registry_arg is registry
            assert device == "device:cuda:source"
            assert stream.device == "device:cuda:source"
            self.stream = stream

        def snapshot_from_gpu_frame(self, frame, *, env_idx, stream, include_aabb):
            assert frame is base_frame
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

    def fake_build_bvh(snapshot_arg, *, device, stream, split_strategy):
        assert snapshot_arg is snapshot
        assert device == "device:cuda:source"
        assert stream.device == "device:cuda:source"
        assert split_strategy == "sah"
        return bvh

    monkeypatch.setattr(render_session, "wp", FakeWp)
    monkeypatch.setattr(render_session, "DeviceOpticalSceneCache", FakeCache)
    monkeypatch.setattr(render_session, "build_device_bvh_from_snapshot", fake_build_bvh)
    monkeypatch.setattr(render_session, "GpuDeviceBvhDirectLightOpticalExecutor", FakeExecutor)

    def pack_rgb8(result):
        return ("packed", result)

    pipeline = render_session.OpticalLabRenderPipeline.create_from_source(
        source,
        render_session.OpticalLabRenderOptions(
            device="cuda:source",
            bvh_backend="cpu",
            bvh_split_strategy="sah",
            shadows=False,
        ),
        TimingRecorder(),
        pack_rgb8=pack_rgb8,
    )

    assert pipeline.session.scene is source
    assert pipeline.session.source is source
    assert pipeline.session.gpu_frame is base_frame
    assert pipeline.session.snapshot is snapshot
    assert pipeline.session.bvh is bvh
    assert pipeline.session.executor.shadows is False
    assert pipeline.session.bvh_backend == "cpu"
    assert pipeline.session.bvh_split_strategy == "sah"
    assert pipeline.session.pack_rgb8("rgb") == ("packed", "rgb")
    assert FakeWp.config.quiet is True
    frame = pipeline.begin_frame(env_idx=3)
    assert frame.frame_id == 12
    assert frame.sim_time == 1.2
    assert frame.env_idx == 3


def test_lab_render_pipeline_create_from_source_factory_preserves_scene_view(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = object()
    base_frame = SimpleNamespace(frame_id=13, sim_time=1.3)
    scene = SimpleNamespace(frame=base_frame, registry=registry)
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
        def __init__(self, registry_arg, *, device, stream):
            assert registry_arg is registry
            assert device == "device:cuda:factory"
            self.stream = stream

        def snapshot_from_gpu_frame(self, frame, *, env_idx, stream, include_aabb):
            assert frame is base_frame
            assert env_idx == 0
            assert stream is self.stream
            assert include_aabb is True
            return snapshot

    class FakeExecutor:
        def __init__(self, *, device, stream, shadows, ambient_rgb, background_rgb):
            self.shadows = shadows

    def fake_build_bvh(snapshot_arg, *, device, stream, split_strategy):
        assert snapshot_arg is snapshot
        assert split_strategy == "sort"
        return bvh

    monkeypatch.setattr(render_session, "wp", FakeWp)
    monkeypatch.setattr(render_session, "DeviceOpticalSceneCache", FakeCache)
    monkeypatch.setattr(render_session, "build_device_bvh_from_snapshot", fake_build_bvh)
    monkeypatch.setattr(render_session, "GpuDeviceBvhDirectLightOpticalExecutor", FakeExecutor)

    def source_factory(workspace):
        assert workspace.device == "device:cuda:factory"
        assert workspace.stream.device == "device:cuda:factory"
        return render_session.OpticalLabRenderSource(
            registry=registry,
            base_frame=base_frame,
            metadata={"scene": scene},
        )

    pipeline = render_session.OpticalLabRenderPipeline.create_from_source_factory(
        source_factory,
        render_session.OpticalLabRenderOptions(device="cuda:factory"),
        TimingRecorder(),
        scene_for_source=lambda source: source.metadata["scene"],
    )

    assert pipeline.session.scene is scene
    assert pipeline.session.source.metadata["scene"] is scene
    assert pipeline.session.gpu_frame is base_frame
    assert pipeline.begin_frame().frame_id == 13


def test_static_asset_render_source_builder_wraps_scene_and_base_frame(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = object()
    scene = SimpleNamespace(
        registry=registry,
        frame=SimpleNamespace(frame_id=14, sim_time=1.4),
        bounds_min=(-0.5, -0.5, -0.1),
        bounds_max=(0.5, 0.5, 0.9),
    )
    base_frame = object()

    def fake_build_scene(scene_preset, args):
        assert scene_preset == "synthetic_body_triangle"
        assert args.scene_preset == "synthetic_body_triangle"
        return scene

    def fake_base_gpu_frame(scene_preset, *, frame_id, sim_time, device):
        assert scene_preset == "synthetic_body_triangle"
        assert frame_id == 14
        assert sim_time == 1.4
        assert device == "device"
        return base_frame

    monkeypatch.setattr(static_asset_source, "build_static_asset_scene_for_preset", fake_build_scene)
    monkeypatch.setattr(static_asset_source, "base_gpu_frame_for_static_asset_scene", fake_base_gpu_frame)

    assert not hasattr(go2_backend, "build_go2_render_source")
    assert not hasattr(go2_backend, "build_go2_static_asset_render_source")
    source = static_asset_source.build_static_asset_render_source(
        SimpleNamespace(scene_preset="synthetic_body_triangle"),
        workspace=go2_backend.OpticalLabRenderWorkspace(device="device", stream="stream"),
    )

    assert source.registry is registry
    assert source.base_frame is base_frame
    assert source.bounds_min == (-0.5, -0.5, -0.1)
    assert source.bounds_max == (0.5, 0.5, 0.9)
    assert source.metadata["scene"] is scene
    assert source.metadata["scene_preset"] == "synthetic_body_triangle"
    assert source.metadata["source_kind"] == "static_asset"
    assert static_asset_source.scene_from_static_asset_render_source(source) is scene


def test_go2_render_options_map_args_to_generic_options():
    options = go2_backend._render_options_from_args(
        SimpleNamespace(
            device="cuda:2",
            bvh_backend="cuda_lbvh",
            bvh_split_strategy="median",
            no_shadows=True,
            verbose_warp=True,
        )
    )

    assert isinstance(options, render_session.OpticalLabRenderOptions)
    assert options.device == "cuda:2"
    assert options.bvh_backend == "cuda_lbvh"
    assert options.bvh_split_strategy == "median"
    assert options.shadows is False
    assert options.verbose_warp is True


def test_lab_render_session_accepts_workspace_with_device_stream_compatibility():
    workspace = go2_backend.OpticalLabRenderWorkspace(device="device", stream="stream")
    session = go2_backend.OpticalLabRenderSession(
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


def test_go2_render_aliases_are_removed_after_cleanup():
    assert not hasattr(go2_backend, "Go2RenderWorkspace")
    assert not hasattr(go2_backend, "Go2RenderSession")
    assert not hasattr(go2_backend, "Go2RenderFrameContext")
    assert not hasattr(go2_backend, "Go2RenderPipeline")

    with pytest.raises(ModuleNotFoundError):
        __import__("tools.optical_pipeline_lab.go2_session")


def test_lab_render_pipeline_frame_context_wraps_render_result(monkeypatch: pytest.MonkeyPatch):
    compute = SimpleNamespace(ready_event=object())
    monkeypatch.setattr(render_session, "wp", SimpleNamespace(synchronize_event=lambda event: None))

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
    pipeline = go2_backend.OpticalLabRenderPipeline(session=session)
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


def test_lab_render_pipeline_static_begin_frame_accepts_session_frame_inputs():
    session = SimpleNamespace(
        scene=SimpleNamespace(frame=SimpleNamespace(frame_id=1, sim_time=0.0)),
        gpu_frame=object(),
    )
    pipeline = go2_backend.OpticalLabRenderPipeline(session=session)

    frame = pipeline.begin_frame(frame_inputs=session.gpu_frame, env_idx=3)

    assert frame.snapshot is None
    assert frame.bvh is None
    assert frame.env_idx == 3
    assert math.isnan(float(frame.prepare_timing["snapshot_ms"]))


def test_lab_render_pipeline_dynamic_begin_frame_delegates_workspace_prepare():
    frame_inputs = SimpleNamespace(frame_id=9, sim_time=0.9)
    snapshot = object()
    bvh = object()

    class FakeWorkspace:
        def __init__(self):
            self.calls = []

        def prepare_dynamic_frame(
            self,
            frame,
            *,
            env_idx,
            cache,
            base_bvh,
            bvh_backend,
            bvh_split_strategy,
        ):
            self.calls.append((frame, env_idx, cache, base_bvh, bvh_backend, bvh_split_strategy))
            return render_session.OpticalLabPreparedFrame(
                snapshot=snapshot,
                bvh=bvh,
                prepare_timing={
                    "snapshot_ms": 1.0,
                    "accel_refit_ms": 2.0,
                    "accel_rebuild_ms": float("nan"),
                },
            )

    workspace = FakeWorkspace()
    session = SimpleNamespace(
        scene=SimpleNamespace(frame=SimpleNamespace(frame_id=1, sim_time=0.0)),
        gpu_frame=object(),
        workspace=workspace,
        cache="cache",
        bvh="base_bvh",
        bvh_backend="cpu",
        bvh_split_strategy="sort",
    )

    frame = go2_backend.OpticalLabRenderPipeline(session=session).begin_frame(
        frame_inputs=frame_inputs,
        env_idx=4,
    )

    assert frame.snapshot is snapshot
    assert frame.bvh is bvh
    assert frame.env_idx == 4
    assert frame.frame is frame_inputs
    assert frame.frame_id == 9
    assert frame.sim_time == 0.9
    assert frame.prepare_timing["snapshot_ms"] == 1.0
    assert workspace.calls == [(frame_inputs, 4, "cache", "base_bvh", "cpu", "sort")]


def test_lab_render_pipeline_dynamic_begin_frame_refits_frame_specific_snapshot(
    monkeypatch: pytest.MonkeyPatch,
):
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
        workspace=go2_backend.OpticalLabRenderWorkspace(device="cuda:fake", stream="stream"),
        cache=FakeCache(),
        bvh=SimpleNamespace(stats=SimpleNamespace(supports_refit=True)),
        bvh_backend="cpu",
        bvh_split_strategy="sort",
    )
    monkeypatch.setattr(
        render_session,
        "wp",
        SimpleNamespace(synchronize_event=lambda event: sync_events.append(event)),
    )
    monkeypatch.setattr(render_session, "refit_device_bvh_from_snapshot", fake_refit)

    frame = go2_backend.OpticalLabRenderPipeline(session=session).begin_frame(
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


def test_lab_render_pipeline_dynamic_begin_frame_rebuilds_when_refit_is_unavailable(
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
        workspace=go2_backend.OpticalLabRenderWorkspace(device="cuda:fake", stream="stream"),
        cache=FakeCache(),
        bvh=SimpleNamespace(stats=SimpleNamespace(supports_refit=False)),
        bvh_backend="cpu",
        bvh_split_strategy="partition",
    )
    monkeypatch.setattr(render_session, "wp", SimpleNamespace(synchronize_event=lambda event: None))
    monkeypatch.setattr(render_session, "build_device_bvh_from_snapshot", fake_build)

    frame = go2_backend.OpticalLabRenderPipeline(session=session).begin_frame(
        frame_inputs=frame_inputs,
        env_idx=0,
    )

    assert frame.snapshot is snapshot
    assert frame.bvh is rebuilt_bvh
    assert math.isnan(float(frame.prepare_timing["accel_refit_ms"]))
    assert frame.prepare_timing["accel_rebuild_ms"] >= 0.0


def test_lab_render_pipeline_dynamic_begin_frame_rebuilds_cuda_lbvh_when_configured(
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
        workspace=go2_backend.OpticalLabRenderWorkspace(device="cuda:fake", stream="stream"),
        cache=FakeCache(),
        bvh=SimpleNamespace(stats=SimpleNamespace(supports_refit=False)),
        bvh_backend="cuda_lbvh",
        bvh_split_strategy="partition",
    )
    monkeypatch.setattr(render_session, "wp", SimpleNamespace(synchronize_event=lambda event: None))
    monkeypatch.setattr(render_session, "build_cuda_lbvh_from_snapshot", fake_cuda_build)

    frame = go2_backend.OpticalLabRenderPipeline(session=session).begin_frame(
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


def test_run_video_benchmark_with_frame_contexts_uses_provider_and_delivery(tmp_path: Path):
    provider_calls: list[tuple[object, ...]] = []
    render_requests: list[object] = []

    class FakeFrameContext:
        prepare_timing = {
            "snapshot_ms": 0.5,
            "accel_refit_ms": 0.25,
            "accel_rebuild_ms": float("nan"),
        }

        def render(self, request):
            render_requests.append(request)
            return RuntimeRenderResult(
                compute=SimpleNamespace(ready_event=object()),
                timing={
                    "render_execute_ms": 1.5,
                    **go2_backend._render_profile_row(None),
                },
            )

    class FakeScope:
        def __init__(self, frame_index: int, env_idx: int):
            self.frame_index = frame_index
            self.env_idx = env_idx

        def __enter__(self):
            provider_calls.append(("enter", self.frame_index, self.env_idx))
            return FakeFrameContext()

        def __exit__(self, exc_type, exc, tb):
            provider_calls.append(("exit", self.frame_index))

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, env_idx: int = 0):
            provider_calls.append(("begin_frame", frame_index, env_idx))
            return FakeScope(frame_index, env_idx)

    def build_camera(scene, args, frame_index):
        return go2_backend.OpticalPinholeCameraSpec(
            frame_id=frame_index,
            sim_time=float(frame_index) / 30.0,
            env_idx=0,
            sensor_id="camera",
            width=16,
            height=8,
            fx=10.0,
            fy=10.0,
            cx=7.5,
            cy=3.5,
        )

    rows = video_loop.run_video_benchmark_with_frame_contexts(
        object(),
        SimpleNamespace(
            video_readback_delivery="sync",
            write_frames=False,
            video_readback="none",
            video_readback_ring_depth=2,
            video_raygen="gpu",
            video_ray_cache="off",
            video_frames=2,
            video_fps=30.0,
            render_profile=False,
            fail_on_overflow=False,
            progress_every=0,
            frame_timing_csv=str(tmp_path / "frame_timing.csv"),
            lab_frame_defaults={
                "readback_payload": "none",
                "delivery_policy": "sync",
            },
        ),
        tmp_path,
        frame_provider=FakeProvider(),
        build_video_camera=build_camera,
        pack_rgb8=lambda result: result,
        synchronize_event=lambda event: None,
        frame_identity_for_index=lambda frame_index: video_loop.FrameIdentity(
            frame_id=100 + frame_index,
            sim_time=10.0 + frame_index,
            env_idx=3 + frame_index,
        ),
        geometry_mode_for_index=lambda frame_index, frame_identity: "dynamic_rigid",
    )

    assert provider_calls == [
        ("begin_frame", 0, 3),
        ("enter", 0, 3),
        ("exit", 0),
        ("begin_frame", 1, 4),
        ("enter", 1, 4),
        ("exit", 1),
    ]
    assert [request.frame_id for request in render_requests] == [100, 101]
    assert [request.sim_time for request in render_requests] == [10.0, 11.0]
    assert [request.env_idx for request in render_requests] == [3, 4]
    assert [row["frame_index"] for row in rows._rows] == [0, 1]
    assert [row["geometry_mode"] for row in rows._rows] == ["dynamic_rigid", "dynamic_rigid"]
    assert rows._rows[0]["snapshot_ms"] == 0.5
    assert rows._rows[0]["accel_refit_ms"] == 0.25
    assert (tmp_path / "frame_timing.csv").exists()


def test_provider_backed_torch_async_warmup_uses_provider_lifecycle():
    compute = SimpleNamespace(ready_event=object())
    provider_calls: list[tuple[object, ...]] = []
    render_requests: list[object] = []

    class FakeFrameContext:
        prepare_timing = {}

        def render(self, request):
            render_requests.append(request)
            return RuntimeRenderResult(
                compute=compute,
                timing={
                    "render_execute_ms": 1.0,
                    **go2_backend._render_profile_row(None),
                },
            )

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, env_idx: int = 0):
            provider_calls.append(("begin_frame", frame_index, env_idx))

            class Scope:
                def __enter__(self_inner):
                    provider_calls.append(("enter", frame_index, env_idx))
                    return FakeFrameContext()

                def __exit__(self_inner, exc_type, exc, tb):
                    provider_calls.append(("exit", frame_index))

            return Scope()

    def build_camera(scene, args, frame_index):
        return go2_backend.OpticalPinholeCameraSpec(
            frame_id=frame_index,
            sim_time=float(frame_index),
            env_idx=7,
            sensor_id="camera",
            width=16,
            height=8,
            fx=10.0,
            fy=10.0,
            cx=7.5,
            cy=3.5,
        )

    result, include_shadow = video_loop.build_provider_backed_torch_async_warmup_result(
        object(),
        SimpleNamespace(
            video_raygen="gpu",
            video_ray_cache="off",
            video_readback="rgb",
            render_profile=True,
            fail_on_overflow=False,
        ),
        frame_provider=FakeProvider(),
        build_video_camera=build_camera,
    )

    assert result is compute
    assert include_shadow is True
    assert provider_calls == [
        ("begin_frame", 0, 7),
        ("enter", 0, 7),
        ("exit", 0),
    ]
    assert len(render_requests) == 1
    assert render_requests[0].camera.env_idx == 7
    assert render_requests[0].diagnostics.traversal_counters is True


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
    assert config.frame_source is FrameSourceKind.STATIC_ASSET_BUILDER
    assert config.clock_owner is ClockOwnerKind.RUNNER
    assert config.geometry_mode is GeometryMode.STATIC
    assert config.delivery_policy is DeliveryPolicy.SYNC
    assert config.width == DEFAULT_RENDER_WIDTH
    assert config.height == DEFAULT_RENDER_HEIGHT
    config.validate_implemented()


def test_synthetic_dynamic_smoke_preset_is_currently_implemented():
    config = get_preset("synthetic_body_triangle_dynamic_smoke")

    assert config.scenario_family is OpticalLabScenarioFamily.VIDEO_ORDERED_EXPORT
    assert config.scene_preset == "synthetic_body_triangle"
    assert config.frame_source is FrameSourceKind.SYNTHETIC_FRAME_SEQUENCE
    assert config.clock_owner is ClockOwnerKind.RUNNER
    assert config.geometry_mode is GeometryMode.DYNAMIC_RIGID
    assert config.accel_policy is AccelPolicy.REFIT_EACH_FRAME
    assert config.readback_payload is ReadbackPayload.RGB
    config.validate_implemented()


def test_physics_body_triangle_video_smoke_preset_is_implemented_by_explicit_lab_path():
    config = get_preset("physics_body_triangle_video_smoke")

    assert config.scenario_family is OpticalLabScenarioFamily.VIDEO_ORDERED_EXPORT
    assert config.scene_preset == "synthetic_body_triangle"
    assert config.frame_source is FrameSourceKind.PHYSICS_PUBLISHED_FRAME
    assert config.clock_owner is ClockOwnerKind.EXTERNAL_PHYSICS_RUNTIME
    assert config.geometry_mode is GeometryMode.DYNAMIC_RIGID
    assert config.accel_policy is AccelPolicy.REFIT_EACH_FRAME
    assert config.camera_mode == "fixed_view"
    config.validate_implemented()


def test_legacy_physics_runtime_frame_source_still_validates_for_explicit_lab_path(tmp_path: Path):
    config = replace(
        get_preset("physics_body_triangle_video_smoke"),
        frame_source=FrameSourceKind.PHYSICS_RUNTIME,
    )

    validate_physics_video_run(config, LabRunOptions(out=tmp_path / "physics"))


def test_run_scenario_support_predicate_separates_lab_support_from_runner_ownership(tmp_path: Path):
    static_config = get_preset("go2_video_ordered_static")
    dynamic_config = get_preset("synthetic_body_triangle_dynamic_smoke")
    physics_config = get_preset("physics_body_triangle_video_smoke")

    physics_config.validate_implemented()
    validate_run(physics_config, LabRunOptions(out=tmp_path / "physics"))

    assert can_run_scenario(static_config) is True
    assert can_run_scenario(dynamic_config) is True
    assert can_run_scenario(physics_config) is False
    with pytest.raises(RunScenarioUnsupportedError, match="cannot construct a physics engine"):
        validate_run_scenario_supported(physics_config)

    invalid_config = OpticalLabScenarioConfig(
        scenario_name="invalid_render_backend",
        scenario_family=OpticalLabScenarioFamily.RENDER_BENCH,
        render_backend=RenderBackend.OPTIX_FIRST_HIT,
    )
    with pytest.raises(NotImplementedError, match="render_backend"):
        can_run_scenario(invalid_config)

    reserved_scene_config = OpticalLabScenarioConfig(
        scenario_name="reserved_static_scene",
        scenario_family=OpticalLabScenarioFamily.RENDER_BENCH,
        scene_preset="reserved_scene",
    )
    assert can_run_scenario(reserved_scene_config) is False
    with pytest.raises(RunScenarioUnsupportedError, match="scene_preset"):
        validate_run_scenario_supported(reserved_scene_config)


def test_default_render_resolution_is_1080p():
    config = OpticalLabScenarioConfig(
        scenario_name="default_resolution",
        scenario_family=OpticalLabScenarioFamily.RENDER_BENCH,
    )

    assert DEFAULT_RENDER_WIDTH == 1920
    assert DEFAULT_RENDER_HEIGHT == 1080
    assert config.width == 1920
    assert config.height == 1080


def test_physics_runtime_frame_source_is_reserved_outside_explicit_smoke_path():
    config = OpticalLabScenarioConfig(
        scenario_name="physics_published_frame_reserved",
        scenario_family=OpticalLabScenarioFamily.SENSOR_ORDERED,
        frame_source=FrameSourceKind.PHYSICS_PUBLISHED_FRAME,
        clock_owner=ClockOwnerKind.EXTERNAL_PHYSICS_RUNTIME,
        geometry_mode=GeometryMode.DYNAMIC_RIGID,
    )

    with pytest.raises(NotImplementedError, match="reserved outside"):
        config.validate_implemented()


def test_external_physics_clock_owner_is_reserved_outside_physics_smoke_path():
    config = OpticalLabScenarioConfig(
        scenario_name="external_clock_reserved",
        scenario_family=OpticalLabScenarioFamily.SENSOR_ORDERED,
        clock_owner=ClockOwnerKind.EXTERNAL_PHYSICS_RUNTIME,
    )

    with pytest.raises(NotImplementedError, match="clock_owner"):
        config.validate_implemented()


def test_synthetic_frame_source_is_reserved_outside_dynamic_smoke_preset():
    config = OpticalLabScenarioConfig(
        scenario_name="synthetic_source_reserved",
        scenario_family=OpticalLabScenarioFamily.VIDEO_ORDERED_EXPORT,
        frame_source=FrameSourceKind.SYNTHETIC_FRAME_SEQUENCE,
        geometry_mode=GeometryMode.STATIC,
    )

    with pytest.raises(NotImplementedError, match="synthetic_frame_sequence"):
        config.validate_implemented()


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


def test_lab_runner_builds_render_options_from_go2_config(tmp_path: Path):
    config = apply_run_overrides(
        get_preset("go2_video_ordered_static"),
        device="cuda:2",
        shadows=False,
    )
    options = LabRunOptions(out=tmp_path / "run", verbose_warp=True)

    render_options = render_options_for_config(config, options)

    assert isinstance(render_options, render_session.OpticalLabRenderOptions)
    assert render_options.device == "cuda:2"
    assert render_options.bvh_backend == "cuda_lbvh"
    assert render_options.bvh_split_strategy == "sort"
    assert render_options.shadows is False
    assert render_options.verbose_warp is True


def test_lab_runner_builds_render_options_from_dynamic_smoke_config(tmp_path: Path):
    config = get_preset("synthetic_body_triangle_dynamic_smoke")
    options = LabRunOptions(out=tmp_path / "dynamic")

    render_options = render_options_for_config(config, options)

    assert render_options.device == "cuda:0"
    assert render_options.bvh_backend == "cpu"
    assert render_options.bvh_split_strategy == "sort"
    assert render_options.shadows is False
    assert render_options.verbose_warp is False


def test_lab_runner_render_options_mapping_does_not_enable_reserved_frame_source(tmp_path: Path):
    config = OpticalLabScenarioConfig(
        scenario_name="physics_published_frame_reserved",
        scenario_family=OpticalLabScenarioFamily.SENSOR_ORDERED,
        frame_source=FrameSourceKind.PHYSICS_PUBLISHED_FRAME,
        geometry_mode=GeometryMode.DYNAMIC_RIGID,
        device="cuda:physics",
        shadows=False,
    )
    options = LabRunOptions(out=tmp_path / "physics")

    render_options = render_options_for_config(config, options)

    assert render_options.device == "cuda:physics"
    assert render_options.bvh_backend == "cuda_lbvh"
    assert render_options.shadows is False
    with pytest.raises(NotImplementedError, match="physics_published_frame"):
        validate_run(config, options)


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
    assert args.lab_frame_defaults["frame_source"] == "static_asset_builder"
    assert args.lab_frame_defaults["clock_owner"] == "runner"
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
    assert args.lab_frame_defaults["frame_source"] == "synthetic_frame_sequence"
    assert args.lab_frame_defaults["clock_owner"] == "runner"
    assert args.lab_frame_defaults["geometry_mode"] == "dynamic_rigid"
    assert args.lab_frame_defaults["accel_policy"] == "refit_each_frame"


def test_lab_runner_translates_physics_smoke_preset_to_video_args(tmp_path: Path):
    config = apply_run_overrides(
        get_preset("physics_body_triangle_video_smoke"),
        width=24,
        height=12,
        readback="full",
    )
    options = LabRunOptions(out=tmp_path / "physics", frames=2, progress_every=0)

    args = build_physics_video_args(config, options)

    assert args.scene_preset == "synthetic_body_triangle"
    assert args.width == 24
    assert args.height == 12
    assert args.bvh_backend == "cpu"
    assert args.video_mode == "fixed_view"
    assert args.video_geometry_mode == "dynamic_rigid"
    assert args.video_readback == "full"
    assert args.frame_timing_csv == str(tmp_path / "physics" / "frame_timing.csv")
    assert args.lab_frame_defaults["scenario_name"] == "physics_body_triangle_video_smoke"
    assert args.lab_frame_defaults["frame_source"] == "physics_published_frame"
    assert args.lab_frame_defaults["clock_owner"] == "external_physics_runtime"
    assert args.lab_frame_defaults["geometry_mode"] == "dynamic_rigid"
    assert args.lab_frame_defaults["readback_payload"] == "full"


def test_menagerie_arg_builder_rejects_physics_runtime_config(tmp_path: Path):
    config = get_preset("physics_body_triangle_video_smoke")

    with pytest.raises(NotImplementedError, match="build_physics_video_args"):
        build_menagerie_example_args(
            config,
            LabRunOptions(out=tmp_path / "physics"),
        )


def test_physics_video_runner_rejects_torch_async_until_warmup_source_exists(tmp_path: Path):
    config = get_preset("physics_body_triangle_video_smoke")

    with pytest.raises(NotImplementedError, match="provider-backed torch_async warmup"):
        validate_physics_video_run(
            config,
            LabRunOptions(
                out=tmp_path / "physics",
                video_readback_delivery="torch_async",
            ),
        )
    with pytest.raises(NotImplementedError, match="provider-backed torch_async warmup"):
        validate_physics_video_product_run(
            config,
            LabRunOptions(
                out=tmp_path / "physics_product",
                video_readback_delivery="torch_async",
            ),
        )


def test_physics_stepped_video_runner_steps_before_provider_borrow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []
    runtime = SimpleNamespace(pipeline=SimpleNamespace(session=SimpleNamespace(scene=object())))

    def fake_create_runtime(*args, **kwargs):
        calls.append(("create_runtime", kwargs["consumer_id"]))
        return runtime

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, published_frame=None, env_idx: int = 0):
            calls.append(("begin_frame", frame_index, published_frame.frame_id, env_idx))

            class Scope:
                def __enter__(self_inner):
                    calls.append(("enter", frame_index))
                    return SimpleNamespace(
                        frame_id=published_frame.frame_id,
                        sim_time=published_frame.sim_time,
                        env_idx=env_idx,
                    )

                def __exit__(self_inner, exc_type, exc, tb):
                    calls.append(("exit", frame_index, exc_type))

            return Scope()

    class FakeDelivery:
        def complete_available(self, *, latest_rendered_frame_index=None):
            calls.append(("complete_available", latest_rendered_frame_index))
            return []

        def submit(self, rendered, *, frame_start):
            calls.append(("submit", rendered.frame_index, frame_start >= 0.0))
            return None

        def flush(self):
            calls.append(("flush",))
            return []

    def fake_provider(runtime_arg, *, delivery_mode):
        calls.append(("provider_factory", runtime_arg is runtime, delivery_mode))
        return FakeProvider()

    def fake_delivery_create(**kwargs):
        calls.append(("delivery_create", kwargs["delivery_policy_label"]))
        return FakeDelivery()

    def fake_build_plan(
        scene, args, frame_index, ray_cache, *, build_video_camera, frame_identity, geometry_mode
    ):
        calls.append(("plan", frame_index, frame_identity.frame_id, frame_identity.sim_time, geometry_mode))
        return SimpleNamespace(request=object(), camera=object())

    def fake_render_from_context(frame_context, plan, *, frame_index):
        calls.append(("render", frame_context.frame_id, frame_index))
        return SimpleNamespace(frame_index=frame_index)

    def step_physics_frame(frame_index: int):
        calls.append(("step", frame_index))
        return SimpleNamespace(frame_id=70 + frame_index, sim_time=7.0 + frame_index)

    monkeypatch.setattr(lab_runner, "create_physics_render_runtime_for_config", fake_create_runtime)
    monkeypatch.setattr(frame_contexts, "physics_frame_context_provider", fake_provider)
    monkeypatch.setattr(lab_runner.VideoDeliveryFacade, "create", staticmethod(fake_delivery_create))
    monkeypatch.setattr(lab_runner, "build_video_render_plan", fake_build_plan)
    monkeypatch.setattr(lab_runner, "render_video_frame_from_context", fake_render_from_context)

    rows = run_physics_stepped_video_scenario(
        get_preset("physics_body_triangle_video_smoke"),
        LabRunOptions(out=tmp_path / "physics", frames=2, progress_every=0),
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=69, sim_time=6.9),
        step_physics_frame=step_physics_frame,
        build_video_camera=lambda scene, args, frame_index: object(),
        synchronize_event=lambda event: None,
        pack_rgb8=lambda result: result,
        consumer_id="stepped_consumer",
    )

    assert isinstance(rows, FrameTimingRecorder)
    assert calls == [
        ("create_runtime", "stepped_consumer"),
        ("provider_factory", True, "sync"),
        ("delivery_create", "sync"),
        ("step", 0),
        ("begin_frame", 0, 70, 0),
        ("enter", 0),
        ("plan", 0, 70, 7.0, "dynamic_rigid"),
        ("render", 70, 0),
        ("exit", 0, None),
        ("complete_available", 0),
        ("submit", 0, True),
        ("complete_available", 0),
        ("step", 1),
        ("begin_frame", 1, 71, 0),
        ("enter", 1),
        ("plan", 1, 71, 8.0, "dynamic_rigid"),
        ("render", 71, 1),
        ("exit", 1, None),
        ("complete_available", 1),
        ("submit", 1, True),
        ("complete_available", 1),
        ("flush",),
    ]


def test_physics_stepped_video_runner_stepper_exception_stops_before_provider_borrow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []
    runtime = SimpleNamespace(pipeline=SimpleNamespace(session=SimpleNamespace(scene=object())))

    def fake_create_runtime(*args, **kwargs):
        calls.append(("create_runtime",))
        return runtime

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, published_frame=None, env_idx: int = 0):
            calls.append(("begin_frame", frame_index, published_frame, env_idx))
            raise AssertionError("stepper failure should stop before provider borrow")

    class FakeDelivery:
        def complete_available(self, *, latest_rendered_frame_index=None):
            raise AssertionError("stepper failure should not reach delivery")

        def submit(self, rendered, *, frame_start):
            raise AssertionError("stepper failure should not submit delivery")

        def flush(self):
            raise AssertionError("stepper failure should not flush delivery")

    monkeypatch.setattr(lab_runner, "create_physics_render_runtime_for_config", fake_create_runtime)
    monkeypatch.setattr(
        frame_contexts,
        "physics_frame_context_provider",
        lambda runtime_arg, *, delivery_mode: FakeProvider(),
    )
    monkeypatch.setattr(
        lab_runner.VideoDeliveryFacade,
        "create",
        staticmethod(lambda **kwargs: FakeDelivery()),
    )

    def step_physics_frame(frame_index: int):
        calls.append(("step", frame_index))
        raise RuntimeError("physics step failed")

    with pytest.raises(RuntimeError, match="physics step failed"):
        run_physics_stepped_video_scenario(
            get_preset("physics_body_triangle_video_smoke"),
            LabRunOptions(out=tmp_path / "physics", frames=1, progress_every=0),
            engine=object(),
            registry=object(),
            base_frame=SimpleNamespace(frame_id=72, sim_time=7.2),
            step_physics_frame=step_physics_frame,
            build_video_camera=lambda scene, args, frame_index: object(),
            synchronize_event=lambda event: None,
            pack_rgb8=lambda result: result,
        )

    assert calls == [("create_runtime",), ("step", 0)]


def test_physics_stepped_video_runner_render_exception_completes_provider_borrow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []
    runtime = SimpleNamespace(pipeline=SimpleNamespace(session=SimpleNamespace(scene=object())))
    published_frame = SimpleNamespace(frame_id=73, sim_time=7.3)

    monkeypatch.setattr(
        lab_runner,
        "create_physics_render_runtime_for_config",
        lambda *args, **kwargs: runtime,
    )

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, published_frame=None, env_idx: int = 0):
            calls.append(("begin_frame", frame_index, published_frame.frame_id, env_idx))

            class Scope:
                def __enter__(self_inner):
                    calls.append(("enter", frame_index))
                    return SimpleNamespace(
                        frame_id=published_frame.frame_id,
                        sim_time=published_frame.sim_time,
                        env_idx=env_idx,
                    )

                def __exit__(self_inner, exc_type, exc, tb):
                    calls.append(("complete_borrow", frame_index, exc_type))

            return Scope()

    class FakeDelivery:
        def complete_available(self, *, latest_rendered_frame_index=None):
            raise AssertionError("render failure should not reach delivery")

        def submit(self, rendered, *, frame_start):
            raise AssertionError("render failure should not submit delivery")

        def flush(self):
            raise AssertionError("render failure should not flush delivery")

    monkeypatch.setattr(
        frame_contexts,
        "physics_frame_context_provider",
        lambda runtime_arg, *, delivery_mode: FakeProvider(),
    )
    monkeypatch.setattr(
        lab_runner.VideoDeliveryFacade,
        "create",
        staticmethod(lambda **kwargs: FakeDelivery()),
    )
    monkeypatch.setattr(
        lab_runner,
        "build_video_render_plan",
        lambda *args, **kwargs: SimpleNamespace(request=object(), camera=object()),
    )

    def fake_render_from_context(frame_context, plan, *, frame_index):
        calls.append(("render", frame_context.frame_id, frame_index))
        raise RuntimeError("render failed")

    monkeypatch.setattr(lab_runner, "render_video_frame_from_context", fake_render_from_context)

    with pytest.raises(RuntimeError, match="render failed"):
        run_physics_stepped_video_scenario(
            get_preset("physics_body_triangle_video_smoke"),
            LabRunOptions(out=tmp_path / "physics", frames=1, progress_every=0),
            engine=object(),
            registry=object(),
            base_frame=SimpleNamespace(frame_id=72, sim_time=7.2),
            step_physics_frame=lambda frame_index: published_frame,
            build_video_camera=lambda scene, args, frame_index: object(),
            synchronize_event=lambda event: None,
            pack_rgb8=lambda result: result,
        )

    assert calls == [
        ("begin_frame", 0, 73, 0),
        ("enter", 0),
        ("render", 73, 0),
        ("complete_borrow", 0, RuntimeError),
    ]


def test_physics_video_product_validation_is_distinct_from_video_only_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        lab_runner,
        "validate_physics_video_run",
        lambda config, options: (_ for _ in ()).throw(RuntimeError("old validation called")),
    )

    validate_physics_video_product_run(
        get_preset("physics_body_triangle_video_smoke"),
        LabRunOptions(out=tmp_path / "physics"),
    )


def test_physics_video_product_runner_uses_shared_product_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []

    class FakeVideoProduct:
        product_name = "video"

        def __init__(self):
            self.rows = FrameTimingRecorder(csv_path=None)

        def begin_run(self):
            calls.append(("begin",))
            return None

        def consume(self, tick):
            calls.append(("consume", tick.frame_index, tick.frame_id))
            return frame_products.FrameProductResult.from_tick(
                product_name=self.product_name,
                tick=tick,
            )

        def end_run(self):
            calls.append(("end",))
            return {"rows": self.rows}

    def fake_build_product(*args, **kwargs):
        calls.append(("build_product", kwargs["consumer_id"]))
        return FakeVideoProduct()

    scenario_runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=99, sim_time=9.9),
        step_frame_fn=lambda frame_index: SimpleNamespace(
            frame_id=100 + frame_index,
            sim_time=10.0 + frame_index,
        ),
    )
    monkeypatch.setattr(lab_runner, "build_physics_video_frame_product", fake_build_product)

    rows = run_physics_stepped_video_product_scenario(
        get_preset("physics_body_triangle_video_smoke"),
        LabRunOptions(out=tmp_path / "physics", frames=1, progress_every=0),
        scenario_runtime=scenario_runtime,
        build_video_camera=lambda scene, args, frame_index: object(),
        synchronize_event=lambda event: None,
        pack_rgb8=lambda result: result,
        consumer_id="shared_builder",
    )

    assert isinstance(rows, FrameTimingRecorder)
    assert calls == [
        ("build_product", "shared_builder"),
        ("begin",),
        ("consume", 0, 100),
        ("end",),
    ]


def test_physics_video_product_runner_steps_tick_before_provider_borrow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []
    render_runtime = SimpleNamespace(pipeline=SimpleNamespace(session=SimpleNamespace(scene=object())))

    def fake_old_helper(*args, **kwargs):
        raise AssertionError("product path must not call run_physics_stepped_video_scenario")

    def fake_create_runtime(*args, **kwargs):
        calls.append(("create_runtime", kwargs["consumer_id"], kwargs["metadata"]["runtime_owner"]))
        return render_runtime

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, published_frame=None, env_idx: int = 0):
            calls.append(("begin_frame", frame_index, published_frame.frame_id, env_idx))

            class Scope:
                def __enter__(self_inner):
                    calls.append(("enter", frame_index))
                    return SimpleNamespace(
                        frame_id=published_frame.frame_id,
                        sim_time=published_frame.sim_time,
                        env_idx=env_idx,
                    )

                def __exit__(self_inner, exc_type, exc, tb):
                    calls.append(("exit", frame_index, exc_type))

            return Scope()

    class FakeDelivery:
        def complete_available(self, *, latest_rendered_frame_index=None):
            calls.append(("complete_available", latest_rendered_frame_index))
            return []

        def submit(self, rendered, *, frame_start):
            calls.append(("submit", rendered.frame_index, frame_start >= 0.0))
            return None

        def flush(self):
            calls.append(("flush",))
            return []

    def fake_provider(runtime_arg, *, delivery_mode):
        calls.append(("provider_factory", runtime_arg is render_runtime, delivery_mode))
        return FakeProvider()

    def fake_delivery_create(**kwargs):
        calls.append(("delivery_create", kwargs["delivery_policy_label"]))
        return FakeDelivery()

    def fake_build_plan(
        scene, args, frame_index, ray_cache, *, build_video_camera, frame_identity, geometry_mode
    ):
        calls.append(("plan", frame_index, frame_identity.frame_id, frame_identity.sim_time, geometry_mode))
        return SimpleNamespace(request=object(), camera=object())

    def fake_render_from_context(frame_context, plan, *, frame_index):
        calls.append(("render", frame_context.frame_id, frame_index))
        return SimpleNamespace(frame_index=frame_index)

    def step_physics_frame(frame_index: int):
        calls.append(("step", frame_index))
        return SimpleNamespace(frame_id=100 + frame_index, sim_time=10.0 + frame_index)

    scenario_runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=99, sim_time=9.9),
        step_frame_fn=step_physics_frame,
        metadata={"runtime_owner": "physics_lab_test"},
    )

    monkeypatch.setattr(lab_runner, "run_physics_stepped_video_scenario", fake_old_helper)
    monkeypatch.setattr(lab_runner, "create_physics_render_runtime_for_config", fake_create_runtime)
    monkeypatch.setattr(frame_contexts, "physics_frame_context_provider", fake_provider)
    monkeypatch.setattr(lab_runner.VideoDeliveryFacade, "create", staticmethod(fake_delivery_create))
    monkeypatch.setattr(lab_runner, "build_video_render_plan", fake_build_plan)
    monkeypatch.setattr(lab_runner, "render_video_frame_from_context", fake_render_from_context)

    rows = run_physics_stepped_video_product_scenario(
        get_preset("physics_body_triangle_video_smoke"),
        LabRunOptions(out=tmp_path / "physics", frames=2, progress_every=0),
        scenario_runtime=scenario_runtime,
        build_video_camera=lambda scene, args, frame_index: object(),
        synchronize_event=lambda event: None,
        pack_rgb8=lambda result: result,
        consumer_id="product_consumer",
    )

    assert isinstance(rows, FrameTimingRecorder)
    assert calls == [
        ("create_runtime", "product_consumer", "physics_lab_test"),
        ("provider_factory", True, "sync"),
        ("delivery_create", "sync"),
        ("step", 0),
        ("begin_frame", 0, 100, 0),
        ("enter", 0),
        ("plan", 0, 100, 10.0, "dynamic_rigid"),
        ("render", 100, 0),
        ("exit", 0, None),
        ("complete_available", 0),
        ("submit", 0, True),
        ("complete_available", 0),
        ("step", 1),
        ("begin_frame", 1, 101, 0),
        ("enter", 1),
        ("plan", 1, 101, 11.0, "dynamic_rigid"),
        ("render", 101, 1),
        ("exit", 1, None),
        ("complete_available", 1),
        ("submit", 1, True),
        ("complete_available", 1),
        ("flush",),
    ]


def test_physics_video_product_runner_can_share_tick_stream_with_debug_product(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []
    render_runtime = SimpleNamespace(pipeline=SimpleNamespace(session=SimpleNamespace(scene=object())))

    def fake_create_runtime(*args, **kwargs):
        calls.append(("create_runtime", kwargs["consumer_id"], kwargs["metadata"]["runtime_owner"]))
        return render_runtime

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, published_frame=None, env_idx: int = 0):
            calls.append(("begin_frame", frame_index, published_frame.frame_id, env_idx))

            class Scope:
                def __enter__(self_inner):
                    calls.append(("enter", frame_index))
                    return SimpleNamespace(
                        frame_id=published_frame.frame_id,
                        sim_time=published_frame.sim_time,
                        env_idx=env_idx,
                    )

                def __exit__(self_inner, exc_type, exc, tb):
                    calls.append(("exit", frame_index, exc_type))

            return Scope()

    class FakeDelivery:
        def complete_available(self, *, latest_rendered_frame_index=None):
            calls.append(("complete_available", latest_rendered_frame_index))
            return []

        def submit(self, rendered, *, frame_start):
            calls.append(("submit", rendered.frame_index, frame_start >= 0.0))
            return None

        def flush(self):
            calls.append(("flush",))
            return []

    class RecordingDebugProduct(frame_products.DebugFrameProduct):
        def consume(self, tick):
            calls.append(("debug", tick.frame_index, tick.frame_id, tick.sim_time, dict(tick.metadata)))
            return super().consume(tick)

    def fake_provider(runtime_arg, *, delivery_mode):
        calls.append(("provider_factory", runtime_arg is render_runtime, delivery_mode))
        return FakeProvider()

    def fake_delivery_create(**kwargs):
        calls.append(("delivery_create", kwargs["delivery_policy_label"]))
        return FakeDelivery()

    def fake_build_plan(
        scene, args, frame_index, ray_cache, *, build_video_camera, frame_identity, geometry_mode
    ):
        calls.append(("plan", frame_index, frame_identity.frame_id, frame_identity.sim_time, geometry_mode))
        return SimpleNamespace(request=object(), camera=object())

    def fake_render_from_context(frame_context, plan, *, frame_index):
        calls.append(("render", frame_context.frame_id, frame_index))
        return SimpleNamespace(frame_index=frame_index)

    def step_physics_frame(frame_index: int):
        calls.append(("step", frame_index))
        return SimpleNamespace(frame_id=120 + frame_index, sim_time=12.0 + frame_index)

    scenario_runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=119, sim_time=11.9),
        step_frame_fn=step_physics_frame,
        metadata={"runtime_owner": "physics_lab_test", "product_set": "video_debug"},
    )
    debug_product = RecordingDebugProduct(product_name="debug", metadata_keys=None)

    monkeypatch.setattr(lab_runner, "create_physics_render_runtime_for_config", fake_create_runtime)
    monkeypatch.setattr(frame_contexts, "physics_frame_context_provider", fake_provider)
    monkeypatch.setattr(lab_runner.VideoDeliveryFacade, "create", staticmethod(fake_delivery_create))
    monkeypatch.setattr(lab_runner, "build_video_render_plan", fake_build_plan)
    monkeypatch.setattr(lab_runner, "render_video_frame_from_context", fake_render_from_context)

    rows = run_physics_stepped_video_product_scenario(
        get_preset("physics_body_triangle_video_smoke"),
        LabRunOptions(out=tmp_path / "physics", frames=2, progress_every=0),
        scenario_runtime=scenario_runtime,
        build_video_camera=lambda scene, args, frame_index: object(),
        synchronize_event=lambda event: None,
        pack_rgb8=lambda result: result,
        consumer_id="product_consumer",
        extra_products=(debug_product,),
    )

    assert isinstance(rows, FrameTimingRecorder)
    assert [record.frame_index for record in debug_product.records] == [0, 1]
    assert [record.frame_id for record in debug_product.records] == [120, 121]
    assert [record.sim_time for record in debug_product.records] == [12.0, 13.0]
    assert debug_product.records[0].metadata == {
        "runtime_owner": "physics_lab_test",
        "product_set": "video_debug",
    }
    assert calls == [
        ("create_runtime", "product_consumer", "physics_lab_test"),
        ("provider_factory", True, "sync"),
        ("delivery_create", "sync"),
        ("step", 0),
        ("begin_frame", 0, 120, 0),
        ("enter", 0),
        ("plan", 0, 120, 12.0, "dynamic_rigid"),
        ("render", 120, 0),
        ("exit", 0, None),
        ("complete_available", 0),
        ("submit", 0, True),
        ("complete_available", 0),
        (
            "debug",
            0,
            120,
            12.0,
            {"runtime_owner": "physics_lab_test", "product_set": "video_debug"},
        ),
        ("step", 1),
        ("begin_frame", 1, 121, 0),
        ("enter", 1),
        ("plan", 1, 121, 13.0, "dynamic_rigid"),
        ("render", 121, 1),
        ("exit", 1, None),
        ("complete_available", 1),
        ("submit", 1, True),
        ("complete_available", 1),
        (
            "debug",
            1,
            121,
            13.0,
            {"runtime_owner": "physics_lab_test", "product_set": "video_debug"},
        ),
        ("flush",),
    ]


def test_physics_video_product_runner_stepper_exception_stops_before_provider_borrow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []
    render_runtime = SimpleNamespace(pipeline=SimpleNamespace(session=SimpleNamespace(scene=object())))

    monkeypatch.setattr(
        lab_runner,
        "create_physics_render_runtime_for_config",
        lambda *args, **kwargs: calls.append(("create_runtime",)) or render_runtime,
    )

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, published_frame=None, env_idx: int = 0):
            raise AssertionError("step failure should stop before provider borrow")

    class FakeDelivery:
        def complete_available(self, *, latest_rendered_frame_index=None):
            raise AssertionError("step failure should not reach delivery")

        def submit(self, rendered, *, frame_start):
            raise AssertionError("step failure should not submit delivery")

        def flush(self):
            raise AssertionError("step failure should not flush delivery")

    monkeypatch.setattr(
        frame_contexts,
        "physics_frame_context_provider",
        lambda runtime_arg, *, delivery_mode: FakeProvider(),
    )
    monkeypatch.setattr(
        lab_runner.VideoDeliveryFacade,
        "create",
        staticmethod(lambda **kwargs: FakeDelivery()),
    )

    def step_physics_frame(frame_index: int):
        calls.append(("step", frame_index))
        raise RuntimeError("product physics step failed")

    scenario_runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=109, sim_time=10.9),
        step_frame_fn=step_physics_frame,
    )

    with pytest.raises(RuntimeError, match="product physics step failed"):
        run_physics_stepped_video_product_scenario(
            get_preset("physics_body_triangle_video_smoke"),
            LabRunOptions(out=tmp_path / "physics", frames=1, progress_every=0),
            scenario_runtime=scenario_runtime,
            build_video_camera=lambda scene, args, frame_index: object(),
            synchronize_event=lambda event: None,
            pack_rgb8=lambda result: result,
        )

    assert calls == [("create_runtime",), ("step", 0)]


def test_physics_video_product_runner_render_exception_completes_provider_borrow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, ...]] = []
    render_runtime = SimpleNamespace(pipeline=SimpleNamespace(session=SimpleNamespace(scene=object())))
    published_frame = SimpleNamespace(frame_id=110, sim_time=11.0)

    monkeypatch.setattr(
        lab_runner,
        "create_physics_render_runtime_for_config",
        lambda *args, **kwargs: render_runtime,
    )

    class FakeProvider:
        def begin_frame(self, frame_index: int, *, published_frame=None, env_idx: int = 0):
            calls.append(("begin_frame", frame_index, published_frame.frame_id, env_idx))

            class Scope:
                def __enter__(self_inner):
                    calls.append(("enter", frame_index))
                    return SimpleNamespace(
                        frame_id=published_frame.frame_id,
                        sim_time=published_frame.sim_time,
                        env_idx=env_idx,
                    )

                def __exit__(self_inner, exc_type, exc, tb):
                    calls.append(("complete_borrow", frame_index, exc_type))

            return Scope()

    class FakeDelivery:
        def complete_available(self, *, latest_rendered_frame_index=None):
            raise AssertionError("render failure should not reach delivery")

        def submit(self, rendered, *, frame_start):
            raise AssertionError("render failure should not submit delivery")

        def flush(self):
            raise AssertionError("render failure should not flush delivery")

    monkeypatch.setattr(
        frame_contexts,
        "physics_frame_context_provider",
        lambda runtime_arg, *, delivery_mode: FakeProvider(),
    )
    monkeypatch.setattr(
        lab_runner.VideoDeliveryFacade,
        "create",
        staticmethod(lambda **kwargs: FakeDelivery()),
    )
    monkeypatch.setattr(
        lab_runner,
        "build_video_render_plan",
        lambda *args, **kwargs: SimpleNamespace(request=object(), camera=object()),
    )

    def fake_render_from_context(frame_context, plan, *, frame_index):
        calls.append(("render", frame_context.frame_id, frame_index))
        raise RuntimeError("product render failed")

    monkeypatch.setattr(lab_runner, "render_video_frame_from_context", fake_render_from_context)

    scenario_runtime = physics_runtime.PhysicsLabScenarioRuntime(
        engine=object(),
        registry=object(),
        base_frame=SimpleNamespace(frame_id=109, sim_time=10.9),
        step_frame_fn=lambda frame_index: published_frame,
    )

    with pytest.raises(RuntimeError, match="product render failed"):
        run_physics_stepped_video_product_scenario(
            get_preset("physics_body_triangle_video_smoke"),
            LabRunOptions(out=tmp_path / "physics", frames=1, progress_every=0),
            scenario_runtime=scenario_runtime,
            build_video_camera=lambda scene, args, frame_index: object(),
            synchronize_event=lambda event: None,
            pack_rgb8=lambda result: result,
        )

    assert calls == [
        ("begin_frame", 0, 110, 0),
        ("enter", 0),
        ("render", 110, 0),
        ("complete_borrow", 0, RuntimeError),
    ]


def test_static_asset_source_configures_synthetic_dynamic_video_frames(monkeypatch: pytest.MonkeyPatch):
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
    monkeypatch.setattr(static_asset_source, "wp", _FakeWpModule)

    static_asset_source.configure_dynamic_video_frame_inputs(
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
    assert payload["scenario"]["frame_source"] == "static_asset_builder"
    assert payload["scenario"]["clock_owner"] == "runner"
    assert payload["scenario"]["readback_payload"] == "rgb"
    assert payload["run_options"]["root"] == str(tmp_path / "run")
    assert payload["run_options"]["frames"] == 2
    assert "_frames_explicit" not in payload["run_options"]


def test_lab_runner_serializes_physics_source_and_clock_metadata(tmp_path: Path):
    config = get_preset("physics_body_triangle_video_smoke")
    options = LabRunOptions(out=tmp_path / "physics", frames=1)
    path = tmp_path / "physics" / "scenario_config.json"

    write_scenario_config(path, config, options)

    payload = json.loads(path.read_text())
    assert payload["scenario"]["frame_source"] == "physics_published_frame"
    assert payload["scenario"]["clock_owner"] == "external_physics_runtime"


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


def test_run_scenario_physics_runtime_requires_explicit_runtime_inputs(tmp_path: Path):
    config = apply_run_overrides(
        get_preset("physics_body_triangle_video_smoke"),
        readback="full",
    )
    options = LabRunOptions(out=tmp_path / "physics", frames=1)

    with pytest.raises(NotImplementedError) as exc_info:
        run_scenario(config, options)

    message = str(exc_info.value)
    assert "cannot construct a physics engine" in message
    assert "run_physics_video_scenario" in message
    assert "run_physics_stepped_video_scenario" in message
    assert not options.out.exists()


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
    assert written[0]["frame_source"] == "static_asset_builder"
    assert written[0]["clock_owner"] == "runner"
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
