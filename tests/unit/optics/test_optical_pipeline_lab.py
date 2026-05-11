import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.optical_pipeline_lab.__main__ as lab_main
import tools.optical_pipeline_lab.async_readback as async_readback
import tools.optical_pipeline_lab.go2_backend as go2_backend
import tools.optical_pipeline_lab.rgb_pack as rgb_pack
from optics.render_api import DeliveryPolicy as RuntimeDeliveryPolicy
from optics.render_api import ReadbackPayload as RuntimeReadbackPayload
from optics.render_api import RenderBackend as RuntimeRenderBackend
from optics.render_api import WritePolicy as RuntimeWritePolicy
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


def test_sync_video_readback_none_row_does_not_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_render_video_frame(session, args, frame_index, ray_cache):
        return go2_backend._RenderedVideoFrame(
            frame_index=frame_index,
            camera=SimpleNamespace(sim_time=0.0),
            result=object(),
            camera_rays_ms=float("nan"),
            render_execute_ms=1.25,
            pack_rgb8_ms=float("nan"),
            render_profile_row=go2_backend._render_profile_row(None),
        )

    def fail_if_staged(*args, **kwargs):
        raise AssertionError("readback=none should not stage host channels")

    monkeypatch.setattr(go2_backend, "_render_video_frame", fake_render_video_frame)
    monkeypatch.setattr(go2_backend, "stage_optical_channels", fail_if_staged)
    monkeypatch.setattr(go2_backend, "stage_optical_compute_result_to_host", fail_if_staged)

    frame_timing_csv = tmp_path / "frame_timing.csv"
    rows = go2_backend._run_video_benchmark(
        session=SimpleNamespace(scene=object()),
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
    assert math.isnan(float(row["readback_host_ms"]))
    assert row["frame_path"] == ""
    assert frame_timing_csv.exists()


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
            "go2_video_ordered_baseline",
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
    assert suite.name == "go2_video_ordered_baseline"
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
