"""Matrix suite runner for Optical Pipeline Lab baselines."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .presets import get_preset
from .runner import DEFAULT_LAB_WARMUP_RENDERS, LabRunOptions, apply_run_overrides, run_scenario
from .scenarios import (
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_WIDTH,
    OpticalLabScenarioConfig,
    ReadbackPayload,
    WritePolicy,
)
from .timing import NAN, percentile

MATRIX_SUMMARY_FIELDNAMES = (
    "case_name",
    "status",
    "error",
    "output_dir",
    "width",
    "height",
    "shadows",
    "readback_payload",
    "write_policy",
    "frames",
    "fps_mean",
    "frame_p50_ms",
    "frame_p90_ms",
    "render_execute_mean_ms",
    "readback_host_mean_ms",
    "image_build_mean_ms",
    "encode_write_mean_ms",
)

AVAILABLE_MATRIX_SUITES = (
    "go2_video_ordered_baseline",
    "go2_video_ordered_legacy_960",
)


@dataclass(frozen=True)
class MatrixCase:
    """One concrete baseline row inside a matrix suite."""

    name: str
    width: int
    height: int
    readback_payload: ReadbackPayload
    shadows: bool = True
    write_policy: WritePolicy = WritePolicy.NONE
    fail_on_overflow: bool | None = None


@dataclass(frozen=True)
class MatrixSuite:
    """Named collection of baseline cases for one preset."""

    name: str
    preset: str
    cases: tuple[MatrixCase, ...]


@dataclass(frozen=True)
class MatrixRunOptions:
    """Execution options shared by all cases in a matrix run."""

    out: Path
    device: str | None = None
    model_dir: str = "out/external/mujoco_menagerie/unitree_go2"
    model_xml: str = "go2.xml"
    frames: int = 10
    fps: float = 30.0
    warmup_renders: int = DEFAULT_LAB_WARMUP_RENDERS
    progress_every: int = 5
    video_raygen: str = "gpu"
    video_ray_cache: str = "off"
    render_profile: bool = False
    verbose_warp: bool = False
    keep_going: bool = True


def go2_video_ordered_baseline_suite(*, include_full_debug: bool = False) -> MatrixSuite:
    """Return the first Go2 ordered-video baseline matrix."""
    cases = [
        MatrixCase(
            name="smoke_160x120_shadow_readback_none",
            width=160,
            height=120,
            readback_payload=ReadbackPayload.NONE,
            shadows=True,
        ),
        MatrixCase(
            name="1080p_shadow_readback_none",
            width=DEFAULT_RENDER_WIDTH,
            height=DEFAULT_RENDER_HEIGHT,
            readback_payload=ReadbackPayload.NONE,
            shadows=True,
        ),
        MatrixCase(
            name="1080p_no_shadow_readback_none",
            width=DEFAULT_RENDER_WIDTH,
            height=DEFAULT_RENDER_HEIGHT,
            readback_payload=ReadbackPayload.NONE,
            shadows=False,
        ),
        MatrixCase(
            name="1080p_shadow_readback_rgb",
            width=DEFAULT_RENDER_WIDTH,
            height=DEFAULT_RENDER_HEIGHT,
            readback_payload=ReadbackPayload.RGB,
            shadows=True,
        ),
    ]
    if include_full_debug:
        cases.append(
            MatrixCase(
                name="1080p_shadow_readback_full",
                width=DEFAULT_RENDER_WIDTH,
                height=DEFAULT_RENDER_HEIGHT,
                readback_payload=ReadbackPayload.FULL,
                shadows=True,
            )
        )
    return MatrixSuite(
        name="go2_video_ordered_baseline",
        preset="go2_video_ordered_static",
        cases=tuple(cases),
    )


def go2_video_ordered_legacy_960_suite(*, include_full_debug: bool = False) -> MatrixSuite:
    """Return the 960x640 suite used for historical VIDEO_ORDERED_EXPORT comparisons."""
    cases = [
        MatrixCase(
            name="legacy_960x640_shadow_readback_none",
            width=960,
            height=640,
            readback_payload=ReadbackPayload.NONE,
            shadows=True,
        ),
        MatrixCase(
            name="legacy_960x640_no_shadow_readback_none",
            width=960,
            height=640,
            readback_payload=ReadbackPayload.NONE,
            shadows=False,
        ),
        MatrixCase(
            name="legacy_960x640_shadow_readback_rgb",
            width=960,
            height=640,
            readback_payload=ReadbackPayload.RGB,
            shadows=True,
        ),
    ]
    if include_full_debug:
        cases.append(
            MatrixCase(
                name="legacy_960x640_shadow_readback_full",
                width=960,
                height=640,
                readback_payload=ReadbackPayload.FULL,
                shadows=True,
            )
        )
    return MatrixSuite(
        name="go2_video_ordered_legacy_960",
        preset="go2_video_ordered_static",
        cases=tuple(cases),
    )


def get_suite(name: str, *, include_full_debug: bool = False) -> MatrixSuite:
    if name == "go2_video_ordered_baseline":
        return go2_video_ordered_baseline_suite(include_full_debug=include_full_debug)
    if name == "go2_video_ordered_legacy_960":
        return go2_video_ordered_legacy_960_suite(include_full_debug=include_full_debug)
    expected = ", ".join(AVAILABLE_MATRIX_SUITES)
    raise ValueError(f"Unknown optical pipeline lab matrix suite {name!r}; expected one of: {expected}")


def run_matrix_suite(
    suite: MatrixSuite,
    options: MatrixRunOptions,
    *,
    run_one: Callable[[OpticalLabScenarioConfig, LabRunOptions], None] = run_scenario,
) -> list[dict[str, object]]:
    """Run a matrix suite serially and write suite/summary artifacts."""
    options.out.mkdir(parents=True, exist_ok=True)
    write_suite_config(options.out / "suite_config.json", suite, options)

    rows: list[dict[str, object]] = []
    for case in suite.cases:
        config = config_for_case(suite, case, device=options.device)
        run_options = run_options_for_case(case, options)
        try:
            run_one(config, run_options)
        except Exception as exc:  # Matrix summaries should retain failed rows.
            row = _summary_row_for_case(case, config, run_options, status="failed", error=str(exc))
            rows.append(row)
            write_matrix_summary(options.out / "matrix_summary.csv", rows)
            if not options.keep_going:
                raise
            continue

        rows.append(_summary_row_for_case(case, config, run_options, status="passed", error=""))
        write_matrix_summary(options.out / "matrix_summary.csv", rows)
    return rows


def config_for_case(
    suite: MatrixSuite,
    case: MatrixCase,
    *,
    device: str | None = None,
) -> OpticalLabScenarioConfig:
    return apply_run_overrides(
        get_preset(suite.preset),
        device=device,
        width=case.width,
        height=case.height,
        readback=case.readback_payload.value,
        shadows=case.shadows,
        write_frames=case.write_policy is WritePolicy.PNG_SEQUENCE,
    )


def run_options_for_case(case: MatrixCase, options: MatrixRunOptions) -> LabRunOptions:
    fail_on_overflow = (
        case.fail_on_overflow
        if case.fail_on_overflow is not None
        else case.readback_payload is not ReadbackPayload.NONE
    )
    return LabRunOptions(
        out=options.out / case.name,
        model_dir=options.model_dir,
        model_xml=options.model_xml,
        frames=options.frames,
        fps=options.fps,
        warmup_renders=options.warmup_renders,
        progress_every=options.progress_every,
        video_raygen=options.video_raygen,
        video_ray_cache=options.video_ray_cache,
        render_profile=options.render_profile,
        fail_on_overflow=bool(fail_on_overflow),
        verbose_warp=options.verbose_warp,
    )


def write_suite_config(path: Path, suite: MatrixSuite, options: MatrixRunOptions) -> None:
    payload = {
        "suite": suite_dict(suite),
        "run_options": {field: _serialize_value(value) for field, value in options.__dict__.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_matrix_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATRIX_SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def suite_dict(suite: MatrixSuite) -> dict[str, object]:
    return {
        "name": suite.name,
        "preset": suite.preset,
        "cases": [case_dict(case) for case in suite.cases],
    }


def case_dict(case: MatrixCase) -> dict[str, object]:
    return {field: _serialize_value(value) for field, value in case.__dict__.items()}


def _summary_row_for_case(
    case: MatrixCase,
    config: OpticalLabScenarioConfig,
    options: LabRunOptions,
    *,
    status: str,
    error: str,
) -> dict[str, object]:
    frame_rows = _read_frame_rows(options.out / "frame_timing.csv") if status == "passed" else []
    frame_total = _series(frame_rows, "frame_total_ms")
    render_execute = _series(frame_rows, "render_execute_ms")
    readback_host = _series(frame_rows, "readback_host_ms")
    image_build = _series(frame_rows, "image_build_ms")
    encode_write = _series(frame_rows, "encode_write_ms")
    if not encode_write:
        # Older frame CSVs used this combined column before encode/write was split.
        encode_write = _series(frame_rows, "encode_or_write_ms")
    return {
        "case_name": case.name,
        "status": status,
        "error": error,
        "output_dir": str(options.out),
        "width": int(config.width),
        "height": int(config.height),
        "shadows": bool(config.shadows),
        "readback_payload": config.readback_payload.value,
        "write_policy": config.write_policy.value,
        "frames": len(frame_rows) if frame_rows else int(options.frames),
        "fps_mean": _fps_mean(frame_total),
        "frame_p50_ms": percentile(frame_total, 0.50),
        "frame_p90_ms": percentile(frame_total, 0.90),
        "render_execute_mean_ms": _mean(render_execute),
        "readback_host_mean_ms": _mean(readback_host),
        "image_build_mean_ms": _mean(image_build),
        "encode_write_mean_ms": _mean(encode_write),
    }


def _read_frame_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _series(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        text = row.get(key, "")
        if text == "":
            continue
        value = float(text)
        if not math.isnan(value):
            values.append(value)
    return values


def _mean(values: list[float]) -> float:
    if not values:
        return NAN
    return sum(values) / float(len(values))


def _fps_mean(frame_total_ms: list[float]) -> float:
    total = sum(frame_total_ms)
    if not frame_total_ms or total <= 0.0:
        return NAN
    return 1000.0 * float(len(frame_total_ms)) / total


def _serialize_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value
