"""Timing and CSV helpers for optical pipeline lab runs."""

from __future__ import annotations

import csv
import math
import statistics
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

NAN = float("nan")

RENDER_PROFILE_PHASES = (
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

FRAME_TIMING_FIELDNAMES = (
    "frame_index",
    "scenario_name",
    "device",
    "width",
    "height",
    "scene_preset",
    "camera_mode",
    "sim_time",
    "video_time",
    "geometry_mode",
    "accel_backend",
    "accel_policy",
    "render_backend",
    "output_profile",
    "readback_payload",
    "delivery_policy",
    "write_policy",
    "raygen_mode",
    "ray_cache_mode",
    "readback_mode",
    "write_mode",
    "camera_rays_ms",
    "snapshot_ms",
    "accel_build_ms",
    "accel_refit_ms",
    "accel_rebuild_ms",
    "render_execute_ms",
    "render_overhead_ms",
    *(f"render_{phase}_ms" for phase in RENDER_PROFILE_PHASES),
    "readback_submit_ms",
    "readback_wait_ms",
    "readback_host_ms",
    "image_build_ms",
    "encode_write_ms",
    "encode_or_write_ms",
    "frame_total_ms",
    "instant_fps",
    "rolling_fps",
    "primary_overflow",
    "shadow_overflow",
    "primary_max_stack",
    "shadow_max_stack",
    "memory_used_mb",
    "readback_lag_frames",
    "readback_ring_depth",
    "readback_ring_block_count",
    "completed_frame_index",
    "overlap_ratio",
    "frame_path",
)

FRAME_TIMING_SUMMARY_PHASES = (
    "camera_rays",
    "snapshot",
    "accel_build",
    "accel_refit",
    "accel_rebuild",
    "render_execute",
    "render_overhead",
    *(f"render_{phase}" for phase in RENDER_PROFILE_PHASES),
    "readback_submit",
    "readback_wait",
    "readback_host",
    "image_build",
    "encode_write",
    "encode_or_write",
    "frame_total",
)


def percentile(samples: Sequence[float], q: float) -> float:
    """Return a linearly interpolated percentile.

    The input may be unsorted; this helper sorts internally so callers do not
    need to rely on a hidden ordering convention.
    """
    if not samples:
        return NAN
    sorted_samples = sorted(float(sample) for sample in samples)
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    pos = q * (len(sorted_samples) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_samples) - 1)
    weight = pos - lo
    return float(sorted_samples[lo]) * (1.0 - weight) + float(sorted_samples[hi]) * weight


class TimingRecorder:
    """Small phase timer for setup and aggregate lab timings."""

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
            rows.append(summary_row(phase, samples))
        return rows

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ("phase", "count", "mean_ms", "p50_ms", "p90_ms", "min_ms", "max_ms")
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.summary_rows())


class FrameTimingRecorder:
    """Per-frame timing table with a stable lab CSV schema."""

    fieldnames = FRAME_TIMING_FIELDNAMES

    def __init__(
        self,
        *,
        csv_path: Path | None,
        default_fields: dict[str, float | int | str] | None = None,
    ) -> None:
        self.csv_path = csv_path
        self._default_fields = dict(default_fields or {})
        self._rows: list[dict[str, float | int | str]] = []

    def add(self, row: dict[str, float | int | str]) -> None:
        normalized = {field: NAN for field in self.fieldnames}
        normalized.update(self._default_fields)
        normalized.update(row)
        self._rows.append(normalized)

    def summary_rows(self) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for phase in FRAME_TIMING_SUMMARY_PHASES:
            key = f"{phase}_ms"
            samples = [
                float(row[key]) for row in self._rows if key in row and not math.isnan(float(row[key]))
            ]
            if samples:
                rows.append(summary_row(phase, samples))
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
            "frame_p90_ms": percentile(sorted_frame_ms, 0.90),
        }

    def write_csv(self) -> None:
        if self.csv_path is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)


def summary_row(phase: str, samples: Sequence[float]) -> dict[str, float | str]:
    sorted_samples = sorted(float(sample) for sample in samples)
    if not sorted_samples:
        raise ValueError("summary_row requires at least one timing sample")
    return {
        "phase": phase,
        "count": float(len(sorted_samples)),
        "mean_ms": statistics.fmean(sorted_samples),
        "p50_ms": statistics.median(sorted_samples),
        "p90_ms": percentile(sorted_samples, 0.90),
        "min_ms": sorted_samples[0],
        "max_ms": sorted_samples[-1],
    }
