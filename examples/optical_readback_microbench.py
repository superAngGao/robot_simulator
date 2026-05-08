"""Microbenchmark optical GPU readback and host materialization costs.

This is an analysis tool for the GPU optical preview path. It renders one
MuJoCo Menagerie frame, then times device-to-host readback, dtype conversion,
RGB preview materialization, and simple output encoding separately.

Example:

    conda run -n env_tilelang_20260119 python examples/optical_readback_microbench.py \
      --width 960 --height 640 \
      --out out/optical_readback_microbench/readback.csv
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.mujoco_menagerie_robot_preview import import_mjcf_visual_scene, make_model_camera
from examples.optical_direct_light_preview import linear_rgb_to_preview_uint8
from optics import (
    DeviceOpticalSceneCache,
    GpuDeviceBvhDirectLightOpticalExecutor,
    OpticalOutputProfile,
    build_cuda_lbvh_from_snapshot,
    build_device_bvh_from_snapshot,
    stage_optical_channels,
    stage_optical_compute_result_to_host,
)
from physics.publish import GpuPublishedFrame
from sensing import build_pinhole_camera_rays

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

_DIAGNOSTIC_CHANNELS = (
    "bvh_stack_overflow_count",
    "bvh_max_stack_depth",
    "shadow_stack_overflow_count",
    "shadow_max_stack_depth",
)
_GEOMETRY_HEAVY_CHANNELS = (
    "range_m",
    "position_world",
    "normal_world",
    "numeric_instance_id",
    "hit_mask",
)
_CHANNEL_TARGET_DTYPES = {
    "hit_mask": np.bool_,
    "range_m": np.float64,
    "position_world": np.float64,
    "normal_world": np.float64,
    "numeric_instance_id": np.int64,
    "material_index": np.int64,
    "rgb": np.float64,
    "intensity": np.float64,
    "bvh_stack_overflow_count": np.int32,
    "bvh_max_stack_depth": np.int32,
    "shadow_stack_overflow_count": np.int32,
    "shadow_max_stack_depth": np.int32,
}
_SYNC_STRATEGY = "wp_synchronize_before_timing"


def main() -> None:
    args = _parse_args()
    if wp is None:
        raise SystemExit(
            "optical_readback_microbench.py requires warp with CUDA support"
        ) from _WARP_IMPORT_ERROR
    if Image is None:
        raise SystemExit("optical_readback_microbench.py requires Pillow") from _PIL_IMPORT_ERROR

    if not args.verbose_warp:
        wp.config.quiet = True
    wp.init()
    device = wp.get_device(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = _render_probe_result(args, device=device)
    rows: list[dict[str, object]] = []
    rows.extend(_bench_selected_channels(result, args))
    rows.extend(_bench_group_readback(result, args))
    rgb_float32 = result.channel("rgb").numpy().astype(np.float32, copy=True)
    rows.extend(_bench_rgb_materialization(rgb_float32, args))
    rows.extend(_bench_encoding(rgb_float32, out_path.parent, args))
    _write_rows(out_path, rows)
    _print_summary(out_path, rows)


def _render_probe_result(args: argparse.Namespace, *, device):
    scene = import_mjcf_visual_scene(Path(args.model_dir), model_xml=args.model_xml)
    stream = wp.Stream(device=device)
    gpu_frame = _static_gpu_frame(frame_id=scene.frame.frame_id, sim_time=scene.frame.sim_time, device=device)
    cache = DeviceOpticalSceneCache(scene.registry, device=device, stream=stream)
    snapshot = cache.snapshot_from_gpu_frame(gpu_frame, env_idx=0, stream=stream, include_aabb=True)
    if args.bvh_backend == "cuda_lbvh":
        bvh = build_cuda_lbvh_from_snapshot(snapshot, device=device, stream=stream)
    else:
        bvh = build_device_bvh_from_snapshot(snapshot, device=device, stream=stream)
    camera = make_model_camera(
        bounds_min=scene.bounds_min,
        bounds_max=scene.bounds_max,
        width=args.width,
        height=args.height,
        frame_id=scene.frame.frame_id,
        sim_time=scene.frame.sim_time,
        view=args.view,
    )
    rays = build_pinhole_camera_rays(camera)
    executor = GpuDeviceBvhDirectLightOpticalExecutor(
        device=device,
        stream=stream,
        shadows=not args.no_shadows,
        ambient_rgb=(0.08, 0.085, 0.09),
        background_rgb=(0.025, 0.03, 0.038),
    )
    for _ in range(max(args.render_warmup, 0)):
        warmup = executor.execute(snapshot, bvh, rays, output_profile=OpticalOutputProfile.DIRECT_LIGHT_FULL)
        wp.synchronize_event(warmup.ready_event)
    result = executor.execute(snapshot, bvh, rays, output_profile=OpticalOutputProfile.DIRECT_LIGHT_FULL)
    wp.synchronize_event(result.ready_event)
    return result


def _bench_selected_channels(result, args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for channel_name in (
        "rgb",
        "range_m",
        "position_world",
        "normal_world",
        "hit_mask",
        "numeric_instance_id",
        "intensity",
        *_DIAGNOSTIC_CHANNELS,
    ):
        value = result.channel(channel_name)
        raw = _read_channel_raw(value)
        rows.append(
            _summary_row(
                bench_group="readback_channel",
                bench_name=f"{channel_name}_raw_numpy",
                channel_names=channel_name,
                samples=_time_samples(args, lambda value=value: _read_channel_raw(value)),
                shape=raw.shape,
                device_dtype=raw.dtype,
                host_dtype=raw.dtype,
                bytes_device=raw.nbytes,
                bytes_host=raw.nbytes,
                args=args,
                notes="value.numpy() after explicit wp.synchronize()",
            )
        )
        target_dtype = np.dtype(_CHANNEL_TARGET_DTYPES.get(channel_name, raw.dtype))
        rows.append(
            _summary_row(
                bench_group="materialization",
                bench_name=f"{channel_name}_copy_{target_dtype.name}",
                channel_names=channel_name,
                samples=_time_samples(
                    args,
                    lambda raw=raw, target_dtype=target_dtype: _copy_as(raw, target_dtype),
                ),
                shape=raw.shape,
                device_dtype=raw.dtype,
                host_dtype=target_dtype,
                bytes_device=raw.nbytes,
                bytes_host=int(raw.size * target_dtype.itemsize),
                args=args,
                notes="host np.asarray(..., dtype=target).copy()",
            )
        )
    return rows


def _bench_group_readback(result, args: argparse.Namespace) -> list[dict[str, object]]:
    groups: tuple[tuple[str, tuple[str, ...], Callable[[], object]], ...] = (
        (
            "rgb_float32_only",
            ("rgb",),
            lambda: result.channel("rgb").numpy().astype(np.float32, copy=True),
        ),
        (
            "rgb_current",
            ("rgb", *_DIAGNOSTIC_CHANNELS),
            lambda: stage_optical_channels(result, ("rgb", *_DIAGNOSTIC_CHANNELS)),
        ),
        (
            "diagnostics_only",
            _DIAGNOSTIC_CHANNELS,
            lambda: stage_optical_channels(result, _DIAGNOSTIC_CHANNELS),
        ),
        (
            "geometry_heavy",
            _GEOMETRY_HEAVY_CHANNELS,
            lambda: stage_optical_channels(result, _GEOMETRY_HEAVY_CHANNELS),
        ),
        (
            "full_current",
            tuple(result.channels),
            lambda: stage_optical_compute_result_to_host(result),
        ),
    )
    rows: list[dict[str, object]] = []
    for bench_name, channel_names, fn in groups:
        bytes_device, bytes_host = _group_bytes(result, channel_names)
        if bench_name == "rgb_float32_only":
            bytes_host = bytes_device
        rows.append(
            _summary_row(
                bench_group="readback_group",
                bench_name=bench_name,
                channel_names=",".join(channel_names),
                samples=_time_samples(args, fn),
                shape="mixed",
                device_dtype="mixed",
                host_dtype="mixed",
                bytes_device=bytes_device,
                bytes_host=bytes_host,
                args=args,
                notes="group readback/materialization",
            )
        )
    return rows


def _bench_rgb_materialization(rgb_float32: np.ndarray, args: argparse.Namespace) -> list[dict[str, object]]:
    rgb_float64 = rgb_float32.astype(np.float64, copy=True)
    rows: list[dict[str, object]] = []
    for label, rgb in (("float32", rgb_float32), ("float64", rgb_float64)):
        clipped = np.clip(rgb, 0.0, 1.0)
        gamma = clipped ** (1.0 / 2.2)
        uint8_rgb = np.rint(gamma * 255.0).astype(np.uint8)
        steps: tuple[tuple[str, Callable[[], object], np.ndarray, object], ...] = (
            ("clip", lambda rgb=rgb: np.clip(rgb, 0.0, 1.0), rgb, rgb.dtype),
            ("gamma", lambda clipped=clipped: clipped ** (1.0 / 2.2), clipped, clipped.dtype),
            (
                "scale_cast_uint8",
                lambda gamma=gamma: np.rint(gamma * 255.0).astype(np.uint8),
                gamma,
                np.uint8,
            ),
            ("pil_fromarray", lambda uint8_rgb=uint8_rgb: Image.fromarray(uint8_rgb), uint8_rgb, np.uint8),
            ("linear_rgb_to_preview_uint8", lambda rgb=rgb: linear_rgb_to_preview_uint8(rgb), rgb, np.uint8),
        )
        for step_name, fn, source, host_dtype in steps:
            host_dtype = np.dtype(host_dtype)
            rows.append(
                _summary_row(
                    bench_group="materialization",
                    bench_name=f"rgb_{label}_{step_name}",
                    channel_names="rgb",
                    samples=_time_samples(args, fn),
                    shape=source.shape,
                    device_dtype=source.dtype,
                    host_dtype=host_dtype,
                    bytes_device=source.nbytes,
                    bytes_host=int(source.size * host_dtype.itemsize),
                    args=args,
                    notes="host RGB materialization substep",
                )
            )
    return rows


def _bench_encoding(
    rgb_float32: np.ndarray,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rgb_uint8 = linear_rgb_to_preview_uint8(rgb_float32)
    png_path = out_dir / "microbench_rgb.png"
    raw_path = out_dir / "microbench_rgb.bin"
    npy_path = out_dir / "microbench_rgb.npy"
    return [
        _summary_row(
            bench_group="encoding",
            bench_name="png_default",
            channel_names="rgb",
            samples=_time_samples(args, lambda: Image.fromarray(rgb_uint8).save(png_path)),
            shape=rgb_uint8.shape,
            device_dtype=rgb_uint8.dtype,
            host_dtype=rgb_uint8.dtype,
            bytes_device=rgb_uint8.nbytes,
            bytes_host=rgb_uint8.nbytes,
            args=args,
            notes=str(png_path),
        ),
        _summary_row(
            bench_group="encoding",
            bench_name="raw_write",
            channel_names="rgb",
            samples=_time_samples(args, lambda: raw_path.write_bytes(rgb_uint8.tobytes())),
            shape=rgb_uint8.shape,
            device_dtype=rgb_uint8.dtype,
            host_dtype=rgb_uint8.dtype,
            bytes_device=rgb_uint8.nbytes,
            bytes_host=rgb_uint8.nbytes,
            args=args,
            notes=str(raw_path),
        ),
        _summary_row(
            bench_group="encoding",
            bench_name="npy_write",
            channel_names="rgb",
            samples=_time_samples(args, lambda: np.save(npy_path, rgb_uint8)),
            shape=rgb_uint8.shape,
            device_dtype=rgb_uint8.dtype,
            host_dtype=rgb_uint8.dtype,
            bytes_device=rgb_uint8.nbytes,
            bytes_host=rgb_uint8.nbytes,
            args=args,
            notes=str(npy_path),
        ),
    ]


def _time_samples(args: argparse.Namespace, fn: Callable[[], object]) -> list[float]:
    for _ in range(max(args.warmup, 0)):
        fn()
    samples: list[float] = []
    for _ in range(max(args.repeat, 0)):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _read_channel_raw(value) -> np.ndarray:
    wp.synchronize()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _copy_as(raw: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    return np.asarray(raw, dtype=target_dtype).copy()


def _group_bytes(result, channel_names: tuple[str, ...]) -> tuple[int, int]:
    bytes_device = 0
    bytes_host = 0
    for name in channel_names:
        raw = _read_channel_raw(result.channel(name))
        target_dtype = np.dtype(_CHANNEL_TARGET_DTYPES.get(name, raw.dtype))
        bytes_device += int(raw.nbytes)
        bytes_host += int(raw.size * target_dtype.itemsize)
    return bytes_device, bytes_host


def _summary_row(
    *,
    bench_group: str,
    bench_name: str,
    channel_names: str,
    samples: list[float],
    shape,
    device_dtype,
    host_dtype,
    bytes_device: int,
    bytes_host: int,
    args: argparse.Namespace,
    notes: str,
) -> dict[str, object]:
    sorted_samples = sorted(samples)
    return {
        "bench_group": bench_group,
        "bench_name": bench_name,
        "resolution": f"{args.width}x{args.height}",
        "num_rays": int(args.width * args.height),
        "channel_names": channel_names,
        "shape": _shape_text(shape),
        "device_dtype": str(device_dtype),
        "host_dtype": str(host_dtype),
        "bytes_device": int(bytes_device),
        "bytes_host": int(bytes_host),
        "transfer_ratio": float(bytes_host) / float(bytes_device) if bytes_device else float("nan"),
        "sync_strategy": _SYNC_STRATEGY,
        "warmup_rounds": int(args.warmup),
        "repeat": int(args.repeat),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(sorted_samples),
        "p90_ms": _percentile(sorted_samples, 0.90),
        "min_ms": sorted_samples[0],
        "max_ms": sorted_samples[-1],
        "notes": notes,
    }


def _shape_text(shape) -> str:
    if isinstance(shape, str):
        return shape
    return "x".join(str(int(dim)) for dim in tuple(shape))


def _percentile(sorted_samples: list[float], q: float) -> float:
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    pos = q * (len(sorted_samples) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_samples) - 1)
    weight = pos - lo
    return sorted_samples[lo] * (1.0 - weight) + sorted_samples[hi] * weight


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "bench_group",
        "bench_name",
        "resolution",
        "num_rays",
        "channel_names",
        "shape",
        "device_dtype",
        "host_dtype",
        "bytes_device",
        "bytes_host",
        "transfer_ratio",
        "sync_strategy",
        "warmup_rounds",
        "repeat",
        "mean_ms",
        "p50_ms",
        "p90_ms",
        "min_ms",
        "max_ms",
        "notes",
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(out_path: Path, rows: list[dict[str, object]]) -> None:
    print(f"Wrote optical readback microbench CSV: {out_path}")
    for row in rows:
        if row["bench_name"] in {
            "rgb_float32_only",
            "rgb_current",
            "geometry_heavy",
            "full_current",
            "rgb_float32_linear_rgb_to_preview_uint8",
            "png_default",
        }:
            print(
                "  "
                f"{row['bench_group']}/{row['bench_name']}: "
                f"p50={float(row['p50_ms']):.3f}ms, "
                f"mean={float(row['mean_ms']):.3f}ms"
            )


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", default="out/external/mujoco_menagerie/unitree_go2")
    parser.add_argument("--model-xml", default="go2.xml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--view", choices=("front", "side", "top"), default="front")
    parser.add_argument("--out", default="out/optical_readback_microbench/readback.csv")
    parser.add_argument("--bvh-backend", choices=("cpu", "cuda_lbvh"), default="cuda_lbvh")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--render-warmup", type=int, default=1)
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--verbose-warp", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be > 0")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0")
    return args


if __name__ == "__main__":
    main()
