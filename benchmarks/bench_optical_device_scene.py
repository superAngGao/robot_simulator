"""Benchmark L5C GPU optical device-scene update and traversal.

This script is intentionally outside pytest. It is a decision aid for L5C.1c
and later acceleration work, not a correctness or CI performance assertion.

Example:

    conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py
    conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py --case smoke
    conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py --case large
    conda run -n env_tilelang_20260119 python benchmarks/bench_optical_device_scene.py \
      --case custom --num-rays 8192 --num-triangles 1048576
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.robot_optical_scene import (
    build_robot_optical_scene,
    make_robot_camera_rays,
    make_robot_gpu_frame,
)
from optics import (
    DeviceOpticalSceneCache,
    GpuDeviceBvhDirectLightOpticalExecutor,
    GpuDeviceBvhOpticalExecutor,
    GpuDeviceSceneOpticalExecutor,
    OpticalInstanceSpec,
    OpticalMaterialSpec,
    OpticalWorldRegistry,
    build_device_bvh_from_snapshot,
    refit_device_bvh_from_snapshot,
    stage_optical_compute_result_to_host,
)
from physics.publish import GpuPublishedFrame
from sensing import OpticalRaySensorSpec

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - script-only guard.
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


@dataclass(frozen=True)
class BenchCase:
    name: str
    num_rays: int
    num_triangles: int
    scene: str = "grid"
    visible_stride: int = 1
    robot_detail: str = "dense"
    num_robots: int = 1
    camera_view: str = "front"
    stage: bool = False


DEFAULT_CASES = (
    BenchCase("few_rays_few_prims", num_rays=128, num_triangles=64),
    BenchCase("camera_rays_few_prims", num_rays=16_384, num_triangles=64),
    BenchCase("camera_rays_mid_prims", num_rays=16_384, num_triangles=512),
    BenchCase("camera_rays_many_prims", num_rays=65_536, num_triangles=2_048),
    BenchCase("role_filtered_many_prims", num_rays=16_384, num_triangles=2_048, visible_stride=8),
)

LARGE_CASES = (
    BenchCase("large_camera_mid_prims", num_rays=262_144, num_triangles=2_048),
    BenchCase("large_camera_many_prims", num_rays=262_144, num_triangles=8_192),
    BenchCase("role_filtered_large_prims", num_rays=262_144, num_triangles=8_192, visible_stride=8),
)

XLARGE_CASES = (
    BenchCase("xlarge_camera_256k_tris", num_rays=65_536, num_triangles=262_144),
    BenchCase("xlarge_mesh_1m_tris", num_rays=8_192, num_triangles=1_048_576),
    BenchCase("role_filtered_xlarge_mesh", num_rays=65_536, num_triangles=1_048_576, visible_stride=32),
)

ROBOT_CASES = (
    BenchCase("robot_proxy_pose", num_rays=16_384, num_triangles=0, scene="robot", robot_detail="proxy"),
    BenchCase("robot_dense_single", num_rays=65_536, num_triangles=0, scene="robot", robot_detail="dense"),
    BenchCase(
        "robot_dense_pack",
        num_rays=65_536,
        num_triangles=0,
        scene="robot",
        robot_detail="dense",
        num_robots=4,
    ),
    BenchCase(
        "robot_ego_camera",
        num_rays=65_536,
        num_triangles=0,
        scene="robot",
        robot_detail="dense",
        camera_view="ego",
    ),
)

SMOKE_CASE = BenchCase("smoke", num_rays=256, num_triangles=64, stage=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument(
        "--verbose-warp",
        action="store_true",
        help="Allow Warp initialization/module-load logs on stdout. By default CSV output is kept clean.",
    )
    parser.add_argument(
        "--case",
        choices=(
            "all",
            "large",
            "xlarge",
            "robot",
            "smoke",
            "custom",
            *(case.name for case in DEFAULT_CASES),
            *(case.name for case in LARGE_CASES),
            *(case.name for case in XLARGE_CASES),
            *(case.name for case in ROBOT_CASES),
        ),
        default="all",
        help="Benchmark case to run.",
    )
    parser.add_argument("--num-rays", type=int, default=None, help="Ray count for --case custom.")
    parser.add_argument("--num-triangles", type=int, default=None, help="Triangle count for --case custom.")
    parser.add_argument(
        "--visible-stride",
        type=int,
        default=1,
        help="For custom cases, only every Nth triangle is visible to the depth role.",
    )
    parser.add_argument("--stage", action="store_true", help="Also time device-result staging to host.")
    parser.add_argument(
        "--use-aabb",
        action="store_true",
        help="Use the L5C.1c per-triangle AABB traversal variant.",
    )
    parser.add_argument(
        "--use-bvh",
        action="store_true",
        help="Use the L5C.2a CPU-build/GPU-traverse BVH variant.",
    )
    parser.add_argument(
        "--refit-bvh",
        action="store_true",
        help="Build BVH topology once, then GPU-refit bounds per snapshot before traversal.",
    )
    parser.add_argument(
        "--direct-light",
        action="store_true",
        help="Run the GPU BVH direct-light executor instead of first-hit only.",
    )
    parser.add_argument(
        "--shadows",
        action="store_true",
        help="Enable shadow rays for --direct-light.",
    )
    parser.add_argument(
        "--compare-direct-light",
        action="store_true",
        help=(
            "In one process, measure first-hit BVH, direct-light without shadows, "
            "and direct-light with shadows on the same snapshots/BVHs."
        ),
    )
    parser.add_argument(
        "--fail-on-overflow",
        action="store_true",
        help="Exit non-zero if any reported primary or shadow BVH stack overflow counter is non-zero.",
    )
    args = parser.parse_args()
    if args.compare_direct_light:
        args.use_bvh = True
    if args.direct_light:
        args.use_bvh = True
    if args.refit_bvh:
        args.use_bvh = True
    if args.shadows and not args.direct_light:
        raise SystemExit("--shadows requires --direct-light")
    if args.compare_direct_light and (args.direct_light or args.shadows):
        raise SystemExit("--compare-direct-light cannot be combined with --direct-light or --shadows")
    if args.use_aabb and args.use_bvh:
        raise SystemExit("--use-aabb and --use-bvh are mutually exclusive traversal modes")
    if wp is None:
        raise SystemExit(
            "bench_optical_device_scene.py requires warp with CUDA support"
        ) from _WARP_IMPORT_ERROR

    if not args.verbose_warp:
        wp.config.quiet = True
    wp.init()
    device = wp.get_device(args.device)

    cases = _select_cases(
        args.case,
        num_rays=args.num_rays,
        num_triangles=args.num_triangles,
        visible_stride=args.visible_stride,
    )
    warmup = int(args.warmup) if args.warmup is not None else _default_warmup(cases)
    repeat = int(args.repeat) if args.repeat is not None else _default_repeat(cases)
    print(
        "case,mode,num_rays,num_triangles,warmup,repeat,"
        "update_ms_mean,update_ms_p50,update_ms_p90,update_ms_std,"
        "cpu_build_ms_mean,cpu_build_ms_p50,cpu_build_ms_p90,cpu_build_ms_std,"
        "gpu_refit_ms_mean,gpu_refit_ms_p50,gpu_refit_ms_p90,gpu_refit_ms_std,"
        "gpu_traverse_ms_mean,gpu_traverse_ms_p50,gpu_traverse_ms_p90,gpu_traverse_ms_std,"
        "stage_ms_mean,stage_ms_p50,stage_ms_p90,stage_ms_std,"
        "bvh_nodes,bvh_max_depth,bvh_sah_quality_cost,"
        "bvh_stack_overflow,bvh_max_stack_observed,"
        "shadow_stack_overflow,shadow_max_stack_observed"
    )
    saw_overflow = False
    for case in cases:
        if args.compare_direct_light:
            rows = run_direct_light_compare_case(
                case,
                device=device,
                warmup=warmup,
                repeat=repeat,
                stage=bool(args.stage or case.stage),
                refit_bvh=bool(args.refit_bvh),
            )
        else:
            stats = run_case(
                case,
                device=device,
                warmup=warmup,
                repeat=repeat,
                stage=bool(args.stage or case.stage),
                use_aabb=bool(args.use_aabb),
                use_bvh=bool(args.use_bvh),
                refit_bvh=bool(args.refit_bvh),
                direct_light=bool(args.direct_light),
                shadows=bool(args.shadows),
            )
            rows = (
                (
                    _mode_name(
                        use_aabb=bool(args.use_aabb),
                        use_bvh=bool(args.use_bvh),
                        refit_bvh=bool(args.refit_bvh),
                        direct_light=bool(args.direct_light),
                        shadows=bool(args.shadows),
                    ),
                    stats,
                ),
            )
        for mode, stats in rows:
            _print_stats(case, mode, stats)
            saw_overflow = saw_overflow or _has_overflow(stats)
    if args.fail_on_overflow and saw_overflow:
        raise SystemExit("benchmark detected BVH stack overflow")


def run_case(
    case: BenchCase,
    *,
    device,
    warmup: int,
    repeat: int,
    stage: bool,
    use_aabb: bool,
    use_bvh: bool,
    refit_bvh: bool,
    direct_light: bool,
    shadows: bool,
) -> dict[str, float]:
    setup = _build_scene_setup(case, device=device)
    registry = setup["registry"]
    cache = DeviceOpticalSceneCache(registry, device=device)
    frame = setup["frame"]
    spec = setup["spec"]
    scene_executor = GpuDeviceSceneOpticalExecutor(device=device, use_aabb=use_aabb)
    if direct_light:
        bvh_executor = GpuDeviceBvhDirectLightOpticalExecutor(device=device, shadows=shadows)
    else:
        bvh_executor = GpuDeviceBvhOpticalExecutor(device=device) if use_bvh else None
    reusable_bvh = None
    if refit_bvh:
        topology_snapshot = cache.snapshot_from_gpu_frame(frame, include_aabb=True)
        wp.synchronize_event(topology_snapshot.ready_event)
        reusable_bvh = build_device_bvh_from_snapshot(topology_snapshot, device=device)

    for _ in range(warmup):
        snapshot = cache.snapshot_from_gpu_frame(frame, include_aabb=use_aabb or use_bvh)
        if use_bvh:
            if refit_bvh:
                bvh = refit_device_bvh_from_snapshot(snapshot, reusable_bvh)
                reusable_bvh = bvh
            else:
                bvh = build_device_bvh_from_snapshot(snapshot, device=device)
            result = bvh_executor.execute(snapshot, bvh, spec)
        else:
            result = scene_executor.execute(snapshot, spec)
        wp.synchronize_event(result.ready_event)

    update_ms: list[float] = []
    cpu_build_ms: list[float] = []
    gpu_refit_ms: list[float] = []
    gpu_traverse_ms: list[float] = []
    stage_ms: list[float] = []
    last_bvh_nodes = 0
    last_bvh_max_depth = 0
    last_bvh_sah_quality_cost = 0.0
    last_bvh_stack_overflow = 0
    last_bvh_max_stack_observed = 0
    last_shadow_stack_overflow = 0
    last_shadow_max_stack_observed = 0

    for _ in range(repeat):
        t0 = time.perf_counter()
        snapshot = cache.snapshot_from_gpu_frame(frame, include_aabb=use_aabb or use_bvh)
        wp.synchronize_event(snapshot.ready_event)
        t1 = time.perf_counter()

        if use_bvh:
            if refit_bvh:
                bvh = refit_device_bvh_from_snapshot(snapshot, reusable_bvh)
                reusable_bvh = bvh
                wp.synchronize_event(bvh.ready_event)
                t_refit = time.perf_counter()
                t_build = t1
            else:
                bvh = build_device_bvh_from_snapshot(snapshot, device=device)
                t_build = time.perf_counter()
                t_refit = t_build
            last_bvh_nodes = bvh.stats.num_nodes
            last_bvh_max_depth = bvh.stats.max_depth
            last_bvh_sah_quality_cost = bvh.stats.sah_quality_cost
            result = bvh_executor.execute(snapshot, bvh, spec)
        else:
            t_build = t1
            t_refit = t1
            result = scene_executor.execute(snapshot, spec)
        wp.synchronize_event(result.ready_event)
        t2 = time.perf_counter()
        diagnostics = _read_result_diagnostics(result)
        last_bvh_stack_overflow = diagnostics["bvh_stack_overflow"]
        last_bvh_max_stack_observed = diagnostics["bvh_max_stack"]
        last_shadow_stack_overflow = diagnostics["shadow_stack_overflow"]
        last_shadow_max_stack_observed = diagnostics["shadow_max_stack"]

        if stage:
            stage_optical_compute_result_to_host(result)
            t3 = time.perf_counter()
            stage_ms.append((t3 - t2) * 1000.0)

        update_ms.append((t1 - t0) * 1000.0)
        cpu_build_ms.append((t_build - t1) * 1000.0)
        gpu_refit_ms.append((t_refit - t_build) * 1000.0)
        gpu_traverse_ms.append((t2 - t_refit) * 1000.0)

    return {
        "warmup": float(warmup),
        "repeat": float(repeat),
        "num_rays": float(spec.num_rays),
        "num_triangles": float(setup["num_triangles"]),
        "bvh_nodes": float(last_bvh_nodes),
        "bvh_max_depth": float(last_bvh_max_depth),
        "bvh_sah_quality_cost": float(last_bvh_sah_quality_cost),
        "bvh_stack_overflow": float(last_bvh_stack_overflow),
        "bvh_max_stack_observed": float(last_bvh_max_stack_observed),
        "shadow_stack_overflow": float(last_shadow_stack_overflow),
        "shadow_max_stack_observed": float(last_shadow_max_stack_observed),
    } | _timing_stats(
        update_ms=update_ms,
        cpu_build_ms=cpu_build_ms,
        gpu_refit_ms=gpu_refit_ms,
        gpu_traverse_ms=gpu_traverse_ms,
        stage_ms=stage_ms,
    )


def run_direct_light_compare_case(
    case: BenchCase,
    *,
    device,
    warmup: int,
    repeat: int,
    stage: bool,
    refit_bvh: bool,
) -> tuple[tuple[str, dict[str, float]], ...]:
    """Measure first-hit/direct/shadow modes in one process on matching BVHs."""

    setup = _build_scene_setup(case, device=device)
    registry = setup["registry"]
    cache = DeviceOpticalSceneCache(registry, device=device)
    frame = setup["frame"]
    spec = setup["spec"]
    executors = {
        _mode_name(
            use_aabb=False,
            use_bvh=True,
            refit_bvh=refit_bvh,
            direct_light=False,
            shadows=False,
        ): GpuDeviceBvhOpticalExecutor(device=device),
        _mode_name(
            use_aabb=False,
            use_bvh=True,
            refit_bvh=refit_bvh,
            direct_light=True,
            shadows=False,
        ): GpuDeviceBvhDirectLightOpticalExecutor(device=device, shadows=False),
        _mode_name(
            use_aabb=False,
            use_bvh=True,
            refit_bvh=refit_bvh,
            direct_light=True,
            shadows=True,
        ): GpuDeviceBvhDirectLightOpticalExecutor(device=device, shadows=True),
    }
    reusable_bvh = None
    if refit_bvh:
        topology_snapshot = cache.snapshot_from_gpu_frame(frame, include_aabb=True)
        wp.synchronize_event(topology_snapshot.ready_event)
        reusable_bvh = build_device_bvh_from_snapshot(topology_snapshot, device=device)

    for _ in range(warmup):
        snapshot = cache.snapshot_from_gpu_frame(frame, include_aabb=True)
        if refit_bvh:
            bvh = refit_device_bvh_from_snapshot(snapshot, reusable_bvh)
            reusable_bvh = bvh
        else:
            bvh = build_device_bvh_from_snapshot(snapshot, device=device)
        for executor in executors.values():
            result = executor.execute(snapshot, bvh, spec)
            wp.synchronize_event(result.ready_event)

    update_ms: list[float] = []
    cpu_build_ms: list[float] = []
    gpu_refit_ms: list[float] = []
    per_mode_ms: dict[str, list[float]] = {mode: [] for mode in executors}
    per_mode_stage_ms: dict[str, list[float]] = {mode: [] for mode in executors}
    per_mode_diagnostics: dict[str, dict[str, int]] = {
        mode: {
            "bvh_stack_overflow": 0,
            "bvh_max_stack": 0,
            "shadow_stack_overflow": 0,
            "shadow_max_stack": 0,
        }
        for mode in executors
    }
    last_bvh_nodes = 0
    last_bvh_max_depth = 0
    last_bvh_sah_quality_cost = 0.0

    for _ in range(repeat):
        t0 = time.perf_counter()
        snapshot = cache.snapshot_from_gpu_frame(frame, include_aabb=True)
        wp.synchronize_event(snapshot.ready_event)
        t1 = time.perf_counter()

        if refit_bvh:
            bvh = refit_device_bvh_from_snapshot(snapshot, reusable_bvh)
            reusable_bvh = bvh
            wp.synchronize_event(bvh.ready_event)
            t_refit = time.perf_counter()
            t_build = t1
        else:
            bvh = build_device_bvh_from_snapshot(snapshot, device=device)
            t_build = time.perf_counter()
            t_refit = t_build

        last_bvh_nodes = bvh.stats.num_nodes
        last_bvh_max_depth = bvh.stats.max_depth
        last_bvh_sah_quality_cost = bvh.stats.sah_quality_cost
        update_ms.append((t1 - t0) * 1000.0)
        cpu_build_ms.append((t_build - t1) * 1000.0)
        gpu_refit_ms.append((t_refit - t_build) * 1000.0)

        for mode, executor in executors.items():
            t_start = time.perf_counter()
            result = executor.execute(snapshot, bvh, spec)
            wp.synchronize_event(result.ready_event)
            t_done = time.perf_counter()
            per_mode_ms[mode].append((t_done - t_start) * 1000.0)
            per_mode_diagnostics[mode] = _read_result_diagnostics(result)
            if stage:
                stage_optical_compute_result_to_host(result)
                t_stage = time.perf_counter()
                per_mode_stage_ms[mode].append((t_stage - t_done) * 1000.0)

    rows: list[tuple[str, dict[str, float]]] = []
    for mode in executors:
        diagnostics = per_mode_diagnostics[mode]
        rows.append(
            (
                mode,
                {
                    "warmup": float(warmup),
                    "repeat": float(repeat),
                    "num_rays": float(spec.num_rays),
                    "num_triangles": float(setup["num_triangles"]),
                    "bvh_nodes": float(last_bvh_nodes),
                    "bvh_max_depth": float(last_bvh_max_depth),
                    "bvh_sah_quality_cost": float(last_bvh_sah_quality_cost),
                    "bvh_stack_overflow": float(diagnostics["bvh_stack_overflow"]),
                    "bvh_max_stack_observed": float(diagnostics["bvh_max_stack"]),
                    "shadow_stack_overflow": float(diagnostics["shadow_stack_overflow"]),
                    "shadow_max_stack_observed": float(diagnostics["shadow_max_stack"]),
                }
                | _timing_stats(
                    update_ms=update_ms,
                    cpu_build_ms=cpu_build_ms,
                    gpu_refit_ms=gpu_refit_ms,
                    gpu_traverse_ms=per_mode_ms[mode],
                    stage_ms=per_mode_stage_ms[mode],
                ),
            )
        )
    return tuple(rows)


def _print_stats(case: BenchCase, mode: str, stats: dict[str, float]) -> None:
    print(
        f"{case.name},{mode},{int(stats['num_rays'])},{int(stats['num_triangles'])},"
        f"{int(stats['warmup'])},{int(stats['repeat'])},"
        f"{stats['update_mean_ms']:.4f},{stats['update_p50_ms']:.4f},"
        f"{stats['update_p90_ms']:.4f},{stats['update_std_ms']:.4f},"
        f"{stats['cpu_build_mean_ms']:.4f},{stats['cpu_build_p50_ms']:.4f},"
        f"{stats['cpu_build_p90_ms']:.4f},{stats['cpu_build_std_ms']:.4f},"
        f"{stats['gpu_refit_mean_ms']:.4f},{stats['gpu_refit_p50_ms']:.4f},"
        f"{stats['gpu_refit_p90_ms']:.4f},{stats['gpu_refit_std_ms']:.4f},"
        f"{stats['gpu_traverse_mean_ms']:.4f},{stats['gpu_traverse_p50_ms']:.4f},"
        f"{stats['gpu_traverse_p90_ms']:.4f},{stats['gpu_traverse_std_ms']:.4f},"
        f"{stats['stage_mean_ms']:.4f},{stats['stage_p50_ms']:.4f},"
        f"{stats['stage_p90_ms']:.4f},{stats['stage_std_ms']:.4f},"
        f"{int(stats['bvh_nodes'])},{int(stats['bvh_max_depth'])},"
        f"{stats['bvh_sah_quality_cost']:.4f},"
        f"{int(stats['bvh_stack_overflow'])},{int(stats['bvh_max_stack_observed'])},"
        f"{int(stats['shadow_stack_overflow'])},{int(stats['shadow_max_stack_observed'])}"
    )


def _timing_stats(
    *,
    update_ms: list[float],
    cpu_build_ms: list[float],
    gpu_refit_ms: list[float],
    gpu_traverse_ms: list[float],
    stage_ms: list[float],
) -> dict[str, float]:
    return (
        _series_stats("update", update_ms)
        | _series_stats("cpu_build", cpu_build_ms)
        | _series_stats("gpu_refit", gpu_refit_ms)
        | _series_stats("gpu_traverse", gpu_traverse_ms)
        | _series_stats("stage", stage_ms)
    )


def _series_stats(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean_ms": 0.0,
            f"{prefix}_p50_ms": 0.0,
            f"{prefix}_p90_ms": 0.0,
            f"{prefix}_std_ms": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean_ms": float(np.mean(array)),
        f"{prefix}_p50_ms": float(np.percentile(array, 50.0)),
        f"{prefix}_p90_ms": float(np.percentile(array, 90.0)),
        f"{prefix}_std_ms": float(np.std(array)),
    }


def _has_overflow(stats: dict[str, float]) -> bool:
    return bool(stats["bvh_stack_overflow"] > 0.0 or stats["shadow_stack_overflow"] > 0.0)


def _read_result_diagnostics(result) -> dict[str, int]:
    return {
        "bvh_stack_overflow": _read_scalar_channel(result, "bvh_stack_overflow_count"),
        "bvh_max_stack": _read_scalar_channel(result, "bvh_max_stack_depth"),
        "shadow_stack_overflow": _read_scalar_channel(result, "shadow_stack_overflow_count"),
        "shadow_max_stack": _read_scalar_channel(result, "shadow_max_stack_depth"),
    }


def _read_scalar_channel(result, name: str) -> int:
    channel = result.channels.get(name)
    if channel is None:
        return 0
    if hasattr(channel, "numpy"):
        return int(channel.numpy()[0])
    return int(np.asarray(channel).reshape(-1)[0])


def _mode_name(
    *,
    use_aabb: bool,
    use_bvh: bool,
    refit_bvh: bool,
    direct_light: bool,
    shadows: bool,
) -> str:
    if direct_light:
        base = "bvh_refit" if refit_bvh else "bvh"
        suffix = "shadow" if shadows else "direct"
        return f"{base}_{suffix}"
    if refit_bvh:
        return "bvh_refit"
    if use_bvh:
        return "bvh"
    if use_aabb:
        return "aabb"
    return "linear"


def _select_cases(
    name: str,
    *,
    num_rays: int | None,
    num_triangles: int | None,
    visible_stride: int,
) -> tuple[BenchCase, ...]:
    if name == "all":
        return DEFAULT_CASES
    if name == "large":
        return LARGE_CASES
    if name == "xlarge":
        return XLARGE_CASES
    if name == "robot":
        return ROBOT_CASES
    if name == "smoke":
        return (SMOKE_CASE,)
    if name == "custom":
        if num_rays is None or num_triangles is None:
            raise SystemExit("--case custom requires --num-rays and --num-triangles")
        return (
            BenchCase(
                "custom",
                num_rays=int(num_rays),
                num_triangles=int(num_triangles),
                visible_stride=int(visible_stride),
            ),
        )
    return tuple(
        case for case in (*DEFAULT_CASES, *LARGE_CASES, *XLARGE_CASES, *ROBOT_CASES) if case.name == name
    )


def _build_scene_setup(case: BenchCase, *, device) -> dict[str, object]:
    if case.scene == "robot":
        robot_scene = build_robot_optical_scene(detail=case.robot_detail, num_robots=case.num_robots)
        frame = make_robot_gpu_frame(robot_scene, device=device)
        spec = make_robot_camera_rays(
            num_rays=case.num_rays,
            frame_id=frame.frame_id,
            sim_time=frame.sim_time,
            num_robots=case.num_robots,
            view=case.camera_view,
        )
        return {
            "registry": robot_scene.registry,
            "frame": frame,
            "spec": spec,
            "num_triangles": robot_scene.num_triangles,
        }
    if case.scene != "grid":
        raise ValueError(f"Unknown benchmark scene: {case.scene}")
    return {
        "registry": _build_triangle_registry(case.num_triangles, visible_stride=case.visible_stride),
        "frame": _static_gpu_frame(device=device),
        "spec": _build_downward_ray_spec(case.num_rays),
        "num_triangles": case.num_triangles,
    }


def _default_warmup(cases: tuple[BenchCase, ...]) -> int:
    max_work = max(
        case.num_rays * max(case.num_triangles, _estimated_robot_triangles(case)) for case in cases
    )
    if max_work >= 8_000_000_000:
        return 1
    return 3


def _default_repeat(cases: tuple[BenchCase, ...]) -> int:
    max_work = max(
        case.num_rays * max(case.num_triangles, _estimated_robot_triangles(case)) for case in cases
    )
    if max_work >= 8_000_000_000:
        return 3
    return 10


def _estimated_robot_triangles(case: BenchCase) -> int:
    if case.scene != "robot":
        return 0
    if case.robot_detail == "proxy":
        return 5_000 * case.num_robots
    if case.robot_detail == "xlarge":
        return 520_000 * case.num_robots
    return 230_000 * case.num_robots


def _build_triangle_registry(num_triangles: int, *, visible_stride: int = 1) -> OpticalWorldRegistry:
    vertices, triangles = _grid_triangles(num_triangles)
    registry = OpticalWorldRegistry()
    registry.add_material(OpticalMaterialSpec("mat_bench"))
    if visible_stride <= 1:
        registry.add_triangle_mesh_geometry("bench_mesh", vertices_local=vertices, triangles=triangles)
        registry.add_instance(
            OpticalInstanceSpec(
                "bench_mesh",
                "bench_mesh",
                "mat_bench",
                roles=frozenset({"depth", "bench"}),
            )
        )
        return registry

    visible = np.arange(num_triangles) % visible_stride == 0
    visible_vertices, visible_triangles = _subset_mesh(vertices, triangles, visible)
    hidden_vertices, hidden_triangles = _subset_mesh(vertices, triangles, ~visible)
    registry.add_triangle_mesh_geometry(
        "visible_mesh",
        vertices_local=visible_vertices,
        triangles=visible_triangles,
    )
    registry.add_triangle_mesh_geometry(
        "hidden_mesh",
        vertices_local=hidden_vertices,
        triangles=hidden_triangles,
    )
    registry.add_instance(
        OpticalInstanceSpec(
            "visible_mesh",
            "visible_mesh",
            "mat_bench",
            roles=frozenset({"depth", "bench"}),
        )
    )
    registry.add_instance(
        OpticalInstanceSpec(
            "hidden_mesh",
            "hidden_mesh",
            "mat_bench",
            roles=frozenset({"hidden"}),
        )
    )
    return registry


def _grid_triangles(num_triangles: int) -> tuple[np.ndarray, np.ndarray]:
    side = int(math.ceil(math.sqrt(num_triangles)))
    spacing = 0.05
    half = side * spacing * 0.5
    indices = np.arange(num_triangles, dtype=np.int64)
    rows = indices // side
    cols = indices % side
    x = cols.astype(np.float64) * spacing - half
    y = rows.astype(np.float64) * spacing - half

    vertices = np.empty((num_triangles * 3, 3), dtype=np.float64)
    vertices[0::3, 0] = x
    vertices[0::3, 1] = y
    vertices[0::3, 2] = 0.0
    vertices[1::3, 0] = x + spacing * 0.8
    vertices[1::3, 1] = y
    vertices[1::3, 2] = 0.0
    vertices[2::3, 0] = x
    vertices[2::3, 1] = y + spacing * 0.8
    vertices[2::3, 2] = 0.0

    triangles = np.arange(num_triangles * 3, dtype=np.int64).reshape(num_triangles, 3)
    return vertices, triangles


def _subset_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = triangles[mask]
    if selected.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)
    if _has_unique_triangle_vertices(vertices, triangles):
        out_vertices = vertices[selected.reshape(-1)].copy()
        out_triangles = np.arange(out_vertices.shape[0], dtype=np.int64).reshape(-1, 3)
        return out_vertices, out_triangles
    remap: dict[int, int] = {}
    out_vertices: list[np.ndarray] = []
    out_triangles: list[list[int]] = []
    for triangle in selected:
        out_triangle: list[int] = []
        for old_index in triangle:
            old_key = int(old_index)
            if old_key not in remap:
                remap[old_key] = len(out_vertices)
                out_vertices.append(vertices[old_key])
            out_triangle.append(remap[old_key])
        out_triangles.append(out_triangle)
    return np.asarray(out_vertices, dtype=np.float64), np.asarray(out_triangles, dtype=np.int64)


def _has_unique_triangle_vertices(vertices: np.ndarray, triangles: np.ndarray) -> bool:
    if vertices.shape[0] != triangles.shape[0] * 3:
        return False
    expected = np.arange(triangles.shape[0] * 3, dtype=triangles.dtype).reshape(-1, 3)
    return bool(np.array_equal(triangles, expected))


def _build_downward_ray_spec(num_rays: int) -> OpticalRaySensorSpec:
    side = int(math.ceil(math.sqrt(num_rays)))
    xs = np.linspace(-1.0, 1.0, side, dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, side, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    origins = np.stack([xx.ravel(), yy.ravel(), np.full(side * side, 2.0)], axis=1)[:num_rays]
    directions = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float64), (num_rays, 1))
    return OpticalRaySensorSpec(
        frame_id=1,
        sim_time=0.0,
        env_idx=0,
        sensor_id="bench_depth",
        origins_world=origins,
        directions_world=directions,
        max_distance=10.0,
        sensor_role="depth",
    )


def _static_gpu_frame(*, device) -> GpuPublishedFrame:
    x_world_R = wp.zeros((1, 0, 3, 3), dtype=wp.float32, device=device)
    x_world_r = wp.zeros((1, 0, 3), dtype=wp.float32, device=device)
    return GpuPublishedFrame(
        slot_id=0,
        frame_id=1,
        sim_time=0.0,
        step_index=1,
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


if __name__ == "__main__":
    main()
