"""Temporary source coverage manifest for pre-a3b product entry points.

This module does not claim runtime observation. The durable runtime wrappers
live in :mod:`benchmarks.optical.instrumentation` and are enforced by the
single-lane runner.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from dataclasses import dataclass
from typing import Any, Literal

from ..eligibility import EntryPointAudit, source_manifest_entry_point_audit


@dataclass(frozen=True)
class _SourceCoverage:
    operation: str
    entry_point: str
    source_symbols: tuple[str, ...]
    known_contaminations: tuple[str, ...] = ()


_SHARED_SOURCE_COVERAGE = (
    _SourceCoverage(
        operation="scene_update",
        entry_point=(
            "tools.optical_pipeline_lab.render_session.OpticalLabRenderWorkspace.prepare_dynamic_frame"
        ),
        source_symbols=(
            "tools.optical_pipeline_lab.render_session.OpticalLabRenderWorkspace.prepare_dynamic_frame",
        ),
        known_contaminations=("allocation", "host_synchronize"),
    ),
    _SourceCoverage(
        operation="delivery",
        entry_point="tools.optical_pipeline_lab.delivery.VideoDeliveryFacade.submit",
        source_symbols=(
            "tools.optical_pipeline_lab.delivery.VideoDeliveryFacade.submit",
            "tools.optical_pipeline_lab.delivery.VideoDeliveryFacade._deliver_sync",
            "tools.optical_pipeline_lab.delivery.readback_video_result",
        ),
        known_contaminations=("allocation", "host_materialization"),
    ),
    _SourceCoverage(
        operation="reclaim",
        entry_point="tools.optical_pipeline_lab.delivery.VideoDeliveryFacade.flush",
        source_symbols=(
            "tools.optical_pipeline_lab.delivery.VideoDeliveryFacade.flush",
            "tools.optical_pipeline_lab.delivery.VideoDeliveryFacade._complete_job",
        ),
    ),
    _SourceCoverage(
        operation="pack",
        entry_point="tools.optical_pipeline_lab.rgb_pack.pack_linear_rgb_to_preview_uint8",
        source_symbols=("tools.optical_pipeline_lab.rgb_pack.pack_linear_rgb_to_preview_uint8",),
        known_contaminations=("allocation",),
    ),
)

_LANE_SOURCE_COVERAGE = {
    "warp": (
        _SourceCoverage(
            operation="acceleration",
            entry_point="optics.device_bvh.refit_device_bvh_from_snapshot",
            source_symbols=("optics.device_bvh.refit_device_bvh_from_snapshot",),
        ),
        _SourceCoverage(
            operation="render",
            entry_point=("optics.warp_execution.GpuDeviceBvhDirectLightOpticalExecutor.execute_camera"),
            source_symbols=(
                "optics.warp_execution.GpuDeviceBvhDirectLightOpticalExecutor.execute_camera",
                "optics.warp_execution.GpuDeviceBvhDirectLightOpticalExecutor._shade_geometry",
            ),
            known_contaminations=("allocation", "host_materialization"),
        ),
    ),
    "cuda": (
        _SourceCoverage(
            operation="acceleration",
            entry_point="optics.cuda_lbvh.build_cuda_lbvh_from_snapshot",
            source_symbols=("optics.cuda_lbvh.build_cuda_lbvh_from_snapshot",),
            known_contaminations=(
                "allocation",
                "global_device_synchronize",
                "host_synchronize",
                "warp_to_torch_bridge",
                "torch_to_warp_bridge",
            ),
        ),
        _SourceCoverage(
            operation="render",
            entry_point=("optics.cuda_direct_light.CudaDeviceBvhDirectLightOpticalExecutor.execute_camera"),
            source_symbols=(
                "optics.cuda_direct_light.CudaDeviceBvhDirectLightOpticalExecutor.execute_camera",
                (
                    "optics.cuda_direct_light.CudaDeviceBvhDirectLightOpticalExecutor."
                    "_shade_geometry_no_shadow"
                ),
                (
                    "optics.cuda_direct_light.CudaDeviceBvhDirectLightOpticalExecutor."
                    "_shade_geometry_with_shadows"
                ),
            ),
            known_contaminations=(
                "allocation",
                "global_device_synchronize",
                "host_synchronize",
                "warp_to_torch_bridge",
            ),
        ),
    ),
}


def current_product_source_entry_point_audits(
    lane: Literal["warp", "cuda"],
) -> tuple[EntryPointAudit, ...]:
    """Return coverage-only audits that remain deliberately uninstrumented."""

    if lane not in _LANE_SOURCE_COVERAGE:
        raise ValueError("lane must be 'warp' or 'cuda'")
    return tuple(
        source_manifest_entry_point_audit(
            operation=coverage.operation,
            entry_point=coverage.entry_point,
            contaminations=coverage.known_contaminations,
            source_manifest_sha256=_source_manifest_sha256(coverage),
        )
        for coverage in (*_SHARED_SOURCE_COVERAGE, *_LANE_SOURCE_COVERAGE[lane])
    )


def verify_current_product_source_manifest(
    lane: Literal["warp", "cuda"],
) -> tuple[EntryPointAudit, ...]:
    """Resolve exact callables and hash their bodies as a non-runtime checklist."""

    return current_product_source_entry_point_audits(lane)


def _source_manifest_sha256(coverage: _SourceCoverage) -> str:
    sources = []
    for symbol in coverage.source_symbols:
        resolved = _resolve_symbol(symbol)
        try:
            source = inspect.getsource(resolved)
        except (OSError, TypeError) as exc:
            raise RuntimeError(f"audited product source is unavailable: {symbol}") from exc
        sources.append(
            {
                "symbol": symbol,
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            }
        )
    payload = {
        "entry_point": coverage.entry_point,
        "operation": coverage.operation,
        "source_symbols": sources,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_symbol(path: str) -> Any:
    parts = path.split(".")
    for index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:index])
        try:
            resolved: Any = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name and not module_name.startswith(f"{exc.name}."):
                raise
            continue
        for name in parts[index:]:
            resolved = getattr(resolved, name)
        return resolved
    raise ImportError(f"cannot resolve source symbol {path!r}")
