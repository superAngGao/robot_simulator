"""Temporary source-bound audit of the pre-a3b product entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..eligibility import EntryPointAudit

_CURRENT_PRODUCT_SOURCE_PROBES = {
    ("tools.optical_pipeline_lab.render_session.OpticalLabRenderWorkspace.prepare_dynamic_frame"): (
        "tools/optical_pipeline_lab/render_session.py",
        ("def prepare_dynamic_frame(", "wp.synchronize_event(snapshot.ready_event)"),
    ),
    "tools.optical_pipeline_lab.delivery.VideoDeliveryFacade.submit": (
        "tools/optical_pipeline_lab/delivery.py",
        ("def submit(", "return self._deliver_sync(", "host_channels = readback_video_result("),
    ),
    "tools.optical_pipeline_lab.delivery.VideoDeliveryFacade.flush": (
        "tools/optical_pipeline_lab/delivery.py",
        ("def flush(", "self._complete_pending("),
    ),
    "tools.optical_pipeline_lab.rgb_pack.pack_linear_rgb_to_preview_uint8": (
        "tools/optical_pipeline_lab/rgb_pack.py",
        ("def pack_linear_rgb_to_preview_uint8(", "rgb8 = wp.empty("),
    ),
    "optics.device_bvh.refit_device_bvh_from_snapshot": (
        "optics/device_bvh.py",
        ("def refit_device_bvh_from_snapshot(", "_refit_bvh_leaf_bounds_kernel"),
    ),
    "optics.warp_execution.GpuDeviceBvhDirectLightOpticalExecutor.execute_camera": (
        "optics/warp_execution.py",
        ("class GpuDeviceBvhDirectLightOpticalExecutor:", "rgb = wp.empty("),
    ),
    "optics.cuda_lbvh.build_cuda_lbvh_from_snapshot": (
        "optics/cuda_lbvh.py",
        (
            "def build_cuda_lbvh_from_snapshot(",
            "wp.to_torch(",
            "torch.cuda.synchronize()",
            "wp.from_torch(",
        ),
    ),
    "optics.cuda_direct_light.CudaDeviceBvhDirectLightOpticalExecutor.execute_camera": (
        "optics/cuda_direct_light.py",
        (
            "class CudaDeviceBvhDirectLightOpticalExecutor:",
            "channel_to_torch(",
            "torch.cuda.synchronize(tensor_device)",
            "torch.empty(",
        ),
    ),
}


def current_product_entry_point_audits(lane: Literal["warp", "cuda"]) -> tuple[EntryPointAudit, ...]:
    """Return the reviewed a3a audit of current, deliberately ineligible calls."""

    shared = (
        EntryPointAudit(
            operation="scene_update",
            entry_point=(
                "tools.optical_pipeline_lab.render_session.OpticalLabRenderWorkspace.prepare_dynamic_frame"
            ),
            instrumented=False,
            contaminations=("allocation", "host_synchronize"),
        ),
        EntryPointAudit(
            operation="delivery",
            entry_point="tools.optical_pipeline_lab.delivery.VideoDeliveryFacade.submit",
            instrumented=False,
            contaminations=("allocation", "host_materialization"),
        ),
        EntryPointAudit(
            operation="reclaim",
            entry_point="tools.optical_pipeline_lab.delivery.VideoDeliveryFacade.flush",
            instrumented=False,
        ),
        EntryPointAudit(
            operation="pack",
            entry_point="tools.optical_pipeline_lab.rgb_pack.pack_linear_rgb_to_preview_uint8",
            instrumented=False,
            contaminations=("allocation",),
        ),
    )
    if lane == "warp":
        return shared + (
            EntryPointAudit(
                operation="acceleration",
                entry_point="optics.device_bvh.refit_device_bvh_from_snapshot",
                instrumented=False,
            ),
            EntryPointAudit(
                operation="render",
                entry_point=("optics.warp_execution.GpuDeviceBvhDirectLightOpticalExecutor.execute_camera"),
                instrumented=False,
                contaminations=("allocation",),
            ),
        )
    if lane == "cuda":
        return shared + (
            EntryPointAudit(
                operation="acceleration",
                entry_point="optics.cuda_lbvh.build_cuda_lbvh_from_snapshot",
                instrumented=False,
                contaminations=(
                    "allocation",
                    "global_device_synchronize",
                    "warp_to_torch_bridge",
                    "torch_to_warp_bridge",
                ),
            ),
            EntryPointAudit(
                operation="render",
                entry_point=(
                    "optics.cuda_direct_light.CudaDeviceBvhDirectLightOpticalExecutor.execute_camera"
                ),
                instrumented=False,
                contaminations=(
                    "allocation",
                    "global_device_synchronize",
                    "warp_to_torch_bridge",
                ),
            ),
        )
    raise ValueError("lane must be 'warp' or 'cuda'")


def verify_current_product_entry_point_audits(
    repo_root: Path,
    lane: Literal["warp", "cuda"],
) -> tuple[EntryPointAudit, ...]:
    """Bind the static a3a audit to signatures in the actual product source."""

    audits = current_product_entry_point_audits(lane)
    root = repo_root.resolve()
    for audit in audits:
        try:
            relative_path, tokens = _CURRENT_PRODUCT_SOURCE_PROBES[audit.entry_point]
        except KeyError as exc:
            raise RuntimeError(f"no source probe exists for {audit.entry_point}") from exc
        source_path = (root / relative_path).resolve()
        if root not in source_path.parents or not source_path.is_file():
            raise RuntimeError(f"audited product source is unavailable: {relative_path}")
        source = source_path.read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in source]
        if missing:
            raise RuntimeError(f"product entry-point audit is stale for {audit.entry_point}: {missing}")
    return audits
