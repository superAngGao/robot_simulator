"""Fail-closed audit adapter for the current Warp optical product calls."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..assets import AssetPreflightError, preflight_case_asset
from ..eligibility import (
    LaneEligibilityEvidence,
    evaluate_lane_eligibility,
)
from ..native import PreparedNativeInput
from .audit import verify_current_product_entry_point_audits


class CurrentWarpLanePreflight:
    """Audit current Warp calls; intentionally has no sample submission API."""

    lane_id = "warp"

    def __init__(self, *, repo_root: Path) -> None:
        self._repo_root = repo_root.resolve()

    def preflight_sample(
        self,
        *,
        row: Mapping[str, Any],
        prepared: PreparedNativeInput,
    ) -> LaneEligibilityEvidence:
        if prepared.backend != self.lane_id:
            raise ValueError("Warp preflight requires Warp-owned native input")
        asset_hash: str | None = None
        try:
            asset = preflight_case_asset(str(row["case_id"]), repo_root=self._repo_root)
        except AssetPreflightError:
            asset = None
        if asset is not None:
            asset_hash = asset.manifest_sha256
        return evaluate_lane_eligibility(
            row=row,
            lane="warp",
            entry_point_audits=verify_current_product_entry_point_audits(self._repo_root, "warp"),
            asset_manifest_sha256=asset_hash,
        )
