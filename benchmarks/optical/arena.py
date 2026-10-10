"""Frozen-gate adapter for the production native optical arena contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from optics.native_arena import (
    NativeOpticalArenaSpec,
)

from .native import NativeReadinessEvidence, PreparedNativeInput


def native_optical_arena_spec_from_frozen_row(
    *,
    gate: Mapping[str, Any],
    row_id: str,
    prepared: PreparedNativeInput,
    readiness: NativeReadinessEvidence,
    execution_stream_token: str,
    primitive_count: int,
    node_capacity: int,
    diagnostic_mode: str = "primary_unprofiled",
) -> NativeOpticalArenaSpec:
    """Derive one production arena spec after exact frozen-row validation.

    Physical slot allocation and C3 stage binding are C4b/C4c work.  This
    adapter only freezes the logical policy and provenance that those builders
    must satisfy.
    """

    # Local import avoids making the production arena depend on benchmark
    # schema code while still requiring the caller's gate to be canonical.
    from .schema import validate_frozen_gate

    validate_frozen_gate(gate)
    rows = {str(row["row_id"]): row for row in gate["mandatory_rows"] if isinstance(row, Mapping)}
    row = rows.get(row_id)
    if row is None:
        raise KeyError(f"unknown frozen native optical row {row_id!r}")
    from optical_worker.arena import native_optical_arena_spec_from_row

    return native_optical_arena_spec_from_row(row=row, case_contract=gate["cases"][row["case_id"]],
        prepared=prepared, readiness=readiness, execution_stream_token=execution_stream_token,
        primitive_count=primitive_count, node_capacity=node_capacity, diagnostic_mode=diagnostic_mode)
