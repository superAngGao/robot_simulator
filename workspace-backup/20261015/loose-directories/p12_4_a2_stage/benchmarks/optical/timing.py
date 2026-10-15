"""Timed-region guardrails for the durable optical benchmark core.

The guard is intentionally small: it gives lane implementations one shared
place to reject operations that invalidate a native timing boundary.  A lane
must still instrument the concrete runtime entry points it uses; the a3 lane
tests are responsible for proving that coverage.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

NATIVE_TIMING_CONTRACT_VERSION = "q54.p12.4.native-timing.v1"

FORBIDDEN_TIMED_OPERATIONS = frozenset(
    {
        "allocation",
        "compile",
        "cross_lane_wait",
        "global_device_synchronize",
        "host_materialization",
        "host_upload",
        "torch_to_warp_bridge",
        "warp_to_torch_bridge",
    }
)


@dataclass(frozen=True)
class ActiveTimedBoundary:
    """Identity of the native timing region active in the current context."""

    row_id: str
    lane: str
    boundary: str


_ACTIVE_BOUNDARY: ContextVar[ActiveTimedBoundary | None] = ContextVar(
    "q54_p12_4_active_native_timed_boundary",
    default=None,
)


class TimedBoundaryViolation(RuntimeError):
    """Raised before an operation can contaminate a native timing sample."""


def active_timed_boundary() -> ActiveTimedBoundary | None:
    """Return the current boundary, or ``None`` outside primary submission."""

    return _ACTIVE_BOUNDARY.get()


def assert_operation_allowed(operation: str) -> None:
    """Reject a declared forbidden operation while a timed boundary is active."""

    if operation not in FORBIDDEN_TIMED_OPERATIONS:
        raise ValueError(f"unknown Q54 timed operation {operation!r}")
    active = active_timed_boundary()
    if active is not None:
        raise TimedBoundaryViolation(
            f"{operation} is forbidden inside {active.boundary} timed submission "
            f"for row {active.row_id!r} on lane {active.lane!r}"
        )


def assert_outside_timed_boundary(operation: str) -> None:
    """Require an operation to run outside every native timing region."""

    active = active_timed_boundary()
    if active is not None:
        raise TimedBoundaryViolation(
            f"{operation} must run outside {active.boundary} timed submission "
            f"for row {active.row_id!r} on lane {active.lane!r}"
        )


@contextmanager
def native_timed_submission(
    *,
    row_id: str,
    lane: str,
    boundary: str,
) -> Iterator[ActiveTimedBoundary]:
    """Mark only the enqueue/submission portion of one native sample.

    Event resolution and receipt construction happen after this context exits,
    so a lane never needs a host wait merely to return from timed submission.
    """

    if active_timed_boundary() is not None:
        raise TimedBoundaryViolation("nested native timed submissions are forbidden")
    if lane not in {"warp", "cuda"}:
        raise ValueError("native timed lane must be 'warp' or 'cuda'")
    if not row_id or not boundary:
        raise ValueError("native timed row_id and boundary must be non-empty")
    active = ActiveTimedBoundary(row_id=row_id, lane=lane, boundary=boundary)
    token = _ACTIVE_BOUNDARY.set(active)
    try:
        yield active
    finally:
        _ACTIVE_BOUNDARY.reset(token)
