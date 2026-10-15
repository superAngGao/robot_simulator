"""Timed-region guardrails and concrete operation tracing for P12.4."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

NATIVE_TIMING_CONTRACT_VERSION = "q54.p12.4.native-timing.v4"
OPERATION_TRACE_SCHEMA_VERSION = "q54.p12.4.operation-trace.v3"

RUNTIME_OPERATION_KINDS = frozenset(
    {
        "acceleration",
        "arena_acquire",
        "allocation",
        "consumer_completion",
        "delivery",
        "extension_load",
        "materialization",
        "pack",
        "physics_step",
        "publication",
        "reclaim",
        "render",
        "scene_update",
        "stream_wait",
        "synchronization",
        "terminal_completion",
        "upload",
        "view_binding",
    }
)

FORBIDDEN_TIMED_OPERATIONS = frozenset(
    {
        "allocation",
        "bridge_creation",
        "compile",
        "cross_lane_wait",
        "extension_load",
        "global_device_synchronize",
        "host_materialization",
        "host_synchronize",
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


class TimedBoundaryViolation(RuntimeError):
    """Raised before an operation can contaminate a native timing sample."""


@dataclass(frozen=True)
class RuntimeOperationRecord:
    """One lane-instrumented operation observed during timed submission."""

    sequence: int
    operation: str
    entry_point: str
    count: int = 1
    byte_count: int = 0
    capture_source: str = "lane"
    runtime_hook_id: str | None = None

    @classmethod
    def from_payload(cls, payload: object) -> "RuntimeOperationRecord":
        expected = {
            "sequence",
            "operation",
            "entry_point",
            "count",
            "byte_count",
            "capture_source",
            "runtime_hook_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("operation trace record fields are invalid")
        sequence = payload.get("sequence")
        operation = payload.get("operation")
        entry_point = payload.get("entry_point")
        count = payload.get("count")
        byte_count = payload.get("byte_count")
        capture_source = payload.get("capture_source")
        runtime_hook_id = payload.get("runtime_hook_id")
        for name, value in (
            ("sequence", sequence),
            ("byte_count", byte_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"operation trace {name} must be a non-negative integer")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("operation trace count must be a positive integer")
        if operation not in RUNTIME_OPERATION_KINDS:
            raise ValueError("operation trace kind is invalid")
        if not isinstance(entry_point, str) or not entry_point:
            raise ValueError("operation trace entry_point must be non-empty")
        if capture_source not in {"lane", "runtime_hook"}:
            raise ValueError("operation trace capture_source is invalid")
        if capture_source == "runtime_hook":
            if not isinstance(runtime_hook_id, str) or not runtime_hook_id:
                raise ValueError("runtime-hook trace record requires runtime_hook_id")
        elif runtime_hook_id is not None:
            raise ValueError("lane trace record cannot claim runtime_hook_id")
        return cls(
            sequence=sequence,
            operation=operation,
            entry_point=entry_point,
            count=count,
            byte_count=byte_count,
            capture_source=capture_source,
            runtime_hook_id=runtime_hook_id,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "operation": self.operation,
            "entry_point": self.entry_point,
            "count": self.count,
            "byte_count": self.byte_count,
            "capture_source": self.capture_source,
            "runtime_hook_id": self.runtime_hook_id,
        }


@dataclass(frozen=True)
class RuntimeOperationTraceEvidence:
    """Immutable trace closed after lane submission returns."""

    schema_version: str
    row_id: str
    lane: str
    boundary: str
    records: tuple[RuntimeOperationRecord, ...]

    @classmethod
    def from_payload(cls, payload: object) -> "RuntimeOperationTraceEvidence":
        expected = {"schema_version", "row_id", "lane", "boundary", "records"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("operation trace evidence fields are invalid")
        if payload.get("schema_version") != OPERATION_TRACE_SCHEMA_VERSION:
            raise ValueError("operation trace schema version is invalid")
        records_payload = payload.get("records")
        if not isinstance(records_payload, Sequence) or isinstance(records_payload, (str, bytes)):
            raise ValueError("operation trace records must be an array")
        records = tuple(RuntimeOperationRecord.from_payload(item) for item in records_payload)
        if any(record.sequence != index for index, record in enumerate(records)):
            raise ValueError("operation trace sequence must be contiguous and ordered")
        for name in ("row_id", "lane", "boundary"):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise ValueError(f"operation trace {name} must be non-empty")
        return cls(
            schema_version=OPERATION_TRACE_SCHEMA_VERSION,
            row_id=payload["row_id"],
            lane=payload["lane"],
            boundary=payload["boundary"],
            records=records,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "row_id": self.row_id,
            "lane": self.lane,
            "boundary": self.boundary,
            "records": [record.to_payload() for record in self.records],
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_payload(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def observed_operations(self) -> frozenset[str]:
        return frozenset(record.operation for record in self.records)


@dataclass
class RuntimeOperationTrace:
    """Mutable trace visible to a lane only while its timer is active."""

    row_id: str
    lane: str
    boundary: str
    _records: list[RuntimeOperationRecord] = field(default_factory=list, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def _start(self) -> None:
        if self._active or self._closed:
            raise TimedBoundaryViolation("operation trace cannot be reused")
        self._active = True

    def _finish(self) -> None:
        if not self._active:
            raise TimedBoundaryViolation("operation trace is not active")
        self._active = False
        self._closed = True

    def record(
        self,
        operation: str,
        *,
        entry_point: str,
        count: int = 1,
        byte_count: int = 0,
        contamination: str | None = None,
        capture_source: str = "lane",
        runtime_hook_id: str | None = None,
        _runtime_hook_capability: object | None = None,
    ) -> None:
        if not self._active or self._closed:
            raise TimedBoundaryViolation("runtime operation must be recorded inside its active trace")
        if operation not in RUNTIME_OPERATION_KINDS:
            raise ValueError(f"unknown Q54 runtime operation {operation!r}")
        if not entry_point:
            raise ValueError("runtime operation entry_point must be non-empty")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("runtime operation count must be a positive integer")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            raise ValueError("runtime operation byte_count must be a non-negative integer")
        if contamination is not None:
            assert_operation_allowed(contamination)
        if capture_source not in {"lane", "runtime_hook"}:
            raise ValueError("runtime operation capture_source is invalid")
        if capture_source == "runtime_hook":
            capabilities = _ACTIVE_RUNTIME_HOOK_CAPABILITIES.get()
            if (
                not runtime_hook_id
                or runtime_hook_id not in capabilities
                or capabilities[runtime_hook_id] is not _runtime_hook_capability
            ):
                raise TimedBoundaryViolation("runtime operation hook is not installed")
        elif runtime_hook_id is not None or _runtime_hook_capability is not None:
            raise ValueError("lane runtime operation cannot claim hook provenance")
        self._records.append(
            RuntimeOperationRecord(
                sequence=len(self._records),
                operation=operation,
                entry_point=entry_point,
                count=count,
                byte_count=byte_count,
                capture_source=capture_source,
                runtime_hook_id=runtime_hook_id,
            )
        )

    def evidence(self) -> RuntimeOperationTraceEvidence:
        if not self._closed:
            raise TimedBoundaryViolation("operation trace evidence is unavailable before close")
        return RuntimeOperationTraceEvidence(
            schema_version=OPERATION_TRACE_SCHEMA_VERSION,
            row_id=self.row_id,
            lane=self.lane,
            boundary=self.boundary,
            records=tuple(self._records),
        )


_ACTIVE_BOUNDARY: ContextVar[ActiveTimedBoundary | None] = ContextVar(
    "q54_p12_4_active_native_timed_boundary",
    default=None,
)

_ACTIVE_OPERATION_TRACE: ContextVar[RuntimeOperationTrace | None] = ContextVar(
    "q54_p12_4_active_runtime_operation_trace",
    default=None,
)

_ACTIVE_RUNTIME_HOOK_CAPABILITIES: ContextVar[Mapping[str, object]] = ContextVar(
    "q54_p12_4_active_runtime_hook_capabilities",
    default={},
)


def active_timed_boundary() -> ActiveTimedBoundary | None:
    """Return the current boundary, or ``None`` outside primary submission."""

    return _ACTIVE_BOUNDARY.get()


def active_operation_trace() -> RuntimeOperationTrace | None:
    """Return the lane-visible trace for the current timed submission."""

    return _ACTIVE_OPERATION_TRACE.get()


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


def record_runtime_operation(
    operation: str,
    *,
    entry_point: str,
    count: int = 1,
    byte_count: int = 0,
    contamination: str | None = None,
) -> None:
    """Record one concrete runtime entry and reject declared contamination."""

    trace = active_operation_trace()
    if trace is None:
        raise TimedBoundaryViolation("no runtime operation trace is active")
    trace.record(
        operation,
        entry_point=entry_point,
        count=count,
        byte_count=byte_count,
        contamination=contamination,
    )


def record_runtime_hook_operation(
    operation: str,
    *,
    entry_point: str,
    runtime_hook_id: str,
    count: int = 1,
    byte_count: int = 0,
    _capability: object,
) -> None:
    """Record a high-level operation from an installed runtime wrapper."""

    trace = active_operation_trace()
    if trace is None:
        raise TimedBoundaryViolation("no runtime operation trace is active")
    trace.record(
        operation,
        entry_point=entry_point,
        count=count,
        byte_count=byte_count,
        capture_source="runtime_hook",
        runtime_hook_id=runtime_hook_id,
        _runtime_hook_capability=_capability,
    )


@contextmanager
def activate_runtime_hooks(hook_ids: frozenset[str]) -> Iterator[Mapping[str, object]]:
    """Authorize only wrappers installed by the current instrumentation scope."""

    if _ACTIVE_RUNTIME_HOOK_CAPABILITIES.get():
        raise TimedBoundaryViolation("nested runtime hook scopes are forbidden")
    if not hook_ids or any(not isinstance(hook_id, str) or not hook_id for hook_id in hook_ids):
        raise ValueError("runtime hook scope requires non-empty hook IDs")
    capabilities = {hook_id: object() for hook_id in hook_ids}
    token = _ACTIVE_RUNTIME_HOOK_CAPABILITIES.set(capabilities)
    try:
        yield capabilities
    finally:
        _ACTIVE_RUNTIME_HOOK_CAPABILITIES.reset(token)


def validate_required_operation_order(
    evidence: RuntimeOperationTraceEvidence,
    required_operations: Sequence[str],
) -> None:
    """Require the frozen high-level transaction to appear in exact order."""

    required = tuple(required_operations)
    if not required:
        raise ValueError("required operation order must be non-empty")
    if evidence.records and evidence.records[0].operation != required[0]:
        raise ValueError(f"operation trace does not start at the frozen timed boundary {required[0]!r}")
    cursor = 0
    for record in evidence.records:
        if cursor < len(required) and record.operation == required[cursor]:
            cursor += 1
    if cursor != len(required):
        raise ValueError(
            "operation trace does not contain the frozen ordered transaction; "
            f"matched {cursor} of {len(required)} operations"
        )


@contextmanager
def native_timed_submission(
    *,
    row_id: str,
    lane: str,
    boundary: str,
    operation_trace: RuntimeOperationTrace | None = None,
) -> Iterator[ActiveTimedBoundary]:
    """Mark only the enqueue/submission portion of one native sample."""

    if active_timed_boundary() is not None:
        raise TimedBoundaryViolation("nested native timed submissions are forbidden")
    if lane not in {"warp", "cuda"}:
        raise ValueError("native timed lane must be 'warp' or 'cuda'")
    if not row_id or not boundary:
        raise ValueError("native timed row_id and boundary must be non-empty")
    if operation_trace is not None and (
        operation_trace.row_id != row_id
        or operation_trace.lane != lane
        or operation_trace.boundary != boundary
    ):
        raise ValueError("operation trace identity does not match timed submission")
    active = ActiveTimedBoundary(row_id=row_id, lane=lane, boundary=boundary)
    if operation_trace is not None:
        operation_trace._start()
    boundary_token = _ACTIVE_BOUNDARY.set(active)
    trace_token = _ACTIVE_OPERATION_TRACE.set(operation_trace)
    try:
        yield active
    finally:
        try:
            if operation_trace is not None:
                operation_trace._finish()
        finally:
            _ACTIVE_OPERATION_TRACE.reset(trace_token)
            _ACTIVE_BOUNDARY.reset(boundary_token)
