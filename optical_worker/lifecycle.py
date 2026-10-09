"""Backend-neutral setup ownership and transport-terminal lifecycle.

This module is intentionally stdlib-only.  Device ordinals are validated as
plain metadata: importing it, creating a descriptor, or replaying evidence
must never initialize CUDA or consult an ambient/default device.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

DEVICE_SCOPE_SCHEMA_VERSION: Final = "q54.p12.4.closed-worker-device-scope.v1"
SETUP_QUIESCENCE_SCHEMA_VERSION: Final = "q54.p12.4.setup-quiescence.v1"
GUARDIAN_COVERAGE_SCHEMA_VERSION: Final = "q54.p12.4.worker-guardian-coverage.v1"
TRANSPORT_TERMINATION_SCHEMA_VERSION: Final = "q54.p12.4.transport-termination.v2"
SETUP_LIFECYCLE_VERSION: Final = "q54.p12.4.setup-lifecycle.v2"

UNKNOWN_DEVICE_OWNER: Final = "UNKNOWN_DEVICE_OWNER"
HOST_OWNER: Final = "HOST_OWNER"
NO_SETUP_QUIESCENCE: Final = "NO_SETUP_QUIESCENCE"
NO_ARENA_YET: Final = "NO_ARENA_YET"

DeviceOwnerScope = int | Literal["UNKNOWN_DEVICE_OWNER", "HOST_OWNER"]
SetupState = Literal[
    "NO_SETUP_GPU_WORK",
    "SETUP_DIRTY",
    "SETUP_QUIESCENT",
    "POISONED",
    "ABORTED_SYNCED",
    "QUARANTINED_UNTIL_PROCESS_EXIT",
    "PROCESS_EXITED",
]
GuardianState = Literal[
    "CONSTRUCTING",
    "SEALED",
    "PARTIAL_FROZEN",
    "TRANSFERRED_TO_QUARANTINE",
    "RELEASE_AUTHORIZED",
    "RELEASED",
]
TerminationDisposition = Literal[
    "cancelled_before_submission_exit",
    "aborted_synced_exit",
    "quarantined_process_exit",
    "multi_or_unknown_device_quarantine_exit",
]

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_LEAF_KINDS = frozenset(
    {
        "runtime_import_or_init",
        "allocation",
        "host_to_device_copy",
        "device_to_host_copy",
        "compile_or_jit",
        "event_create",
        "event_record",
        "event_wait",
        "warmup_launch",
        "device_synchronize",
    }
)
_FAULT_KINDS = frozenset(
    {
        "channel_eof",
        "truncated_frame",
        "broken_pipe",
        "parent_lost",
        "control_idle_timeout",
        "response_write_timeout",
        "response_failure",
        "request_protocol_failure",
        "dispatcher_failure",
        "setup_failure",
        "device_scope_compromised",
    }
)


class WorkerSetupProtocolError(RuntimeError):
    """The setup/guardian lifecycle was used outside its frozen state graph."""


class DeviceScopeCompromisedError(WorkerSetupProtocolError):
    """A setup leaf could have dirtied a foreign or unknown CUDA device."""


class _DeviceLeafPermit:
    __slots__ = ("descriptor_sha256", "lifecycle", "operation_sequence", "used")

    def __init__(
        self,
        *,
        lifecycle: "NativeWorkerSetupLifecycle",
        operation_sequence: int,
        descriptor_sha256: str,
    ) -> None:
        self.lifecycle = lifecycle
        self.operation_sequence = operation_sequence
        self.descriptor_sha256 = descriptor_sha256
        self.used = False


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) is not None


def _array_operation_id(prefix: str, array_name: str) -> str:
    """Bounded diagnostic ID without changing/case-folding the array identity.

    Canonical arrays include distinct x_world_R/x_world_r names. Guardian IDs
    are lowercase and bounded; the full UTF-8 name digest keeps those separate.
    This ID is not storage authority: the exact manifest/objects still bind it.
    """
    if type(array_name) is not str or not array_name or type(prefix) is not str:
        raise ValueError("array operation requires an exact nonempty name and prefix")
    result = f"{prefix}_{hashlib.sha256(array_name.encode('utf-8')).hexdigest()}"
    if not _valid_identifier(result):
        raise ValueError("array operation prefix is invalid or too long")
    return result


@dataclass(frozen=True)
class DeviceScopeDescriptor:
    """Pure-CPU proof declared immediately before one CUDA-touching leaf."""

    schema_version: str
    operation_id: str
    leaf_kind: str
    device_ordinals: tuple[int, ...]
    scope_source: str
    runtime_identity: str
    unreturned_resource_possible: bool

    @classmethod
    def exact_target(
        cls,
        *,
        operation_id: str,
        leaf_kind: str,
        device_ordinal: int,
        scope_source: str,
        runtime_identity: str,
        unreturned_resource_possible: bool = False,
    ) -> "DeviceScopeDescriptor":
        return cls(
            schema_version=DEVICE_SCOPE_SCHEMA_VERSION,
            operation_id=operation_id,
            leaf_kind=leaf_kind,
            device_ordinals=(device_ordinal,),
            scope_source=scope_source,
            runtime_identity=runtime_identity,
            unreturned_resource_possible=unreturned_resource_possible,
        ).validated()

    @classmethod
    def unknown(
        cls,
        *,
        operation_id: str,
        leaf_kind: str,
        scope_source: str,
        runtime_identity: str,
    ) -> "DeviceScopeDescriptor":
        return cls(
            schema_version=DEVICE_SCOPE_SCHEMA_VERSION,
            operation_id=operation_id,
            leaf_kind=leaf_kind,
            device_ordinals=(),
            scope_source=scope_source,
            runtime_identity=runtime_identity,
            unreturned_resource_possible=True,
        ).validated()

    def validated(self) -> "DeviceScopeDescriptor":
        if self.schema_version != DEVICE_SCOPE_SCHEMA_VERSION:
            raise WorkerSetupProtocolError("device-scope schema version is invalid")
        if not _valid_identifier(self.operation_id):
            raise WorkerSetupProtocolError("device-scope operation_id is invalid")
        if self.leaf_kind not in _LEAF_KINDS:
            raise WorkerSetupProtocolError("device-scope leaf_kind is invalid")
        if not _valid_identifier(self.scope_source) or not _valid_identifier(self.runtime_identity):
            raise WorkerSetupProtocolError("device-scope source/runtime identity is invalid")
        if not isinstance(self.unreturned_resource_possible, bool):
            raise WorkerSetupProtocolError("device-scope unreturned-resource flag is invalid")
        if tuple(sorted(set(self.device_ordinals))) != self.device_ordinals:
            raise WorkerSetupProtocolError("device-scope ordinals must be sorted and unique")
        for ordinal in self.device_ordinals:
            if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 0 <= ordinal <= 1023:
                raise WorkerSetupProtocolError("device-scope ordinal is invalid")
        if not self.device_ordinals and not self.unreturned_resource_possible:
            raise WorkerSetupProtocolError("empty device scope must be marked unknown")
        return self

    def to_payload(self) -> dict[str, object]:
        self.validated()
        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "leaf_kind": self.leaf_kind,
            "device_ordinals": list(self.device_ordinals),
            "scope_source": self.scope_source,
            "runtime_identity": self.runtime_identity,
            "unreturned_resource_possible": self.unreturned_resource_possible,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_payload())


@dataclass(frozen=True)
class GuardianResourceRecord:
    sequence: int
    role: str
    owner_type: str
    device_scope: DeviceOwnerScope
    operation_id: str

    def to_payload(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "role": self.role,
            "owner_type": self.owner_type,
            "device_scope": self.device_scope,
            "operation_id": self.operation_id,
        }


@dataclass(frozen=True)
class SetupLifecycleTransition:
    sequence: int
    operation: str
    epoch: int
    details: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "operation": self.operation,
            "epoch": self.epoch,
            "details": dict(self.details),
        }


class NativeWorkerResourceGuardian:
    """Strongly own a partial or sealed worker resource graph as one unit."""

    __slots__ = (
        "_authority",
        "_lock",
        "_owners",
        "_records",
        "_state",
        "_session_identity",
    )

    def __init__(self, *, session_identity: str) -> None:
        if not _valid_identifier(session_identity):
            raise WorkerSetupProtocolError("guardian session identity is invalid")
        self._authority = object()
        self._lock = threading.RLock()
        self._owners: list[object] = []
        self._records: list[GuardianResourceRecord] = []
        self._state: GuardianState = "CONSTRUCTING"
        self._session_identity = session_identity

    @property
    def state(self) -> GuardianState:
        return self._state

    @property
    def resource_count(self) -> int:
        return len(self._owners)

    @property
    def records(self) -> tuple[GuardianResourceRecord, ...]:
        return tuple(self._records)

    def contains_exact_owner(self, owner: object) -> bool:
        return any(candidate is owner for candidate in self._owners)

    def adopt(
        self,
        owner: object,
        *,
        role: str,
        device_scope: DeviceOwnerScope,
        operation_id: str,
    ) -> GuardianResourceRecord:
        with self._lock:
            if self._state != "CONSTRUCTING":
                raise WorkerSetupProtocolError("guardian additions require CONSTRUCTING state")
            if owner is None:
                raise WorkerSetupProtocolError("guardian cannot adopt a null owner")
            if not _valid_identifier(role) or not _valid_identifier(operation_id):
                raise WorkerSetupProtocolError("guardian resource role/operation is invalid")
            if isinstance(device_scope, int):
                if isinstance(device_scope, bool) or not 0 <= device_scope <= 1023:
                    raise WorkerSetupProtocolError("guardian resource device ordinal is invalid")
            elif device_scope not in {UNKNOWN_DEVICE_OWNER, HOST_OWNER}:
                raise WorkerSetupProtocolError("guardian resource device scope is invalid")
            if self.contains_exact_owner(owner):
                raise WorkerSetupProtocolError("guardian owner was already adopted")
            record = GuardianResourceRecord(
                sequence=len(self._records),
                role=role,
                owner_type=f"{type(owner).__module__}.{type(owner).__qualname__}",
                device_scope=device_scope,
                operation_id=operation_id,
            )
            self._owners.append(owner)
            self._records.append(record)
            return record

    def seal(self, authority: object) -> None:
        with self._lock:
            self._require_authority(authority)
            if self._state != "CONSTRUCTING":
                raise WorkerSetupProtocolError("guardian can seal exactly once")
            self._state = "SEALED"

    def freeze_partial(self, authority: object) -> None:
        with self._lock:
            self._require_authority(authority)
            if self._state == "PARTIAL_FROZEN":
                return
            if self._state not in {"CONSTRUCTING", "SEALED"}:
                raise WorkerSetupProtocolError("guardian cannot freeze partial resources now")
            self._state = "PARTIAL_FROZEN"

    def transfer_to_quarantine(self, authority: object) -> None:
        with self._lock:
            self._require_authority(authority)
            if self._state == "TRANSFERRED_TO_QUARANTINE":
                return
            if self._state not in {"PARTIAL_FROZEN", "SEALED"}:
                raise WorkerSetupProtocolError("guardian quarantine transfer requires frozen resources")
            self._state = "TRANSFERRED_TO_QUARANTINE"
            _PROCESS_QUARANTINE.append(self)

    def authorize_release(self, authority: object) -> None:
        with self._lock:
            self._require_authority(authority)
            if self._state not in {"SEALED", "PARTIAL_FROZEN"}:
                raise WorkerSetupProtocolError("guardian release cannot be authorized now")
            self._state = "RELEASE_AUTHORIZED"

    def release_whole_graph(self, authority: object) -> None:
        with self._lock:
            self._require_authority(authority)
            if self._state != "RELEASE_AUTHORIZED":
                raise WorkerSetupProtocolError("guardian release requires explicit authorization")
            self._owners.clear()
            self._state = "RELEASED"

    def coverage_payload(self) -> dict[str, object]:
        by_device: dict[str, int] = {}
        for record in self._records:
            key = str(record.device_scope)
            by_device[key] = by_device.get(key, 0) + 1
        return {
            "schema_version": GUARDIAN_COVERAGE_SCHEMA_VERSION,
            "session_identity": self._session_identity,
            "state": self._state,
            "records": [record.to_payload() for record in self._records],
            "per_device_resource_counts": dict(sorted(by_device.items())),
        }

    @property
    def coverage_sha256(self) -> str:
        return _sha256(self.coverage_payload())

    def _control_authority(self) -> object:
        return self._authority

    def _require_authority(self, authority: object) -> None:
        if authority is not self._authority:
            raise WorkerSetupProtocolError("guardian control authority is invalid")


_PROCESS_QUARANTINE: list[NativeWorkerResourceGuardian] = []


@dataclass(frozen=True)
class SetupQuiescenceEvidence:
    schema_version: str
    lifecycle_version: str
    session_identity: str
    lane_id: str
    target_device_ordinal: int
    epoch: int
    possible_dirty_device_ordinals: tuple[int, ...]
    unknown_device_scope: bool
    proven_quiescent_device_ordinals: tuple[int, ...]
    setup_operation_count: int
    setup_work_counts: Mapping[str, int]
    a2_readiness_sha256: str
    prepared_graph_sha256: str
    guardian_coverage_sha256: str
    scope_expansion_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lifecycle_version": self.lifecycle_version,
            "session_identity": self.session_identity,
            "lane_id": self.lane_id,
            "target_device_ordinal": self.target_device_ordinal,
            "epoch": self.epoch,
            "possible_dirty_device_ordinals": list(self.possible_dirty_device_ordinals),
            "unknown_device_scope": self.unknown_device_scope,
            "proven_quiescent_device_ordinals": list(self.proven_quiescent_device_ordinals),
            "setup_operation_count": self.setup_operation_count,
            "setup_work_counts": dict(self.setup_work_counts),
            "a2_readiness_sha256": self.a2_readiness_sha256,
            "prepared_graph_sha256": self.prepared_graph_sha256,
            "guardian_coverage_sha256": self.guardian_coverage_sha256,
            "scope_expansion_sha256": self.scope_expansion_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_payload())


class NativeWorkerSetupLifecycle:
    """Single-session authority for setup scope, ownership, and termination."""

    __slots__ = (
        "_current_operation_id",
        "_epoch",
        "_emergency_sync_notifier",
        "_guardian",
        "_guardian_authority",
        "_lane_id",
        "_lock",
        "_measured_sample_submission_latched",
        "_operation_count",
        "_pending_leaf_permit",
        "_possible_dirty",
        "_proven_quiescent",
        "_quiescence",
        "_scope_compromised",
        "_session_identity",
        "_setup_work_counts",
        "_state",
        "_target",
        "_trace",
        "_unknown_scope",
    )

    def __init__(
        self,
        *,
        lane_id: str,
        target_device_ordinal: int,
        emergency_sync_notifier: Callable[[int], None],
        session_identity: str | None = None,
    ) -> None:
        if lane_id not in {"warp", "cuda"}:
            raise WorkerSetupProtocolError("setup lane_id is invalid")
        if (
            not isinstance(target_device_ordinal, int)
            or isinstance(target_device_ordinal, bool)
            or not 0 <= target_device_ordinal <= 1023
        ):
            raise WorkerSetupProtocolError("setup target device ordinal is invalid")
        if not callable(emergency_sync_notifier):
            raise WorkerSetupProtocolError("emergency-sync notifier must be callable")
        self._session_identity = session_identity or f"worker-{secrets.token_hex(16)}"
        if not _valid_identifier(self._session_identity):
            raise WorkerSetupProtocolError("setup session identity is invalid")
        self._lane_id = lane_id
        self._target = target_device_ordinal
        self._emergency_sync_notifier = emergency_sync_notifier
        self._guardian = NativeWorkerResourceGuardian(session_identity=self._session_identity)
        self._guardian_authority = self._guardian._control_authority()
        self._lock = threading.RLock()
        self._state: SetupState = "NO_SETUP_GPU_WORK"
        self._epoch = 0
        self._possible_dirty: set[int] = set()
        self._proven_quiescent: set[int] = set()
        self._unknown_scope = False
        self._scope_compromised = False
        self._operation_count = 0
        self._pending_leaf_permit: _DeviceLeafPermit | None = None
        self._setup_work_counts: dict[str, int] = {}
        self._current_operation_id = "bootstrap"
        self._quiescence: SetupQuiescenceEvidence | None = None
        self._measured_sample_submission_latched = False
        self._trace: list[SetupLifecycleTransition] = []
        self._append("guardian_created", {"lane_id": lane_id, "target_device_ordinal": target_device_ordinal})

    @property
    def guardian(self) -> NativeWorkerResourceGuardian:
        return self._guardian

    @property
    def state(self) -> SetupState:
        return self._state

    @property
    def target_device_ordinal(self) -> int:
        return self._target

    @property
    def possible_dirty_device_ordinals(self) -> tuple[int, ...]:
        return tuple(sorted(self._possible_dirty))

    @property
    def proven_quiescent_device_ordinals(self) -> tuple[int, ...]:
        return tuple(sorted(self._proven_quiescent))

    @property
    def unknown_device_scope(self) -> bool:
        return self._unknown_scope

    @property
    def quiescence_evidence(self) -> SetupQuiescenceEvidence | None:
        return self._quiescence

    @property
    def trace(self) -> tuple[SetupLifecycleTransition, ...]:
        return tuple(self._trace)

    def begin_device_leaf(self, descriptor: DeviceScopeDescriptor) -> _DeviceLeafPermit:
        """Record possible dirty scope before the caller invokes the leaf."""

        descriptor.validated()
        with self._lock:
            if self._state in {
                "POISONED",
                "ABORTED_SYNCED",
                "QUARANTINED_UNTIL_PROCESS_EXIT",
                "PROCESS_EXITED",
            }:
                raise WorkerSetupProtocolError("setup lifecycle is terminal")
            if self._guardian.state != "CONSTRUCTING":
                raise WorkerSetupProtocolError("setup GPU work cannot begin after guardian seal")
            if self._pending_leaf_permit is not None:
                raise WorkerSetupProtocolError("previous setup leaf never reached its low-level hook")
            if self._state in {"NO_SETUP_GPU_WORK", "SETUP_QUIESCENT"}:
                self._epoch += 1
                self._possible_dirty.clear()
                self._proven_quiescent.clear()
                self._unknown_scope = False
                self._quiescence = None
            self._state = "SETUP_DIRTY"
            self._current_operation_id = descriptor.operation_id
            self._operation_count += 1
            self._setup_work_counts[descriptor.leaf_kind] = (
                self._setup_work_counts.get(descriptor.leaf_kind, 0) + 1
            )
            self._append(
                "setup_operation_started",
                {
                    "descriptor_sha256": descriptor.sha256,
                    "operation_id": descriptor.operation_id,
                    "leaf_kind": descriptor.leaf_kind,
                },
            )
            for ordinal in descriptor.device_ordinals:
                self._add_dirty_ordinal(
                    ordinal,
                    source=descriptor.scope_source,
                    reason="pre_leaf_descriptor",
                )
            if descriptor.unreturned_resource_possible:
                self._set_unknown_scope(
                    source=descriptor.scope_source,
                    reason="unreturned_resource_possible",
                )
            self._raise_if_compromised()
            permit = _DeviceLeafPermit(
                lifecycle=self,
                operation_sequence=self._operation_count,
                descriptor_sha256=descriptor.sha256,
            )
            self._pending_leaf_permit = permit
            return permit

    def enter_device_leaf(self, permit: object) -> None:
        """Consume the exact permit at the low-level wrapper before the API call."""

        with self._lock:
            if (
                type(permit) is not _DeviceLeafPermit
                or permit is not self._pending_leaf_permit
                or permit.lifecycle is not self
                or permit.used
                or permit.operation_sequence != self._operation_count
            ):
                raise WorkerSetupProtocolError("low-level setup leaf permit is invalid")
            permit.used = True
            self._pending_leaf_permit = None
            self._append(
                "low_level_leaf_entered",
                {
                    "descriptor_sha256": permit.descriptor_sha256,
                    "operation_sequence": permit.operation_sequence,
                },
            )

    def adopt_host_owner(self, owner: object, *, role: str, operation_id: str) -> None:
        with self._lock:
            record = self._guardian.adopt(
                owner,
                role=role,
                device_scope=HOST_OWNER,
                operation_id=operation_id,
            )
            self._append("owner_adopted", record.to_payload())

    def adopt_returned_device_owner(
        self,
        owner: object,
        *,
        actual_device_ordinal: int | None,
        role: str,
        operation_id: str,
    ) -> None:
        """Adopt first, then validate the returned owner's metadata ordinal."""

        with self._lock:
            scope: DeviceOwnerScope = (
                UNKNOWN_DEVICE_OWNER if actual_device_ordinal is None else actual_device_ordinal
            )
            record = self._guardian.adopt(
                owner,
                role=role,
                device_scope=scope,
                operation_id=operation_id,
            )
            self._append("owner_adopted", record.to_payload())
            if actual_device_ordinal is None:
                self._set_unknown_scope(source="returned_owner", reason="owner_device_unknown")
            else:
                self._add_dirty_ordinal(
                    actual_device_ordinal,
                    source="returned_owner",
                    reason="returned_owner_device",
                )
            self._raise_if_compromised()

    def require_adopted(self, *owners: object) -> None:
        for owner in owners:
            if not self._guardian.contains_exact_owner(owner):
                raise WorkerSetupProtocolError("device leaf operand is not guardian-owned")

    def mark_unknown_device_scope(self, *, source: str, reason: str) -> None:
        """Irreversibly compromise setup when a leaf's device set is unknowable."""

        with self._lock:
            self._set_unknown_scope(source=source, reason=reason)

    def seal_guardian(self) -> None:
        with self._lock:
            self._raise_if_compromised()
            self._guardian.seal(self._guardian_authority)
            self._append("guardian_sealed", {"coverage_sha256": self._guardian.coverage_sha256})

    def commit_target_barrier(
        self,
        *,
        a2_readiness_sha256: str,
        prepared_graph_sha256: str,
    ) -> SetupQuiescenceEvidence:
        """Commit evidence only after the caller's exact target sync returned."""

        with self._lock:
            if self._guardian.state != "SEALED":
                raise WorkerSetupProtocolError("setup barrier requires a sealed guardian")
            if self._state != "SETUP_DIRTY":
                raise WorkerSetupProtocolError("setup barrier requires dirty setup work")
            self._raise_if_compromised()
            if self._possible_dirty != {self._target}:
                raise WorkerSetupProtocolError("setup barrier covers only exact target scope")
            for digest in (a2_readiness_sha256, prepared_graph_sha256):
                if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                    raise WorkerSetupProtocolError("setup quiescence digest is invalid")
            self._proven_quiescent = {self._target}
            scope_trace = [
                item.to_payload()
                for item in self._trace
                if item.epoch == self._epoch
                and item.operation in {"dirty_ordinal_added", "unknown_scope_set"}
            ]
            evidence = SetupQuiescenceEvidence(
                schema_version=SETUP_QUIESCENCE_SCHEMA_VERSION,
                lifecycle_version=SETUP_LIFECYCLE_VERSION,
                session_identity=self._session_identity,
                lane_id=self._lane_id,
                target_device_ordinal=self._target,
                epoch=self._epoch,
                possible_dirty_device_ordinals=(self._target,),
                unknown_device_scope=False,
                proven_quiescent_device_ordinals=(self._target,),
                setup_operation_count=self._operation_count,
                setup_work_counts=MappingProxyType(dict(sorted(self._setup_work_counts.items()))),
                a2_readiness_sha256=a2_readiness_sha256,
                prepared_graph_sha256=prepared_graph_sha256,
                guardian_coverage_sha256=self._guardian.coverage_sha256,
                scope_expansion_sha256=_sha256(scope_trace),
            )
            self._quiescence = evidence
            self._state = "SETUP_QUIESCENT"
            self._append("setup_barrier_succeeded", {"evidence_sha256": evidence.sha256})
            return evidence

    def latch_measured_sample_submission(self) -> None:
        with self._lock:
            if self._state != "SETUP_QUIESCENT":
                raise WorkerSetupProtocolError("measured sample requires current setup quiescence")
            if not self._measured_sample_submission_latched:
                self._measured_sample_submission_latched = True
                self._append("sample_submission_latched", {})

    def commit_clean_close(self) -> None:
        """Release the whole idle graph only after CLOSE_ACK was written."""

        with self._lock:
            if self._measured_sample_submission_latched or self._state not in {
                "NO_SETUP_GPU_WORK",
                "SETUP_QUIESCENT",
            }:
                raise WorkerSetupProtocolError("clean close requires idle, quiescent setup")
            if self._guardian.state == "CONSTRUCTING":
                self._guardian.seal(self._guardian_authority)
                self._append("guardian_sealed", {"coverage_sha256": self._guardian.coverage_sha256})
            self._guardian.authorize_release(self._guardian_authority)
            self._append("resource_release_authorized", {"reason": "close_ack_committed"})
            self._append("ordinary_cleanup_started", {})
            self._guardian.release_whole_graph(self._guardian_authority)
            self._append("guardian_released", {})
            self._append("close_ack_committed", {})
            self._state = "PROCESS_EXITED"

    def terminate(
        self,
        *,
        fault: str,
        detection_phase: str,
        emergency_target_synchronize: Callable[[], None] | None,
    ) -> "TransportTerminationEvidence":
        """Reach exactly one non-reusable transport-terminal disposition."""

        if fault not in _FAULT_KINDS:
            raise WorkerSetupProtocolError("transport fault enum is invalid")
        if not _valid_identifier(detection_phase):
            raise WorkerSetupProtocolError("transport detection phase is invalid")
        with self._lock:
            if self._state in {
                "ABORTED_SYNCED",
                "QUARANTINED_UNTIL_PROCESS_EXIT",
                "PROCESS_EXITED",
            }:
                raise WorkerSetupProtocolError("transport termination is single-use")
            pre_state = self._state
            self._pending_leaf_permit = None
            quiescence_current = self._state == "SETUP_QUIESCENT" and self._quiescence is not None
            no_setup = self._state == "NO_SETUP_GPU_WORK"
            clean_cancel = not self._measured_sample_submission_latched and (no_setup or quiescence_current)
            self._state = "POISONED"
            self._append("setup_poisoned", {"fault": fault, "detection_phase": detection_phase})
            if self._guardian.state in {"CONSTRUCTING", "SEALED"}:
                self._guardian.freeze_partial(self._guardian_authority)
                self._append("guardian_partial_frozen", {"coverage_sha256": self._guardian.coverage_sha256})

            emergency_attempt_count = 0
            emergency_attempted_ordinal: int | None = None
            emergency_outcome = "not_attempted"
            per_device: dict[str, str] = {}
            release_authorized = False
            ordinary_cleanup_started = False

            if clean_cancel:
                if quiescence_current:
                    per_device[str(self._target)] = "already_quiescent"
                self._guardian.authorize_release(self._guardian_authority)
                self._append("resource_release_authorized", {"reason": "clean_cancel"})
                release_authorized = True
                ordinary_cleanup_started = True
                self._append("ordinary_cleanup_started", {})
                self._guardian.release_whole_graph(self._guardian_authority)
                self._append("guardian_released", {})
                disposition: TerminationDisposition = "cancelled_before_submission_exit"
                self._state = "ABORTED_SYNCED"
            else:
                uncovered = set(self._possible_dirty)
                if self._measured_sample_submission_latched:
                    uncovered.add(self._target)
                exact_target_only = not self._unknown_scope and uncovered == {self._target}
                if exact_target_only:
                    if emergency_target_synchronize is None:
                        raise WorkerSetupProtocolError(
                            "target-only dirty termination requires the exact emergency sync"
                        )
                    emergency_attempted_ordinal = self._target
                    try:
                        self._emergency_sync_notifier(self._target)
                    except BaseException as exc:
                        emergency_outcome = "watchdog_arm_failed"
                        per_device[str(self._target)] = "watchdog_arm_failed"
                        self._append(
                            "abort_watchdog_arm_failed",
                            {"device_ordinal": self._target, "exception_type": type(exc).__name__},
                        )
                        self._guardian.transfer_to_quarantine(self._guardian_authority)
                        self._append(
                            "guardian_quarantine_transferred",
                            {"coverage_sha256": self._guardian.coverage_sha256},
                        )
                        disposition = "quarantined_process_exit"
                        self._state = "QUARANTINED_UNTIL_PROCESS_EXIT"
                    else:
                        self._append("abort_watchdog_armed", {"device_ordinal": self._target})
                        emergency_attempt_count = 1
                        self._append("emergency_sync_started", {"device_ordinal": self._target})
                        try:
                            emergency_target_synchronize()
                        except BaseException as exc:
                            emergency_outcome = "failed"
                            per_device[str(self._target)] = "emergency_sync_failed"
                            self._append(
                                "emergency_sync_failed",
                                {
                                    "device_ordinal": self._target,
                                    "exception_type": type(exc).__name__,
                                },
                            )
                            self._guardian.transfer_to_quarantine(self._guardian_authority)
                            self._append(
                                "guardian_quarantine_transferred",
                                {"coverage_sha256": self._guardian.coverage_sha256},
                            )
                            disposition = "quarantined_process_exit"
                            self._state = "QUARANTINED_UNTIL_PROCESS_EXIT"
                        else:
                            emergency_outcome = "succeeded"
                            self._proven_quiescent.add(self._target)
                            per_device[str(self._target)] = "emergency_sync_succeeded"
                            self._append("emergency_sync_succeeded", {"device_ordinal": self._target})
                            self._guardian.authorize_release(self._guardian_authority)
                            self._append("resource_release_authorized", {"reason": "emergency_sync"})
                            release_authorized = True
                            ordinary_cleanup_started = True
                            self._append("ordinary_cleanup_started", {})
                            self._guardian.release_whole_graph(self._guardian_authority)
                            self._append("guardian_released", {})
                            disposition = "aborted_synced_exit"
                            self._state = "ABORTED_SYNCED"
                else:
                    for ordinal in sorted(uncovered):
                        per_device[str(ordinal)] = "process_teardown_only"
                    if self._unknown_scope:
                        per_device[UNKNOWN_DEVICE_OWNER] = "process_teardown_only"
                    self._guardian.transfer_to_quarantine(self._guardian_authority)
                    self._append(
                        "guardian_quarantine_transferred",
                        {"coverage_sha256": self._guardian.coverage_sha256},
                    )
                    disposition = "multi_or_unknown_device_quarantine_exit"
                    self._state = "QUARANTINED_UNTIL_PROCESS_EXIT"

            self._append("terminated", {"disposition": disposition})
            evidence = TransportTerminationEvidence(
                schema_version=TRANSPORT_TERMINATION_SCHEMA_VERSION,
                lifecycle_version=SETUP_LIFECYCLE_VERSION,
                session_identity=self._session_identity,
                lane_id=self._lane_id,
                target_device_ordinal=self._target,
                fault=fault,
                detection_phase=detection_phase,
                pre_fault_state=pre_state,
                final_state=self._state,
                setup_epoch=self._epoch,
                setup_operation_count=self._operation_count,
                setup_work_counts=MappingProxyType(dict(sorted(self._setup_work_counts.items()))),
                setup_quiescence_sha256=(
                    self._quiescence.sha256 if quiescence_current else NO_SETUP_QUIESCENCE
                ),
                possible_dirty_device_ordinals=tuple(sorted(self._possible_dirty)),
                unknown_device_scope=self._unknown_scope,
                proven_quiescent_device_ordinals=tuple(sorted(self._proven_quiescent)),
                measured_sample_submission_latched=self._measured_sample_submission_latched,
                guardian_state=self._guardian.state,
                guardian_coverage=MappingProxyType(self._guardian.coverage_payload()),
                guardian_coverage_sha256=self._guardian.coverage_sha256,
                guardian_transfer_committed=self._guardian.state == "TRANSFERRED_TO_QUARANTINE",
                emergency_sync_attempted_ordinal=emergency_attempted_ordinal,
                emergency_sync_attempt_count=emergency_attempt_count,
                emergency_sync_outcome=emergency_outcome,
                per_device_disposition=MappingProxyType(dict(sorted(per_device.items()))),
                resource_release_authorized=release_authorized,
                ordinary_cleanup_started=ordinary_cleanup_started,
                arena_state_sha256=NO_ARENA_YET,
                disposition=disposition,
                transition_trace=tuple(self._trace),
            )
            validate_transport_termination_evidence(evidence)
            return evidence

    def _add_dirty_ordinal(self, ordinal: int, *, source: str, reason: str) -> None:
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 0 <= ordinal <= 1023:
            self._set_unknown_scope(source=source, reason="invalid_device_ordinal")
            return
        if ordinal not in self._possible_dirty:
            self._possible_dirty.add(ordinal)
            self._append(
                "dirty_ordinal_added",
                {"device_ordinal": ordinal, "source": source, "reason": reason},
            )
        if ordinal != self._target:
            self._scope_compromised = True

    def _set_unknown_scope(self, *, source: str, reason: str) -> None:
        if not self._unknown_scope:
            self._unknown_scope = True
            self._append("unknown_scope_set", {"source": source, "reason": reason})
        self._scope_compromised = True

    def _raise_if_compromised(self) -> None:
        if (
            self._scope_compromised
            or self._unknown_scope
            or any(ordinal != self._target for ordinal in self._possible_dirty)
        ):
            raise DeviceScopeCompromisedError("setup device scope contains a foreign or unknown device")

    def _append(self, operation: str, details: Mapping[str, object]) -> None:
        self._trace.append(
            SetupLifecycleTransition(
                sequence=len(self._trace),
                operation=operation,
                epoch=self._epoch,
                details=MappingProxyType(dict(details)),
            )
        )


@dataclass(frozen=True)
class TransportTerminationEvidence:
    schema_version: str
    lifecycle_version: str
    session_identity: str
    lane_id: str
    target_device_ordinal: int
    fault: str
    detection_phase: str
    pre_fault_state: str
    final_state: str
    setup_epoch: int
    setup_operation_count: int
    setup_work_counts: Mapping[str, int]
    setup_quiescence_sha256: str
    possible_dirty_device_ordinals: tuple[int, ...]
    unknown_device_scope: bool
    proven_quiescent_device_ordinals: tuple[int, ...]
    measured_sample_submission_latched: bool
    guardian_state: str
    guardian_coverage: Mapping[str, object]
    guardian_coverage_sha256: str
    guardian_transfer_committed: bool
    emergency_sync_attempted_ordinal: int | None
    emergency_sync_attempt_count: int
    emergency_sync_outcome: str
    per_device_disposition: Mapping[str, str]
    resource_release_authorized: bool
    ordinary_cleanup_started: bool
    arena_state_sha256: str
    disposition: TerminationDisposition
    transition_trace: tuple[SetupLifecycleTransition, ...]

    @classmethod
    def from_payload(cls, payload: object) -> "TransportTerminationEvidence":
        fields = {
            "schema_version",
            "lifecycle_version",
            "session_identity",
            "lane_id",
            "target_device_ordinal",
            "fault",
            "detection_phase",
            "pre_fault_state",
            "final_state",
            "setup_epoch",
            "setup_operation_count",
            "setup_work_counts",
            "setup_quiescence_sha256",
            "possible_dirty_device_ordinals",
            "unknown_device_scope",
            "proven_quiescent_device_ordinals",
            "measured_sample_submission_latched",
            "guardian_state",
            "guardian_coverage",
            "guardian_coverage_sha256",
            "guardian_transfer_committed",
            "emergency_sync_attempted_ordinal",
            "emergency_sync_attempt_count",
            "emergency_sync_outcome",
            "per_device_disposition",
            "resource_release_authorized",
            "ordinary_cleanup_started",
            "arena_state_sha256",
            "disposition",
            "transition_trace",
        }
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise WorkerSetupProtocolError("termination evidence fields are invalid")
        trace_payload = payload["transition_trace"]
        if not isinstance(trace_payload, list):
            raise WorkerSetupProtocolError("termination transition trace is invalid")
        transitions: list[SetupLifecycleTransition] = []
        for item in trace_payload:
            if not isinstance(item, Mapping) or set(item) != {
                "sequence",
                "operation",
                "epoch",
                "details",
            }:
                raise WorkerSetupProtocolError("termination transition fields are invalid")
            if not isinstance(item["details"], Mapping):
                raise WorkerSetupProtocolError("termination transition details are invalid")
            transitions.append(
                SetupLifecycleTransition(
                    sequence=item["sequence"],
                    operation=item["operation"],
                    epoch=item["epoch"],
                    details=MappingProxyType(dict(item["details"])),
                )
            )
        for name in ("setup_work_counts", "guardian_coverage", "per_device_disposition"):
            if not isinstance(payload[name], Mapping):
                raise WorkerSetupProtocolError(f"termination {name} is invalid")
        for name in ("possible_dirty_device_ordinals", "proven_quiescent_device_ordinals"):
            if not isinstance(payload[name], list):
                raise WorkerSetupProtocolError(f"termination {name} is invalid")
        evidence = cls(
            schema_version=payload["schema_version"],
            lifecycle_version=payload["lifecycle_version"],
            session_identity=payload["session_identity"],
            lane_id=payload["lane_id"],
            target_device_ordinal=payload["target_device_ordinal"],
            fault=payload["fault"],
            detection_phase=payload["detection_phase"],
            pre_fault_state=payload["pre_fault_state"],
            final_state=payload["final_state"],
            setup_epoch=payload["setup_epoch"],
            setup_operation_count=payload["setup_operation_count"],
            setup_work_counts=MappingProxyType(dict(payload["setup_work_counts"])),
            setup_quiescence_sha256=payload["setup_quiescence_sha256"],
            possible_dirty_device_ordinals=tuple(payload["possible_dirty_device_ordinals"]),
            unknown_device_scope=payload["unknown_device_scope"],
            proven_quiescent_device_ordinals=tuple(payload["proven_quiescent_device_ordinals"]),
            measured_sample_submission_latched=payload["measured_sample_submission_latched"],
            guardian_state=payload["guardian_state"],
            guardian_coverage=MappingProxyType(dict(payload["guardian_coverage"])),
            guardian_coverage_sha256=payload["guardian_coverage_sha256"],
            guardian_transfer_committed=payload["guardian_transfer_committed"],
            emergency_sync_attempted_ordinal=payload["emergency_sync_attempted_ordinal"],
            emergency_sync_attempt_count=payload["emergency_sync_attempt_count"],
            emergency_sync_outcome=payload["emergency_sync_outcome"],
            per_device_disposition=MappingProxyType(dict(payload["per_device_disposition"])),
            resource_release_authorized=payload["resource_release_authorized"],
            ordinary_cleanup_started=payload["ordinary_cleanup_started"],
            arena_state_sha256=payload["arena_state_sha256"],
            disposition=payload["disposition"],
            transition_trace=tuple(transitions),
        )
        validate_transport_termination_evidence(evidence)
        return evidence

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "lifecycle_version": self.lifecycle_version,
            "session_identity": self.session_identity,
            "lane_id": self.lane_id,
            "target_device_ordinal": self.target_device_ordinal,
            "fault": self.fault,
            "detection_phase": self.detection_phase,
            "pre_fault_state": self.pre_fault_state,
            "final_state": self.final_state,
            "setup_epoch": self.setup_epoch,
            "setup_operation_count": self.setup_operation_count,
            "setup_work_counts": dict(self.setup_work_counts),
            "setup_quiescence_sha256": self.setup_quiescence_sha256,
            "possible_dirty_device_ordinals": list(self.possible_dirty_device_ordinals),
            "unknown_device_scope": self.unknown_device_scope,
            "proven_quiescent_device_ordinals": list(self.proven_quiescent_device_ordinals),
            "measured_sample_submission_latched": self.measured_sample_submission_latched,
            "guardian_state": self.guardian_state,
            "guardian_coverage": dict(self.guardian_coverage),
            "guardian_coverage_sha256": self.guardian_coverage_sha256,
            "guardian_transfer_committed": self.guardian_transfer_committed,
            "emergency_sync_attempted_ordinal": self.emergency_sync_attempted_ordinal,
            "emergency_sync_attempt_count": self.emergency_sync_attempt_count,
            "emergency_sync_outcome": self.emergency_sync_outcome,
            "per_device_disposition": dict(self.per_device_disposition),
            "resource_release_authorized": self.resource_release_authorized,
            "ordinary_cleanup_started": self.ordinary_cleanup_started,
            "arena_state_sha256": self.arena_state_sha256,
            "disposition": self.disposition,
            "transition_trace": [item.to_payload() for item in self.transition_trace],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_payload())


def validate_transport_termination_evidence(evidence: TransportTerminationEvidence) -> None:
    """Replay the closed subset of setup/abort transitions and summaries."""

    if type(evidence) is not TransportTerminationEvidence:
        raise WorkerSetupProtocolError("termination evidence type is invalid")
    if evidence.schema_version != TRANSPORT_TERMINATION_SCHEMA_VERSION:
        raise WorkerSetupProtocolError("termination evidence schema version is invalid")
    if evidence.lifecycle_version != SETUP_LIFECYCLE_VERSION:
        raise WorkerSetupProtocolError("termination lifecycle version is invalid")
    if evidence.lane_id not in {"warp", "cuda"} or evidence.fault not in _FAULT_KINDS:
        raise WorkerSetupProtocolError("termination lane/fault is invalid")
    if not _valid_identifier(evidence.session_identity) or not _valid_identifier(evidence.detection_phase):
        raise WorkerSetupProtocolError("termination identity/phase is invalid")
    if tuple(sorted(set(evidence.possible_dirty_device_ordinals))) != (
        evidence.possible_dirty_device_ordinals
    ):
        raise WorkerSetupProtocolError("termination dirty-device set is not canonical")
    if tuple(sorted(set(evidence.proven_quiescent_device_ordinals))) != (
        evidence.proven_quiescent_device_ordinals
    ):
        raise WorkerSetupProtocolError("termination quiescent-device set is not canonical")
    if not set(evidence.proven_quiescent_device_ordinals).issubset(
        set(evidence.possible_dirty_device_ordinals) | {evidence.target_device_ordinal}
    ):
        raise WorkerSetupProtocolError("termination quiescence exceeds observed device scope")
    if _SHA256_RE.fullmatch(evidence.guardian_coverage_sha256) is None:
        raise WorkerSetupProtocolError("termination guardian digest is invalid")
    coverage = dict(evidence.guardian_coverage)
    if set(coverage) != {
        "schema_version",
        "session_identity",
        "state",
        "records",
        "per_device_resource_counts",
    }:
        raise WorkerSetupProtocolError("termination guardian coverage fields are invalid")
    if (
        coverage["schema_version"] != GUARDIAN_COVERAGE_SCHEMA_VERSION
        or coverage["session_identity"] != evidence.session_identity
        or coverage["state"] != evidence.guardian_state
        or _sha256(coverage) != evidence.guardian_coverage_sha256
    ):
        raise WorkerSetupProtocolError("termination guardian coverage does not match its summary")
    records = coverage["records"]
    per_device_counts = coverage["per_device_resource_counts"]
    if not isinstance(records, list) or not isinstance(per_device_counts, Mapping):
        raise WorkerSetupProtocolError("termination guardian resource records are invalid")
    replayed_device_counts: dict[str, int] = {}
    canonical_records: list[dict[str, object]] = []
    for expected_sequence, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "sequence",
            "role",
            "owner_type",
            "device_scope",
            "operation_id",
        }:
            raise WorkerSetupProtocolError("termination guardian record fields are invalid")
        if record["sequence"] != expected_sequence:
            raise WorkerSetupProtocolError("termination guardian record sequence is invalid")
        if not _valid_identifier(record["role"]) or not _valid_identifier(record["operation_id"]):
            raise WorkerSetupProtocolError("termination guardian record identity is invalid")
        if not isinstance(record["owner_type"], str) or not record["owner_type"]:
            raise WorkerSetupProtocolError("termination guardian owner type is invalid")
        scope = record["device_scope"]
        if isinstance(scope, int):
            if isinstance(scope, bool) or not 0 <= scope <= 1023:
                raise WorkerSetupProtocolError("termination guardian record scope is invalid")
        elif scope not in {HOST_OWNER, UNKNOWN_DEVICE_OWNER}:
            raise WorkerSetupProtocolError("termination guardian record scope is invalid")
        key = str(scope)
        replayed_device_counts[key] = replayed_device_counts.get(key, 0) + 1
        canonical_records.append(dict(record))
    if dict(per_device_counts) != dict(sorted(replayed_device_counts.items())):
        raise WorkerSetupProtocolError("termination guardian device counts do not replay")

    dirty: set[int] = set()
    unknown = False
    operation_count = 0
    work_counts: dict[str, int] = {}
    operation_descriptors: dict[int, object] = {}
    entered_operation_sequences: set[int] = set()
    sample_latched = False
    guardian_state = "CONSTRUCTING"
    emergency_started = 0
    emergency_succeeded = 0
    emergency_failed = 0
    watchdog_armed = 0
    watchdog_arm_failed = 0
    release_authorized = False
    cleanup_started = False
    terminated_disposition: str | None = None
    adopted_records: list[dict[str, object]] = []
    previous_epoch = 0
    for expected_sequence, item in enumerate(evidence.transition_trace):
        if item.sequence != expected_sequence or item.epoch < previous_epoch:
            raise WorkerSetupProtocolError("termination transition sequence/epoch is invalid")
        previous_epoch = item.epoch
        details = dict(item.details)
        if item.operation == "guardian_created":
            if expected_sequence != 0 or details != {
                "lane_id": evidence.lane_id,
                "target_device_ordinal": evidence.target_device_ordinal,
            }:
                raise WorkerSetupProtocolError("termination guardian creation is invalid")
        elif item.operation == "setup_operation_started":
            if set(details) != {"descriptor_sha256", "operation_id", "leaf_kind"}:
                raise WorkerSetupProtocolError("termination setup operation is invalid")
            operation_count += 1
            operation_descriptors[operation_count] = details["descriptor_sha256"]
            leaf_kind = details["leaf_kind"]
            if leaf_kind not in _LEAF_KINDS:
                raise WorkerSetupProtocolError("termination setup leaf kind is invalid")
            work_counts[leaf_kind] = work_counts.get(leaf_kind, 0) + 1
        elif item.operation == "low_level_leaf_entered":
            if set(details) != {"descriptor_sha256", "operation_sequence"}:
                raise WorkerSetupProtocolError("termination low-level leaf entry is invalid")
            operation_sequence = details["operation_sequence"]
            if (
                not isinstance(operation_sequence, int)
                or isinstance(operation_sequence, bool)
                or operation_sequence <= 0
                or operation_sequence > operation_count
                or operation_sequence in entered_operation_sequences
                or details["descriptor_sha256"] != operation_descriptors[operation_sequence]
            ):
                raise WorkerSetupProtocolError("termination low-level leaf sequence is invalid")
            entered_operation_sequences.add(operation_sequence)
        elif item.operation == "dirty_ordinal_added":
            if set(details) != {"device_ordinal", "source", "reason"}:
                raise WorkerSetupProtocolError("termination dirty expansion is invalid")
            ordinal = details["device_ordinal"]
            if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal in dirty:
                raise WorkerSetupProtocolError("termination dirty expansion is duplicated/invalid")
            dirty.add(ordinal)
        elif item.operation == "unknown_scope_set":
            if unknown:
                raise WorkerSetupProtocolError("termination unknown scope was set twice")
            unknown = True
        elif item.operation == "owner_adopted":
            if guardian_state != "CONSTRUCTING":
                raise WorkerSetupProtocolError("termination owner was adopted after guardian freeze")
            adopted_records.append(details)
        elif item.operation == "guardian_sealed":
            if guardian_state != "CONSTRUCTING":
                raise WorkerSetupProtocolError("termination guardian seal predecessor is invalid")
            guardian_state = "SEALED"
        elif item.operation == "guardian_partial_frozen":
            if guardian_state not in {"CONSTRUCTING", "SEALED"}:
                raise WorkerSetupProtocolError("termination partial freeze predecessor is invalid")
            guardian_state = "PARTIAL_FROZEN"
        elif item.operation == "guardian_quarantine_transferred":
            if guardian_state not in {"PARTIAL_FROZEN", "SEALED"}:
                raise WorkerSetupProtocolError("termination quarantine predecessor is invalid")
            guardian_state = "TRANSFERRED_TO_QUARANTINE"
        elif item.operation == "resource_release_authorized":
            if guardian_state not in {"PARTIAL_FROZEN", "SEALED"} or release_authorized:
                raise WorkerSetupProtocolError("termination release authorization is invalid")
            guardian_state = "RELEASE_AUTHORIZED"
            release_authorized = True
        elif item.operation == "ordinary_cleanup_started":
            if not release_authorized or cleanup_started:
                raise WorkerSetupProtocolError("termination cleanup transition is invalid")
            cleanup_started = True
        elif item.operation == "guardian_released":
            if guardian_state != "RELEASE_AUTHORIZED" or not cleanup_started:
                raise WorkerSetupProtocolError("termination guardian release is invalid")
            guardian_state = "RELEASED"
        elif item.operation == "sample_submission_latched":
            if sample_latched:
                raise WorkerSetupProtocolError("termination sample latch is not monotonic")
            sample_latched = True
        elif item.operation == "abort_watchdog_armed":
            watchdog_armed += 1
        elif item.operation == "abort_watchdog_arm_failed":
            watchdog_arm_failed += 1
        elif item.operation == "emergency_sync_started":
            emergency_started += 1
        elif item.operation == "emergency_sync_succeeded":
            emergency_succeeded += 1
        elif item.operation == "emergency_sync_failed":
            emergency_failed += 1
        elif item.operation == "terminated":
            if expected_sequence != len(evidence.transition_trace) - 1 or terminated_disposition is not None:
                raise WorkerSetupProtocolError("termination final transition is misplaced")
            terminated_disposition = details.get("disposition")
        elif item.operation not in {"setup_barrier_succeeded", "setup_poisoned"}:
            raise WorkerSetupProtocolError(f"termination operation {item.operation!r} is invalid")

    if operation_count != evidence.setup_operation_count or dict(evidence.setup_work_counts) != dict(
        sorted(work_counts.items())
    ):
        raise WorkerSetupProtocolError("termination setup counters do not replay")
    if dirty != set(evidence.possible_dirty_device_ordinals) or unknown != evidence.unknown_device_scope:
        raise WorkerSetupProtocolError("termination dirty-device summary does not replay")
    if sample_latched != evidence.measured_sample_submission_latched:
        raise WorkerSetupProtocolError("termination sample latch does not replay")
    if guardian_state != evidence.guardian_state:
        raise WorkerSetupProtocolError("termination guardian state does not replay")
    if evidence.guardian_transfer_committed != (evidence.guardian_state == "TRANSFERRED_TO_QUARANTINE"):
        raise WorkerSetupProtocolError("termination guardian transfer flag is invalid")
    if adopted_records != canonical_records:
        raise WorkerSetupProtocolError("termination guardian records do not replay from transitions")
    if release_authorized != evidence.resource_release_authorized:
        raise WorkerSetupProtocolError("termination release authorization does not replay")
    if cleanup_started != evidence.ordinary_cleanup_started:
        raise WorkerSetupProtocolError("termination cleanup state does not replay")
    if terminated_disposition != evidence.disposition:
        raise WorkerSetupProtocolError("termination disposition does not replay")
    if emergency_started != evidence.emergency_sync_attempt_count or emergency_started > 1:
        raise WorkerSetupProtocolError("termination emergency-sync count does not replay")
    if watchdog_armed != emergency_started or watchdog_armed + watchdog_arm_failed > 1:
        raise WorkerSetupProtocolError("termination abort-watchdog state does not replay")
    if emergency_succeeded and emergency_failed:
        raise WorkerSetupProtocolError("termination emergency sync has conflicting outcomes")
    expected_outcome = "not_attempted"
    if watchdog_arm_failed:
        expected_outcome = "watchdog_arm_failed"
    elif emergency_succeeded:
        expected_outcome = "succeeded"
    elif emergency_failed:
        expected_outcome = "failed"
    if evidence.emergency_sync_outcome != expected_outcome:
        raise WorkerSetupProtocolError("termination emergency-sync outcome does not replay")

    foreign_or_unknown = evidence.unknown_device_scope or any(
        ordinal != evidence.target_device_ordinal for ordinal in evidence.possible_dirty_device_ordinals
    )
    if foreign_or_unknown:
        if (
            evidence.disposition != "multi_or_unknown_device_quarantine_exit"
            or evidence.final_state != "QUARANTINED_UNTIL_PROCESS_EXIT"
            or evidence.emergency_sync_attempt_count != 0
            or evidence.resource_release_authorized
            or evidence.ordinary_cleanup_started
            or evidence.guardian_state != "TRANSFERRED_TO_QUARANTINE"
            or evidence.guardian_transfer_committed is not True
        ):
            raise WorkerSetupProtocolError("foreign/unknown termination did not fail closed")
        if any(value != "process_teardown_only" for value in evidence.per_device_disposition.values()):
            raise WorkerSetupProtocolError("foreign/unknown per-device disposition is invalid")
    elif evidence.disposition == "aborted_synced_exit":
        if (
            evidence.final_state != "ABORTED_SYNCED"
            or evidence.emergency_sync_attempt_count != 1
            or evidence.emergency_sync_attempted_ordinal != evidence.target_device_ordinal
            or evidence.emergency_sync_outcome != "succeeded"
            or not evidence.resource_release_authorized
            or evidence.guardian_state != "RELEASED"
        ):
            raise WorkerSetupProtocolError("synced abort terminal proof is incomplete")
    elif evidence.disposition == "quarantined_process_exit":
        if (
            evidence.final_state != "QUARANTINED_UNTIL_PROCESS_EXIT"
            or evidence.emergency_sync_outcome not in {"failed", "watchdog_arm_failed"}
            or evidence.guardian_state != "TRANSFERRED_TO_QUARANTINE"
            or evidence.resource_release_authorized
        ):
            raise WorkerSetupProtocolError("target sync-failure quarantine proof is incomplete")
    elif evidence.disposition == "cancelled_before_submission_exit":
        if (
            evidence.final_state != "ABORTED_SYNCED"
            or evidence.measured_sample_submission_latched
            or evidence.emergency_sync_attempt_count != 0
            or not evidence.resource_release_authorized
            or evidence.guardian_state != "RELEASED"
        ):
            raise WorkerSetupProtocolError("clean cancellation proof is incomplete")
    else:
        raise WorkerSetupProtocolError("termination disposition is invalid")
