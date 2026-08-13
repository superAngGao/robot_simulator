"""Physics-owned CUDA publication binding and resource lifetime contracts.

This module deliberately contains no Torch or optical-backend imports.  It
coordinates an external CUDA consumer stream with :class:`PublishedRing`
through runtime and control-plane callbacks supplied by ``GpuEngine``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
import warnings
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from .publish import ConsumerState, GpuPublishedFrame

PUBLICATION_CUDA_BINDING_CONTRACT_VERSION = "q54.p12.4.publication-cuda-binding.v1"

GuardianState = Literal[
    "sealed_unbound",
    "bound",
    "released_to_session",
    "released_after_emergency_sync",
    "quarantined",
    "closed",
]
BindingState = Literal[
    "open",
    "poisoned",
    "aborting",
    "closed",
    "aborted_synced",
    "quarantined",
]


class CudaPublicationError(RuntimeError):
    """Base error for the external-CUDA-stream publication contract."""


class CudaPublicationStateError(CudaPublicationError):
    """Raised when a binding, guardian, or lease transition is invalid."""


class CudaPublicationPoisonedError(CudaPublicationError):
    """Raised when a CUDA timeline failure poisons a publication binding."""


class CudaPublicationCompletionRequiredError(CudaPublicationStateError):
    """Raised after safely completing a lease omitted on normal context exit."""


class CudaPublicationQuarantinedError(CudaPublicationError):
    """Raised when a process-terminal CUDA resource quarantine is active."""


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _opaque_token(*parts: object) -> str:
    return _canonical_sha256([str(part) for part in parts])


def _device_ordinal(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("CUDA device ordinal must not be boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("CUDA device ordinal must be >= 0")
        return value

    index = getattr(value, "index", None)
    if isinstance(index, int) and index >= 0:
        return index
    ordinal = getattr(value, "ordinal", None)
    if isinstance(ordinal, int) and ordinal >= 0:
        return ordinal

    text = str(value)
    if text.startswith("cuda:"):
        try:
            parsed = int(text.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device {value!r}") from exc
        if parsed >= 0:
            return parsed
    raise ValueError(f"Cannot determine explicit CUDA device ordinal from {value!r}")


class CudaConsumerResourceGuardian:
    """Single-use strong owner for one CUDA consumer session's resources.

    Covered resources are opaque to physics.  The guardian's job is solely to
    keep their Python/session owners alive until publication returns the
    lifetime hold or transfers it into process-terminal quarantine.
    """

    def __init__(
        self,
        *,
        device_ordinal: int,
        cuda_stream_owner: object,
        resources: tuple[object, ...],
    ) -> None:
        self._lock = threading.RLock()
        self._guardian_id = uuid.uuid4().hex
        self._device_ordinal = _device_ordinal(device_ordinal)
        self._stream_owner: object | None = cuda_stream_owner
        self._resources: tuple[object, ...] = resources
        self._state: GuardianState = "sealed_unbound"
        self._binding_id: str | None = None

        if cuda_stream_owner is None:
            raise ValueError("cuda_stream_owner is required")
        if not resources:
            raise ValueError("guardian requires at least one covered CUDA resource")
        if any(resource is None for resource in resources):
            raise ValueError("guardian resources must not contain None")
        identities = [id(resource) for resource in resources]
        if len(identities) != len(set(identities)):
            raise ValueError("guardian resources must be identity-unique")

        owner_handle = self._owner_stream_handle()
        owner_device = self._owner_device_ordinal()
        if owner_device != self._device_ordinal:
            raise ValueError(
                "guardian stream owner device does not match device_ordinal: "
                f"{owner_device} != {self._device_ordinal}"
            )
        self._cuda_stream_handle = owner_handle
        resource_manifest = [
            {
                "index": index,
                "token": _opaque_token(
                    self._guardian_id,
                    "resource",
                    index,
                    type(resource).__module__,
                    type(resource).__qualname__,
                    id(resource),
                ),
                "type": f"{type(resource).__module__}.{type(resource).__qualname__}",
            }
            for index, resource in enumerate(resources)
        ]
        self._resource_manifest_sha256 = _canonical_sha256(resource_manifest)
        self._stream_token = _opaque_token(
            self._guardian_id,
            "stream",
            self._device_ordinal,
            owner_handle,
        )

    @classmethod
    def seal(
        cls,
        *,
        device_ordinal: int,
        cuda_stream_owner: object,
        resources: tuple[object, ...] | list[object],
    ) -> "CudaConsumerResourceGuardian":
        return cls(
            device_ordinal=device_ordinal,
            cuda_stream_owner=cuda_stream_owner,
            resources=tuple(resources),
        )

    @property
    def guardian_id(self) -> str:
        return self._guardian_id

    @property
    def device_ordinal(self) -> int:
        return self._device_ordinal

    @property
    def cuda_stream_handle(self) -> int:
        return self._cuda_stream_handle

    @property
    def stream_token(self) -> str:
        return self._stream_token

    @property
    def resource_count(self) -> int:
        return len(self._resources)

    @property
    def resource_manifest_sha256(self) -> str:
        return self._resource_manifest_sha256

    @property
    def state(self) -> GuardianState:
        return self._state

    @property
    def binding_id(self) -> str | None:
        return self._binding_id

    def validate_identity(self, *, device_ordinal: int, cuda_stream_handle: int) -> None:
        with self._lock:
            if self._state in {"quarantined", "closed"}:
                raise CudaPublicationStateError(f"guardian {self._guardian_id} is {self._state}")
            if _device_ordinal(device_ordinal) != self._device_ordinal:
                raise CudaPublicationStateError("guardian CUDA device ordinal changed")
            if int(cuda_stream_handle) != self._cuda_stream_handle:
                raise CudaPublicationStateError("guardian CUDA stream handle mismatch")
            if self._owner_stream_handle() != self._cuda_stream_handle:
                raise CudaPublicationStateError("guardian stream owner handle changed")
            if self._owner_device_ordinal() != self._device_ordinal:
                raise CudaPublicationStateError("guardian stream owner device changed")

    def close(self) -> None:
        """Release guardian references when no publication binding owns them."""
        with self._lock:
            if self._state == "closed":
                return
            if self._state == "bound":
                raise CudaPublicationStateError("cannot close a bound CUDA resource guardian")
            if self._state == "quarantined":
                raise CudaPublicationQuarantinedError(
                    "quarantined CUDA resources must remain alive until process teardown"
                )
            self._resources = ()
            self._stream_owner = None
            self._state = "closed"

    def assert_resource_disposal_allowed(self, resource: object) -> None:
        """Guard an arena/session's explicit member cleanup operation.

        Future CUDA optical resource owners must call this before destructive
        close/free/resize.  Strong references protect ordinary Python lifetime;
        this check protects explicit cleanup performed by cooperative owners.
        """
        with self._lock:
            if not any(candidate is resource for candidate in self._resources):
                raise CudaPublicationStateError("resource is not covered by this guardian")
            if self._state in {"sealed_unbound", "bound"}:
                raise CudaPublicationStateError(
                    f"cannot dispose covered CUDA resource while guardian is {self._state}"
                )
            if self._state == "quarantined":
                raise CudaPublicationQuarantinedError(
                    "quarantined CUDA resource cannot be individually disposed"
                )
            if self._state == "closed":
                raise CudaPublicationStateError("CUDA resource guardian is closed")

    def _bind(self, *, binding_id: str, device_ordinal: int, cuda_stream_handle: int) -> None:
        with self._lock:
            if self._state != "sealed_unbound":
                raise CudaPublicationStateError(
                    f"guardian must be sealed_unbound before bind, got {self._state}"
                )
            self.validate_identity(
                device_ordinal=device_ordinal,
                cuda_stream_handle=cuda_stream_handle,
            )
            self._binding_id = binding_id
            self._state = "bound"

    def _rollback_bind(self, binding_id: str) -> None:
        with self._lock:
            if self._state != "bound" or self._binding_id != binding_id:
                raise CudaPublicationStateError("guardian bind rollback identity mismatch")
            self._binding_id = None
            self._state = "sealed_unbound"

    def _release_to_session(self, binding_id: str) -> None:
        self._release_binding(binding_id, "released_to_session")

    def _release_after_emergency_sync(self, binding_id: str) -> None:
        self._release_binding(binding_id, "released_after_emergency_sync")

    def _release_binding(self, binding_id: str, state: GuardianState) -> None:
        with self._lock:
            if self._state != "bound" or self._binding_id != binding_id:
                raise CudaPublicationStateError("guardian release identity mismatch")
            self._state = state

    def _quarantine(self, binding_id: str) -> None:
        with self._lock:
            if self._state == "quarantined" and self._binding_id == binding_id:
                return
            if self._state != "bound" or self._binding_id != binding_id:
                raise CudaPublicationStateError("guardian quarantine identity mismatch")
            self._state = "quarantined"

    def _owner_stream_handle(self) -> int:
        owner = self._stream_owner
        handle = getattr(owner, "cuda_stream", None)
        if isinstance(handle, bool) or not isinstance(handle, int) or handle < 0:
            raise ValueError("cuda_stream_owner.cuda_stream must be a non-negative integer")
        return handle

    def _owner_device_ordinal(self) -> int:
        owner = self._stream_owner
        if not hasattr(owner, "device"):
            raise ValueError("cuda_stream_owner.device is required")
        return _device_ordinal(getattr(owner, "device"))


class CudaPublicationRuntime(Protocol):
    """Backend operations required by the publication state machine."""

    def device_scope(self) -> AbstractContextManager[object]: ...

    def create_external_stream(self, cuda_stream_handle: int) -> object: ...

    def wait_event(self, stream: object, event: object) -> None: ...

    def allocate_event(self) -> object: ...

    def record_event(self, stream: object, event: object) -> None: ...

    def producer_wait_event(self, event: object) -> None: ...

    def emergency_synchronize(self) -> None: ...

    def event_native_handle(self, event: object) -> int: ...


@dataclass(frozen=True)
class CudaPublicationLeaseEvidence:
    contract_version: str
    binding_id: str
    guardian_id: str
    guardian_resource_manifest_sha256: str
    guardian_resource_count: int
    guardian_state: GuardianState
    physics_run_id: str
    consumer_id: str
    frame_id: int
    slot_id: int
    step_index: int
    physics_step_count: int
    device_ordinal: int
    stream_token: str
    publish_event_token: str
    publish_event_recorded: bool
    producer_publish_completion_recorded: bool
    publish_wait_enqueued: bool
    borrow_ordinal: int
    total_binding_borrow_count: int
    done_event_token: str | None
    consumer_completion_recorded: bool
    slot_logically_reclaimable: bool
    done_event_retained: bool
    producer_reuse_wait_enqueued: bool
    last_reader_operation: str | None
    poison_reason_category: str | None
    binding_state: BindingState
    final_drain_wait_enqueued: bool
    emergency_sync_attempt_count: int
    emergency_sync_success_count: int
    emergency_sync_failure_category: str | None
    teardown_disposition: str | None
    quarantine_accepted: bool
    process_terminal: bool


@dataclass(frozen=True)
class CudaPublicationAbortDisposition:
    binding_id: str
    guardian_id: str
    disposition: Literal["emergency_sync_succeeded", "quarantined_until_process_exit"]
    emergency_sync_attempted: bool
    emergency_sync_succeeded: bool
    emergency_sync_failure_category: str | None
    quarantine_accepted: bool
    process_terminal: bool


@dataclass
class _LeaseEvidenceState:
    frame_id: int
    slot_id: int
    step_index: int
    publish_event_token: str
    borrow_ordinal: int
    publish_wait_enqueued: bool = True
    done_event_token: str | None = None
    consumer_completion_recorded: bool = False
    slot_logically_reclaimable: bool = False
    done_event_retained: bool = False
    producer_reuse_wait_enqueued: bool = False
    last_reader_operation: str | None = None
    poison_reason_category: str | None = None


@dataclass
class _CompletionRecord:
    coverage_start_frame_id: int
    frame_id: int
    event: object
    event_token: str
    producer_wait_frame_ids: set[int]


@dataclass(frozen=True)
class _QuarantineCapsule:
    device_ordinal: int
    binding_id: str
    reason_category: str
    engine_owner: object
    binding: "CudaDeviceConsumerBinding"
    guardian: CudaConsumerResourceGuardian


class CudaPublicationQuarantineRegistry:
    """Append-only strong owner for process-terminal CUDA resource graphs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._capsules: list[_QuarantineCapsule] = []
        self._device_ordinals: set[int] = set()

    def require_device_available(self, device_ordinal: int) -> None:
        with self._lock:
            if device_ordinal in self._device_ordinals:
                raise CudaPublicationQuarantinedError(
                    f"CUDA device {device_ordinal} is quarantined; terminate this process"
                )

    def accept(
        self,
        *,
        device_ordinal: int,
        binding: "CudaDeviceConsumerBinding",
        guardian: CudaConsumerResourceGuardian,
        engine_owner: object,
        reason_category: str,
    ) -> None:
        with self._lock:
            if any(capsule.binding_id == binding.binding_id for capsule in self._capsules):
                return
            guardian._quarantine(binding.binding_id)
            capsule = _QuarantineCapsule(
                device_ordinal=device_ordinal,
                binding_id=binding.binding_id,
                reason_category=reason_category,
                engine_owner=engine_owner,
                binding=binding,
                guardian=guardian,
            )
            self._capsules.append(capsule)
            self._device_ordinals.add(device_ordinal)

    @property
    def capsule_count(self) -> int:
        with self._lock:
            return len(self._capsules)

    def contains_binding(self, binding_id: str) -> bool:
        with self._lock:
            return any(capsule.binding_id == binding_id for capsule in self._capsules)


PROCESS_CUDA_PUBLICATION_QUARANTINE = CudaPublicationQuarantineRegistry()


FrameResolver = Callable[[int | None], GpuPublishedFrame]
BorrowCommit = Callable[[ConsumerState, int, str], None]
CompletionCommit = Callable[[ConsumerState, int, str, object], None]
PoisonCommit = Callable[[ConsumerState, tuple[int, ...], str], None]
ClearDoneEvent = Callable[[ConsumerState, object], None]
BindingStateCommit = Callable[[ConsumerState, str, BindingState], None]


class CudaPublicationLease:
    """One ordered borrow of a physics-published frame."""

    def __init__(
        self,
        *,
        binding: "CudaDeviceConsumerBinding",
        frame: GpuPublishedFrame,
    ) -> None:
        self._binding = binding
        self._frame: GpuPublishedFrame | None = frame
        self._frame_id = frame.frame_id
        self._active = True
        self._completed = False
        self._ephemeral_resources: list[object] = []
        self._last_reader_operation: str | None = None

    def __enter__(self) -> "CudaPublicationLease":
        self._require_active()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._completed:
            return False
        if exc is None:
            frame_id = self._frame_id
            self._binding._complete_lease(
                self,
                last_reader_operation=(self._last_reader_operation or "no_consumer_reader_enqueued"),
                exceptional=False,
            )
            raise CudaPublicationCompletionRequiredError(
                f"CUDA publication lease for frame {frame_id} required explicit complete()"
            )
        try:
            self._binding._complete_lease(
                self,
                last_reader_operation=(self._last_reader_operation or "no_consumer_reader_enqueued"),
                exceptional=True,
            )
        except Exception as completion_error:
            raise completion_error from exc
        return False

    @property
    def frame(self) -> GpuPublishedFrame:
        self._require_active()
        assert self._frame is not None
        return self._frame

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def active(self) -> bool:
        return self._active

    @property
    def completed(self) -> bool:
        return self._completed

    def retain_ephemeral(self, *resources: object) -> None:
        self._require_active()
        if any(resource is None for resource in resources):
            raise ValueError("ephemeral lease resources must not contain None")
        self._ephemeral_resources.extend(resources)

    def mark_last_reader_enqueued(self, operation: str) -> None:
        self._require_active()
        if not isinstance(operation, str) or not operation:
            raise ValueError("last reader operation must be non-empty")
        self._last_reader_operation = operation

    def complete(self, *, last_reader_operation: str) -> CudaPublicationLeaseEvidence:
        if self._completed:
            raise CudaPublicationStateError("duplicate CUDA lease completion rejected")
        self.mark_last_reader_enqueued(last_reader_operation)
        return self._binding._complete_lease(
            self,
            last_reader_operation=self._last_reader_operation,
            exceptional=False,
        )

    def evidence(self) -> CudaPublicationLeaseEvidence:
        return self._binding.evidence(self.frame_id)

    def _mark_completed(self) -> None:
        self._completed = True
        self._active = False
        self._frame = None
        self._ephemeral_resources.clear()

    def _invalidate_for_quarantine(self) -> None:
        self._active = False

    def _invalidate_after_emergency_sync(self) -> None:
        self._active = False
        self._frame = None
        self._ephemeral_resources.clear()

    def _require_active(self) -> None:
        if not self._active or self._frame is None:
            raise CudaPublicationStateError("CUDA publication lease is no longer active")


class CudaDeviceConsumerBinding:
    """Stable ordered binding from physics publication to an external CUDA stream."""

    def __init__(
        self,
        *,
        consumer: ConsumerState,
        device_ordinal: int,
        cuda_stream_handle: int,
        guardian: CudaConsumerResourceGuardian,
        runtime: CudaPublicationRuntime,
        frame_resolver: FrameResolver,
        borrow_commit: BorrowCommit,
        completion_commit: CompletionCommit,
        poison_commit: PoisonCommit,
        clear_done_event: ClearDoneEvent,
        binding_state_commit: BindingStateCommit,
        engine_owner: object,
        physics_run_id: str,
        quarantine_registry: CudaPublicationQuarantineRegistry = PROCESS_CUDA_PUBLICATION_QUARANTINE,
    ) -> None:
        self._lock = threading.RLock()
        self._binding_id = uuid.uuid4().hex
        self._consumer = consumer
        self._consumer_id = consumer.consumer_id
        self._device_ordinal = _device_ordinal(device_ordinal)
        self._cuda_stream_handle = int(cuda_stream_handle)
        self._guardian: CudaConsumerResourceGuardian | None = guardian
        self._guardian_id = guardian.guardian_id
        self._guardian_state: GuardianState = guardian.state
        self._guardian_manifest_sha256 = guardian.resource_manifest_sha256
        self._guardian_resource_count = guardian.resource_count
        self._stream_token = guardian.stream_token
        self._runtime = runtime
        self._frame_resolver = frame_resolver
        self._borrow_commit = borrow_commit
        self._completion_commit = completion_commit
        self._poison_commit = poison_commit
        self._clear_done_event = clear_done_event
        self._binding_state_commit = binding_state_commit
        self._engine_owner = engine_owner
        if not isinstance(physics_run_id, str) or not physics_run_id:
            raise ValueError("physics_run_id must be non-empty")
        self._physics_run_id = physics_run_id
        self._quarantine_registry = quarantine_registry
        self._state: BindingState = "open"
        self._external_stream: object | None = None
        self._active_leases: deque[CudaPublicationLease] = deque()
        self._completion_records: deque[_CompletionRecord] = deque()
        self._failed_events: list[object] = []
        self._evidence: dict[int, _LeaseEvidenceState] = {}
        self._last_borrowed_frame_id = -1
        self._last_completed_frame_id = consumer.device_completed_frame_id
        self._borrow_count = 0
        self._poison_reason_category: str | None = None
        self._abort_disposition: CudaPublicationAbortDisposition | None = None
        self._final_drain_wait_enqueued = False

        guardian._bind(
            binding_id=self._binding_id,
            device_ordinal=self._device_ordinal,
            cuda_stream_handle=self._cuda_stream_handle,
        )
        try:
            with runtime.device_scope():
                self._external_stream = runtime.create_external_stream(self._cuda_stream_handle)
            self._binding_state_commit(self._consumer, self._binding_id, "open")
            self._guardian_state = guardian.state
        except Exception:
            guardian._rollback_bind(self._binding_id)
            self._guardian = None
            raise

    @property
    def binding_id(self) -> str:
        return self._binding_id

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    @property
    def device_ordinal(self) -> int:
        return self._device_ordinal

    @property
    def cuda_stream_handle(self) -> int:
        return self._cuda_stream_handle

    @property
    def state(self) -> BindingState:
        return self._state

    @property
    def active_lease_count(self) -> int:
        return len(self._active_leases)

    @property
    def retained_done_event_count(self) -> int:
        return len(self._completion_records)

    @property
    def abort_disposition(self) -> CudaPublicationAbortDisposition | None:
        return self._abort_disposition

    def borrow(self, frame_id: int | None = None) -> CudaPublicationLease:
        with self._lock:
            self._require_open()
            self._validate_guardian_identity_or_poison(())
            frame = self._frame_resolver(frame_id)
            if frame.frame_id <= self._last_borrowed_frame_id:
                raise CudaPublicationStateError(
                    "CUDA publication borrows must be strictly increasing and unique"
                )
            publish_event = frame.ready_event
            if publish_event is None:
                self._poison("missing_publish_event", (frame.slot_id,))
                raise CudaPublicationPoisonedError("published CUDA frame has no publish event")
            external_stream = self._external_stream
            if external_stream is None:
                self._poison("external_stream_unavailable", (frame.slot_id,))
                raise CudaPublicationPoisonedError("external CUDA stream is unavailable")

            try:
                with self._runtime.device_scope():
                    self._runtime.wait_event(external_stream, publish_event)
            except Exception as exc:
                self._poison("publish_wait_failure", (frame.slot_id,))
                raise CudaPublicationPoisonedError(
                    f"failed to enqueue publish wait for frame {frame.frame_id}"
                ) from exc

            try:
                self._borrow_commit(self._consumer, frame.frame_id, self._binding_id)
            except Exception as exc:
                self._poison("borrow_commit_failure", (frame.slot_id,))
                raise CudaPublicationPoisonedError(
                    f"failed to commit CUDA borrow for frame {frame.frame_id}"
                ) from exc

            lease = CudaPublicationLease(binding=self, frame=frame)
            self._active_leases.append(lease)
            self._last_borrowed_frame_id = frame.frame_id
            self._borrow_count += 1
            self._evidence[frame.frame_id] = _LeaseEvidenceState(
                frame_id=frame.frame_id,
                slot_id=frame.slot_id,
                step_index=frame.step_index,
                publish_event_token=self._event_token("publish", publish_event),
                borrow_ordinal=self._borrow_count,
            )
            return lease

    def complete(
        self,
        frame_id: int,
        *,
        last_reader_operation: str,
    ) -> CudaPublicationLeaseEvidence:
        with self._lock:
            lease = next(
                (candidate for candidate in self._active_leases if candidate.frame_id == frame_id),
                None,
            )
            if lease is None:
                raise CudaPublicationStateError(f"no active CUDA publication borrow for frame {frame_id}")
        return self._complete_lease(
            lease,
            last_reader_operation=last_reader_operation,
            exceptional=False,
        )

    def evidence(self, frame_id: int) -> CudaPublicationLeaseEvidence:
        with self._lock:
            state = self._evidence.get(frame_id)
            if state is None:
                raise KeyError(f"No CUDA publication evidence for frame {frame_id}")
            guardian_state = self._guardian.state if self._guardian is not None else self._guardian_state
            disposition = self._abort_disposition
            return CudaPublicationLeaseEvidence(
                contract_version=PUBLICATION_CUDA_BINDING_CONTRACT_VERSION,
                binding_id=self._binding_id,
                guardian_id=self._guardian_id,
                guardian_resource_manifest_sha256=self._guardian_manifest_sha256,
                guardian_resource_count=self._guardian_resource_count,
                guardian_state=guardian_state,
                physics_run_id=self._physics_run_id,
                consumer_id=self._consumer_id,
                frame_id=state.frame_id,
                slot_id=state.slot_id,
                step_index=state.step_index,
                physics_step_count=state.step_index + 1,
                device_ordinal=self._device_ordinal,
                stream_token=self._stream_token,
                publish_event_token=state.publish_event_token,
                publish_event_recorded=True,
                producer_publish_completion_recorded=True,
                publish_wait_enqueued=state.publish_wait_enqueued,
                borrow_ordinal=state.borrow_ordinal,
                total_binding_borrow_count=self._borrow_count,
                done_event_token=state.done_event_token,
                consumer_completion_recorded=state.consumer_completion_recorded,
                slot_logically_reclaimable=state.slot_logically_reclaimable,
                done_event_retained=state.done_event_retained,
                producer_reuse_wait_enqueued=state.producer_reuse_wait_enqueued,
                last_reader_operation=state.last_reader_operation,
                poison_reason_category=state.poison_reason_category,
                binding_state=self._state,
                final_drain_wait_enqueued=self._final_drain_wait_enqueued,
                emergency_sync_attempt_count=(
                    int(disposition.emergency_sync_attempted) if disposition is not None else 0
                ),
                emergency_sync_success_count=(
                    int(disposition.emergency_sync_succeeded) if disposition is not None else 0
                ),
                emergency_sync_failure_category=(
                    disposition.emergency_sync_failure_category if disposition is not None else None
                ),
                teardown_disposition=(disposition.disposition if disposition is not None else None),
                quarantine_accepted=(disposition.quarantine_accepted if disposition is not None else False),
                process_terminal=(disposition.process_terminal if disposition is not None else False),
            )

    def prepare_slot_reuse(self, frame_id: int) -> None:
        """Enqueue the producer wait covering ``frame_id`` without a host wait."""
        with self._lock:
            self._require_open_or_closed_for_reuse()
            if self._state == "closed":
                return
            record = next(
                (
                    candidate
                    for candidate in self._completion_records
                    if candidate.coverage_start_frame_id <= frame_id <= candidate.frame_id
                ),
                None,
            )
            if record is None:
                raise CudaPublicationStateError(f"no retained completion event covers frame {frame_id}")
            try:
                with self._runtime.device_scope():
                    self._runtime.producer_wait_event(record.event)
            except Exception as exc:
                slots = tuple(
                    evidence.slot_id for evidence in self._evidence.values() if evidence.frame_id == frame_id
                )
                self._poison("producer_reuse_wait_failure", slots)
                raise CudaPublicationPoisonedError(
                    f"failed to enqueue producer reuse wait for frame {frame_id}"
                ) from exc
            record.producer_wait_frame_ids.add(frame_id)
            for evidence in self._evidence.values():
                if evidence.frame_id == frame_id:
                    evidence.producer_reuse_wait_enqueued = True

    def release_reuse_safe_records(self, live_slot_frame_ids: tuple[int, ...]) -> None:
        """Drop events only after every covered ring frame has left its slot."""
        with self._lock:
            self._require_open()
            live = set(live_slot_frame_ids)
            retained: deque[_CompletionRecord] = deque()
            released_events: list[object] = []
            for record in self._completion_records:
                covered_live = {
                    frame_id
                    for frame_id in live
                    if record.coverage_start_frame_id <= frame_id <= record.frame_id
                }
                if covered_live:
                    retained.append(record)
                    continue
                released_events.append(record.event)
                for evidence in self._evidence.values():
                    if record.coverage_start_frame_id <= evidence.frame_id <= record.frame_id:
                        evidence.done_event_retained = False
            self._completion_records = retained
            for event in released_events:
                self._clear_done_event(self._consumer, event)

    def close(self) -> None:
        with self._lock:
            if self._state == "closed":
                return
            if self._state in {"poisoned", "aborting", "quarantined"}:
                raise CudaPublicationStateError(f"normal close is not allowed while binding is {self._state}")
            if self._state == "aborted_synced":
                return
            if self._active_leases:
                raise CudaPublicationStateError("cannot close binding with active CUDA leases")
            self._validate_guardian_identity_or_poison(())
            newest = self._completion_records[-1] if self._completion_records else None
            if newest is not None:
                try:
                    with self._runtime.device_scope():
                        self._runtime.producer_wait_event(newest.event)
                except Exception as exc:
                    self._poison("final_drain_wait_failure", ())
                    raise CudaPublicationPoisonedError(
                        "failed to enqueue CUDA publication final-drain wait"
                    ) from exc
                for evidence in self._evidence.values():
                    if evidence.frame_id <= newest.frame_id:
                        evidence.producer_reuse_wait_enqueued = True
                        evidence.done_event_retained = False
                self._final_drain_wait_enqueued = True
                self._clear_done_event(self._consumer, newest.event)
            self._completion_records.clear()
            guardian = self._require_guardian()
            guardian._release_to_session(self._binding_id)
            self._guardian_state = guardian.state
            self._binding_state_commit(self._consumer, self._binding_id, "closed")
            self._external_stream = None
            self._guardian = None
            self._state = "closed"

    def abort_poisoned(self) -> CudaPublicationAbortDisposition:
        with self._lock:
            if self._abort_disposition is not None:
                return self._abort_disposition
            if self._state == "aborting":
                raise CudaPublicationStateError("re-entrant CUDA publication abort rejected")
            if self._state != "poisoned":
                raise CudaPublicationStateError(
                    f"abort_poisoned requires poisoned binding, got {self._state}"
                )
            self._state = "aborting"
            guardian = self._require_guardian()
            reason = self._poison_reason_category or "unknown_cuda_failure"

            try:
                with self._runtime.device_scope():
                    self._runtime.emergency_synchronize()
            except Exception as exc:
                failure_category = type(exc).__name__
                self._quarantine_registry.accept(
                    device_ordinal=self._device_ordinal,
                    binding=self,
                    guardian=guardian,
                    engine_owner=self._engine_owner,
                    reason_category=reason,
                )
                for lease in self._active_leases:
                    lease._invalidate_for_quarantine()
                self._state = "quarantined"
                self._guardian_state = guardian.state
                self._binding_state_commit(self._consumer, self._binding_id, "quarantined")
                disposition = CudaPublicationAbortDisposition(
                    binding_id=self._binding_id,
                    guardian_id=self._guardian_id,
                    disposition="quarantined_until_process_exit",
                    emergency_sync_attempted=True,
                    emergency_sync_succeeded=False,
                    emergency_sync_failure_category=failure_category,
                    quarantine_accepted=True,
                    process_terminal=True,
                )
                self._abort_disposition = disposition
                return disposition

            for lease in self._active_leases:
                lease._invalidate_after_emergency_sync()
            self._active_leases.clear()
            for record in self._completion_records:
                self._clear_done_event(self._consumer, record.event)
            for evidence in self._evidence.values():
                evidence.done_event_retained = False
            self._completion_records.clear()
            self._failed_events.clear()
            guardian._release_after_emergency_sync(self._binding_id)
            self._guardian_state = guardian.state
            self._binding_state_commit(self._consumer, self._binding_id, "aborted_synced")
            self._external_stream = None
            self._guardian = None
            self._state = "aborted_synced"
            disposition = CudaPublicationAbortDisposition(
                binding_id=self._binding_id,
                guardian_id=self._guardian_id,
                disposition="emergency_sync_succeeded",
                emergency_sync_attempted=True,
                emergency_sync_succeeded=True,
                emergency_sync_failure_category=None,
                quarantine_accepted=False,
                process_terminal=False,
            )
            self._abort_disposition = disposition
            return disposition

    def quarantine_from_finalizer(self, reason_category: str) -> None:
        """Non-synchronizing fail-closed ownership transfer for ``__del__``."""
        with self._lock:
            if self._state in {"closed", "aborted_synced", "quarantined"}:
                return
            guardian = self._guardian
            if guardian is None:
                return
            self._poison_reason_category = reason_category
            for evidence in self._evidence.values():
                evidence.poison_reason_category = reason_category
            self._quarantine_registry.accept(
                device_ordinal=self._device_ordinal,
                binding=self,
                guardian=guardian,
                engine_owner=self._engine_owner,
                reason_category=reason_category,
            )
            for lease in self._active_leases:
                lease._invalidate_for_quarantine()
            self._state = "quarantined"
            self._guardian_state = guardian.state
            self._binding_state_commit(self._consumer, self._binding_id, "quarantined")
            self._abort_disposition = CudaPublicationAbortDisposition(
                binding_id=self._binding_id,
                guardian_id=self._guardian_id,
                disposition="quarantined_until_process_exit",
                emergency_sync_attempted=False,
                emergency_sync_succeeded=False,
                emergency_sync_failure_category=None,
                quarantine_accepted=True,
                process_terminal=True,
            )
            warnings.warn(
                f"CUDA publication binding {self._binding_id} quarantined during finalization",
                ResourceWarning,
                stacklevel=2,
            )

    def _complete_lease(
        self,
        lease: CudaPublicationLease,
        *,
        last_reader_operation: str,
        exceptional: bool,
    ) -> CudaPublicationLeaseEvidence:
        with self._lock:
            self._require_open()
            if not isinstance(last_reader_operation, str) or not last_reader_operation:
                raise ValueError("last_reader_operation must be non-empty")
            if not lease.active or lease.completed:
                raise CudaPublicationStateError("duplicate CUDA lease completion rejected")
            if lease not in self._active_leases:
                raise CudaPublicationStateError("completion without active CUDA borrow rejected")
            head = self._active_leases[0]
            if head is not lease:
                if exceptional:
                    slot_ids = tuple(active.frame.slot_id for active in self._active_leases)
                    self._poison("exceptional_out_of_order_completion", slot_ids)
                    raise CudaPublicationPoisonedError(
                        "ordinary exception could not safely complete a non-head CUDA lease"
                    )
                raise CudaPublicationStateError("out-of-order CUDA lease completion rejected")

            frame = lease.frame
            evidence = self._evidence[frame.frame_id]
            self._validate_guardian_identity_or_poison((frame.slot_id,))
            external_stream = self._external_stream
            if external_stream is None:
                self._poison("external_stream_unavailable", (frame.slot_id,))
                raise CudaPublicationPoisonedError("external CUDA stream is unavailable")

            try:
                done_event: object | None = None
                with self._runtime.device_scope():
                    done_event = self._runtime.allocate_event()
                    self._runtime.record_event(external_stream, done_event)
            except Exception as exc:
                if done_event is not None:
                    self._failed_events.append(done_event)
                self._poison("done_event_record_failure", (frame.slot_id,))
                raise CudaPublicationPoisonedError(
                    f"failed to record CUDA consumer completion for frame {frame.frame_id}"
                ) from exc

            event_token = self._event_token("done", done_event)
            coverage_start = self._last_completed_frame_id + 1
            record = _CompletionRecord(
                coverage_start_frame_id=coverage_start,
                frame_id=frame.frame_id,
                event=done_event,
                event_token=event_token,
                producer_wait_frame_ids=set(),
            )
            self._completion_records.append(record)
            evidence.done_event_token = event_token
            evidence.done_event_retained = True
            evidence.last_reader_operation = last_reader_operation

            try:
                self._completion_commit(
                    self._consumer,
                    frame.frame_id,
                    self._binding_id,
                    done_event,
                )
            except Exception as exc:
                self._poison("ring_completion_commit_failure", (frame.slot_id,))
                raise CudaPublicationPoisonedError(
                    f"failed to commit CUDA consumer completion for frame {frame.frame_id}"
                ) from exc

            self._last_completed_frame_id = frame.frame_id
            evidence.consumer_completion_recorded = True
            evidence.slot_logically_reclaimable = True
            self._active_leases.popleft()
            lease._mark_completed()
            return self.evidence(frame.frame_id)

    def _validate_guardian_identity(self) -> None:
        guardian = self._require_guardian()
        guardian.validate_identity(
            device_ordinal=self._device_ordinal,
            cuda_stream_handle=self._cuda_stream_handle,
        )

    def _validate_guardian_identity_or_poison(self, slot_ids: tuple[int, ...]) -> None:
        try:
            self._validate_guardian_identity()
        except Exception as exc:
            self._poison("stream_identity_failure", slot_ids)
            raise CudaPublicationPoisonedError("CUDA stream/guardian identity validation failed") from exc

    def _event_token(self, kind: str, event: object) -> str:
        try:
            native_handle = self._runtime.event_native_handle(event)
        except Exception:
            native_handle = id(event)
        return _opaque_token(
            self._binding_id,
            kind,
            self._device_ordinal,
            native_handle,
        )

    def _poison(self, reason_category: str, slot_ids: tuple[int, ...]) -> None:
        if self._state in {"closed", "aborted_synced", "quarantined"}:
            raise CudaPublicationStateError(f"cannot poison binding in state {self._state}")
        poisoned_slot_ids = set(slot_ids)
        poisoned_slot_ids.update(lease.frame.slot_id for lease in self._active_leases)
        poisoned_slot_ids.update(
            evidence.slot_id for evidence in self._evidence.values() if evidence.done_event_retained
        )
        self._poison_reason_category = reason_category
        self._state = "poisoned"
        for evidence in self._evidence.values():
            evidence.poison_reason_category = reason_category
        self._poison_commit(self._consumer, tuple(sorted(poisoned_slot_ids)), reason_category)

    def _require_guardian(self) -> CudaConsumerResourceGuardian:
        guardian = self._guardian
        if guardian is None:
            raise CudaPublicationStateError("CUDA resource guardian is no longer bound")
        return guardian

    def _require_open(self) -> None:
        if self._state == "poisoned":
            raise CudaPublicationPoisonedError(
                f"CUDA publication binding is poisoned: {self._poison_reason_category}"
            )
        if self._state != "open":
            raise CudaPublicationStateError(f"CUDA publication binding is {self._state}")

    def _require_open_or_closed_for_reuse(self) -> None:
        if self._state == "poisoned":
            raise CudaPublicationPoisonedError(
                f"CUDA publication binding is poisoned: {self._poison_reason_category}"
            )
        if self._state not in {"open", "closed"}:
            raise CudaPublicationStateError(f"CUDA publication binding is {self._state}")


__all__ = [
    "CudaConsumerResourceGuardian",
    "CudaDeviceConsumerBinding",
    "CudaPublicationAbortDisposition",
    "CudaPublicationCompletionRequiredError",
    "CudaPublicationError",
    "CudaPublicationLease",
    "CudaPublicationLeaseEvidence",
    "CudaPublicationPoisonedError",
    "CudaPublicationQuarantineRegistry",
    "CudaPublicationQuarantinedError",
    "CudaPublicationRuntime",
    "CudaPublicationStateError",
    "PROCESS_CUDA_PUBLICATION_QUARANTINE",
    "PUBLICATION_CUDA_BINDING_CONTRACT_VERSION",
]
