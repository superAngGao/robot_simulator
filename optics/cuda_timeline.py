"""Composable Torch-native CUDA execution transactions for optical work."""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .native_arena import C4ActiveOutputLease, NativeOpticalArenaProtocolError

try:  # pragma: no cover - exercised on CUDA hosts.
    import torch
except Exception:  # pragma: no cover - keeps CPU-only imports available.
    torch = None


class CudaOpticalTimelineError(RuntimeError):
    """Base error for an invalid CUDA optical timeline operation."""


class CudaOpticalTransactionPoisonedError(CudaOpticalTimelineError):
    """Raised after an enqueued transaction loses safe reuse evidence."""


_CONTROLLED_TEMPLATE_CAPABILITIES: weakref.WeakKeyDictionary[object, object] = weakref.WeakKeyDictionary()
_CONTROLLED_TRANSACTION_CAPABILITIES: weakref.WeakKeyDictionary[object, object] = weakref.WeakKeyDictionary()
_CONTROLLED_TRANSACTION_EXECUTION_AUTHORITIES: weakref.WeakKeyDictionary[
    object, weakref.ReferenceType[object]
] = weakref.WeakKeyDictionary()
_AUTHORIZED_CONTROLLED_TRANSACTIONS: weakref.WeakKeyDictionary[
    object, tuple[weakref.ReferenceType[object], str]
] = weakref.WeakKeyDictionary()
_CONTROLLED_RESOURCE_CAPABILITIES: dict[int, weakref.ReferenceType[object]] = {}
_MINTED_CUDA_ARENA_EXECUTION_AUTHORITIES: weakref.WeakKeyDictionary[object, tuple[object, object]] = (
    weakref.WeakKeyDictionary()
)
_CUDA_ARENA_EXECUTION_AUTHORITY_KEY = object()


class _CudaOpticalTransactionCapability:
    __slots__ = ("__weakref__",)


class _CudaArenaLiveExecutionAuthority:
    """Exact timeline-owned authority for one acquired C4 lease."""

    __slots__ = (
        "_active_owner",
        "_evidence_sha256",
        "_expected_binding",
        "_frozen_signature",
        "_generation",
        "_lease",
        "_phase",
        "_sealed",
        "_transaction",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        transaction: "CudaOpticalTransaction",
        lease: C4ActiveOutputLease,
        active_owner: object,
        expected_binding: tuple[tuple[str, object], ...],
        _key: object,
    ) -> None:
        if _key is not _CUDA_ARENA_EXECUTION_AUTHORITY_KEY:
            raise TypeError("CUDA arena execution authorities are timeline-minted")
        if type(lease) is not C4ActiveOutputLease or lease.state != "LEASED":
            raise NativeOpticalArenaProtocolError(
                "CUDA execution authority requires an active leased C4 generation"
            )
        if (
            getattr(active_owner, "_transaction", None) is not transaction
            or getattr(active_owner, "_control_lease", object()) is not None
            or getattr(active_owner, "_state", None) != "DORMANT"
        ):
            raise NativeOpticalArenaProtocolError("CUDA execution authority owner is invalid")
        evidence = lease.evidence
        if any(getattr(evidence, name, object()) != value for name, value in expected_binding):
            raise NativeOpticalArenaProtocolError(
                "CUDA execution authority lease does not match its sealed plan"
            )
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "_transaction", transaction)
        object.__setattr__(self, "_lease", lease)
        object.__setattr__(self, "_active_owner", active_owner)
        object.__setattr__(self, "_expected_binding", expected_binding)
        object.__setattr__(self, "_generation", evidence.generation)
        object.__setattr__(self, "_evidence_sha256", evidence.sha256)
        object.__setattr__(self, "_phase", "ACQUIRED")
        object.__setattr__(self, "_frozen_signature", self._static_signature())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CUDA arena execution authority is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CUDA arena execution authority is immutable")

    def _begin(self, transaction: "CudaOpticalTransaction", *, operation: str) -> None:
        self._assert_registered(transaction)
        lease = self._require_live_lease()
        if operation == "enqueue":
            if (
                self._phase != "ACQUIRED"
                or getattr(self._active_owner, "_state", None) != "ACQUIRED"
                or lease.state != "LEASED"
                or transaction.state != "validated_open"
            ):
                raise NativeOpticalArenaProtocolError("CUDA enqueue authority requires its acquired C4 lease")
            lease.begin_enqueue()
            if lease.state != "ENQUEUING":
                raise NativeOpticalArenaProtocolError("CUDA C4 lease did not enter enqueue state")
            object.__setattr__(self, "_phase", "ENQUEUE_AUTHORIZED")
            return
        if operation == "finish":
            if (
                self._phase != "ENQUEUED"
                or getattr(self._active_owner, "_state", None) != "ENQUEUED"
                or lease.state != "ENQUEUING"
                or transaction.state != "enqueuing"
            ):
                raise NativeOpticalArenaProtocolError("CUDA finish authority requires its enqueued C4 lease")
            object.__setattr__(self, "_phase", "FINISH_AUTHORIZED")
            return
        raise NativeOpticalArenaProtocolError("CUDA execution authority operation is invalid")

    def _assert_execution(
        self,
        transaction: "CudaOpticalTransaction",
        *,
        operation: str,
    ) -> None:
        self._assert_registered(transaction)
        lease = self._require_live_lease()
        expected_phase = "ENQUEUE_AUTHORIZED" if operation == "enqueue" else "FINISH_AUTHORIZED"
        expected_owner_state = "ACQUIRED" if operation == "enqueue" else "ENQUEUED"
        if (
            self._phase != expected_phase
            or getattr(self._active_owner, "_state", None) != expected_owner_state
            or lease.state != "ENQUEUING"
        ):
            raise NativeOpticalArenaProtocolError(
                "CUDA transaction execution lost its live C4 lease authority"
            )

    def _end(
        self,
        transaction: "CudaOpticalTransaction",
        *,
        operation: str,
        failure: BaseException | None,
    ) -> None:
        if failure is not None:
            self._poison(f"CUDA {operation} execution failed: {failure}")
            return
        try:
            self._assert_registered(transaction)
            lease = self._require_live_lease()
            if operation == "enqueue":
                if (
                    self._phase != "ENQUEUE_AUTHORIZED"
                    or transaction.state != "enqueuing"
                    or transaction._next_stage != len(transaction._stages)
                    or lease.state != "ENQUEUING"
                ):
                    raise NativeOpticalArenaProtocolError(
                        "CUDA enqueue did not complete the exact registered stage graph"
                    )
                object.__setattr__(self, "_phase", "ENQUEUED")
                object.__setattr__(self._active_owner, "_state", "ENQUEUED")
                return
            if operation == "finish":
                if self._phase != "FINISH_AUTHORIZED" or transaction.state != "finished":
                    raise NativeOpticalArenaProtocolError(
                        "CUDA finish did not record its exact terminal event"
                    )
                completion = transaction.completion
                lease.record_completion()
                object.__setattr__(self, "_phase", "COMPLETION_RECORDED")
                object.__setattr__(self._active_owner, "_completion", completion)
                object.__setattr__(self._active_owner, "_state", "COMPLETION_RECORDED")
                return
            raise NativeOpticalArenaProtocolError("CUDA execution authority operation is invalid")
        except BaseException as exc:
            self._poison(f"CUDA {operation} lifecycle commit failed: {exc}")
            raise

    def retire(self, *, expected_phase: str, terminal_phase: str) -> None:
        self._assert_registered(self._transaction)
        if self._phase != expected_phase:
            raise NativeOpticalArenaProtocolError("CUDA execution authority retirement is out of order")
        object.__setattr__(self, "_phase", terminal_phase)

    def _assert_registered(self, transaction: "CudaOpticalTransaction") -> None:
        registration = _MINTED_CUDA_ARENA_EXECUTION_AUTHORITIES.get(self)
        if (
            type(self) is not _CudaArenaLiveExecutionAuthority
            or registration is None
            or registration[0] is not transaction
            or registration[1] is not self._lease
            or transaction is not self._transaction
            or getattr(self._active_owner, "_transaction", None) is not transaction
            or self._static_signature() != self._frozen_signature
        ):
            raise NativeOpticalArenaProtocolError(
                "CUDA execution authority is not the exact timeline-minted identity"
            )

    def _require_live_lease(self) -> C4ActiveOutputLease:
        lease = self._lease
        if (
            type(lease) is not C4ActiveOutputLease
            or lease is not getattr(self._active_owner, "_control_lease", None)
            or lease.evidence.generation != self._generation
            or lease.evidence.sha256 != self._evidence_sha256
            or any(getattr(lease.evidence, name, object()) != value for name, value in self._expected_binding)
        ):
            raise NativeOpticalArenaProtocolError("CUDA live execution C4 lease binding changed")
        return lease

    def _poison(self, reason: str) -> None:
        try:
            self._lease.mark_submitted_failure(reason)
        except Exception:
            pass
        object.__setattr__(self, "_phase", "POISONED")
        object.__setattr__(self._active_owner, "_state", "POISONED")

    def _static_signature(self) -> tuple[object, ...]:
        return (
            id(self._transaction),
            id(self._lease),
            id(self._active_owner),
            self._expected_binding,
            self._generation,
            self._evidence_sha256,
        )


def _controlled_resource_capability(resource: object) -> object | None:
    reference = _CONTROLLED_RESOURCE_CAPABILITIES.get(id(resource))
    if reference is None:
        return None
    capability = reference()
    if capability is None:
        _CONTROLLED_RESOURCE_CAPABILITIES.pop(id(resource), None)
    return capability


@dataclass(frozen=True)
class CudaDeferredProfile:
    """Timing events whose elapsed values are resolved after terminal wait."""

    event_pairs: tuple[tuple[str, Any, Any], ...]

    def resolve_ms(self) -> tuple[tuple[str, float], ...]:
        return tuple((name, float(start.elapsed_time(end))) for name, start, end in self.event_pairs)


@dataclass(frozen=True)
class CudaOpticalCompletion:
    """One recorded, not necessarily physically complete, transaction event."""

    event: Any
    device: Any
    stream_handle: int
    stage_names: tuple[str, ...]
    stage_evidence: tuple[object, ...]
    resource_owners: tuple[object, ...]
    deferred_profile: CudaDeferredProfile | None = None
    recorded: bool = True


class CudaOpticalExecutionContext:
    """Strongly owned indexed Torch device/stream identity."""

    def __init__(self, *, device: Any, stream: Any) -> None:
        _require_torch_cuda()
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda" or resolved_device.index is None:
            raise ValueError("CUDA optical execution requires an indexed CUDA device")
        stream_device = torch.device(getattr(stream, "device", "cpu"))
        if stream_device != resolved_device:
            raise ValueError("CUDA optical stream device must match the indexed target device")
        stream_handle = getattr(stream, "cuda_stream", None)
        if not isinstance(stream_handle, int) or stream_handle == 0:
            raise ValueError("CUDA optical execution requires a live Torch CUDA stream")
        self.device = resolved_device
        self.stream = stream
        self.stream_handle = int(stream_handle)

    def begin_transaction(
        self,
        *,
        stages: Sequence[object],
        completion_event: object,
        diagnostic_events: Mapping[str, tuple[object, object]] | None = None,
    ) -> "CudaOpticalTransaction":
        """Validate and freeze a complete ordered transaction before enqueue."""

        return self.prepare_transaction_template(
            stages=stages,
            completion_event=completion_event,
            diagnostic_events=diagnostic_events,
        ).activate()

    def prepare_transaction_template(
        self,
        *,
        stages: Sequence[object],
        completion_event: object,
        diagnostic_events: Mapping[str, tuple[object, object]] | None = None,
        _activation_capability: object | None = None,
    ) -> "CudaOpticalTransactionTemplate":
        """Fully validate an immutable stage/event graph for later activations."""

        self._validate_identity()
        return CudaOpticalTransactionTemplate(
            context=self,
            stages=stages,
            completion_event=completion_event,
            diagnostic_events=diagnostic_events,
            _activation_capability=_activation_capability,
        )

    def _new_controlled_transaction_capability(self) -> object:
        return _CudaOpticalTransactionCapability()

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Select the exact indexed stream and restore ambient CUDA state."""

        self._validate_identity()
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            yield

    def _validate_identity(self) -> None:
        if torch.device(getattr(self.stream, "device", "cpu")) != self.device:
            raise CudaOpticalTimelineError("CUDA optical stream device identity changed")
        if int(getattr(self.stream, "cuda_stream", 0)) != self.stream_handle:
            raise CudaOpticalTimelineError("CUDA optical stream handle identity changed")


class CudaOpticalTransactionTemplate:
    """Reusable immutable validation result for single-use transactions.

    The template owns no generation or slot policy.  C4 is responsible for
    exposing activations only after an exact arena lease is acquired and for
    preventing event reuse until that generation is reclaimed.
    """

    __slots__ = (
        "_completion_event",
        "_context",
        "_diagnostic_events",
        "_frozen_signature",
        "_sealed",
        "_stage_evidence",
        "_stage_names",
        "_stages",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        context: CudaOpticalExecutionContext,
        stages: Sequence[object],
        completion_event: object,
        diagnostic_events: Mapping[str, tuple[object, object]] | None,
        _activation_capability: object | None = None,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        frozen_stages = tuple(stages)
        if not frozen_stages:
            raise ValueError("CUDA optical transaction requires at least one stage")
        if len({id(stage) for stage in frozen_stages}) != len(frozen_stages):
            raise ValueError("CUDA optical transaction stages must be identity-unique")
        _validate_cuda_event(completion_event, device=context.device, name="completion_event")

        frozen_diagnostics: list[tuple[str, object, object]] = []
        seen_event_ids = {id(completion_event)}
        for name, pair in (diagnostic_events or {}).items():
            if not isinstance(name, str) or not name:
                raise ValueError("CUDA diagnostic phase name must be non-empty")
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError(f"CUDA diagnostic phase {name!r} requires a start/end pair")
            start, end = pair
            _validate_cuda_event(start, device=context.device, name=f"{name}.start")
            _validate_cuda_event(end, device=context.device, name=f"{name}.end")
            if id(start) in seen_event_ids or id(end) in seen_event_ids or start is end:
                raise ValueError("CUDA transaction terminal/diagnostic events must be identity-unique")
            seen_event_ids.update((id(start), id(end)))
            frozen_diagnostics.append((name, start, end))

        stage_names: list[str] = []
        stage_evidence: list[object] = []
        for stage in frozen_stages:
            validator = getattr(stage, "_validate_cuda_transaction_stage", None)
            if not callable(validator):
                raise TypeError("CUDA optical transaction stage does not implement validation")
            validator(context)
            stability_check = getattr(stage, "_assert_cuda_transaction_stage_stable", None)
            if callable(stability_check):
                stability_check(context)
            stage_name = getattr(stage, "cuda_transaction_stage_name", None)
            if not isinstance(stage_name, str) or not stage_name:
                raise ValueError("CUDA optical transaction stage name must be non-empty")
            stage_names.append(stage_name)
            stage_evidence.append(_stage_evidence(stage))
        if len(set(stage_names)) != len(stage_names):
            raise ValueError("CUDA optical transaction stage names must be unique")

        controlled_resources = (
            *frozen_stages,
            completion_event,
            *(event for _, start, end in frozen_diagnostics for event in (start, end)),
        )
        if _activation_capability is not None and not isinstance(
            _activation_capability, _CudaOpticalTransactionCapability
        ):
            raise TypeError("controlled CUDA transaction capability is invalid")
        for resource in controlled_resources:
            registered = _controlled_resource_capability(resource)
            if registered is not None and registered is not _activation_capability:
                raise CudaOpticalTimelineError(
                    "CUDA arena-controlled stage/event cannot be rebound to another template"
                )

        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_stages", frozen_stages)
        object.__setattr__(self, "_stage_names", tuple(stage_names))
        object.__setattr__(self, "_stage_evidence", tuple(stage_evidence))
        object.__setattr__(self, "_completion_event", completion_event)
        object.__setattr__(self, "_diagnostic_events", tuple(frozen_diagnostics))
        object.__setattr__(self, "_frozen_signature", self._current_signature())
        object.__setattr__(self, "_sealed", True)
        if _activation_capability is not None:
            _CONTROLLED_TEMPLATE_CAPABILITIES[self] = _activation_capability
            capability_ref = weakref.ref(_activation_capability)
            for resource in controlled_resources:
                _CONTROLLED_RESOURCE_CAPABILITIES[id(resource)] = capability_ref

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CUDA optical transaction template is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CUDA optical transaction template is immutable")
        object.__delattr__(self, name)

    @property
    def context(self) -> CudaOpticalExecutionContext:
        return self._context

    @property
    def stages(self) -> tuple[object, ...]:
        return self._stages

    @property
    def stage_names(self) -> tuple[str, ...]:
        return self._stage_names

    @property
    def stage_evidence(self) -> tuple[object, ...]:
        return self._stage_evidence

    @property
    def completion_event(self) -> object:
        return self._completion_event

    @property
    def diagnostic_events(self) -> tuple[tuple[str, object, object], ...]:
        return self._diagnostic_events

    def validate_bound(self) -> None:
        """Recheck only identity/stability facts, without rebuilding evidence."""

        self._context._validate_identity()
        if self._current_signature() != self._frozen_signature:
            raise CudaOpticalTimelineError("CUDA optical transaction template binding changed")
        for index, stage in enumerate(self._stages):
            stability_check = getattr(stage, "_assert_cuda_transaction_stage_stable", None)
            if callable(stability_check):
                stability_check(self._context)
            if _stage_evidence(stage) != self._stage_evidence[index]:
                raise CudaOpticalTimelineError("CUDA optical transaction stage evidence changed")

    def activate(self) -> "CudaOpticalTransaction":
        """Create fresh single-generation host state from this validated graph."""

        if self in _CONTROLLED_TEMPLATE_CAPABILITIES:
            raise CudaOpticalTimelineError(
                "controlled CUDA transaction template requires its arena-issued capability"
            )
        self.validate_bound()
        return CudaOpticalTransaction(_template=self)

    def _activate_with_capability(self, capability: object) -> "CudaOpticalTransaction":
        required = _CONTROLLED_TEMPLATE_CAPABILITIES.get(self)
        if required is None or capability is not required:
            raise CudaOpticalTimelineError("controlled CUDA transaction activation capability is invalid")
        self.validate_bound()
        return CudaOpticalTransaction(
            _template=self,
            _construction_capability=required,
        )

    def _current_signature(self) -> tuple[object, ...]:
        return (
            id(self._context),
            self._context.device,
            self._context.stream_handle,
            id(self._context.stream),
            tuple(id(stage) for stage in self._stages),
            self._stage_names,
            self._stage_evidence,
            id(self._completion_event),
            tuple((name, id(start), id(end)) for name, start, end in self._diagnostic_events),
        )


class CudaOpticalTransaction:
    """Single-use ordered enqueue state plus one owner-controlled finish."""

    def __init__(
        self,
        *,
        _template: CudaOpticalTransactionTemplate,
        _construction_capability: object | None = None,
    ) -> None:
        required = _CONTROLLED_TEMPLATE_CAPABILITIES.get(_template)
        if required is not None and _construction_capability is not required:
            raise CudaOpticalTimelineError(
                "controlled CUDA transaction construction requires its arena-issued capability"
            )
        self._context = _template.context
        self._stages = _template.stages
        self._stage_names = _template.stage_names
        self._stage_evidence = _template.stage_evidence
        self._completion_event = _template.completion_event
        self._diagnostic_events = _template.diagnostic_events
        self._diagnostic_state = {name: 0 for name, _, _ in self._diagnostic_events}
        self._next_stage = 0
        self._active_stage: object | None = None
        self._state = "validated_open"
        self._poison_reason: str | None = None
        self._completion: CudaOpticalCompletion | None = None
        if required is not None:
            _CONTROLLED_TRANSACTION_CAPABILITIES[self] = required

    @property
    def state(self) -> str:
        return self._state

    @property
    def stage_names(self) -> tuple[str, ...]:
        return self._stage_names

    @property
    def stage_evidence(self) -> tuple[object, ...]:
        return self._stage_evidence

    @property
    def completion(self) -> CudaOpticalCompletion:
        if self._completion is None:
            raise CudaOpticalTimelineError("completion is unavailable before successful finish")
        return self._completion

    @property
    def poison_reason(self) -> str | None:
        return self._poison_reason

    def has_diagnostic_phase(self, name: str) -> bool:
        return name in self._diagnostic_state

    @contextmanager
    def enqueue_stage(self, stage: object) -> Iterator[None]:
        """Enter the exact next registered stage on the selected CUDA stream."""

        self._require_execution_authority(operation="enqueue")
        self._require_enqueueable(stage)
        stability_check = getattr(stage, "_assert_cuda_transaction_stage_stable", None)
        try:
            if callable(stability_check):
                stability_check(self._context)
            if _stage_evidence(stage) != self._stage_evidence[self._next_stage]:
                raise CudaOpticalTimelineError("CUDA optical transaction stage evidence changed")
            self._active_stage = stage
            self._state = "enqueuing"
            with self._context.activate():
                yield
        except BaseException as exc:
            self._poison(f"stage {self._stage_names[self._next_stage]!r} failed: {exc}")
            raise
        else:
            self._next_stage += 1
        finally:
            self._active_stage = None

    def record_diagnostic_start(self, name: str) -> None:
        pair = self._diagnostic_pair(name)
        if pair is None:
            return
        if self._active_stage is None or self._diagnostic_state[name] != 0:
            self._protocol_failure(f"diagnostic phase {name!r} start is out of order")
        try:
            pair[0].record(self._context.stream)
        except BaseException as exc:
            self._poison(f"diagnostic phase {name!r} start record failed: {exc}")
            raise
        self._diagnostic_state[name] = 1

    def record_diagnostic_end(self, name: str) -> None:
        pair = self._diagnostic_pair(name)
        if pair is None:
            return
        if self._active_stage is None or self._diagnostic_state[name] != 1:
            self._protocol_failure(f"diagnostic phase {name!r} end is out of order")
        try:
            pair[1].record(self._context.stream)
        except BaseException as exc:
            self._poison(f"diagnostic phase {name!r} end record failed: {exc}")
            raise
        self._diagnostic_state[name] = 2

    def finish(self) -> CudaOpticalCompletion:
        """Record the one optical terminal event after every registered stage."""

        self._require_execution_authority(operation="finish")
        if self._state == "finished":
            raise CudaOpticalTimelineError("CUDA optical transaction is already finished")
        if self._state == "poisoned":
            raise CudaOpticalTransactionPoisonedError(self._poison_reason or "transaction is poisoned")
        if self._active_stage is not None:
            self._protocol_failure("cannot finish while a transaction stage is active")
        if self._next_stage != len(self._stages):
            self._protocol_failure("cannot finish before all registered stages enqueue")
        incomplete_diagnostics = [name for name, state in self._diagnostic_state.items() if state != 2]
        if incomplete_diagnostics:
            self._protocol_failure(
                f"cannot finish before diagnostic phases end: {sorted(incomplete_diagnostics)!r}"
            )

        try:
            with self._context.activate():
                self._completion_event.record(self._context.stream)
        except BaseException as exc:
            self._poison(f"terminal completion event record failed: {exc}")
            raise

        deferred_profile = None
        if self._diagnostic_events:
            deferred_profile = CudaDeferredProfile(self._diagnostic_events)
        completion = CudaOpticalCompletion(
            event=self._completion_event,
            device=self._context.device,
            stream_handle=self._context.stream_handle,
            stage_names=self._stage_names,
            stage_evidence=self._stage_evidence,
            resource_owners=self._stages,
            deferred_profile=deferred_profile,
        )
        self._completion = completion
        self._state = "finished"
        return completion

    def _bind_arena_execution_lease(
        self,
        construction_capability: object,
        lease: C4ActiveOutputLease,
        *,
        active_owner: object,
        expected_binding: tuple[tuple[str, object], ...],
    ) -> _CudaArenaLiveExecutionAuthority:
        required = _CONTROLLED_TRANSACTION_CAPABILITIES.get(self)
        if required is None or construction_capability is not required:
            raise CudaOpticalTimelineError("CUDA arena transaction construction capability is invalid")
        if self in _CONTROLLED_TRANSACTION_EXECUTION_AUTHORITIES:
            raise CudaOpticalTimelineError("CUDA arena execution authority is already bound")
        if not isinstance(expected_binding, tuple) or any(
            not isinstance(entry, tuple) or len(entry) != 2 or not isinstance(entry[0], str) or not entry[0]
            for entry in expected_binding
        ):
            raise CudaOpticalTimelineError("CUDA arena execution binding is invalid")
        execution_authority = _CudaArenaLiveExecutionAuthority(
            transaction=self,
            lease=lease,
            active_owner=active_owner,
            expected_binding=expected_binding,
            _key=_CUDA_ARENA_EXECUTION_AUTHORITY_KEY,
        )
        _MINTED_CUDA_ARENA_EXECUTION_AUTHORITIES[execution_authority] = (self, lease)
        reference = weakref.ref(execution_authority)
        _CONTROLLED_TRANSACTION_EXECUTION_AUTHORITIES[self] = reference
        return execution_authority

    @contextmanager
    def _authorize_arena_execution(
        self,
        execution_authority: object,
        *,
        operation: str,
    ) -> Iterator[None]:
        if operation not in {"enqueue", "finish"}:
            raise CudaOpticalTimelineError("CUDA arena execution operation is invalid")
        if type(execution_authority) is not _CudaArenaLiveExecutionAuthority:
            raise CudaOpticalTimelineError("CUDA arena execution requires an exact timeline-minted authority")
        reference = _CONTROLLED_TRANSACTION_EXECUTION_AUTHORITIES.get(self)
        if reference is None or reference() is not execution_authority:
            raise CudaOpticalTimelineError("controlled CUDA transaction requires its live C4 lease authority")
        registration = _MINTED_CUDA_ARENA_EXECUTION_AUTHORITIES.get(execution_authority)
        if registration is None or registration[0] is not self:
            raise CudaOpticalTimelineError(
                "CUDA arena execution authority is not registered for this transaction"
            )
        if self in _AUTHORIZED_CONTROLLED_TRANSACTIONS:
            raise CudaOpticalTimelineError("CUDA arena execution authority is already active")
        execution_authority._begin(self, operation=operation)
        active_reference = weakref.ref(execution_authority)
        _AUTHORIZED_CONTROLLED_TRANSACTIONS[self] = (active_reference, operation)
        try:
            yield
        except BaseException as exc:
            try:
                execution_authority._end(self, operation=operation, failure=exc)
            finally:
                _AUTHORIZED_CONTROLLED_TRANSACTIONS.pop(self, None)
            raise
        else:
            try:
                execution_authority._end(self, operation=operation, failure=None)
            finally:
                _AUTHORIZED_CONTROLLED_TRANSACTIONS.pop(self, None)

    def _require_execution_authority(self, *, operation: str) -> None:
        if self not in _CONTROLLED_TRANSACTION_CAPABILITIES:
            return
        authorized = _AUTHORIZED_CONTROLLED_TRANSACTIONS.get(self)
        if authorized is None or authorized[1] != operation:
            raise CudaOpticalTimelineError(
                "controlled CUDA transaction execution requires its active arena lease"
            )
        execution_authority = authorized[0]()
        bound = _CONTROLLED_TRANSACTION_EXECUTION_AUTHORITIES.get(self)
        if execution_authority is None or bound is None or bound() is not execution_authority:
            raise CudaOpticalTimelineError("CUDA arena live execution authority expired")
        if type(execution_authority) is not _CudaArenaLiveExecutionAuthority:
            raise CudaOpticalTimelineError("CUDA arena live execution authority type changed")
        registration = _MINTED_CUDA_ARENA_EXECUTION_AUTHORITIES.get(execution_authority)
        if registration is None or registration[0] is not self:
            raise CudaOpticalTimelineError("CUDA arena live execution authority lost registration")
        execution_authority._assert_execution(self, operation=operation)

    def _require_enqueueable(self, stage: object) -> None:
        if self._state == "finished":
            raise CudaOpticalTimelineError("cannot enqueue after CUDA optical transaction finish")
        if self._state == "poisoned":
            raise CudaOpticalTransactionPoisonedError(self._poison_reason or "transaction is poisoned")
        if self._active_stage is not None:
            self._protocol_failure("CUDA optical transaction stages cannot be nested")
        if self._next_stage >= len(self._stages) or self._stages[self._next_stage] is not stage:
            self._protocol_failure("CUDA optical transaction stage enqueue is out of order")

    def _diagnostic_pair(self, name: str) -> tuple[object, object] | None:
        for phase_name, start, end in self._diagnostic_events:
            if phase_name == name:
                return start, end
        return None

    def _protocol_failure(self, message: str) -> None:
        self._poison(message)
        raise CudaOpticalTransactionPoisonedError(message)

    def _poison(self, reason: str) -> None:
        if self._state != "finished":
            self._state = "poisoned"
            if self._poison_reason is None:
                self._poison_reason = reason


def _validate_cuda_event(event: object, *, device: Any, name: str) -> None:
    if not isinstance(event, torch.cuda.Event):
        raise TypeError(f"{name} must be a Torch CUDA event")
    event_device = getattr(event, "device", None)
    if event_device is None:
        raise ValueError(f"{name} must be initialized on the target CUDA device before transaction")
    if torch.device(event_device) != torch.device(device):
        raise ValueError(f"{name} device must match the CUDA optical transaction device")


def _stage_evidence(stage: object) -> object:
    evidence_factory = getattr(stage, "_cuda_transaction_stage_evidence", None)
    if callable(evidence_factory):
        return evidence_factory()
    return getattr(stage, "cuda_transaction_stage_name")


def _require_torch_cuda() -> None:
    if torch is None:
        raise ImportError("CUDA optical execution requires PyTorch")
    if not torch.cuda.is_available():
        raise ImportError("CUDA optical execution requires Torch CUDA availability")
