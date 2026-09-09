"""Composable Torch-native CUDA execution transactions for optical work."""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

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
_AUTHORIZED_CONTROLLED_TRANSACTIONS: weakref.WeakSet[object] = weakref.WeakSet()
_CONTROLLED_RESOURCE_CAPABILITIES: dict[int, weakref.ReferenceType[object]] = {}


class _CudaOpticalTransactionCapability:
    __slots__ = ("__weakref__",)


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

        self._require_execution_authority()
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

        self._require_execution_authority()
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

    @contextmanager
    def _authorize_arena_execution(self, capability: object) -> Iterator[None]:
        required = _CONTROLLED_TRANSACTION_CAPABILITIES.get(self)
        if required is None or capability is not required:
            raise CudaOpticalTimelineError("CUDA arena execution capability is invalid")
        if self in _AUTHORIZED_CONTROLLED_TRANSACTIONS:
            raise CudaOpticalTimelineError("CUDA arena execution capability is already active")
        _AUTHORIZED_CONTROLLED_TRANSACTIONS.add(self)
        try:
            yield
        finally:
            _AUTHORIZED_CONTROLLED_TRANSACTIONS.discard(self)

    def _require_execution_authority(self) -> None:
        if self in _CONTROLLED_TRANSACTION_CAPABILITIES and self not in _AUTHORIZED_CONTROLLED_TRANSACTIONS:
            raise CudaOpticalTimelineError(
                "controlled CUDA transaction execution requires its active arena lease"
            )

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
