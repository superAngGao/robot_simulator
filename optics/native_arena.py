"""CPU-safe ownership state for reusable native optical output arenas.

This module deliberately owns no benchmark policy and imports no CUDA runtime.
It provides the C4 control-plane contract used later by CUDA, Warp, and B3
resource builders.  Constructing an arena is not result activation: the live
lease type is arena-issued but remains non-result-capable until the separate
C4/B3 activation review.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION = "optics.native-optical-arena-layout.v1"
NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION = "optics.native-optical-arena-spec.v1"
NATIVE_OPTICAL_SLOT_RESOURCE_SCHEMA_VERSION = "optics.native-optical-slot-resource.v1"
NATIVE_OPTICAL_RESOURCE_MANIFEST_SCHEMA_VERSION = "optics.native-optical-arena-resource-manifest.v1"
C4_PREPARED_SAMPLE_PLAN_SCHEMA_VERSION = "q54.p12.4.c4-prepared-sample-plan.v2"
C4_ACTIVE_LEASE_BINDING_SCHEMA_VERSION = "q54.p12.4.c4-active-lease-binding.v2"
C4_ARENA_LIFECYCLE_SCHEMA_VERSION = "q54.p12.4.c4-arena-lifecycle.v2"

_ARENA_STATES = frozenset(
    {
        "OPEN",
        "DRAINING",
        "CLOSED",
        "POISONED",
        "ABORTING_DEVICE",
        "SYNCED_WAITING_EXTERNAL_HOLDS",
        "ABORTED_SYNCED",
        "QUARANTINED_UNTIL_PROCESS_EXIT",
    }
)
_SLOT_STATES = frozenset(
    {
        "FREE",
        "LEASED",
        "ENQUEUING",
        "COMPLETION_RECORDED",
        "PAYLOAD_READY",
        "RECLAIMABLE",
        "RETAINED_POISONED",
    }
)
C4_LIFECYCLE_OPERATION_KINDS = (
    "arena_opened",
    "acquire_backpressure",
    "acquired",
    "cancelled_generation_burned",
    "enqueue_started",
    "completion_recorded",
    "completion_resolved",
    "device_result_hold_released",
    "slot_reclaimable",
    "slot_reclaimed",
    "drain_started",
    "final_drain_complete",
    "arena_closed",
    "arena_poisoned",
)
C4_SHUTDOWN_DISPOSITIONS = (
    "synced_waiting_external_holds",
    "aborted_synced_release_authorized",
    "quarantined_until_process_exit",
)
_LIFECYCLE_OPERATION_KIND_SET = frozenset(C4_LIFECYCLE_OPERATION_KINDS)
_SHUTDOWN_DISPOSITION_SET = frozenset(C4_SHUTDOWN_DISPOSITIONS)
_SUPPORTED_OUTPUT_PROFILES = frozenset({"direct_light_full", "rgb_preview", "render_only"})
_SUPPORTED_DIAGNOSTIC_MODES = frozenset({"primary_unprofiled", "diagnostic_profiled"})
_SUPPORTED_PAYLOAD_POLICIES = frozenset(
    {
        ("native_only", "native_only"),
        ("none", "device_only"),
        ("rgb", "sync_host"),
        ("rgb", "torch_async_ordered"),
        ("full", "sync_host"),
        ("rgb8", "sync_host"),
        ("rgb8", "torch_async_ordered"),
    }
)
_LIVE_AUTHORITY_KEY = object()


class NativeOpticalArenaError(RuntimeError):
    """Base error for invalid native arena ownership or state."""


class NativeOpticalArenaProtocolError(NativeOpticalArenaError):
    """Raised when a lease attempts an illegal state transition."""


class NativeOpticalArenaPoisonedError(NativeOpticalArenaError):
    """Raised after a submitted generation loses safe reuse evidence."""


class NativeOpticalArenaNotActivatedError(NativeOpticalArenaError):
    """Raised for C4/B3 surfaces that are intentionally still closed."""


@dataclass(frozen=True, slots=True)
class NativeOpticalArenaBackpressure:
    """Non-mutating acquire result when the exact next FIFO slot is occupied."""

    arena_token: str
    slot_id: int
    slot_state: str
    active_generation: int
    backpressure_count: int


@dataclass(frozen=True, slots=True)
class NativeOpticalArenaSpec:
    """Immutable logical capacity and policy for one native optical lane."""

    schema_version: str
    row_id: str
    case_id: str
    case_sha256: str
    boundary: str
    lane: str
    device_ordinal: int
    execution_stream_token: str
    input_contract: str
    native_input_manifest_sha256: str
    native_readiness_sha256: str
    scene_contract_sha256: str
    transform_replay_sha256: str | None
    transform_frame_count: int
    fixed_render_fixture_sha256: str | None
    renderer_id: str
    acceleration_class: str
    acceleration_id: str
    width: int
    height: int
    num_rays: int
    primitive_count: int
    node_capacity: int
    output_profile: str
    shadows: bool
    diagnostic_mode: str
    payload: str
    delivery_policy: str
    payload_contract_id: str | None
    ring_depth: int
    ordered_delivery: bool
    arena_layout_schema_version: str

    def __post_init__(self) -> None:
        _validate_arena_spec_payload(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> "NativeOpticalArenaSpec":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("native optical arena spec fields are invalid")
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


@dataclass(frozen=True, slots=True)
class NativeOpticalSlotResourceManifest:
    """Opaque physical-resource identities owned by one stable arena slot."""

    schema_version: str
    slot_id: int
    slot_storage_manifest_sha256: str
    terminal_event_token: str
    resource_tokens: tuple[str, ...]
    resource_owner_tokens: tuple[str, ...]
    resource_nbytes: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_slot_manifest_payload(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> "NativeOpticalSlotResourceManifest":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("native optical slot resource fields are invalid")
        return cls(
            schema_version=payload["schema_version"],
            slot_id=payload["slot_id"],
            slot_storage_manifest_sha256=payload["slot_storage_manifest_sha256"],
            terminal_event_token=payload["terminal_event_token"],
            resource_tokens=tuple(payload["resource_tokens"]),
            resource_owner_tokens=tuple(payload["resource_owner_tokens"]),
            resource_nbytes=tuple(payload["resource_nbytes"]),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "slot_id": self.slot_id,
            "slot_storage_manifest_sha256": self.slot_storage_manifest_sha256,
            "terminal_event_token": self.terminal_event_token,
            "resource_tokens": list(self.resource_tokens),
            "resource_owner_tokens": list(self.resource_owner_tokens),
            "resource_nbytes": list(self.resource_nbytes),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())

    @property
    def total_resource_nbytes(self) -> int:
        return sum(self.resource_nbytes)


@dataclass(frozen=True, slots=True)
class NativeOpticalArenaResourceManifest:
    """Complete session/slot resource inventory retained by an arena."""

    schema_version: str
    arena_spec_sha256: str
    session_resource_tokens: tuple[str, ...]
    session_resource_owner_tokens: tuple[str, ...]
    session_resource_nbytes: tuple[int, ...]
    slots: tuple[NativeOpticalSlotResourceManifest, ...]
    total_resource_nbytes: int

    def __post_init__(self) -> None:
        _validate_resource_manifest(self)

    @classmethod
    def from_payload(cls, payload: object) -> "NativeOpticalArenaResourceManifest":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("native optical arena resource manifest fields are invalid")
        slots_payload = payload["slots"]
        if not isinstance(slots_payload, Sequence) or isinstance(slots_payload, (str, bytes)):
            raise ValueError("native optical arena slots must be an array")
        return cls(
            schema_version=payload["schema_version"],
            arena_spec_sha256=payload["arena_spec_sha256"],
            session_resource_tokens=tuple(payload["session_resource_tokens"]),
            session_resource_owner_tokens=tuple(payload["session_resource_owner_tokens"]),
            session_resource_nbytes=tuple(payload["session_resource_nbytes"]),
            slots=tuple(NativeOpticalSlotResourceManifest.from_payload(value) for value in slots_payload),
            total_resource_nbytes=payload["total_resource_nbytes"],
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "arena_spec_sha256": self.arena_spec_sha256,
            "session_resource_tokens": list(self.session_resource_tokens),
            "session_resource_owner_tokens": list(self.session_resource_owner_tokens),
            "session_resource_nbytes": list(self.session_resource_nbytes),
            "slots": [slot.to_payload() for slot in self.slots],
            "total_resource_nbytes": self.total_resource_nbytes,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


@dataclass(frozen=True, slots=True)
class C4PreparedSamplePlanEvidence:
    """Serialized identity for an untimed, non-active exact slot/frame plan."""

    schema_version: str
    arena_spec_sha256: str
    arena_resource_manifest_sha256: str
    arena_token: str
    ring_depth: int
    slot_id: int
    slot_storage_manifest_sha256: str
    expected_generation: int
    row_id: str
    case_id: str
    boundary: str
    lane: str
    device_ordinal: int
    execution_stream_token: str
    input_contract: str
    native_input_manifest_sha256: str
    native_readiness_sha256: str
    scene_contract_sha256: str
    transform_replay_sha256: str | None
    fixed_render_fixture_sha256: str | None
    renderer_id: str
    acceleration_class: str
    acceleration_id: str
    output_profile: str
    shadows: bool
    diagnostic_mode: str
    payload: str
    delivery_policy: str
    transform_frame_index: int | None
    c3_one_shot_binding_sha256: str
    ordered_stage_evidence_sha256: str
    dataflow_binding_sha256: str
    terminal_event_token: str
    planned_acquisition_offset: int

    def __post_init__(self) -> None:
        _validate_prepared_evidence_payload(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> "C4PreparedSamplePlanEvidence":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("C4 prepared sample plan fields are invalid")
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


@dataclass(frozen=True, slots=True)
class C4ActiveLeaseBindingEvidence:
    """Serialized evidence emitted only after the measured acquire transition."""

    schema_version: str
    arena_spec_sha256: str
    arena_resource_manifest_sha256: str
    arena_token: str
    ring_depth: int
    slot_id: int
    slot_storage_manifest_sha256: str
    generation: int
    lease_token: str
    prepared_slot_plan_sha256: str
    row_id: str
    case_id: str
    boundary: str
    lane: str
    device_ordinal: int
    execution_stream_token: str
    input_contract: str
    native_input_manifest_sha256: str
    native_readiness_sha256: str
    scene_contract_sha256: str
    transform_replay_sha256: str | None
    fixed_render_fixture_sha256: str | None
    renderer_id: str
    acceleration_class: str
    acceleration_id: str
    output_profile: str
    shadows: bool
    diagnostic_mode: str
    payload: str
    delivery_policy: str
    transform_frame_index: int | None
    c3_one_shot_binding_sha256: str
    ordered_stage_evidence_sha256: str
    dataflow_binding_sha256: str
    terminal_event_token: str
    planned_acquisition_offset: int
    result_capable: bool

    def __post_init__(self) -> None:
        _validate_active_evidence_payload(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> "C4ActiveLeaseBindingEvidence":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("C4 active lease binding fields are invalid")
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


@dataclass(frozen=True, slots=True)
class C4ArenaLifecycleTransition:
    """One arena-produced state transition; lanes cannot synthesize it."""

    sequence: int
    operation: str
    arena_state: str
    slot_id: int | None
    generation: int | None
    slot_state: str | None
    transaction_hold_count: int
    device_result_hold_count: int
    delivery_device_hold_count: int
    external_hold_count: int

    @classmethod
    def from_payload(cls, payload: object) -> "C4ArenaLifecycleTransition":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("C4 lifecycle transition fields are invalid")
        value = cls(**{name: payload[name] for name in cls.__dataclass_fields__})
        value._validate()
        return value

    def _validate(self) -> None:
        if not _nonnegative_int(self.sequence):
            raise ValueError("C4 lifecycle transition sequence is invalid")
        if self.operation not in _LIFECYCLE_OPERATION_KIND_SET:
            raise ValueError("C4 lifecycle transition operation is invalid")
        if self.arena_state not in _ARENA_STATES:
            raise ValueError("C4 lifecycle transition arena state is invalid")
        if (self.slot_id is None) != (self.generation is None):
            raise ValueError("C4 lifecycle transition slot/generation must appear together")
        if self.slot_id is None:
            if self.slot_state is not None:
                raise ValueError("arena-only C4 transition cannot carry a slot state")
        else:
            if not _nonnegative_int(self.slot_id) or not _positive_int(self.generation):
                raise ValueError("C4 lifecycle transition slot identity is invalid")
            if self.slot_state not in _SLOT_STATES:
                raise ValueError("C4 lifecycle transition slot state is invalid")
        for name in (
            "transaction_hold_count",
            "device_result_hold_count",
            "delivery_device_hold_count",
            "external_hold_count",
        ):
            if not _nonnegative_int(getattr(self, name)):
                raise ValueError(f"C4 lifecycle transition {name} is invalid")

    def to_payload(self) -> dict[str, object]:
        self._validate()
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class C4ArenaSlotLifecycleSnapshot:
    """Final state of one slot, checked against the replayed transition trace."""

    slot_id: int
    generation: int
    state: str
    terminal_event_record_count: int
    completion_resolved: bool
    payload_ready: bool
    transaction_hold_count: int
    device_result_hold_count: int
    delivery_device_hold_count: int
    external_hold_count: int

    def __post_init__(self) -> None:
        _validate_slot_lifecycle_snapshot(self)

    @classmethod
    def from_payload(cls, payload: object) -> "C4ArenaSlotLifecycleSnapshot":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("C4 arena slot lifecycle snapshot fields are invalid")
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class C4ArenaLifecycleEvidence:
    """Immutable snapshot generated from the arena's transition authority."""

    schema_version: str
    arena_spec_sha256: str
    arena_resource_manifest_sha256: str
    arena_token: str
    arena_state: str
    ring_depth: int
    next_acquire_slot_id: int
    backpressure_count: int
    terminal_event_record_count: int
    completion_resolution_count: int
    slot_reclaim_count: int
    final_drain_complete: bool
    final_reclaim_complete: bool
    external_hold_count: int
    external_hold_count_at_poison: int | None
    external_hold_count_after_device_sync: int | None
    external_hold_discharge_count: int
    device_emergency_sync_attempt_count: int
    device_emergency_sync_succeeded: bool
    device_quiesced: bool
    waiting_external_holds: bool
    external_holds_discharged: bool
    resource_release_authorized: bool
    poisoned_slot_count: int
    retained_pinned_storage_count: int
    shutdown_disposition: str | None
    shutdown_transition_trace_sha256: str | None
    slots: tuple[C4ArenaSlotLifecycleSnapshot, ...]
    transitions: tuple[C4ArenaLifecycleTransition, ...]
    transition_trace_sha256: str

    def __post_init__(self) -> None:
        _validate_lifecycle_evidence(self)

    @classmethod
    def from_payload(cls, payload: object) -> "C4ArenaLifecycleEvidence":
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("C4 arena lifecycle evidence fields are invalid")
        transitions = payload["transitions"]
        slots = payload["slots"]
        if not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes)):
            raise ValueError("C4 lifecycle transitions must be an array")
        if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
            raise ValueError("C4 lifecycle slots must be an array")
        return cls(
            **{
                name: payload[name]
                for name in cls.__dataclass_fields__
                if name not in {"slots", "transitions"}
            },
            slots=tuple(C4ArenaSlotLifecycleSnapshot.from_payload(value) for value in slots),
            transitions=tuple(C4ArenaLifecycleTransition.from_payload(value) for value in transitions),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in {"slots", "transitions"}
            },
            "slots": [slot.to_payload() for slot in self.slots],
            "transitions": [transition.to_payload() for transition in self.transitions],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


@dataclass(slots=True)
class _SlotRecord:
    manifest: NativeOpticalSlotResourceManifest
    generation: int = 0
    state: str = "FREE"
    lease: C4ActiveOutputLease | None = None
    lease_evidence: C4ActiveLeaseBindingEvidence | None = None
    first_enqueue_started: bool = False
    terminal_event_record_count: int = 0
    completion_resolved: bool = False
    payload_ready: bool = False
    transaction_hold_count: int = 0
    device_result_hold_count: int = 0
    delivery_device_hold_count: int = 0
    external_hold_count: int = 0


class C4PreparedSamplePlan:
    """Arena-issued prevalidated plan that cannot itself authorize output."""

    __slots__ = ("_arena_authority", "_evidence", "_live_lease", "_sealed", "__weakref__")

    def __init__(
        self,
        *,
        arena_authority: object,
        evidence: C4PreparedSamplePlanEvidence,
        live_lease: "C4ActiveOutputLease",
        _key: object,
    ) -> None:
        if _key is not _LIVE_AUTHORITY_KEY:
            raise TypeError("C4 prepared plans can only be created by an arena")
        object.__setattr__(self, "_arena_authority", arena_authority)
        object.__setattr__(self, "_evidence", evidence)
        object.__setattr__(self, "_live_lease", live_lease)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("C4 prepared sample plans are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("C4 prepared sample plans are immutable")
        object.__delattr__(self, name)

    @property
    def evidence(self) -> C4PreparedSamplePlanEvidence:
        return self._evidence

    @property
    def result_capable(self) -> bool:
        return False


class C4ActiveOutputLease:
    """Live arena authority for one exact slot generation."""

    __slots__ = ("_arena", "_arena_authority", "_evidence", "_sealed")

    def __init__(
        self,
        *,
        arena: "NativeOpticalArena",
        arena_authority: object,
        evidence: C4ActiveLeaseBindingEvidence,
        _key: object,
    ) -> None:
        if _key is not _LIVE_AUTHORITY_KEY:
            raise TypeError("C4 active leases can only be created by an arena")
        object.__setattr__(self, "_arena", arena)
        object.__setattr__(self, "_arena_authority", arena_authority)
        object.__setattr__(self, "_evidence", evidence)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("C4 active output leases are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("C4 active output leases are immutable")
        object.__delattr__(self, name)

    @property
    def evidence(self) -> C4ActiveLeaseBindingEvidence:
        return self._evidence

    @property
    def result_capable(self) -> bool:
        return False

    @property
    def state(self) -> str:
        return self._arena._lease_state(self)

    def cancel_before_enqueue(self) -> None:
        self._arena._cancel(self)

    def begin_enqueue(self) -> None:
        self._arena._begin_enqueue(self)

    def record_completion(self) -> None:
        self._arena._record_completion(self)

    def resolve_terminal_completion(self) -> None:
        self._arena._resolve_terminal_completion(self)

    def release_device_result(self) -> None:
        self._arena._release_device_result(self)

    def reclaim(self) -> None:
        self._arena._reclaim(self)


class NativeOpticalArena:
    """Sealed FIFO arena control plane with exact slot/generation leases."""

    _IMMUTABLE_CONFIGURATION_FIELDS = frozenset(
        {
            "_lock",
            "_authority",
            "_arena_token",
            "_spec",
            "_spec_snapshot",
            "_resource_manifest",
            "_resource_manifest_snapshot",
            "_resource_owners",
            "_owner_binding_snapshot",
            "_slots",
            "_frozen_signature",
            "_sealed",
        }
    )

    def __init__(
        self,
        *,
        spec: NativeOpticalArenaSpec,
        resource_manifest: NativeOpticalArenaResourceManifest,
        resource_owners: Mapping[str, object],
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        canonical_spec = NativeOpticalArenaSpec.from_payload(spec.to_payload())
        canonical_manifest = NativeOpticalArenaResourceManifest.from_payload(resource_manifest.to_payload())
        if canonical_manifest.arena_spec_sha256 != canonical_spec.sha256:
            raise ValueError("arena resource manifest does not bind the exact arena spec")
        expected_slot_ids = tuple(range(canonical_spec.ring_depth))
        if tuple(slot.slot_id for slot in canonical_manifest.slots) != expected_slot_ids:
            raise ValueError("arena resource manifest must contain every FIFO slot in order")
        required_owner_tokens = set(canonical_manifest.session_resource_owner_tokens)
        for slot in canonical_manifest.slots:
            required_owner_tokens.update(slot.resource_owner_tokens)
        if set(resource_owners) != required_owner_tokens:
            raise ValueError("arena resource owners must match the complete manifest")
        if any(value is None for value in resource_owners.values()):
            raise ValueError("arena resource owners cannot be null")

        self._lock = threading.RLock()
        self._authority = object()
        self._arena_token = f"arena:{uuid.uuid4().hex}"
        self._spec = canonical_spec
        self._spec_snapshot = NativeOpticalArenaSpec.from_payload(canonical_spec.to_payload())
        self._resource_manifest = canonical_manifest
        self._resource_manifest_snapshot = NativeOpticalArenaResourceManifest.from_payload(
            canonical_manifest.to_payload()
        )
        self._resource_owners = MappingProxyType(dict(resource_owners))
        self._owner_binding_snapshot = tuple(
            (token, self._resource_owners[token]) for token in sorted(self._resource_owners)
        )
        self._slots = tuple(_SlotRecord(manifest=slot) for slot in canonical_manifest.slots)
        self._prepared_registry: weakref.WeakKeyDictionary[
            C4PreparedSamplePlan,
            tuple[
                C4PreparedSamplePlanEvidence,
                C4ActiveOutputLease,
                C4ActiveLeaseBindingEvidence,
            ],
        ] = weakref.WeakKeyDictionary()
        self._active_fifo: list[C4ActiveOutputLease] = []
        self._next_acquire_slot_id = 0
        self._state = "OPEN"
        self._transitions: list[C4ArenaLifecycleTransition] = []
        self._backpressure_count = 0
        self._completion_resolution_count = 0
        self._slot_reclaim_count = 0
        self._final_drain_complete = False
        self._final_reclaim_complete = False
        self._poison_reason: str | None = None
        self._external_hold_count_at_poison: int | None = None
        self._frozen_signature = self._current_signature()
        self._record_transition("arena_opened")
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False) and name in self._IMMUTABLE_CONFIGURATION_FIELDS:
            raise AttributeError("native optical arena configuration is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False) and name in self._IMMUTABLE_CONFIGURATION_FIELDS:
            raise AttributeError("native optical arena configuration is immutable")
        object.__delattr__(self, name)

    @property
    def spec(self) -> NativeOpticalArenaSpec:
        return self._spec

    @property
    def resource_manifest(self) -> NativeOpticalArenaResourceManifest:
        return self._resource_manifest

    @property
    def arena_token(self) -> str:
        return self._arena_token

    @property
    def state(self) -> str:
        return self._state

    @property
    def poison_reason(self) -> str | None:
        return self._poison_reason

    def prepare_plan(
        self,
        *,
        transform_frame_index: int | None,
        c3_one_shot_binding_sha256: str,
        ordered_stage_evidence_sha256: str,
        dataflow_binding_sha256: str,
        planned_acquisition_offset: int = 0,
    ) -> C4PreparedSamplePlan:
        """Build an immutable next-slot plan outside measured acquisition."""

        with self._lock:
            self._require_open_for_prepare()
            self._validate_frozen()
            _require_sha256(c3_one_shot_binding_sha256, "C3 one-shot binding digest")
            _require_sha256(ordered_stage_evidence_sha256, "ordered stage evidence digest")
            _require_sha256(dataflow_binding_sha256, "dataflow binding digest")
            self._validate_frame_index(transform_frame_index)
            if not _nonnegative_int(planned_acquisition_offset):
                raise ValueError("planned acquisition offset must be non-negative")
            planned_position = self._next_acquire_slot_id + planned_acquisition_offset
            slot = self._slots[planned_position % self._spec.ring_depth]
            expected_generation = slot.generation + 1 + planned_position // self._spec.ring_depth
            prepared_evidence = C4PreparedSamplePlanEvidence(
                schema_version=C4_PREPARED_SAMPLE_PLAN_SCHEMA_VERSION,
                arena_spec_sha256=self._spec.sha256,
                arena_resource_manifest_sha256=self._resource_manifest.sha256,
                arena_token=self._arena_token,
                ring_depth=self._spec.ring_depth,
                slot_id=slot.manifest.slot_id,
                slot_storage_manifest_sha256=slot.manifest.slot_storage_manifest_sha256,
                expected_generation=expected_generation,
                row_id=self._spec.row_id,
                case_id=self._spec.case_id,
                boundary=self._spec.boundary,
                lane=self._spec.lane,
                device_ordinal=self._spec.device_ordinal,
                execution_stream_token=self._spec.execution_stream_token,
                input_contract=self._spec.input_contract,
                native_input_manifest_sha256=self._spec.native_input_manifest_sha256,
                native_readiness_sha256=self._spec.native_readiness_sha256,
                scene_contract_sha256=self._spec.scene_contract_sha256,
                transform_replay_sha256=self._spec.transform_replay_sha256,
                fixed_render_fixture_sha256=self._spec.fixed_render_fixture_sha256,
                renderer_id=self._spec.renderer_id,
                acceleration_class=self._spec.acceleration_class,
                acceleration_id=self._spec.acceleration_id,
                output_profile=self._spec.output_profile,
                shadows=self._spec.shadows,
                diagnostic_mode=self._spec.diagnostic_mode,
                payload=self._spec.payload,
                delivery_policy=self._spec.delivery_policy,
                transform_frame_index=transform_frame_index,
                c3_one_shot_binding_sha256=c3_one_shot_binding_sha256,
                ordered_stage_evidence_sha256=ordered_stage_evidence_sha256,
                dataflow_binding_sha256=dataflow_binding_sha256,
                terminal_event_token=slot.manifest.terminal_event_token,
                planned_acquisition_offset=planned_acquisition_offset,
            )
            lease_token = _canonical_sha256(
                {
                    "arena_token": self._arena_token,
                    "slot_id": slot.manifest.slot_id,
                    "generation": expected_generation,
                    "prepared_slot_plan_sha256": prepared_evidence.sha256,
                    "nonce": uuid.uuid4().hex,
                }
            )
            active_evidence = C4ActiveLeaseBindingEvidence(
                schema_version=C4_ACTIVE_LEASE_BINDING_SCHEMA_VERSION,
                arena_spec_sha256=prepared_evidence.arena_spec_sha256,
                arena_resource_manifest_sha256=(prepared_evidence.arena_resource_manifest_sha256),
                arena_token=self._arena_token,
                ring_depth=self._spec.ring_depth,
                slot_id=slot.manifest.slot_id,
                slot_storage_manifest_sha256=slot.manifest.slot_storage_manifest_sha256,
                generation=expected_generation,
                lease_token=lease_token,
                prepared_slot_plan_sha256=prepared_evidence.sha256,
                row_id=self._spec.row_id,
                case_id=self._spec.case_id,
                boundary=self._spec.boundary,
                lane=self._spec.lane,
                device_ordinal=self._spec.device_ordinal,
                execution_stream_token=self._spec.execution_stream_token,
                input_contract=self._spec.input_contract,
                native_input_manifest_sha256=self._spec.native_input_manifest_sha256,
                native_readiness_sha256=self._spec.native_readiness_sha256,
                scene_contract_sha256=self._spec.scene_contract_sha256,
                transform_replay_sha256=self._spec.transform_replay_sha256,
                fixed_render_fixture_sha256=self._spec.fixed_render_fixture_sha256,
                renderer_id=self._spec.renderer_id,
                acceleration_class=self._spec.acceleration_class,
                acceleration_id=self._spec.acceleration_id,
                output_profile=self._spec.output_profile,
                shadows=self._spec.shadows,
                diagnostic_mode=self._spec.diagnostic_mode,
                payload=self._spec.payload,
                delivery_policy=self._spec.delivery_policy,
                transform_frame_index=transform_frame_index,
                c3_one_shot_binding_sha256=c3_one_shot_binding_sha256,
                ordered_stage_evidence_sha256=ordered_stage_evidence_sha256,
                dataflow_binding_sha256=dataflow_binding_sha256,
                terminal_event_token=slot.manifest.terminal_event_token,
                planned_acquisition_offset=planned_acquisition_offset,
                result_capable=False,
            )
            lease = C4ActiveOutputLease(
                arena=self,
                arena_authority=self._authority,
                evidence=active_evidence,
                _key=_LIVE_AUTHORITY_KEY,
            )
            plan = C4PreparedSamplePlan(
                arena_authority=self._authority,
                evidence=prepared_evidence,
                live_lease=lease,
                _key=_LIVE_AUTHORITY_KEY,
            )
            self._prepared_registry[plan] = (
                prepared_evidence,
                lease,
                active_evidence,
            )
            return plan

    def acquire(self, plan: C4PreparedSamplePlan) -> C4ActiveOutputLease | NativeOpticalArenaBackpressure:
        """Perform measured O(1) identity/state activation for an exact plan."""

        with self._lock:
            self._require_arena_state("OPEN", operation="acquire")
            self._validate_frozen_lightweight()
            if not isinstance(plan, C4PreparedSamplePlan):
                raise NativeOpticalArenaProtocolError("prepared plan is not owned by this arena")
            registered = self._prepared_registry.get(plan)
            if registered is None or plan._arena_authority is not self._authority:
                raise NativeOpticalArenaProtocolError("prepared plan is not owned by this arena")
            if plan.evidence != registered[0]:
                raise NativeOpticalArenaProtocolError("prepared plan evidence changed after seal")
            if plan._live_lease is not registered[1] or plan._live_lease.evidence != registered[2]:
                raise NativeOpticalArenaProtocolError("prepared plan live lease changed after seal")
            evidence = plan.evidence
            slot = self._slots[self._next_acquire_slot_id]
            self._validate_plan_for_slot(evidence, slot)
            if slot.state != "FREE":
                self._backpressure_count += 1
                self._record_transition("acquire_backpressure", slot=slot)
                return NativeOpticalArenaBackpressure(
                    arena_token=self._arena_token,
                    slot_id=slot.manifest.slot_id,
                    slot_state=slot.state,
                    active_generation=slot.generation,
                    backpressure_count=self._backpressure_count,
                )

            lease = plan._live_lease
            if lease._arena_authority is not self._authority:
                raise NativeOpticalArenaProtocolError("prepared plan live lease authority changed")
            if lease.evidence.prepared_slot_plan_sha256 != evidence.sha256:
                raise NativeOpticalArenaProtocolError("prepared plan live lease binding changed")
            slot.generation = evidence.expected_generation
            slot.state = "LEASED"
            slot.lease = lease
            slot.lease_evidence = registered[2]
            slot.first_enqueue_started = False
            slot.terminal_event_record_count = 0
            slot.completion_resolved = False
            slot.payload_ready = False
            slot.transaction_hold_count = 1
            slot.device_result_hold_count = 1
            slot.delivery_device_hold_count = int(self._spec.payload != "native_only")
            slot.external_hold_count = 0
            self._active_fifo.append(lease)
            self._next_acquire_slot_id = (slot.manifest.slot_id + 1) % self._spec.ring_depth
            del self._prepared_registry[plan]
            self._record_transition("acquired", slot=slot)
            return lease

    def begin_drain(self) -> None:
        with self._lock:
            self._require_arena_state("OPEN", operation="begin_drain")
            self._prepared_registry.clear()
            self._state = "DRAINING"
            self._record_transition("drain_started")
            self._maybe_complete_drain()

    def close(self) -> None:
        with self._lock:
            if self._state == "CLOSED":
                return
            if self._state not in {"OPEN", "DRAINING"}:
                raise NativeOpticalArenaProtocolError(
                    "normal close cannot release a poisoned or aborted arena"
                )
            if self._active_fifo or any(slot.state != "FREE" for slot in self._slots):
                raise NativeOpticalArenaProtocolError("normal close requires every arena slot to be free")
            if self._state == "DRAINING":
                self._final_drain_complete = True
                self._final_reclaim_complete = True
            self._prepared_registry.clear()
            self._state = "CLOSED"
            self._record_transition("arena_closed")

    def lifecycle_evidence(self) -> C4ArenaLifecycleEvidence:
        with self._lock:
            transitions = tuple(self._transitions)
            external_hold_count = sum(slot.external_hold_count for slot in self._slots)
            return C4ArenaLifecycleEvidence(
                schema_version=C4_ARENA_LIFECYCLE_SCHEMA_VERSION,
                arena_spec_sha256=self._spec.sha256,
                arena_resource_manifest_sha256=self._resource_manifest.sha256,
                arena_token=self._arena_token,
                arena_state=self._state,
                ring_depth=self._spec.ring_depth,
                next_acquire_slot_id=self._next_acquire_slot_id,
                backpressure_count=self._backpressure_count,
                terminal_event_record_count=sum(
                    transition.operation == "completion_recorded" for transition in transitions
                ),
                completion_resolution_count=self._completion_resolution_count,
                slot_reclaim_count=self._slot_reclaim_count,
                final_drain_complete=self._final_drain_complete,
                final_reclaim_complete=self._final_reclaim_complete,
                external_hold_count=external_hold_count,
                external_hold_count_at_poison=self._external_hold_count_at_poison,
                external_hold_count_after_device_sync=None,
                external_hold_discharge_count=0,
                device_emergency_sync_attempt_count=0,
                device_emergency_sync_succeeded=False,
                device_quiesced=False,
                waiting_external_holds=False,
                external_holds_discharged=external_hold_count == 0,
                resource_release_authorized=False,
                poisoned_slot_count=sum(slot.state == "RETAINED_POISONED" for slot in self._slots),
                retained_pinned_storage_count=0,
                shutdown_disposition=None,
                shutdown_transition_trace_sha256=None,
                slots=tuple(_slot_lifecycle_snapshot(slot) for slot in self._slots),
                transitions=transitions,
                transition_trace_sha256=_canonical_sha256(
                    [transition.to_payload() for transition in transitions]
                ),
            )

    def _lease_state(self, lease: C4ActiveOutputLease) -> str:
        with self._lock:
            return self._slot_for_live_lease(lease).state

    def _cancel(self, lease: C4ActiveOutputLease) -> None:
        with self._lock:
            slot = self._slot_for_live_lease(lease)
            if slot.state != "LEASED" or slot.first_enqueue_started:
                self._protocol_failure(slot, "only a pre-enqueue lease can be cancelled")
            slot.transaction_hold_count = 0
            slot.device_result_hold_count = 0
            slot.delivery_device_hold_count = 0
            slot.state = "FREE"
            slot.lease = None
            slot.lease_evidence = None
            self._active_fifo.remove(lease)
            self._record_transition("cancelled_generation_burned", slot=slot)
            self._maybe_complete_drain()

    def _begin_enqueue(self, lease: C4ActiveOutputLease) -> None:
        with self._lock:
            slot = self._slot_for_live_lease(lease)
            self._require_running_arena(operation="begin_enqueue")
            if slot.state != "LEASED":
                self._protocol_failure(slot, "generation can begin enqueue exactly once")
            for earlier in self._active_fifo:
                if earlier is lease:
                    break
                earlier_slot = self._slot_for_live_lease(earlier)
                if earlier_slot.state in {"LEASED", "ENQUEUING"}:
                    self._protocol_failure(slot, "generation enqueue is out of FIFO order")
            slot.first_enqueue_started = True
            slot.state = "ENQUEUING"
            self._record_transition("enqueue_started", slot=slot)

    def _record_completion(self, lease: C4ActiveOutputLease) -> None:
        with self._lock:
            slot = self._slot_for_live_lease(lease)
            self._require_running_arena(operation="record_completion")
            if slot.state != "ENQUEUING" or slot.terminal_event_record_count != 0:
                self._protocol_failure(slot, "terminal completion must be recorded exactly once")
            slot.terminal_event_record_count = 1
            slot.state = "COMPLETION_RECORDED"
            self._record_transition("completion_recorded", slot=slot)

    def _resolve_terminal_completion(self, lease: C4ActiveOutputLease) -> None:
        with self._lock:
            slot = self._slot_for_live_lease(lease)
            self._require_running_arena(operation="resolve_terminal_completion")
            self._require_fifo_head(lease, operation="completion resolution")
            if slot.state != "COMPLETION_RECORDED" or slot.completion_resolved:
                self._protocol_failure(slot, "terminal completion resolution is out of order")
            slot.completion_resolved = True
            slot.transaction_hold_count = 0
            self._completion_resolution_count += 1
            if self._spec.payload in {"native_only", "none"}:
                slot.payload_ready = True
                slot.delivery_device_hold_count = 0
                slot.state = "PAYLOAD_READY"
            self._record_transition("completion_resolved", slot=slot)
            self._update_reclaimable(slot)

    def _release_device_result(self, lease: C4ActiveOutputLease) -> None:
        with self._lock:
            slot = self._slot_for_live_lease(lease)
            self._require_running_arena(operation="release_device_result")
            if not slot.completion_resolved or slot.device_result_hold_count != 1:
                self._protocol_failure(slot, "device-result hold release is out of order")
            slot.device_result_hold_count = 0
            self._record_transition("device_result_hold_released", slot=slot)
            self._update_reclaimable(slot)

    def _reclaim(self, lease: C4ActiveOutputLease) -> None:
        with self._lock:
            slot = self._slot_for_live_lease(lease)
            self._require_running_arena(operation="reclaim")
            self._require_fifo_head(lease, operation="reclaim")
            if slot.state != "RECLAIMABLE":
                self._protocol_failure(slot, "slot is not logically reclaimable")
            slot.state = "FREE"
            slot.lease = None
            slot.lease_evidence = None
            self._active_fifo.pop(0)
            self._slot_reclaim_count += 1
            self._record_transition("slot_reclaimed", slot=slot)
            self._maybe_complete_drain()

    def _update_reclaimable(self, slot: _SlotRecord) -> None:
        if (
            slot.payload_ready
            and slot.transaction_hold_count == 0
            and slot.device_result_hold_count == 0
            and slot.delivery_device_hold_count == 0
            and slot.external_hold_count == 0
        ):
            slot.state = "RECLAIMABLE"
            self._record_transition("slot_reclaimable", slot=slot)

    def _slot_for_live_lease(self, lease: C4ActiveOutputLease) -> _SlotRecord:
        if not isinstance(lease, C4ActiveOutputLease):
            raise NativeOpticalArenaProtocolError("operation requires a live C4 lease")
        if lease._arena is not self or lease._arena_authority is not self._authority:
            raise NativeOpticalArenaProtocolError("C4 lease belongs to another arena")
        evidence = lease.evidence
        if evidence.result_capable:
            raise NativeOpticalArenaProtocolError("C4a lease cannot claim result capability")
        if evidence.slot_id >= len(self._slots):
            raise NativeOpticalArenaProtocolError("C4 lease slot is outside this arena")
        slot = self._slots[evidence.slot_id]
        if slot.lease is not lease or slot.generation != evidence.generation:
            raise NativeOpticalArenaProtocolError("C4 lease is stale or not active")
        if slot.lease_evidence is None or evidence != slot.lease_evidence:
            raise NativeOpticalArenaProtocolError("C4 live lease evidence changed after activation")
        return slot

    def _validate_plan_for_slot(self, evidence: C4PreparedSamplePlanEvidence, slot: _SlotRecord) -> None:
        expected = {
            "arena_spec_sha256": self._spec.sha256,
            "arena_resource_manifest_sha256": self._resource_manifest.sha256,
            "arena_token": self._arena_token,
            "ring_depth": self._spec.ring_depth,
            "slot_id": slot.manifest.slot_id,
            "slot_storage_manifest_sha256": slot.manifest.slot_storage_manifest_sha256,
            "expected_generation": slot.generation + 1,
            "row_id": self._spec.row_id,
            "case_id": self._spec.case_id,
            "boundary": self._spec.boundary,
            "lane": self._spec.lane,
            "device_ordinal": self._spec.device_ordinal,
            "execution_stream_token": self._spec.execution_stream_token,
            "input_contract": self._spec.input_contract,
            "native_input_manifest_sha256": self._spec.native_input_manifest_sha256,
            "native_readiness_sha256": self._spec.native_readiness_sha256,
            "scene_contract_sha256": self._spec.scene_contract_sha256,
            "transform_replay_sha256": self._spec.transform_replay_sha256,
            "fixed_render_fixture_sha256": self._spec.fixed_render_fixture_sha256,
            "renderer_id": self._spec.renderer_id,
            "acceleration_class": self._spec.acceleration_class,
            "acceleration_id": self._spec.acceleration_id,
            "output_profile": self._spec.output_profile,
            "shadows": self._spec.shadows,
            "diagnostic_mode": self._spec.diagnostic_mode,
            "payload": self._spec.payload,
            "delivery_policy": self._spec.delivery_policy,
            "terminal_event_token": slot.manifest.terminal_event_token,
        }
        for name, value in expected.items():
            if getattr(evidence, name) != value:
                raise NativeOpticalArenaProtocolError(f"prepared plan {name} is stale or not arena-owned")
        self._validate_frame_index(evidence.transform_frame_index)

    def _validate_frame_index(self, frame_index: int | None) -> None:
        if self._spec.boundary == "R0":
            if frame_index is not None:
                raise ValueError("R0 arena plans cannot select a replay frame")
            return
        if not _nonnegative_int(frame_index) or frame_index >= self._spec.transform_frame_count:
            raise ValueError("dynamic arena plan replay frame index is invalid")

    def _require_fifo_head(self, lease: C4ActiveOutputLease, *, operation: str) -> None:
        if not self._active_fifo or self._active_fifo[0] is not lease:
            slot = self._slot_for_live_lease(lease)
            self._protocol_failure(slot, f"{operation} is out of FIFO order")

    def _require_open_for_prepare(self) -> None:
        if self._state != "OPEN":
            raise NativeOpticalArenaProtocolError("prepared plans require an open arena")

    def _require_running_arena(self, *, operation: str) -> None:
        if self._state in {"POISONED", "ABORTING_DEVICE", "SYNCED_WAITING_EXTERNAL_HOLDS"}:
            raise NativeOpticalArenaPoisonedError(
                self._poison_reason or f"cannot {operation} after arena poison"
            )
        if self._state not in {"OPEN", "DRAINING"}:
            raise NativeOpticalArenaProtocolError(f"cannot {operation} while arena state is {self._state}")

    def _require_arena_state(self, expected: str, *, operation: str) -> None:
        if self._state != expected:
            if self._state == "POISONED":
                raise NativeOpticalArenaPoisonedError(self._poison_reason or "arena is poisoned")
            raise NativeOpticalArenaProtocolError(f"cannot {operation} while arena state is {self._state}")

    def _protocol_failure(self, slot: _SlotRecord, message: str) -> None:
        if slot.first_enqueue_started and self._state in {"OPEN", "DRAINING"}:
            self._poison_reason = message
            self._external_hold_count_at_poison = sum(
                active_slot.external_hold_count for active_slot in self._slots
            )
            self._state = "POISONED"
            for active_slot in self._slots:
                if active_slot.state != "FREE":
                    active_slot.state = "RETAINED_POISONED"
            self._record_transition("arena_poisoned", slot=slot)
            raise NativeOpticalArenaPoisonedError(message)
        raise NativeOpticalArenaProtocolError(message)

    def _maybe_complete_drain(self) -> None:
        if self._state == "DRAINING" and not self._active_fifo:
            self._final_drain_complete = True
            self._final_reclaim_complete = True
            self._record_transition("final_drain_complete")

    def _record_transition(self, operation: str, *, slot: _SlotRecord | None = None) -> None:
        transition = C4ArenaLifecycleTransition(
            sequence=len(self._transitions),
            operation=operation,
            arena_state=self._state,
            slot_id=None if slot is None else slot.manifest.slot_id,
            generation=None if slot is None else slot.generation,
            slot_state=None if slot is None else slot.state,
            transaction_hold_count=0 if slot is None else slot.transaction_hold_count,
            device_result_hold_count=0 if slot is None else slot.device_result_hold_count,
            delivery_device_hold_count=0 if slot is None else slot.delivery_device_hold_count,
            external_hold_count=0 if slot is None else slot.external_hold_count,
        )
        transition._validate()
        self._transitions.append(transition)

    def _current_signature(self) -> str:
        return _canonical_sha256(
            {
                "spec": self._spec.to_payload(),
                "resource_manifest": self._resource_manifest.to_payload(),
                "arena_token": self._arena_token,
                "owner_tokens": sorted(self._resource_owners),
                "owner_bindings": [
                    [token, id(self._resource_owners[token])] for token in sorted(self._resource_owners)
                ],
            }
        )

    def _validate_frozen(self) -> None:
        if self._current_signature() != self._frozen_signature:
            raise NativeOpticalArenaProtocolError("sealed arena specification/resources changed")

    def _validate_frozen_lightweight(self) -> None:
        """Check the prevalidated immutable graph without rebuilding a manifest."""

        if self._spec != self._spec_snapshot or self._resource_manifest != self._resource_manifest_snapshot:
            raise NativeOpticalArenaProtocolError("sealed arena specification/resources changed")
        if len(self._resource_owners) != len(self._owner_binding_snapshot):
            raise NativeOpticalArenaProtocolError("sealed arena resource owners changed")
        for token, owner in self._owner_binding_snapshot:
            if self._resource_owners.get(token) is not owner:
                raise NativeOpticalArenaProtocolError("sealed arena resource owners changed")


def _validate_arena_spec_payload(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION:
        raise ValueError("native optical arena spec schema version is invalid")
    if payload.get("arena_layout_schema_version") != NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION:
        raise ValueError("native optical arena layout schema version is invalid")
    for name in (
        "row_id",
        "case_id",
        "boundary",
        "lane",
        "execution_stream_token",
        "input_contract",
        "renderer_id",
        "acceleration_class",
        "acceleration_id",
        "output_profile",
        "diagnostic_mode",
        "payload",
        "delivery_policy",
    ):
        _require_token(payload.get(name), f"native optical arena spec {name}")
    for name in (
        "case_sha256",
        "native_input_manifest_sha256",
        "native_readiness_sha256",
        "scene_contract_sha256",
    ):
        _require_sha256(payload.get(name), f"native optical arena spec {name}")
    for name in ("transform_replay_sha256", "fixed_render_fixture_sha256"):
        value = payload.get(name)
        if value is not None:
            _require_sha256(value, f"native optical arena spec {name}")
    if payload.get("boundary") not in {"R0", "R1", "R2", "S"}:
        raise ValueError("native optical arena boundary is invalid")
    if payload.get("lane") not in {"warp", "cuda"}:
        raise ValueError("native optical arena lane is invalid")
    if payload.get("output_profile") not in _SUPPORTED_OUTPUT_PROFILES:
        raise ValueError("native optical arena output profile is invalid")
    if payload.get("diagnostic_mode") not in _SUPPORTED_DIAGNOSTIC_MODES:
        raise ValueError("native optical arena diagnostic mode is invalid")
    if not isinstance(payload.get("shadows"), bool):
        raise ValueError("native optical arena shadows must be boolean")
    for name in ("device_ordinal", "transform_frame_count", "primitive_count", "node_capacity"):
        if not _nonnegative_int(payload.get(name)):
            raise ValueError(f"native optical arena spec {name} must be non-negative")
    for name in ("width", "height", "num_rays", "ring_depth"):
        if not _positive_int(payload.get(name)):
            raise ValueError(f"native optical arena spec {name} must be positive")
    if payload["num_rays"] != payload["width"] * payload["height"]:
        raise ValueError("native optical arena num_rays must equal width * height")
    if payload["ring_depth"] not in {1, 2}:
        raise ValueError("native optical arena ring depth must be one or two")
    if not isinstance(payload.get("ordered_delivery"), bool):
        raise ValueError("native optical arena ordered_delivery must be boolean")
    if payload.get("ordered_delivery") is not True:
        raise ValueError("native optical arena delivery/reclaim order must be FIFO")
    pair = (payload.get("payload"), payload.get("delivery_policy"))
    if pair not in _SUPPORTED_PAYLOAD_POLICIES:
        raise ValueError("native optical arena payload/delivery policy is invalid")
    payload_contract = payload.get("payload_contract_id")
    if pair == ("native_only", "native_only"):
        if payload_contract is not None or payload["ring_depth"] != 1:
            raise ValueError("native-only arena requires no payload contract and depth one")
    else:
        _require_token(payload_contract, "native optical arena payload contract")
    if payload.get("delivery_policy") == "torch_async_ordered":
        if payload["ring_depth"] != 2 or payload.get("ordered_delivery") is not True:
            raise ValueError("ordered async arena requires ring depth two")
    elif payload["ring_depth"] != 1:
        raise ValueError("non-async native arena requires ring depth one")
    if payload.get("boundary") == "R0":
        if payload.get("input_contract") != "fixed_native_scene_v1":
            raise ValueError("R0 arena input contract is invalid")
        if payload.get("fixed_render_fixture_sha256") is None:
            raise ValueError("R0 arena requires fixed render fixture provenance")
        if payload.get("transform_replay_sha256") is not None or payload["transform_frame_count"] != 0:
            raise ValueError("R0 arena forbids transform replay provenance")
    elif payload.get("boundary") in {"R1", "R2"}:
        if payload.get("input_contract") != "canonical_native_replay_v1":
            raise ValueError("R1/R2 arena input contract is invalid")
        if payload.get("transform_replay_sha256") is None or payload["transform_frame_count"] <= 0:
            raise ValueError("R1/R2 arena requires transform replay provenance")
        if payload.get("fixed_render_fixture_sha256") is not None:
            raise ValueError("R1/R2 arena forbids fixed render fixture provenance")
    else:
        if payload.get("input_contract") != "gpu_engine_published_frame_v1":
            raise ValueError("shipping arena input contract is invalid")
        if (
            payload.get("transform_replay_sha256") is not None
            or payload["transform_frame_count"] != 0
            or payload.get("fixed_render_fixture_sha256") is not None
        ):
            raise ValueError("shipping arena forbids replay and fixed fixture provenance")
    if payload.get("boundary") in {"R0", "R1"} and pair != (
        "native_only",
        "native_only",
    ):
        raise ValueError("R0/R1 arenas require native-only payload policy")
    if payload.get("boundary") in {"R2", "S"} and pair == (
        "native_only",
        "native_only",
    ):
        raise ValueError("R2/S arenas require an explicit delivery policy")


def _validate_slot_manifest_payload(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != NATIVE_OPTICAL_SLOT_RESOURCE_SCHEMA_VERSION:
        raise ValueError("native optical slot resource schema version is invalid")
    if not _nonnegative_int(payload.get("slot_id")):
        raise ValueError("native optical slot ID must be non-negative")
    _require_sha256(payload.get("slot_storage_manifest_sha256"), "slot storage manifest")
    _require_token(payload.get("terminal_event_token"), "terminal event token")
    resource_tokens = payload.get("resource_tokens")
    owner_tokens = payload.get("resource_owner_tokens")
    resource_nbytes = payload.get("resource_nbytes")
    for name, values in (
        ("resource_tokens", resource_tokens),
        ("resource_owner_tokens", owner_tokens),
    ):
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise ValueError(f"native optical slot {name} must be a non-empty array")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"native optical slot {name} must contain non-empty strings")
    if len(set(resource_tokens)) != len(resource_tokens):
        raise ValueError("native optical slot resource tokens must be unique")
    if not isinstance(resource_nbytes, Sequence) or isinstance(resource_nbytes, (str, bytes)):
        raise ValueError("native optical slot resource bytes must be an array")
    if any(not _nonnegative_int(value) for value in resource_nbytes):
        raise ValueError("native optical slot resource bytes must be non-negative integers")
    if len(resource_tokens) != len(owner_tokens) or len(resource_tokens) != len(resource_nbytes):
        raise ValueError("native optical slot resource, owner, and byte arrays must align")
    if payload["terminal_event_token"] not in resource_tokens:
        raise ValueError("arena terminal event must be listed as a slot resource")


def _validate_resource_manifest(manifest: NativeOpticalArenaResourceManifest) -> None:
    if manifest.schema_version != NATIVE_OPTICAL_RESOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("native optical arena resource manifest schema version is invalid")
    _require_sha256(manifest.arena_spec_sha256, "arena resource spec digest")
    if not manifest.slots:
        raise ValueError("native optical arena resource manifest requires slots")
    for name, values in (
        ("session_resource_tokens", manifest.session_resource_tokens),
        ("session_resource_owner_tokens", manifest.session_resource_owner_tokens),
    ):
        if not values or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"arena {name} must contain non-empty strings")
    if len(set(manifest.session_resource_tokens)) != len(manifest.session_resource_tokens):
        raise ValueError("arena session resource tokens must be unique")
    if any(not _nonnegative_int(value) for value in manifest.session_resource_nbytes):
        raise ValueError("arena session resource bytes must be non-negative integers")
    if not (
        len(manifest.session_resource_tokens)
        == len(manifest.session_resource_owner_tokens)
        == len(manifest.session_resource_nbytes)
    ):
        raise ValueError("arena session resource, owner, and byte arrays must align")
    all_resources = list(manifest.session_resource_tokens)
    event_tokens: list[str] = []
    for slot in manifest.slots:
        all_resources.extend(slot.resource_tokens)
        event_tokens.append(slot.terminal_event_token)
    if len(set(all_resources)) != len(all_resources):
        raise ValueError("arena resource tokens must be globally unique")
    if len(set(event_tokens)) != len(event_tokens):
        raise ValueError("arena terminal event tokens must be unique per slot")
    if not _nonnegative_int(manifest.total_resource_nbytes):
        raise ValueError("arena total resource bytes must be non-negative")
    if manifest.total_resource_nbytes != sum(manifest.session_resource_nbytes) + sum(
        slot.total_resource_nbytes for slot in manifest.slots
    ):
        raise ValueError("arena total resource bytes do not match slot manifests")


def _validate_prepared_evidence_payload(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != C4_PREPARED_SAMPLE_PLAN_SCHEMA_VERSION:
        raise ValueError("C4 prepared sample plan schema version is invalid")
    for name in (
        "arena_spec_sha256",
        "arena_resource_manifest_sha256",
        "slot_storage_manifest_sha256",
        "c3_one_shot_binding_sha256",
        "ordered_stage_evidence_sha256",
        "dataflow_binding_sha256",
        "native_input_manifest_sha256",
        "native_readiness_sha256",
        "scene_contract_sha256",
    ):
        _require_sha256(payload.get(name), f"C4 prepared sample plan {name}")
    for name in ("transform_replay_sha256", "fixed_render_fixture_sha256"):
        value = payload.get(name)
        if value is not None:
            _require_sha256(value, f"C4 prepared sample plan {name}")
    for name in (
        "arena_token",
        "row_id",
        "case_id",
        "boundary",
        "lane",
        "execution_stream_token",
        "input_contract",
        "renderer_id",
        "acceleration_class",
        "acceleration_id",
        "output_profile",
        "diagnostic_mode",
        "payload",
        "delivery_policy",
        "terminal_event_token",
    ):
        _require_token(payload.get(name), f"C4 prepared sample plan {name}")
    for name in ("ring_depth", "expected_generation"):
        if not _positive_int(payload.get(name)):
            raise ValueError(f"C4 prepared sample plan {name} must be positive")
    for name in ("slot_id", "planned_acquisition_offset"):
        if not _nonnegative_int(payload.get(name)):
            raise ValueError(f"C4 prepared sample plan {name} must be non-negative")
    if not _nonnegative_int(payload.get("device_ordinal")):
        raise ValueError("C4 prepared sample plan device ordinal must be non-negative")
    if not isinstance(payload.get("shadows"), bool):
        raise ValueError("C4 prepared sample plan shadows must be boolean")
    frame_index = payload.get("transform_frame_index")
    if frame_index is not None and not _nonnegative_int(frame_index):
        raise ValueError("C4 prepared sample plan frame index is invalid")


def _validate_active_evidence_payload(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != C4_ACTIVE_LEASE_BINDING_SCHEMA_VERSION:
        raise ValueError("C4 active lease binding schema version is invalid")
    prepared_fields = {
        name: payload[name]
        for name in C4PreparedSamplePlanEvidence.__dataclass_fields__
        if name not in {"schema_version", "expected_generation"}
    }
    prepared_fields.update(
        {
            "schema_version": C4_PREPARED_SAMPLE_PLAN_SCHEMA_VERSION,
            "expected_generation": payload.get("generation"),
        }
    )
    _validate_prepared_evidence_payload(prepared_fields)
    if not _positive_int(payload.get("generation")):
        raise ValueError("C4 active lease generation must be positive")
    _require_sha256(payload.get("lease_token"), "C4 active lease token")
    _require_sha256(payload.get("prepared_slot_plan_sha256"), "C4 prepared plan digest")
    if payload.get("result_capable") is not False:
        raise ValueError("C4a active lease binding must remain non-result-capable")


def _slot_lifecycle_snapshot(slot: _SlotRecord) -> C4ArenaSlotLifecycleSnapshot:
    return C4ArenaSlotLifecycleSnapshot(
        slot_id=slot.manifest.slot_id,
        generation=slot.generation,
        state=slot.state,
        terminal_event_record_count=slot.terminal_event_record_count,
        completion_resolved=slot.completion_resolved,
        payload_ready=slot.payload_ready,
        transaction_hold_count=slot.transaction_hold_count,
        device_result_hold_count=slot.device_result_hold_count,
        delivery_device_hold_count=slot.delivery_device_hold_count,
        external_hold_count=slot.external_hold_count,
    )


def _validate_slot_lifecycle_snapshot(snapshot: C4ArenaSlotLifecycleSnapshot) -> None:
    if not _nonnegative_int(snapshot.slot_id) or not _nonnegative_int(snapshot.generation):
        raise ValueError("C4 arena slot lifecycle identity is invalid")
    if snapshot.state not in _SLOT_STATES:
        raise ValueError("C4 arena slot lifecycle state is invalid")
    for name in (
        "terminal_event_record_count",
        "transaction_hold_count",
        "device_result_hold_count",
        "delivery_device_hold_count",
        "external_hold_count",
    ):
        if not _nonnegative_int(getattr(snapshot, name)):
            raise ValueError(f"C4 arena slot lifecycle {name} is invalid")
    for name in ("completion_resolved", "payload_ready"):
        if not isinstance(getattr(snapshot, name), bool):
            raise ValueError(f"C4 arena slot lifecycle {name} must be boolean")
    if snapshot.generation == 0 and (
        snapshot.state != "FREE"
        or snapshot.terminal_event_record_count != 0
        or snapshot.completion_resolved
        or snapshot.payload_ready
        or snapshot.transaction_hold_count
        or snapshot.device_result_hold_count
        or snapshot.delivery_device_hold_count
        or snapshot.external_hold_count
    ):
        raise ValueError("an unused C4 arena slot must remain in its zero state")


def _transition_matches_slot(
    transition: C4ArenaLifecycleTransition,
    slot: dict[str, object],
) -> bool:
    return (
        transition.slot_id == slot["slot_id"]
        and transition.generation == slot["generation"]
        and transition.slot_state == slot["state"]
        and transition.transaction_hold_count == slot["transaction_hold_count"]
        and transition.device_result_hold_count == slot["device_result_hold_count"]
        and transition.delivery_device_hold_count == slot["delivery_device_hold_count"]
        and transition.external_hold_count == slot["external_hold_count"]
    )


def _require_arena_only_transition(transition: C4ArenaLifecycleTransition) -> None:
    if (
        transition.slot_id is not None
        or transition.slot_state is not None
        or transition.transaction_hold_count
        or transition.device_result_hold_count
        or transition.delivery_device_hold_count
        or transition.external_hold_count
    ):
        raise ValueError(f"C4 lifecycle {transition.operation} must be arena-only")


def _replay_lifecycle_evidence(evidence: C4ArenaLifecycleEvidence) -> dict[str, object]:
    slots: list[dict[str, object]] = [
        {
            "slot_id": slot_id,
            "generation": 0,
            "state": "FREE",
            "terminal_event_record_count": 0,
            "completion_resolved": False,
            "payload_ready": False,
            "transaction_hold_count": 0,
            "device_result_hold_count": 0,
            "delivery_device_hold_count": 0,
            "external_hold_count": 0,
        }
        for slot_id in range(evidence.ring_depth)
    ]
    arena_state: str | None = None
    active_fifo: list[int] = []
    next_acquire_slot_id = 0
    backpressure_count = 0
    terminal_event_record_count = 0
    completion_resolution_count = 0
    slot_reclaim_count = 0
    final_drain_complete = False
    external_hold_count_at_poison: int | None = None

    for index, transition in enumerate(evidence.transitions):
        if transition.sequence != index:
            raise ValueError("C4 lifecycle transition sequence is not contiguous")
        operation = transition.operation
        if operation == "arena_opened":
            if index != 0 or arena_state is not None or transition.arena_state != "OPEN":
                raise ValueError("C4 lifecycle must begin with exactly one arena_opened transition")
            _require_arena_only_transition(transition)
            arena_state = "OPEN"
            continue
        if arena_state is None:
            raise ValueError("C4 lifecycle transition occurred before arena_opened")

        if operation in {"drain_started", "final_drain_complete", "arena_closed"}:
            _require_arena_only_transition(transition)
            if operation == "drain_started":
                if arena_state != "OPEN" or transition.arena_state != "DRAINING":
                    raise ValueError("C4 drain_started transition is illegal")
                arena_state = "DRAINING"
            elif operation == "final_drain_complete":
                if (
                    arena_state != "DRAINING"
                    or active_fifo
                    or transition.arena_state != "DRAINING"
                    or final_drain_complete
                ):
                    raise ValueError("C4 final_drain_complete transition is illegal")
                final_drain_complete = True
            else:
                if (
                    arena_state not in {"OPEN", "DRAINING"}
                    or active_fifo
                    or any(slot["state"] != "FREE" for slot in slots)
                    or transition.arena_state != "CLOSED"
                ):
                    raise ValueError("C4 arena_closed transition is illegal")
                arena_state = "CLOSED"
            continue

        if transition.slot_id is None or transition.slot_id >= evidence.ring_depth:
            raise ValueError(f"C4 lifecycle {operation} has an invalid slot")
        slot = slots[transition.slot_id]

        if operation == "acquire_backpressure":
            if (
                arena_state != "OPEN"
                or transition.arena_state != "OPEN"
                or transition.slot_id != next_acquire_slot_id
                or slot["state"] == "FREE"
                or not _transition_matches_slot(transition, slot)
            ):
                raise ValueError("C4 acquire_backpressure transition is illegal")
            backpressure_count += 1
            continue

        if operation == "acquired":
            if (
                arena_state != "OPEN"
                or transition.arena_state != "OPEN"
                or transition.slot_id != next_acquire_slot_id
                or slot["state"] != "FREE"
                or transition.generation != slot["generation"] + 1
                or transition.slot_state != "LEASED"
                or transition.transaction_hold_count != 1
                or transition.device_result_hold_count != 1
                or transition.delivery_device_hold_count not in {0, 1}
                or transition.external_hold_count != 0
            ):
                raise ValueError("C4 acquired transition is illegal")
            slot.update(
                {
                    "generation": transition.generation,
                    "state": "LEASED",
                    "terminal_event_record_count": 0,
                    "completion_resolved": False,
                    "payload_ready": False,
                    "transaction_hold_count": 1,
                    "device_result_hold_count": 1,
                    "delivery_device_hold_count": transition.delivery_device_hold_count,
                    "external_hold_count": 0,
                }
            )
            active_fifo.append(transition.slot_id)
            next_acquire_slot_id = (transition.slot_id + 1) % evidence.ring_depth
            continue

        if transition.generation != slot["generation"] or transition.slot_id not in active_fifo:
            raise ValueError(f"C4 lifecycle {operation} targets a stale generation")

        if operation == "cancelled_generation_burned":
            if (
                arena_state not in {"OPEN", "DRAINING"}
                or slot["state"] != "LEASED"
                or transition.arena_state != arena_state
                or transition.slot_state != "FREE"
                or any(
                    (
                        transition.transaction_hold_count,
                        transition.device_result_hold_count,
                        transition.delivery_device_hold_count,
                        transition.external_hold_count,
                    )
                )
            ):
                raise ValueError("C4 cancelled generation transition is illegal")
            slot.update(
                {
                    "state": "FREE",
                    "transaction_hold_count": 0,
                    "device_result_hold_count": 0,
                    "delivery_device_hold_count": 0,
                    "external_hold_count": 0,
                }
            )
            active_fifo.remove(transition.slot_id)
            continue

        if operation == "enqueue_started":
            if (
                arena_state not in {"OPEN", "DRAINING"}
                or slot["state"] != "LEASED"
                or transition.arena_state != arena_state
            ):
                raise ValueError("C4 enqueue_started transition is illegal")
            position = active_fifo.index(transition.slot_id)
            if any(slots[slot_id]["state"] in {"LEASED", "ENQUEUING"} for slot_id in active_fifo[:position]):
                raise ValueError("C4 enqueue_started transition violates FIFO")
            slot["state"] = "ENQUEUING"
        elif operation == "completion_recorded":
            if (
                arena_state not in {"OPEN", "DRAINING"}
                or slot["state"] != "ENQUEUING"
                or slot["terminal_event_record_count"] != 0
                or transition.arena_state != arena_state
            ):
                raise ValueError("C4 completion_recorded transition is illegal")
            slot["terminal_event_record_count"] = 1
            slot["state"] = "COMPLETION_RECORDED"
            terminal_event_record_count += 1
        elif operation == "completion_resolved":
            if (
                arena_state not in {"OPEN", "DRAINING"}
                or not active_fifo
                or active_fifo[0] != transition.slot_id
                or slot["state"] != "COMPLETION_RECORDED"
                or slot["terminal_event_record_count"] != 1
                or slot["completion_resolved"]
                or transition.arena_state != arena_state
            ):
                raise ValueError("C4 completion_resolved transition is illegal")
            slot["completion_resolved"] = True
            slot["transaction_hold_count"] = 0
            completion_resolution_count += 1
            if transition.slot_state == "PAYLOAD_READY":
                slot["state"] = "PAYLOAD_READY"
                slot["payload_ready"] = True
                slot["delivery_device_hold_count"] = 0
            elif transition.slot_state != "COMPLETION_RECORDED":
                raise ValueError("C4 completion_resolved target state is invalid")
        elif operation == "device_result_hold_released":
            if (
                arena_state not in {"OPEN", "DRAINING"}
                or not slot["completion_resolved"]
                or slot["device_result_hold_count"] != 1
                or transition.arena_state != arena_state
            ):
                raise ValueError("C4 device-result release transition is illegal")
            slot["device_result_hold_count"] = 0
        elif operation == "slot_reclaimable":
            if (
                arena_state not in {"OPEN", "DRAINING"}
                or not slot["payload_ready"]
                or any(
                    slot[name]
                    for name in (
                        "transaction_hold_count",
                        "device_result_hold_count",
                        "delivery_device_hold_count",
                        "external_hold_count",
                    )
                )
                or transition.arena_state != arena_state
            ):
                raise ValueError("C4 slot_reclaimable transition is illegal")
            slot["state"] = "RECLAIMABLE"
        elif operation == "slot_reclaimed":
            if (
                arena_state not in {"OPEN", "DRAINING"}
                or not active_fifo
                or active_fifo[0] != transition.slot_id
                or slot["state"] != "RECLAIMABLE"
                or transition.arena_state != arena_state
            ):
                raise ValueError("C4 slot_reclaimed transition is illegal")
            slot["state"] = "FREE"
            active_fifo.pop(0)
            slot_reclaim_count += 1
        elif operation == "arena_poisoned":
            if arena_state not in {"OPEN", "DRAINING"} or slot["state"] == "LEASED":
                raise ValueError("C4 arena_poisoned transition is illegal")
            external_hold_count_at_poison = sum(int(value["external_hold_count"]) for value in slots)
            arena_state = "POISONED"
            for active_slot_id in active_fifo:
                slots[active_slot_id]["state"] = "RETAINED_POISONED"
        else:
            raise ValueError(f"unsupported C4 lifecycle transition {operation!r}")

        if transition.arena_state != arena_state or not _transition_matches_slot(transition, slot):
            raise ValueError(f"C4 lifecycle {operation} snapshot does not match replayed state")

    if arena_state is None:
        raise ValueError("C4 lifecycle transition trace is empty")
    return {
        "arena_state": arena_state,
        "slots": slots,
        "next_acquire_slot_id": next_acquire_slot_id,
        "backpressure_count": backpressure_count,
        "terminal_event_record_count": terminal_event_record_count,
        "completion_resolution_count": completion_resolution_count,
        "slot_reclaim_count": slot_reclaim_count,
        "final_drain_complete": final_drain_complete,
        "final_reclaim_complete": final_drain_complete,
        "external_hold_count": sum(int(slot["external_hold_count"]) for slot in slots),
        "external_hold_count_at_poison": external_hold_count_at_poison,
        "poisoned_slot_count": sum(slot["state"] == "RETAINED_POISONED" for slot in slots),
    }


def _validate_lifecycle_evidence(evidence: C4ArenaLifecycleEvidence) -> None:
    if evidence.schema_version != C4_ARENA_LIFECYCLE_SCHEMA_VERSION:
        raise ValueError("C4 arena lifecycle schema version is invalid")
    for value, label in (
        (evidence.arena_spec_sha256, "arena spec"),
        (evidence.arena_resource_manifest_sha256, "arena resource manifest"),
        (evidence.transition_trace_sha256, "transition trace"),
    ):
        _require_sha256(value, f"C4 lifecycle {label} digest")
    _require_token(evidence.arena_token, "C4 lifecycle arena token")
    if evidence.arena_state not in _ARENA_STATES:
        raise ValueError("C4 lifecycle arena state is invalid")
    if not _positive_int(evidence.ring_depth) or evidence.ring_depth not in {1, 2}:
        raise ValueError("C4 lifecycle ring depth is invalid")
    if len(evidence.slots) != evidence.ring_depth or tuple(slot.slot_id for slot in evidence.slots) != tuple(
        range(evidence.ring_depth)
    ):
        raise ValueError("C4 lifecycle must contain every slot snapshot in order")
    for name in (
        "next_acquire_slot_id",
        "backpressure_count",
        "terminal_event_record_count",
        "completion_resolution_count",
        "slot_reclaim_count",
        "external_hold_count",
        "external_hold_discharge_count",
        "device_emergency_sync_attempt_count",
        "poisoned_slot_count",
        "retained_pinned_storage_count",
    ):
        if not _nonnegative_int(getattr(evidence, name)):
            raise ValueError(f"C4 lifecycle {name} is invalid")
    for name in ("external_hold_count_at_poison", "external_hold_count_after_device_sync"):
        value = getattr(evidence, name)
        if value is not None and not _nonnegative_int(value):
            raise ValueError(f"C4 lifecycle {name} is invalid")
    for name in (
        "final_drain_complete",
        "final_reclaim_complete",
        "device_emergency_sync_succeeded",
        "device_quiesced",
        "waiting_external_holds",
        "external_holds_discharged",
        "resource_release_authorized",
    ):
        if not isinstance(getattr(evidence, name), bool):
            raise ValueError(f"C4 lifecycle {name} must be boolean")
    expected_trace = _canonical_sha256([transition.to_payload() for transition in evidence.transitions])
    if evidence.transition_trace_sha256 != expected_trace:
        raise ValueError("C4 lifecycle transition trace digest is invalid")
    replayed = _replay_lifecycle_evidence(evidence)
    for name in (
        "arena_state",
        "next_acquire_slot_id",
        "backpressure_count",
        "terminal_event_record_count",
        "completion_resolution_count",
        "slot_reclaim_count",
        "final_drain_complete",
        "final_reclaim_complete",
        "external_hold_count",
        "external_hold_count_at_poison",
        "poisoned_slot_count",
    ):
        if getattr(evidence, name) != replayed[name]:
            raise ValueError(f"C4 lifecycle {name} does not match the replayed transition trace")
    if tuple(slot.to_payload() for slot in evidence.slots) != tuple(replayed["slots"]):
        raise ValueError("C4 lifecycle slot summaries do not match the replayed transition trace")
    if evidence.external_hold_discharge_count != 0 or evidence.retained_pinned_storage_count != 0:
        raise ValueError("C4a transition trace cannot claim B3 external-hold activity")
    if (
        evidence.shutdown_disposition is not None
        and evidence.shutdown_disposition not in _SHUTDOWN_DISPOSITION_SET
    ):
        raise ValueError("C4 lifecycle shutdown disposition is invalid")
    if evidence.resource_release_authorized and not (
        evidence.device_quiesced
        and evidence.device_emergency_sync_succeeded
        and evidence.external_hold_count == 0
        and evidence.shutdown_disposition == "aborted_synced_release_authorized"
    ):
        raise ValueError("C4 lifecycle release authorization lacks both safety proofs")
    if evidence.waiting_external_holds and not (
        evidence.device_quiesced
        and evidence.device_emergency_sync_succeeded
        and evidence.external_hold_count > 0
        and evidence.retained_pinned_storage_count > 0
        and not evidence.resource_release_authorized
    ):
        raise ValueError("C4 lifecycle external-hold waiting evidence is inconsistent")
    if (
        evidence.shutdown_disposition == "quarantined_until_process_exit"
        and evidence.resource_release_authorized
    ):
        raise ValueError("C4 quarantined lifecycle cannot authorize resource release")
    if evidence.external_hold_count_at_poison is None and evidence.poisoned_slot_count:
        raise ValueError("C4 poisoned slots require the hold count captured at poison")
    if evidence.device_emergency_sync_succeeded:
        if (
            evidence.device_emergency_sync_attempt_count != 1
            or not evidence.device_quiesced
            or evidence.external_hold_count_after_device_sync is None
        ):
            raise ValueError("C4 successful emergency sync evidence is incomplete")
    elif evidence.device_quiesced or evidence.external_hold_count_after_device_sync is not None:
        raise ValueError("C4 device quiescence cannot be claimed without successful emergency sync")
    if evidence.external_hold_count_after_device_sync is not None and (
        evidence.external_hold_count_after_device_sync < evidence.external_hold_count
    ):
        raise ValueError("C4 current external hold count cannot exceed its post-sync count")
    if evidence.external_holds_discharged != (evidence.external_hold_count == 0):
        raise ValueError("C4 external hold discharge evidence is inconsistent")
    expected_disposition_by_state = {
        "SYNCED_WAITING_EXTERNAL_HOLDS": "synced_waiting_external_holds",
        "ABORTED_SYNCED": "aborted_synced_release_authorized",
        "QUARANTINED_UNTIL_PROCESS_EXIT": "quarantined_until_process_exit",
    }
    expected_disposition = expected_disposition_by_state.get(evidence.arena_state)
    if evidence.shutdown_disposition != expected_disposition:
        raise ValueError("C4 lifecycle state and shutdown disposition are inconsistent")
    if evidence.resource_release_authorized != (evidence.arena_state == "ABORTED_SYNCED"):
        raise ValueError("C4 aborted-synced state and release authorization must agree")
    if evidence.waiting_external_holds != (evidence.arena_state == "SYNCED_WAITING_EXTERNAL_HOLDS"):
        raise ValueError("C4 waiting state and external-hold flag must agree")
    if evidence.shutdown_disposition is None:
        if evidence.shutdown_transition_trace_sha256 is not None:
            raise ValueError("C4 shutdown trace requires a shutdown disposition")
    else:
        _require_sha256(
            evidence.shutdown_transition_trace_sha256,
            "C4 lifecycle shutdown transition trace",
        )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_token(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def _require_sha256(value: object, label: str) -> None:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
