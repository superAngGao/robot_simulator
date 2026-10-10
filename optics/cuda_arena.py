"""Torch-native CUDA storage arena and prevalidated transaction plans.

This module is the C4b data-plane layer over :mod:`optics.native_arena`.
It allocates and seals the exact CUDA storage/event graph outside measured
submission, then couples one prebuilt C2 transaction to each arena-issued C4
lease.  It deliberately supports only R0/R1 and the R2 ``none/device_only``
row; host payload storage and result activation remain B3 work.
"""

from __future__ import annotations

import hashlib
import json
import math
import weakref
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from types import MappingProxyType

from optical_worker.native_contract import (
    CudaPreparedNativeInput,
    NativeReadinessEvidence,
    assert_cuda_prepared_authority,
)

from .cuda_direct_light import (
    CudaCameraParameters,
    CudaDirectLightBuffers,
    CudaDirectLightNativeRenderer,
    CudaFirstHitInputs,
    CudaShadeInputs,
    allocate_cuda_direct_light_buffers,
)
from .cuda_direct_light import (
    _dynamic_first_hit_inputs as _dynamic_first_hit_inputs,
)
from .cuda_direct_light import (
    _shade_inputs as _shade_inputs,
)
from .cuda_lbvh import (
    CUDA_LBVH_ACCELERATION_ID,
    CudaLbvhBuffers,
    CudaLbvhRebuildStageFactory,
    allocate_cuda_lbvh_buffers,
    allocate_cuda_lbvh_workspace,
)
from .cuda_scene import (
    CudaSceneTransformInputs,
    CudaSceneUpdateStageFactory,
    CudaWorldSceneBuffers,
    allocate_cuda_world_scene_buffers,
)
from .cuda_timeline import (
    CudaOpticalCompletion,
    CudaOpticalExecutionContext,
    CudaOpticalTransaction,
    CudaOpticalTransactionTemplate,
)
from .execution import OpticalOutputProfile, normalize_output_profile
from .native_arena import (
    NATIVE_OPTICAL_RESOURCE_MANIFEST_SCHEMA_VERSION,
    NATIVE_OPTICAL_SLOT_RESOURCE_SCHEMA_VERSION,
    C4ActiveOutputLease,
    C4PreparedSamplePlan,
    NativeOpticalArena,
    NativeOpticalArenaBackpressure,
    NativeOpticalArenaProtocolError,
    NativeOpticalArenaResourceManifest,
    NativeOpticalArenaSpec,
    NativeOpticalSlotResourceManifest,
    _retain_failure_without_masking,
    _retain_or_cancel_enqueue_failure_without_masking,
    plan_native_optical_acquisition,
)
from .scene_contract import SCENE_ARRAY_SEMANTICS

CUDA_NATIVE_ARENA_PLAN_SCHEMA_VERSION = "q54.p12.4.cuda-arena-plan.v2"
CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION = "q54.p12.4.cuda-arena-resource.v1"
CUDA_DEVICE_ONLY_DELIVERY_SCHEMA_VERSION = "q54.p12.4.cuda-device-only-delivery.v1"
_SUPPORTED_BOUNDARIES = frozenset({"R0", "R1", "R2"})
_SHA256_LENGTH = 64
_C3_PROVENANCE_AUTHORITY = object()
_FIXED_DERIVED_ARRAY_NAMES = (
    "triangle_v0_world",
    "triangle_e1_world",
    "triangle_e2_world",
    "triangle_normal_world",
    "triangle_aabb_min",
    "triangle_aabb_max",
    "plane_normal_world",
    "plane_point_world",
    "bvh_bounds_min",
    "bvh_bounds_max",
    "bvh_left",
    "bvh_right",
    "bvh_start",
    "bvh_count",
    "bvh_node_depth",
    "bvh_level_ranges",
    "bvh_prim_ids",
)


@dataclass(frozen=True, slots=True)
class CudaDeviceOnlyDeliveryEvidence:
    """Identity of the completion-neutral R2 device-only endpoint stage."""

    schema_version: str
    payload: str
    delivery_policy: str
    output_profile: str
    device_ordinal: int
    stream_handle: int
    output_storage_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != CUDA_DEVICE_ONLY_DELIVERY_SCHEMA_VERSION:
            raise ValueError("CUDA device-only delivery schema version is invalid")
        if (self.payload, self.delivery_policy) != ("none", "device_only"):
            raise ValueError("CUDA device-only delivery identity is invalid")
        if not isinstance(self.output_profile, str) or not self.output_profile:
            raise ValueError("CUDA device-only delivery profile is invalid")
        for name in ("device_ordinal", "stream_handle"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"CUDA device-only delivery {name} is invalid")
        _require_sha256(self.output_storage_sha256, "output storage digest")


class CudaDeviceOnlyDeliveryStage:
    """Explicit no-copy R2 endpoint registered before the terminal event."""

    cuda_transaction_stage_name = "cuda_device_only_delivery"

    __slots__ = ("_binding_signature", "_sealed", "buffers", "context", "evidence")

    def __init__(
        self,
        *,
        context: CudaOpticalExecutionContext,
        buffers: CudaDirectLightBuffers,
        output_profile: OpticalOutputProfile,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        evidence = CudaDeviceOnlyDeliveryEvidence(
            schema_version=CUDA_DEVICE_ONLY_DELIVERY_SCHEMA_VERSION,
            payload="none",
            delivery_policy="device_only",
            output_profile=output_profile.value,
            device_ordinal=int(context.device.index),
            stream_handle=context.stream_handle,
            output_storage_sha256=_tensor_sequence_sha256(buffers.resource_tuple()),
        )
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "buffers", buffers)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "_binding_signature", self._current_signature())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CUDA device-only delivery stage is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CUDA device-only delivery stage is immutable")
        object.__delattr__(self, name)

    def enqueue(self, transaction: CudaOpticalTransaction) -> None:
        with transaction.enqueue_stage(self):
            return

    def _validate_cuda_transaction_stage(self, context: CudaOpticalExecutionContext) -> None:
        if context is not self.context:
            raise ValueError("CUDA device-only delivery context identity mismatch")
        self._assert_cuda_transaction_stage_stable(context)

    def _assert_cuda_transaction_stage_stable(self, context: CudaOpticalExecutionContext) -> None:
        if context is not self.context or self._current_signature() != self._binding_signature:
            raise ValueError("CUDA device-only delivery binding changed after validation")

    def _cuda_transaction_stage_evidence(self) -> CudaDeviceOnlyDeliveryEvidence:
        return self.evidence

    def _current_signature(self) -> tuple[object, ...]:
        return (
            id(self.context),
            self.context.device,
            self.context.stream_handle,
            id(self.buffers),
            _tensor_identity_tuple(self.buffers.resource_tuple()),
            self.evidence,
        )


@dataclass(frozen=True, slots=True)
class CudaArenaPlanEvidence:
    """C4b proof that one C2/C3 stage graph is bound to one arena slot."""

    schema_version: str
    arena_spec_sha256: str
    slot_id: int
    slot_storage_manifest_sha256: str
    slot_resource_contract_sha256: str
    transform_frame_index: int | None
    stage_names: tuple[str, ...]
    ordered_runtime_stage_evidence_sha256: str
    terminal_event_token: str
    diagnostic_phase_names: tuple[str, ...]
    plan_binding_sha256: str
    render_sample_index: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != CUDA_NATIVE_ARENA_PLAN_SCHEMA_VERSION:
            raise ValueError("CUDA arena plan schema version is invalid")
        for name in (
            "arena_spec_sha256",
            "slot_storage_manifest_sha256",
            "slot_resource_contract_sha256",
            "ordered_runtime_stage_evidence_sha256",
            "plan_binding_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.slot_id, int) or isinstance(self.slot_id, bool) or self.slot_id < 0:
            raise ValueError("CUDA arena plan slot ID is invalid")
        if self.transform_frame_index is not None and (
            not isinstance(self.transform_frame_index, int)
            or isinstance(self.transform_frame_index, bool)
            or self.transform_frame_index < 0
        ):
            raise ValueError("CUDA arena transform frame index is invalid")
        if self.render_sample_index is not None and (
            type(self.render_sample_index) is not int or self.render_sample_index < 0
        ):
            raise ValueError("CUDA arena render sample index is invalid")
        if (
            not isinstance(self.stage_names, tuple)
            or not self.stage_names
            or any(not isinstance(value, str) or not value for value in self.stage_names)
        ):
            raise ValueError("CUDA arena plan stage names are invalid")
        if len(set(self.stage_names)) != len(self.stage_names):
            raise ValueError("CUDA arena plan stage names must be unique")
        if not isinstance(self.diagnostic_phase_names, tuple) or any(
            not isinstance(value, str) or not value for value in self.diagnostic_phase_names
        ):
            raise ValueError("CUDA arena diagnostic phase names are invalid")
        if len(set(self.diagnostic_phase_names)) != len(self.diagnostic_phase_names):
            raise ValueError("CUDA arena diagnostic phase names must be unique")
        if not isinstance(self.terminal_event_token, str) or not self.terminal_event_token:
            raise ValueError("CUDA arena terminal event token is invalid")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(_dataclass_payload(self))


@dataclass(frozen=True, slots=True, init=False)
class CudaArenaC3Provenance:
    """C3 receipt digests mechanically attached to one exact C4b plan target."""

    plan_binding_sha256: str
    ordered_runtime_stage_evidence_sha256: str
    c3_one_shot_binding_sha256: str
    ordered_stage_evidence_sha256: str
    dataflow_binding_sha256: str

    def __init__(
        self,
        *,
        plan_binding_sha256: str,
        ordered_runtime_stage_evidence_sha256: str,
        c3_one_shot_binding_sha256: str,
        ordered_stage_evidence_sha256: str,
        dataflow_binding_sha256: str,
        _authority: object,
    ) -> None:
        if _authority is not _C3_PROVENANCE_AUTHORITY:
            raise TypeError("CUDA arena C3 provenance is minted only from a verified C3 binding")
        values = {
            "plan_binding_sha256": plan_binding_sha256,
            "ordered_runtime_stage_evidence_sha256": ordered_runtime_stage_evidence_sha256,
            "c3_one_shot_binding_sha256": c3_one_shot_binding_sha256,
            "ordered_stage_evidence_sha256": ordered_stage_evidence_sha256,
            "dataflow_binding_sha256": dataflow_binding_sha256,
        }
        for name, value in values.items():
            _require_sha256(value, name)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class CudaArenaPlanTarget:
    """Non-executable slot/frame inspection used to build C3 provenance."""

    evidence: CudaArenaPlanEvidence
    _render_buffers: CudaDirectLightBuffers = field(repr=False)
    slot_resources: tuple["CudaArenaResourceRecord", ...]
    _spec: NativeOpticalArenaSpec = field(repr=False)
    _runtime_scene_arrays: Mapping[str, object] = field(repr=False)
    _derived_fixture_arrays: Mapping[str, object] = field(repr=False)
    _transform_arrays: Mapping[str, object] = field(repr=False)
    _scene_output_arrays: Mapping[str, object] = field(repr=False)
    _acceleration_output_arrays: Mapping[str, object] = field(repr=False)
    _runtime_stage_evidence: tuple[object, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, CudaArenaPlanEvidence):
            raise TypeError("CUDA arena target requires plan evidence")
        if not isinstance(self._render_buffers, CudaDirectLightBuffers):
            raise TypeError("CUDA arena target requires render buffers")
        if not isinstance(self.slot_resources, tuple) or any(
            not isinstance(value, CudaArenaResourceRecord) for value in self.slot_resources
        ):
            raise TypeError("CUDA arena target resources are invalid")
        if not isinstance(self._spec, NativeOpticalArenaSpec):
            raise TypeError("CUDA arena target specification is invalid")
        if not isinstance(self._runtime_scene_arrays, Mapping) or not self._runtime_scene_arrays:
            raise TypeError("CUDA arena target runtime scene arrays are invalid")
        if not isinstance(self._derived_fixture_arrays, Mapping):
            raise TypeError("CUDA arena target derived fixture arrays are invalid")
        for name in (
            "_transform_arrays",
            "_scene_output_arrays",
            "_acceleration_output_arrays",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"CUDA arena target {name[1:]} are invalid")
        if not isinstance(self._runtime_stage_evidence, tuple) or not self._runtime_stage_evidence:
            raise TypeError("CUDA arena target runtime stage evidence is invalid")
        if self.evidence.arena_spec_sha256 != self._spec.sha256:
            raise ValueError("CUDA arena target specification digest is inconsistent")

    @property
    def result_capable(self) -> bool:
        return False

    def _bind_c3_provenance(
        self,
        binding: object,
    ) -> CudaArenaC3Provenance:
        """Canonicalize and bind one real C3 receipt to this exact CUDA graph."""

        canonical = _canonical_c3_binding(binding)
        _validate_c3_binding_identity(self, canonical)
        _validate_c3_binding_storage(self, canonical)
        ordered_stage_sha256 = _canonical_sha256(
            [entry.to_payload() for entry in canonical.ordered_stage_evidence]
        )

        provenance = CudaArenaC3Provenance(
            plan_binding_sha256=self.evidence.plan_binding_sha256,
            ordered_runtime_stage_evidence_sha256=(self.evidence.ordered_runtime_stage_evidence_sha256),
            c3_one_shot_binding_sha256=canonical.sha256,
            ordered_stage_evidence_sha256=ordered_stage_sha256,
            dataflow_binding_sha256=canonical.dataflow_binding_sha256,
            _authority=_C3_PROVENANCE_AUTHORITY,
        )
        return provenance


@dataclass(frozen=True, slots=True)
class CudaArenaLogicalResourceView:
    """One named tensor/event/owner view into an arena physical resource."""

    logical_name: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    active_offset_bytes: int
    active_nbytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.logical_name, str) or not self.logical_name:
            raise ValueError("CUDA arena logical resource name is invalid")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("CUDA arena logical resource dtype is invalid")
        for name, values in (("shape", self.shape), ("stride", self.stride)):
            if not isinstance(values, tuple) or any(
                not isinstance(value, int) or isinstance(value, bool) for value in values
            ):
                raise ValueError(f"CUDA arena logical resource {name} is invalid")
        if any(value < 0 for value in self.shape):
            raise ValueError("CUDA arena logical resource shape is invalid")
        for name in ("active_offset_bytes", "active_nbytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"CUDA arena logical resource {name} is invalid")

    def to_payload(self) -> dict[str, object]:
        return {
            "logical_name": self.logical_name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "active_offset_bytes": self.active_offset_bytes,
            "active_nbytes": self.active_nbytes,
        }


@dataclass(frozen=True, slots=True)
class CudaArenaResourceRecord:
    """Exact physical allocation/event identity plus every logical view."""

    schema_version: str
    resource_kind: str
    device_ordinal: int
    storage_token: str
    logical_views: tuple[CudaArenaLogicalResourceView, ...]
    nbytes: int

    def __post_init__(self) -> None:
        if self.schema_version != CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION:
            raise ValueError("CUDA arena resource schema version is invalid")
        if (
            not isinstance(self.logical_views, tuple)
            or not self.logical_views
            or any(not isinstance(value, CudaArenaLogicalResourceView) for value in self.logical_views)
        ):
            raise ValueError("CUDA arena resource logical views must be non-empty")
        names = tuple(value.logical_name for value in self.logical_views)
        if len(set(names)) != len(names):
            raise ValueError("CUDA arena resource logical names must be unique")
        if self.resource_kind not in {"cuda_storage", "cuda_event", "python_owner"}:
            raise ValueError("CUDA arena resource kind is invalid")
        if (
            not isinstance(self.device_ordinal, int)
            or isinstance(self.device_ordinal, bool)
            or self.device_ordinal < 0
            or not isinstance(self.nbytes, int)
            or isinstance(self.nbytes, bool)
            or self.nbytes < 0
            or not isinstance(self.storage_token, str)
            or not self.storage_token
        ):
            raise ValueError("CUDA arena resource identity is invalid")
        if any(view.active_offset_bytes + view.active_nbytes > self.nbytes for view in self.logical_views):
            raise ValueError("CUDA arena logical view exceeds its physical allocation")
        if self.resource_kind != "cuda_storage" and (
            self.nbytes != 0
            or any(view.active_offset_bytes or view.active_nbytes for view in self.logical_views)
        ):
            raise ValueError("CUDA arena non-storage resources cannot expose byte ranges")

    @property
    def logical_names(self) -> tuple[str, ...]:
        return tuple(value.logical_name for value in self.logical_views)

    def logical_view(self, name: str) -> CudaArenaLogicalResourceView:
        matches = tuple(value for value in self.logical_views if value.logical_name == name)
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "resource_kind": self.resource_kind,
            "device_ordinal": self.device_ordinal,
            "storage_token": self.storage_token,
            "logical_views": [value.to_payload() for value in self.logical_views],
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True, slots=True)
class _CudaExecutableTarget:
    """Arena-private executable graph; never exposed by a plan target."""

    target: CudaArenaPlanTarget
    transaction_template: CudaOpticalTransactionTemplate
    activation_capability: object
    render_stage: object

    def activate(self) -> CudaOpticalTransaction:
        return self.transaction_template._activate_with_capability(self.activation_capability)

    def validate_bound(self) -> None:
        self.transaction_template.validate_bound()


class CudaPreparedArenaSample:
    """Untimed C4/C2 preparation; it cannot submit without arena acquire."""

    __slots__ = (
        "_active",
        "_control_plan",
        "_executable",
        "_execution_capability",
        "_frozen_signature",
        "_provenance",
        "_sealed",
        "_target",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        control_plan: C4PreparedSamplePlan,
        target: CudaArenaPlanTarget,
        provenance: CudaArenaC3Provenance,
        transaction: CudaOpticalTransaction,
        executable: _CudaExecutableTarget,
        arena_acquire_authority: object,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        active = CudaActiveArenaSample(
            target=target,
            provenance=provenance,
            transaction=transaction,
            executable=executable,
            arena_acquire_authority=arena_acquire_authority,
        )
        execution_capability = transaction._prepare_arena_execution_capability(
            active_owner=active,
            plan_binding_sha256=provenance.plan_binding_sha256,
            ordered_runtime_stage_evidence_sha256=(provenance.ordered_runtime_stage_evidence_sha256),
            c3_one_shot_binding_sha256=provenance.c3_one_shot_binding_sha256,
            ordered_stage_evidence_sha256=provenance.ordered_stage_evidence_sha256,
            dataflow_binding_sha256=provenance.dataflow_binding_sha256,
        )
        object.__setattr__(self, "_control_plan", control_plan)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_provenance", provenance)
        object.__setattr__(self, "_executable", executable)
        object.__setattr__(self, "_execution_capability", execution_capability)
        object.__setattr__(self, "_active", active)
        object.__setattr__(self, "_frozen_signature", self._current_signature())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CUDA prepared arena sample is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CUDA prepared arena sample is immutable")
        object.__delattr__(self, name)

    @property
    def plan_evidence(self) -> CudaArenaPlanEvidence:
        return self._target.evidence

    @property
    def c4_evidence(self):
        return self._control_plan.evidence

    @property
    def result_capable(self) -> bool:
        return False

    def _current_signature(self) -> tuple[object, ...]:
        return (
            id(self._control_plan),
            self._control_plan.evidence.sha256,
            id(self._target),
            self._target.evidence.sha256,
            self._provenance,
            id(self._executable),
            id(self._execution_capability),
            id(self._active),
        )

    def _validate_bound(self) -> None:
        if self._current_signature() != self._frozen_signature:
            raise NativeOpticalArenaProtocolError("CUDA prepared sample changed after seal")

    def _validate_acquire_fast(
        self,
        *,
        control_plan: C4PreparedSamplePlan,
        target: CudaArenaPlanTarget,
        provenance: CudaArenaC3Provenance,
        executable: _CudaExecutableTarget,
        execution_capability: object,
        active: "CudaActiveArenaSample",
        arena_acquire_authority: object,
    ) -> None:
        if (
            not self._sealed
            or self._control_plan is not control_plan
            or self._target is not target
            or self._provenance is not provenance
            or self._executable is not executable
            or self._execution_capability is not execution_capability
            or self._active is not active
            or active._arena_acquire_authority is not arena_acquire_authority
        ):
            raise NativeOpticalArenaProtocolError("CUDA prepared sample binding changed")


class CudaDeviceResultView(Mapping[str, object]):
    """Capability-scoped access to one payload-ready CUDA generation."""

    __slots__ = ("_active",)

    def __init__(self, active: "CudaActiveArenaSample") -> None:
        object.__setattr__(self, "_active", active)

    def __getitem__(self, name: str) -> object:
        return self._active._output_mapping()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._active._output_mapping())

    def __len__(self) -> int:
        return len(self._active._output_mapping())


class CudaActiveArenaSample:
    """Single-use C2 transaction coupled to an active C4 slot generation."""

    __slots__ = (
        "_arena_acquire_authority",
        "_completion",
        "_control_lease",
        "_device_result_view",
        "_executable",
        "_execution_authority",
        "_frozen_signature",
        "_provenance",
        "_sealed",
        "_state",
        "_target",
        "_transaction",
    )

    def __init__(
        self,
        *,
        target: CudaArenaPlanTarget,
        provenance: CudaArenaC3Provenance,
        transaction: CudaOpticalTransaction,
        executable: _CudaExecutableTarget,
        arena_acquire_authority: object,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_provenance", provenance)
        object.__setattr__(self, "_transaction", transaction)
        object.__setattr__(self, "_executable", executable)
        object.__setattr__(self, "_arena_acquire_authority", arena_acquire_authority)
        object.__setattr__(self, "_control_lease", None)
        object.__setattr__(self, "_completion", None)
        object.__setattr__(self, "_device_result_view", CudaDeviceResultView(self))
        object.__setattr__(self, "_state", "DORMANT")
        object.__setattr__(self, "_execution_authority", None)
        object.__setattr__(self, "_frozen_signature", self._configuration_signature())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CUDA active arena sample configuration is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CUDA active arena sample configuration is immutable")

    @property
    def state(self) -> str:
        return self._state

    @property
    def c4_evidence(self):
        if self._control_lease is None:
            raise NativeOpticalArenaProtocolError("CUDA arena sample is not active")
        return self._control_lease.evidence

    @property
    def plan_evidence(self) -> CudaArenaPlanEvidence:
        return self._target.evidence

    @property
    def result_capable(self) -> bool:
        return False

    @property
    def completion(self) -> CudaOpticalCompletion:
        if self._completion is None:
            raise NativeOpticalArenaProtocolError("CUDA completion is not recorded")
        return self._completion

    def enqueue(self) -> None:
        self._require_state("ACQUIRED", operation="enqueue")
        lease = self._require_lease()
        self._assert_configuration_stable()
        try:
            with self._transaction._authorize_arena_execution(
                self._execution_authority,
                operation="enqueue",
            ):
                for stage in self._executable.transaction_template.stages:
                    stage.enqueue(self._transaction)
            if self._state != "ENQUEUED" or lease.state != "ENQUEUING":
                raise NativeOpticalArenaProtocolError("CUDA enqueue lifecycle commit is incomplete")
        except BaseException as exc:
            cancelled = _retain_or_cancel_enqueue_failure_without_masking(
                lease,
                f"CUDA transaction stage failure: {exc}",
            )
            if cancelled:
                try:
                    self._transaction._retire_cancelled_arena_execution_lease(lease)
                except Exception:
                    pass
            object.__setattr__(self, "_state", "POISONED")
            raise

    def finish(self) -> CudaOpticalCompletion:
        self._require_state("ENQUEUED", operation="finish")
        lease = self._require_lease()
        try:
            self._assert_configuration_stable()
            with self._transaction._authorize_arena_execution(
                self._execution_authority,
                operation="finish",
            ):
                completion = self._transaction.finish()
            if (
                self._state != "COMPLETION_RECORDED"
                or self._completion is not completion
                or lease.state != "COMPLETION_RECORDED"
            ):
                raise NativeOpticalArenaProtocolError("CUDA finish lifecycle commit is incomplete")
        except BaseException as exc:
            _retain_failure_without_masking(lease, f"CUDA terminal completion failure: {exc}")
            object.__setattr__(self, "_state", "POISONED")
            raise
        return completion

    def resolve(self, *, _guarded_controller=None) -> None:
        """Observe completion, then commit the lease after any guard has disarmed.

        The standalone arena keeps its existing untimed wait. The closed worker
        supplies its actual controller: waiting inside that controller's scope
        must not release transaction holds until COMPLETE/DISARMED succeeds.
        No caller-provided completion flag or serialized ticket is accepted.
        """
        self._require_state("COMPLETION_RECORDED", operation="resolve")
        lease = self._require_lease()
        try:
            self._validate_resolution_binding()
            if _guarded_controller is None:
                self.completion.event.synchronize()
            else:
                from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle

                controller = _guarded_controller
                if type(controller) is not NativeGuardedSetupLifecycle:
                    raise TypeError("CUDA resolution requires the exact guarded controller")
                context = self._transaction._context
                if (controller._lane_id != "cuda"
                        or controller.target_device_ordinal != int(context.device.index)
                        or controller.unknown_device_scope
                        or controller.possible_dirty_device_ordinals != (controller.target_device_ordinal,)):
                    raise ValueError("CUDA resolution differs from the retained exact target scope")
                controller.observe_terminal_completion(owner=self, transaction=self._transaction,
                    event=self.completion.event, stream=context.stream, target=context.device)
            # A successful device wait is only an observation. In particular,
            # a lost DISARMED acknowledgement above leaves the lease unresolved.
            self._validate_resolution_binding()
            lease.resolve_terminal_completion()
        except BaseException as exc:
            _retain_failure_without_masking(lease, f"CUDA terminal resolution failure: {exc}")
            object.__setattr__(self, "_state", "POISONED")
            raise
        object.__setattr__(self, "_state", "PAYLOAD_READY")

    def _validate_resolution_binding(self) -> None:
        """Untimed recheck of the original template, authority and terminal scope."""
        from .cuda_timeline import _validate_cuda_event

        self._assert_configuration_stable()
        self._executable.validate_bound()
        authority = self._execution_authority
        authority._assert_registered(self._transaction)
        if (authority._phase != "COMPLETION_RECORDED"
                or authority._require_live_lease() is not self._require_lease()):
            raise NativeOpticalArenaProtocolError("CUDA resolution lost its recorded lease authority")
        completion = self.completion
        context = self._transaction._context
        if (completion is not self._transaction.completion
                or completion.event is not self._executable.transaction_template.completion_event
                or completion.device != context.device
                or completion.stream_handle != context.stream_handle):
            raise NativeOpticalArenaProtocolError("CUDA resolution differs from its sealed terminal scope")
        _validate_cuda_event(completion.event, device=context.device, name="terminal completion")

    def release_device_result(self) -> None:
        self._require_state("PAYLOAD_READY", operation="device-result release")
        lease = self._require_lease()
        self._assert_after_submission_stable(lease, operation="device-result release")
        lease.release_device_result()
        object.__setattr__(self, "_state", "RECLAIMABLE")

    def reclaim(self) -> None:
        self._require_state("RECLAIMABLE", operation="reclaim")
        lease = self._require_lease()
        self._assert_after_submission_stable(lease, operation="reclaim")
        lease.reclaim()
        self._execution_authority.retire(
            expected_phase="COMPLETION_RECORDED",
            terminal_phase="RECLAIMED",
        )
        object.__setattr__(self, "_state", "RECLAIMED")

    def cancel_before_enqueue(self) -> None:
        self._require_state("ACQUIRED", operation="cancel")
        self._assert_configuration_stable()
        self._require_lease().cancel_before_enqueue()
        self._execution_authority.retire(expected_phase="ACQUIRED", terminal_phase="CANCELLED")
        object.__setattr__(self, "_state", "CANCELLED")

    def output_channels(self) -> CudaDeviceResultView:
        self._output_mapping()
        return self._device_result_view

    def _output_mapping(self) -> Mapping[str, object]:
        if self._state != "PAYLOAD_READY":
            raise NativeOpticalArenaProtocolError(
                "CUDA output channels require a resolved payload-ready generation"
            )
        self._assert_after_submission_stable(self._require_lease(), operation="output access")
        render = self._executable.render_stage
        channels = render.output_channels(
            include_traversal_diagnostics=bool(render.buffers.shadow_traversal_counters)
        )
        return MappingProxyType(channels)

    def _activate(
        self,
        lease: C4ActiveOutputLease,
        *,
        execution_authority: object,
        arena_acquire_authority: object,
    ) -> None:
        self._require_state("DORMANT", operation="activation")
        if arena_acquire_authority is not self._arena_acquire_authority:
            raise NativeOpticalArenaProtocolError("CUDA arena acquire authority is invalid")
        if (
            lease.evidence.slot_id != self._target.evidence.slot_id
            or self._transaction.state != "validated_open"
        ):
            raise NativeOpticalArenaProtocolError("CUDA arena activation binding is invalid")
        execution_authority._assert_registered(self._transaction)
        object.__setattr__(self, "_execution_authority", execution_authority)
        object.__setattr__(self, "_control_lease", lease)
        object.__setattr__(self, "_state", "ACQUIRED")

    def _configuration_signature(self) -> tuple[object, ...]:
        return (
            id(self._target),
            self._target.evidence.sha256,
            self._provenance,
            id(self._transaction),
            id(self._executable),
            id(self._executable.transaction_template),
            id(self._executable.activation_capability),
            id(self._device_result_view),
            tuple(id(stage) for stage in self._executable.transaction_template.stages),
            id(self._executable.transaction_template.completion_event),
            id(self._transaction._context),
            tuple(id(stage) for stage in self._transaction._stages),
            id(self._transaction._completion_event),
            getattr(self._transaction._completion_event, "cuda_event", None),
            tuple((name, id(start), getattr(start, "cuda_event", None),
                   id(end), getattr(end, "cuda_event", None))
                  for name, start, end in self._transaction._diagnostic_events),
        )

    def _assert_configuration_stable(self) -> None:
        if self._configuration_signature() != self._frozen_signature:
            raise NativeOpticalArenaProtocolError("CUDA active sample changed after preparation")

    def _assert_after_submission_stable(
        self,
        lease: C4ActiveOutputLease,
        *,
        operation: str,
    ) -> None:
        try:
            self._assert_configuration_stable()
        except BaseException as exc:
            _retain_failure_without_masking(lease, f"CUDA {operation} binding failure: {exc}")
            object.__setattr__(self, "_state", "POISONED")
            raise

    def _require_lease(self) -> C4ActiveOutputLease:
        if self._control_lease is None:
            raise NativeOpticalArenaProtocolError("CUDA arena sample is not active")
        return self._control_lease

    def _require_state(self, expected: str, *, operation: str) -> None:
        if self._state != expected:
            raise NativeOpticalArenaProtocolError(
                f"CUDA arena {operation} requires state {expected}; found {self._state}"
            )


@dataclass(frozen=True, slots=True)
class _CudaSlotGraph:
    slot_id: int
    world: CudaWorldSceneBuffers | None
    lbvh: CudaLbvhBuffers | None
    render: CudaDirectLightBuffers
    completion_event: object
    diagnostic_events: Mapping[str, tuple[object, object]]
    resource_records: tuple[CudaArenaResourceRecord, ...]
    targets: Mapping[tuple[int | None, int | None], CudaArenaPlanTarget]
    executables: Mapping[tuple[int | None, int | None], _CudaExecutableTarget]
    owners: tuple[object, ...]


class CudaNativeOpticalArena:
    """Sealed CUDA C4b arena with exact preflight and measured activation."""

    __slots__ = (
        "_acquire_authority",
        "_c3_provenance_by_target",
        "_control",
        "_frozen_signature",
        "_execution_by_target_id",
        "_prepared_registry",
        "_sealed",
        "_session_resource_records",
        "_session_owners",
        "_slots",
    )

    def __init__(
        self,
        *,
        control: NativeOpticalArena,
        slots: Sequence[_CudaSlotGraph],
        session_owners: Sequence[object],
        session_resource_records: Sequence[CudaArenaResourceRecord],
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "_control", control)
        object.__setattr__(self, "_slots", tuple(slots))
        executions = {
            id(executable.target): executable
            for slot in self._slots
            for executable in slot.executables.values()
        }
        if len(executions) != sum(len(slot.targets) for slot in self._slots):
            raise ValueError("CUDA arena executable targets must be identity-unique")
        object.__setattr__(self, "_execution_by_target_id", executions)
        object.__setattr__(self, "_acquire_authority", object())
        object.__setattr__(self, "_session_owners", tuple(session_owners))
        object.__setattr__(self, "_session_resource_records", tuple(session_resource_records))
        object.__setattr__(self, "_prepared_registry", weakref.WeakKeyDictionary())
        object.__setattr__(self, "_c3_provenance_by_target", {})
        object.__setattr__(self, "_frozen_signature", self._configuration_signature())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("CUDA native optical arena configuration is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CUDA native optical arena configuration is immutable")

    @property
    def spec(self) -> NativeOpticalArenaSpec:
        return self._control.spec

    @property
    def resource_manifest(self) -> NativeOpticalArenaResourceManifest:
        return self._control.resource_manifest

    @property
    def state(self) -> str:
        return self._control.state

    @property
    def session_resource_records(self) -> tuple[CudaArenaResourceRecord, ...]:
        return self._session_resource_records

    def plan_target(
        self,
        *,
        transform_frame_index: int | None,
        planned_acquisition_offset: int = 0,
        render_sample_index: int | None = None,
    ) -> CudaArenaPlanTarget:
        """Select the deterministic non-reserving slot/frame graph outside timing."""

        for index in (transform_frame_index, render_sample_index):
            if index is not None and (type(index) is not int or index < 0):
                raise ValueError("CUDA arena frame selection requires nonnegative integer indices")
        self._validate_configuration()
        lifecycle = self._control.lifecycle_evidence()
        planned = plan_native_optical_acquisition(
            next_acquire_slot_id=lifecycle.next_acquire_slot_id,
            slot_generations=tuple(slot.generation for slot in lifecycle.slots),
            planned_acquisition_offset=planned_acquisition_offset,
        )
        try:
            return self._slots[planned.slot_id].targets[transform_frame_index, render_sample_index]
        except KeyError as exc:
            raise ValueError("CUDA arena transform frame is not part of the sealed graph") from exc

    def bind_c3_provenance(
        self,
        *,
        target: CudaArenaPlanTarget,
        binding: object,
        prepared_input: object,
        native_readiness: object,
    ) -> CudaArenaC3Provenance:
        """Issue one bounded C3 provenance capability for an arena-owned plan."""

        self._validate_configuration()
        self._execution_for_target(target)
        _validate_live_a2_binding(target, prepared_input, native_readiness)
        _validate_c3_adapter_authority(
            target,
            binding,
            prepared_input,
            native_readiness,
        )
        canonical = _canonical_c3_binding(binding)
        existing = self._c3_provenance_by_target.get(id(target))
        if existing is not None:
            existing_binding_sha256, provenance, existing_prepared, existing_readiness = existing
            if (
                existing_binding_sha256 != canonical.sha256
                or existing_prepared is not prepared_input
                or existing_readiness is not native_readiness
            ):
                raise NativeOpticalArenaProtocolError(
                    "CUDA arena plan target cannot be rebound to different C3/a2 evidence"
                )
            return provenance
        provenance = target._bind_c3_provenance(canonical)
        self._c3_provenance_by_target[id(target)] = (
            canonical.sha256,
            provenance,
            prepared_input,
            native_readiness,
        )
        return provenance

    def prepare_sample(
        self,
        *,
        target: CudaArenaPlanTarget,
        provenance: CudaArenaC3Provenance,
        planned_acquisition_offset: int = 0,
    ) -> CudaPreparedArenaSample:
        """Create all per-generation Python state outside measured acquire."""

        self._validate_configuration()
        expected_target = self.plan_target(
            transform_frame_index=target.evidence.transform_frame_index,
            planned_acquisition_offset=planned_acquisition_offset,
            render_sample_index=target.evidence.render_sample_index,
        )
        if target is not expected_target:
            raise NativeOpticalArenaProtocolError("CUDA plan target is stale or not arena-owned")
        registered_provenance = self._c3_provenance_by_target.get(id(target))
        if registered_provenance is None or registered_provenance[1] is not provenance:
            raise NativeOpticalArenaProtocolError(
                "CUDA C3 provenance was not issued for this arena plan target"
            )
        _validate_c3_provenance(target, provenance)
        executable = self._execution_for_target(target)
        executable.validate_bound()
        transaction = executable.activate()
        control_plan = self._control.prepare_plan(
            transform_frame_index=target.evidence.transform_frame_index,
            c3_one_shot_binding_sha256=provenance.c3_one_shot_binding_sha256,
            ordered_stage_evidence_sha256=provenance.ordered_stage_evidence_sha256,
            dataflow_binding_sha256=provenance.dataflow_binding_sha256,
            planned_acquisition_offset=planned_acquisition_offset,
        )
        if (
            control_plan.evidence.slot_id != target.evidence.slot_id
            or control_plan.evidence.slot_storage_manifest_sha256
            != target.evidence.slot_storage_manifest_sha256
            or control_plan.evidence.terminal_event_token != target.evidence.terminal_event_token
        ):
            raise NativeOpticalArenaProtocolError("C4 control plan and CUDA graph disagree")
        prepared = CudaPreparedArenaSample(
            control_plan=control_plan,
            target=target,
            provenance=provenance,
            transaction=transaction,
            executable=executable,
            arena_acquire_authority=self._acquire_authority,
        )
        self._prepared_registry[prepared] = (
            control_plan,
            target,
            provenance,
            executable,
            prepared._active,
            prepared._execution_capability,
        )
        return prepared

    def acquire(
        self, prepared: CudaPreparedArenaSample
    ) -> CudaActiveArenaSample | NativeOpticalArenaBackpressure:
        """Perform only C4 O(1) identity/state changes in the measured region."""

        if not isinstance(prepared, CudaPreparedArenaSample):
            raise NativeOpticalArenaProtocolError("CUDA prepared sample is not arena-owned")
        if not self._sealed:
            raise NativeOpticalArenaProtocolError("CUDA arena seal is invalid")
        registered = self._prepared_registry.get(prepared)
        if registered is None:
            raise NativeOpticalArenaProtocolError("CUDA prepared sample is not arena-owned")
        prepared._validate_acquire_fast(
            control_plan=registered[0],
            target=registered[1],
            provenance=registered[2],
            executable=registered[3],
            execution_capability=registered[5],
            active=registered[4],
            arena_acquire_authority=self._acquire_authority,
        )
        result = self._control.acquire(prepared._control_plan)
        if isinstance(result, NativeOpticalArenaBackpressure):
            return result
        try:
            execution_authority = prepared._active._transaction._mint_arena_execution_authority(
                registered[5],
                result,
                active_owner=registered[4],
            )
            prepared._active._activate(
                result,
                execution_authority=execution_authority,
                arena_acquire_authority=self._acquire_authority,
            )
        except BaseException:
            if "execution_authority" in locals():
                try:
                    execution_authority.retire(
                        expected_phase="ACQUIRED",
                        terminal_phase="CANCELLED",
                    )
                except Exception:
                    pass
            if result.state == "LEASED":
                result.cancel_before_enqueue()
            self._prepared_registry.pop(prepared, None)
            raise
        del self._prepared_registry[prepared]
        return prepared._active

    def begin_drain(self) -> None:
        self._control.begin_drain()

    def close(self) -> None:
        self._control.close()

    def lifecycle_evidence(self):
        return self._control.lifecycle_evidence()

    def _configuration_signature(self) -> tuple[object, ...]:
        return (
            id(self._control),
            self._control.spec.sha256,
            self._control.resource_manifest.sha256,
            self._configuration_identity(),
        )

    def _configuration_identity(self) -> tuple[object, ...]:
        return (
            tuple(id(slot) for slot in self._slots),
            tuple(
                (
                    slot.slot_id,
                    id(slot.completion_event),
                    getattr(slot.completion_event, "cuda_event", None),
                    tuple((name, id(start), getattr(start, "cuda_event", None),
                           id(end), getattr(end, "cuda_event", None))
                          for name, (start, end) in slot.diagnostic_events.items()),
                    tuple((key, id(value)) for key, value in slot.targets.items()),
                    tuple((key, id(value)) for key, value in slot.executables.items()),
                )
                for slot in self._slots
            ),
            tuple(id(owner) for owner in self._session_owners),
            tuple(record.to_payload()["storage_token"] for record in self._session_resource_records),
        )

    def _validate_configuration(self) -> None:
        if self._configuration_signature() != self._frozen_signature:
            raise NativeOpticalArenaProtocolError("CUDA arena resource graph changed after seal")
        for slot in self._slots:
            if set(slot.targets) != set(slot.executables):
                raise NativeOpticalArenaProtocolError("CUDA arena target/executable keys changed")
            for key, target in slot.targets.items():
                if key != (target.evidence.transform_frame_index, target.evidence.render_sample_index):
                    raise NativeOpticalArenaProtocolError("CUDA arena frame selection changed")
                executable = slot.executables[key]
                executable.validate_bound()
                _validate_target_evidence(self.spec, slot, target, executable)

    def _execution_for_target(self, target: CudaArenaPlanTarget) -> _CudaExecutableTarget:
        executable = self._execution_by_target_id.get(id(target))
        if executable is None or executable.target is not target:
            raise NativeOpticalArenaProtocolError("CUDA C3 target is not owned by this arena")
        return executable


def build_cuda_dynamic_optical_arena(
    *,
    spec: NativeOpticalArenaSpec,
    context: CudaOpticalExecutionContext,
    scene_factory: CudaSceneUpdateStageFactory,
    lbvh_factory: CudaLbvhRebuildStageFactory,
    renderer: CudaDirectLightNativeRenderer,
    transforms: Sequence[CudaSceneTransformInputs],
    camera: CudaCameraParameters,
    sensor_role_mask: int,
    max_distance: float,
    ambient_rgb: tuple[float, float, float],
    background_rgb: tuple[float, float, float],
    shadow_bias: float,
) -> CudaNativeOpticalArena:
    """Allocate and seal an R1 or R2 device-only CUDA arena."""

    _validate_common_build(spec, context, renderer, camera)
    if spec.boundary not in {"R1", "R2"}:
        raise ValueError("dynamic CUDA arena requires R1 or R2")
    if spec.boundary == "R2" and (spec.payload, spec.delivery_policy) != (
        "none",
        "device_only",
    ):
        raise ValueError("C4b only supports the R2 none/device_only row")
    if spec.boundary == "R1" and (spec.payload, spec.delivery_policy) != (
        "native_only",
        "native_only",
    ):
        raise ValueError("R1 CUDA arena requires native-only delivery")
    if spec.acceleration_id != CUDA_LBVH_ACCELERATION_ID:
        raise ValueError("dynamic CUDA arena requires the reviewed CUDA LBVH identity")
    expected_acceleration_class = "production_dynamic" if spec.boundary == "R1" else "production_static"
    if spec.acceleration_class != expected_acceleration_class:
        raise ValueError(f"{spec.boundary} CUDA arena requires {expected_acceleration_class} acceleration")
    if scene_factory.context is not context or lbvh_factory.context is not context:
        raise ValueError("CUDA arena factories must share the exact execution context")
    scene = scene_factory.local_scene
    if scene.device_ordinal != spec.device_ordinal:
        raise ValueError("CUDA arena local scene is on the wrong device")
    if (
        scene.scene_contract_sha256 != spec.scene_contract_sha256
        or scene_factory.native_input_manifest_sha256 != spec.native_input_manifest_sha256
        or lbvh_factory.native_input_manifest_sha256 != spec.native_input_manifest_sha256
        or scene_factory.input_contract != spec.input_contract
        or lbvh_factory.input_contract != spec.input_contract
    ):
        raise ValueError("CUDA arena factory provenance does not match its specification")
    if scene.num_triangles != spec.primitive_count:
        raise ValueError("CUDA arena primitive count does not match its scene")
    if lbvh_factory.primitive_count != spec.primitive_count:
        raise ValueError("CUDA arena LBVH primitive count mismatch")
    if lbvh_factory.requirements.node_capacity != spec.node_capacity:
        raise ValueError("CUDA arena LBVH node capacity mismatch")
    frozen_transforms = tuple(transforms)
    if len(frozen_transforms) != spec.transform_frame_count:
        raise ValueError("CUDA arena requires every replay transform frame")
    for index, transform in enumerate(frozen_transforms):
        if (
            transform.transform_frame_index != index
            or transform.transform_replay_sha256 != spec.transform_replay_sha256
            or transform.input_contract != spec.input_contract
        ):
            raise ValueError("CUDA arena replay transform provenance is incomplete or reordered")
    _validate_render_scalars(sensor_role_mask, max_distance, shadow_bias)

    workspace = allocate_cuda_lbvh_workspace(lbvh_factory.requirements)
    shade_inputs = _shade_inputs(scene)
    diagnostic = spec.diagnostic_mode == "diagnostic_profiled"
    slot_builds: list[dict[str, object]] = []
    for slot_id in range(spec.ring_depth):
        world = allocate_cuda_world_scene_buffers(scene)
        lbvh = allocate_cuda_lbvh_buffers(
            primitive_count=spec.primitive_count,
            device=context.device,
        )
        acceleration_stage = lbvh_factory.bind(world=world, outputs=lbvh, workspace=workspace)
        first_hit = _dynamic_first_hit_inputs(
            scene=scene,
            world=world,
            lbvh=lbvh,
            sensor_role_mask=sensor_role_mask,
            max_distance=max_distance,
        )
        render_buffers = allocate_cuda_direct_light_buffers(
            num_rays=spec.num_rays,
            device=context.device,
            direct_light=True,
            traversal_diagnostics=diagnostic,
        )
        render_stage = renderer.bind_direct_light(
            first_hit_inputs=first_hit,
            shade_inputs=shade_inputs,
            buffers=render_buffers,
            camera=camera,
            output_profile=spec.output_profile,
            shadows=spec.shadows,
            ambient_rgb=ambient_rgb,
            background_rgb=background_rgb,
            shadow_bias=shadow_bias,
        )
        delivery_stage = (
            CudaDeviceOnlyDeliveryStage(
                context=context,
                buffers=render_buffers,
                output_profile=normalize_output_profile(spec.output_profile),
            )
            if spec.boundary == "R2"
            else None
        )
        completion, diagnostic_events = _prepare_events(
            context,
            boundary=spec.boundary,
            diagnostic=diagnostic,
        )
        templates: dict[tuple[int | None, int | None], tuple[CudaOpticalTransactionTemplate, object]] = {}
        activation_capability = context._new_controlled_transaction_capability()
        for frame_index, transform in enumerate(frozen_transforms):
            scene_stage = scene_factory.bind(transforms=transform, outputs=world)
            stages = (scene_stage, acceleration_stage, render_stage)
            if delivery_stage is not None:
                stages += (delivery_stage,)
            template = context.prepare_transaction_template(
                stages=stages,
                completion_event=completion,
                diagnostic_events=diagnostic_events,
                _activation_capability=activation_capability,
            )
            templates[frame_index, None] = (template, activation_capability)
        slot_builds.append(
            {
                "slot_id": slot_id,
                "world": world,
                "lbvh": lbvh,
                "render": render_buffers,
                "completion": completion,
                "diagnostics": diagnostic_events,
                "templates": templates,
                "stage_owners": (acceleration_stage, render_stage, delivery_stage),
            }
        )
    session_objects = (
        context,
        context.stream,
        scene,
        scene_factory,
        lbvh_factory,
        renderer,
        workspace,
        shade_inputs,
        *frozen_transforms,
    )
    session_tensors = (
        tuple((f"local_scene.{name}", value) for name, value in scene.arrays.items())
        + tuple(
            (f"transform[{index}].{name}", value)
            for index, transform in enumerate(frozen_transforms)
            for name, value in (
                ("x_world_R", transform.x_world_R),
                ("x_world_r", transform.x_world_r),
            )
        )
        + (("lbvh.workspace", workspace.storage),)
    )
    return _seal_cuda_arena(
        spec=spec,
        slot_builds=slot_builds,
        session_objects=session_objects,
        session_tensors=session_tensors,
        runtime_scene_arrays=scene.arrays,
        derived_fixture_arrays={},
    )


def build_cuda_fixed_optical_arena(
    *,
    spec: NativeOpticalArenaSpec,
    context: CudaOpticalExecutionContext,
    renderer: CudaDirectLightNativeRenderer,
    first_hit_inputs: CudaFirstHitInputs,
    shade_inputs: CudaShadeInputs,
    camera: CudaCameraParameters,
    ambient_rgb: tuple[float, float, float],
    background_rgb: tuple[float, float, float],
    shadow_bias: float,
    runtime_scene_arrays: Mapping[str, object],
    derived_fixture_arrays: Mapping[str, object],
    resource_owners: Sequence[object] = (),
) -> CudaNativeOpticalArena:
    """Allocate and seal an R0 fixed-fixture CUDA arena."""

    _validate_common_build(spec, context, renderer, camera)
    if spec.boundary != "R0":
        raise ValueError("fixed CUDA arena requires R0")
    if spec.acceleration_class != "fixed_bvh_fixture" or spec.acceleration_id != "fixed_bvh_fixture":
        raise ValueError("R0 CUDA arena requires the fixed BVH fixture identity")
    if int(first_hit_inputs.triangle_v0_world.shape[0]) != spec.primitive_count:
        raise ValueError("R0 CUDA arena primitive count does not match its fixture")
    if first_hit_inputs.num_bvh_nodes != spec.node_capacity:
        raise ValueError("R0 CUDA arena node capacity does not match its fixture")
    frozen_runtime_scene = MappingProxyType(
        {name: runtime_scene_arrays[name] for name in sorted(runtime_scene_arrays)}
    )
    frozen_derived_fixture = MappingProxyType(
        {name: derived_fixture_arrays[name] for name in sorted(derived_fixture_arrays)}
    )
    _validate_fixed_input_bindings(
        context=context,
        first_hit_inputs=first_hit_inputs,
        shade_inputs=shade_inputs,
        runtime_scene_arrays=frozen_runtime_scene,
        derived_fixture_arrays=frozen_derived_fixture,
    )
    diagnostic = spec.diagnostic_mode == "diagnostic_profiled"
    slot_builds: list[dict[str, object]] = []
    for slot_id in range(spec.ring_depth):
        render_buffers = allocate_cuda_direct_light_buffers(
            num_rays=spec.num_rays,
            device=context.device,
            direct_light=True,
            traversal_diagnostics=diagnostic,
        )
        render_stage = renderer.bind_direct_light(
            first_hit_inputs=first_hit_inputs,
            shade_inputs=shade_inputs,
            buffers=render_buffers,
            camera=camera,
            output_profile=spec.output_profile,
            shadows=spec.shadows,
            ambient_rgb=ambient_rgb,
            background_rgb=background_rgb,
            shadow_bias=shadow_bias,
        )
        completion, diagnostic_events = _prepare_events(
            context,
            boundary="R0",
            diagnostic=diagnostic,
        )
        activation_capability = context._new_controlled_transaction_capability()
        template = context.prepare_transaction_template(
            stages=(render_stage,),
            completion_event=completion,
            diagnostic_events=diagnostic_events,
            _activation_capability=activation_capability,
        )
        slot_builds.append(
            {
                "slot_id": slot_id,
                "world": None,
                "lbvh": None,
                "render": render_buffers,
                "completion": completion,
                "diagnostics": diagnostic_events,
                "templates": {(None, None): (template, activation_capability)},
                "stage_owners": (render_stage,),
            }
        )
    session_tensors = tuple((f"scene.{name}", value) for name, value in frozen_runtime_scene.items()) + tuple(
        (f"fixed_render.{name}", value) for name, value in frozen_derived_fixture.items()
    )
    session_objects = (
        context,
        context.stream,
        renderer,
        first_hit_inputs,
        shade_inputs,
        frozen_runtime_scene,
        frozen_derived_fixture,
        *tuple(resource_owners),
    )
    return _seal_cuda_arena(
        spec=spec,
        slot_builds=slot_builds,
        session_objects=session_objects,
        session_tensors=session_tensors,
        runtime_scene_arrays=frozen_runtime_scene,
        derived_fixture_arrays=frozen_derived_fixture,
    )


def cuda_execution_stream_token(context: CudaOpticalExecutionContext) -> str:
    return f"torch-stream:{int(context.device.index)}:{context.stream_handle}"


def _seal_cuda_arena(
    *,
    spec: NativeOpticalArenaSpec,
    slot_builds: Sequence[dict[str, object]],
    session_objects: Sequence[object],
    session_tensors: Sequence[tuple[str, object]],
    runtime_scene_arrays: Mapping[str, object],
    derived_fixture_arrays: Mapping[str, object],
) -> CudaNativeOpticalArena:
    session_records = _tensor_resource_records("session", session_tensors)
    session_records.append(
        CudaArenaResourceRecord(
            schema_version=CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION,
            resource_kind="python_owner",
            device_ordinal=spec.device_ordinal,
            storage_token=f"cuda-session-control:{id(session_objects):x}",
            logical_views=(
                CudaArenaLogicalResourceView(
                    logical_name="session.control_graph",
                    dtype="opaque",
                    shape=(),
                    stride=(),
                    active_offset_bytes=0,
                    active_nbytes=0,
                ),
            ),
            nbytes=0,
        )
    )
    session_owner_token = "cuda-session-owner"
    slot_manifests: list[NativeOpticalSlotResourceManifest] = []
    slot_resource_records: list[list[CudaArenaResourceRecord]] = []
    finalized_slots: list[_CudaSlotGraph] = []
    slot_owner_objects: dict[str, object] = {}

    for build in slot_builds:
        slot_id = int(build["slot_id"])
        world = build["world"]
        lbvh = build["lbvh"]
        render = build["render"]
        completion = build["completion"]
        diagnostics = build["diagnostics"]
        templates = build["templates"]
        tensors: list[tuple[str, object]] = []
        if world is not None:
            tensors.extend((f"world.{name}", value) for name, value in world.tensor_items())
        if lbvh is not None:
            tensors.extend((f"lbvh.{name}", value) for name, value in lbvh.tensor_items())
        tensors.extend(_render_buffer_items(render))
        records = _tensor_resource_records(f"slot{slot_id}", tensors)
        terminal_event_token = _event_token(completion, slot_id=slot_id, name="terminal")
        records.append(
            _event_resource_record(
                completion,
                token=terminal_event_token,
                logical_name="terminal_event",
            )
        )
        for phase_name, (start, end) in diagnostics.items():
            start_token = _event_token(start, slot_id=slot_id, name=f"{phase_name}:start")
            end_token = _event_token(end, slot_id=slot_id, name=f"{phase_name}:end")
            records.extend(
                (
                    _event_resource_record(
                        start,
                        token=start_token,
                        logical_name=f"diagnostic.{phase_name}.start",
                    ),
                    _event_resource_record(
                        end,
                        token=end_token,
                        logical_name=f"diagnostic.{phase_name}.end",
                    ),
                )
            )
        slot_owner_token = f"cuda-slot-owner:{slot_id}"
        manifest = NativeOpticalSlotResourceManifest(
            schema_version=NATIVE_OPTICAL_SLOT_RESOURCE_SCHEMA_VERSION,
            slot_id=slot_id,
            slot_storage_manifest_sha256=_canonical_sha256([record.to_payload() for record in records]),
            terminal_event_token=terminal_event_token,
            resource_tokens=tuple(record.storage_token for record in records),
            resource_owner_tokens=tuple(slot_owner_token for _ in records),
            resource_nbytes=tuple(record.nbytes for record in records),
        )
        slot_manifests.append(manifest)
        slot_resource_records.append(records)
        targets: dict[tuple[int | None, int | None], CudaArenaPlanTarget] = {}
        executables: dict[tuple[int | None, int | None], _CudaExecutableTarget] = {}
        for (frame_index, render_sample_index), controlled_template in templates.items():
            template, activation_capability = controlled_template
            runtime_sha256 = _runtime_stage_evidence_sha256(template)
            plan_payload = {
                "schema_version": CUDA_NATIVE_ARENA_PLAN_SCHEMA_VERSION,
                "arena_spec_sha256": spec.sha256,
                "slot_id": slot_id,
                "slot_storage_manifest_sha256": manifest.slot_storage_manifest_sha256,
                "slot_resource_contract_sha256": manifest.slot_storage_manifest_sha256,
                "transform_frame_index": frame_index,
                "render_sample_index": render_sample_index,
                "stage_names": list(template.stage_names),
                "ordered_runtime_stage_evidence_sha256": runtime_sha256,
                "terminal_event_token": terminal_event_token,
                "diagnostic_phase_names": [name for name, _, _ in template.diagnostic_events],
            }
            evidence = CudaArenaPlanEvidence(
                schema_version=CUDA_NATIVE_ARENA_PLAN_SCHEMA_VERSION,
                arena_spec_sha256=spec.sha256,
                slot_id=slot_id,
                slot_storage_manifest_sha256=manifest.slot_storage_manifest_sha256,
                slot_resource_contract_sha256=manifest.slot_storage_manifest_sha256,
                transform_frame_index=frame_index,
                render_sample_index=render_sample_index,
                stage_names=tuple(template.stage_names),
                ordered_runtime_stage_evidence_sha256=runtime_sha256,
                terminal_event_token=terminal_event_token,
                diagnostic_phase_names=tuple(name for name, _, _ in template.diagnostic_events),
                plan_binding_sha256=_canonical_sha256(plan_payload),
            )
            template._bind_arena_plan_identity(
                activation_capability,
                slot_id=evidence.slot_id,
                terminal_event_token=evidence.terminal_event_token,
                transform_frame_index=evidence.transform_frame_index,
                plan_binding_sha256=evidence.plan_binding_sha256,
                ordered_runtime_stage_evidence_sha256=(evidence.ordered_runtime_stage_evidence_sha256),
            )
            if spec.boundary == "R0":
                render_stage = template.stages[0]
                transform_arrays: dict[str, object] = {}
                scene_output_arrays: dict[str, object] = {}
                acceleration_output_arrays: dict[str, object] = {}
            else:
                scene_stage, acceleration_stage, render_stage = template.stages[:3]
                transform_arrays = {
                    "replay_x_world_R": scene_stage.transforms.x_world_R,
                    "replay_x_world_r": scene_stage.transforms.x_world_r,
                }
                scene_output_arrays = dict(scene_stage.outputs.tensor_items())
                acceleration_output_arrays = {
                    "bvh_bounds_min": acceleration_stage.outputs.node_bounds_min,
                    "bvh_bounds_max": acceleration_stage.outputs.node_bounds_max,
                    "bvh_left": acceleration_stage.outputs.left,
                    "bvh_right": acceleration_stage.outputs.right,
                    "bvh_start": acceleration_stage.outputs.start,
                    "bvh_count": acceleration_stage.outputs.count,
                    "bvh_prim_ids": acceleration_stage.outputs.sorted_primitive_ids,
                }
            target = CudaArenaPlanTarget(
                evidence=evidence,
                _render_buffers=render,
                slot_resources=tuple(records),
                _spec=NativeOpticalArenaSpec.from_payload(spec.to_payload()),
                _runtime_scene_arrays=MappingProxyType(dict(runtime_scene_arrays)),
                _derived_fixture_arrays=MappingProxyType(dict(derived_fixture_arrays)),
                _transform_arrays=MappingProxyType(transform_arrays),
                _scene_output_arrays=MappingProxyType(scene_output_arrays),
                _acceleration_output_arrays=MappingProxyType(acceleration_output_arrays),
                _runtime_stage_evidence=tuple(template.stage_evidence),
            )
            targets[frame_index, render_sample_index] = target
            executables[frame_index, render_sample_index] = _CudaExecutableTarget(
                target=target,
                transaction_template=template,
                activation_capability=activation_capability,
                render_stage=render_stage,
            )
        slot_graph = _CudaSlotGraph(
            slot_id=slot_id,
            world=world,
            lbvh=lbvh,
            render=render,
            completion_event=completion,
            diagnostic_events=MappingProxyType(dict(diagnostics)),
            resource_records=tuple(records),
            targets=MappingProxyType(targets),
            executables=MappingProxyType(executables),
            owners=tuple(build["stage_owners"]) + tuple(executables.values()),
        )
        finalized_slots.append(slot_graph)
        slot_owner_objects[slot_owner_token] = slot_graph

    manifest = NativeOpticalArenaResourceManifest(
        schema_version=NATIVE_OPTICAL_RESOURCE_MANIFEST_SCHEMA_VERSION,
        arena_spec_sha256=spec.sha256,
        session_resource_tokens=tuple(record.storage_token for record in session_records),
        session_resource_owner_tokens=tuple(session_owner_token for _ in session_records),
        session_resource_nbytes=tuple(record.nbytes for record in session_records),
        slots=tuple(slot_manifests),
        total_resource_nbytes=sum(record.nbytes for record in session_records)
        + sum(slot.total_resource_nbytes for slot in slot_manifests),
    )
    _validate_global_storage_disjointness(session_records, slot_resource_records)
    owners = {
        session_owner_token: tuple(session_objects) + tuple(finalized_slots),
        **slot_owner_objects,
    }
    control = NativeOpticalArena(
        spec=spec,
        resource_manifest=manifest,
        resource_owners=owners,
    )
    arena = CudaNativeOpticalArena(
        control=control,
        slots=finalized_slots,
        session_owners=(owners,),
        session_resource_records=session_records,
    )
    arena._validate_configuration()
    return arena


def _validate_common_build(
    spec: NativeOpticalArenaSpec,
    context: CudaOpticalExecutionContext,
    renderer: CudaDirectLightNativeRenderer,
    camera: CudaCameraParameters,
) -> None:
    if not isinstance(spec, NativeOpticalArenaSpec):
        raise TypeError("CUDA arena requires NativeOpticalArenaSpec")
    if spec.lane != "cuda" or spec.boundary not in _SUPPORTED_BOUNDARIES:
        raise ValueError("CUDA arena requires an R0/R1/R2 CUDA specification")
    if spec.renderer_id != "cuda_direct_light":
        raise ValueError("CUDA arena renderer identity is invalid")
    if spec.execution_stream_token != cuda_execution_stream_token(context):
        raise ValueError("CUDA arena stream token does not match its execution context")
    if spec.device_ordinal != context.device.index or renderer.context is not context:
        raise ValueError("CUDA arena device/context identity mismatch")
    if camera.width != spec.width or camera.height != spec.height:
        raise ValueError("CUDA arena camera dimensions do not match its specification")
    if camera.width * camera.height != spec.num_rays:
        raise ValueError("CUDA arena ray capacity does not match its camera")
    normalize_output_profile(spec.output_profile)


def _validate_fixed_input_bindings(
    *,
    context: CudaOpticalExecutionContext,
    first_hit_inputs: CudaFirstHitInputs,
    shade_inputs: CudaShadeInputs,
    runtime_scene_arrays: Mapping[str, object],
    derived_fixture_arrays: Mapping[str, object],
) -> None:
    if set(runtime_scene_arrays) != set(SCENE_ARRAY_SEMANTICS):
        raise ValueError("R0 CUDA arena requires the complete packed-scene allocation set")
    if set(derived_fixture_arrays) != set(_FIXED_DERIVED_ARRAY_NAMES):
        raise ValueError("R0 CUDA arena requires the complete derived-fixture allocation set")
    combined = dict(runtime_scene_arrays) | dict(derived_fixture_arrays)
    for name, tensor in combined.items():
        device = getattr(tensor, "device", None)
        if (
            getattr(device, "type", None) != "cuda"
            or getattr(device, "index", None) != context.device.index
            or not bool(tensor.is_contiguous())
        ):
            raise ValueError(f"R0 CUDA arena storage {name!r} is not exact indexed CUDA storage")
    consumed = dict(first_hit_inputs.tensor_items()) | dict(shade_inputs.tensor_items())
    for name, tensor in consumed.items():
        if combined.get(name) is not tensor:
            raise ValueError(f"R0 CUDA renderer input {name!r} does not reference the prepared allocation")
    nonempty_storage: set[tuple[int, int]] = set()
    for name, tensor in combined.items():
        storage = tensor.untyped_storage()
        if storage.nbytes() == 0:
            continue
        key = (int(getattr(storage, "_cdata", 0)), int(storage.data_ptr()))
        if key in nonempty_storage:
            raise ValueError(f"R0 CUDA storage {name!r} aliases another logical allocation")
        nonempty_storage.add(key)


def _validate_render_scalars(sensor_role_mask: int, max_distance: float, shadow_bias: float) -> None:
    if not isinstance(sensor_role_mask, int) or isinstance(sensor_role_mask, bool) or sensor_role_mask < 0:
        raise ValueError("CUDA arena sensor role mask is invalid")
    if not math.isfinite(float(max_distance)) or float(max_distance) <= 0.0:
        raise ValueError("CUDA arena max distance must be finite and positive")
    if not math.isfinite(float(shadow_bias)) or float(shadow_bias) < 0.0:
        raise ValueError("CUDA arena shadow bias must be finite and non-negative")


def _prepare_events(
    context: CudaOpticalExecutionContext,
    *,
    boundary: str,
    diagnostic: bool,
) -> tuple[object, Mapping[str, tuple[object, object]]]:
    import torch

    from optical_worker.native_setup import _prepare_native_events, _render_phase_names

    context._validate_identity()
    completion, pairs = _prepare_native_events(torch, context.device, context.stream, lane="cuda",
        phase_names=_render_phase_names(boundary=boundary, diagnostic=diagnostic))
    return completion, MappingProxyType({name: (start, end) for name, start, end in pairs})


def _runtime_stage_evidence_sha256(template: CudaOpticalTransactionTemplate) -> str:
    return _canonical_sha256([_plain_payload(value) for value in template.stage_evidence])


def _validate_target_evidence(
    spec: NativeOpticalArenaSpec,
    slot: _CudaSlotGraph,
    target: CudaArenaPlanTarget,
    executable: _CudaExecutableTarget,
) -> None:
    evidence = target.evidence
    if target._spec.to_payload() != spec.to_payload():
        raise NativeOpticalArenaProtocolError("CUDA arena target specification changed")
    if target.slot_resources != slot.resource_records or target._render_buffers is not slot.render:
        raise NativeOpticalArenaProtocolError("CUDA arena target resource binding changed")
    executable.validate_bound()
    if evidence.schema_version != CUDA_NATIVE_ARENA_PLAN_SCHEMA_VERSION:
        raise NativeOpticalArenaProtocolError("CUDA arena plan schema changed")
    expected_stages = (
        ("cuda_direct_light_render",)
        if spec.boundary == "R0"
        else (
            "cuda_scene_update",
            "cuda_lbvh_rebuild",
            "cuda_direct_light_render",
        )
        + (("cuda_device_only_delivery",) if spec.boundary == "R2" else ())
    )
    if evidence.stage_names != expected_stages:
        raise NativeOpticalArenaProtocolError(
            f"CUDA arena plan stage order changed: {evidence.stage_names!r} != {expected_stages!r}"
        )
    if evidence.arena_spec_sha256 != spec.sha256:
        raise NativeOpticalArenaProtocolError("CUDA arena plan specification digest changed")
    if (
        evidence.slot_id != slot.slot_id
        or executable.transaction_template.completion_event is not slot.completion_event
    ):
        raise NativeOpticalArenaProtocolError("CUDA arena plan slot/event binding changed")
    if evidence.ordered_runtime_stage_evidence_sha256 != _runtime_stage_evidence_sha256(
        executable.transaction_template
    ):
        raise NativeOpticalArenaProtocolError("CUDA arena runtime stage evidence changed")
    payload = {
        "schema_version": evidence.schema_version,
        "arena_spec_sha256": evidence.arena_spec_sha256,
        "slot_id": evidence.slot_id,
        "slot_storage_manifest_sha256": evidence.slot_storage_manifest_sha256,
        "slot_resource_contract_sha256": evidence.slot_resource_contract_sha256,
        "transform_frame_index": evidence.transform_frame_index,
        "render_sample_index": evidence.render_sample_index,
        "stage_names": list(evidence.stage_names),
        "ordered_runtime_stage_evidence_sha256": evidence.ordered_runtime_stage_evidence_sha256,
        "terminal_event_token": evidence.terminal_event_token,
        "diagnostic_phase_names": list(evidence.diagnostic_phase_names),
    }
    if evidence.plan_binding_sha256 != _canonical_sha256(payload):
        raise NativeOpticalArenaProtocolError("CUDA arena plan binding digest changed")
    resource_sha256 = _canonical_sha256([record.to_payload() for record in target.slot_resources])
    if (
        evidence.slot_resource_contract_sha256 != resource_sha256
        or evidence.slot_storage_manifest_sha256 != resource_sha256
    ):
        raise NativeOpticalArenaProtocolError("CUDA arena slot resource contract changed")


def _validate_c3_provenance(
    target: CudaArenaPlanTarget,
    provenance: CudaArenaC3Provenance,
) -> None:
    if not isinstance(provenance, CudaArenaC3Provenance):
        raise TypeError("CUDA arena requires C3 provenance")
    if (
        provenance.plan_binding_sha256 != target.evidence.plan_binding_sha256
        or provenance.ordered_runtime_stage_evidence_sha256
        != target.evidence.ordered_runtime_stage_evidence_sha256
    ):
        raise NativeOpticalArenaProtocolError("C3 provenance is not bound to this CUDA plan")


def _validate_live_a2_binding(
    target: CudaArenaPlanTarget,
    prepared_input: object,
    native_readiness: object,
) -> None:
    if (
        type(prepared_input) is not CudaPreparedNativeInput
        or type(native_readiness) is not NativeReadinessEvidence
    ):
        raise TypeError("CUDA arena requires the exact live a2 CUDA input/readiness types")
    assert_cuda_prepared_authority(prepared_input, native_readiness)

    spec = target._spec
    manifest = prepared_input.manifest
    expected = {
        "device_ordinal": spec.device_ordinal,
        "manifest_sha256": spec.native_input_manifest_sha256,
        "readiness_sha256": spec.native_readiness_sha256,
        "input_contract": spec.input_contract,
        "scene_contract_sha256": spec.scene_contract_sha256,
        "transform_replay_sha256": spec.transform_replay_sha256,
        "transform_frame_count": spec.transform_frame_count,
        "fixed_render_fixture_sha256": spec.fixed_render_fixture_sha256,
    }
    observed = {
        "device_ordinal": prepared_input.device_ordinal,
        "manifest_sha256": manifest.sha256,
        "readiness_sha256": native_readiness.sha256,
        "input_contract": manifest.input_contract,
        "scene_contract_sha256": manifest.scene_contract_sha256,
        "transform_replay_sha256": manifest.transform_replay_sha256,
        "transform_frame_count": manifest.transform_frame_count,
        "fixed_render_fixture_sha256": manifest.fixed_render_fixture_sha256,
    }
    if observed != expected:
        raise NativeOpticalArenaProtocolError("CUDA arena a2 provenance does not match its plan")

    arrays = prepared_input.arrays
    if spec.boundary == "R0":
        _require_exact_tensor_objects(
            target._runtime_scene_arrays,
            arrays,
            tuple(manifest.scene_array_names),
            label="R0 packed scene",
        )
        _require_exact_tensor_objects(
            target._derived_fixture_arrays,
            arrays,
            tuple(manifest.fixed_render_fixture_arrays),
            label="R0 fixed fixture",
        )
        if target._transform_arrays:
            raise NativeOpticalArenaProtocolError("R0 CUDA arena cannot bind replay storage")
        return

    _require_exact_tensor_objects(
        target._runtime_scene_arrays,
        arrays,
        tuple(manifest.scene_array_names),
        label="dynamic packed scene",
    )
    frame_index = target.evidence.transform_frame_index
    if frame_index is None:
        raise NativeOpticalArenaProtocolError("dynamic CUDA arena lacks a replay frame")
    for name in ("replay_x_world_R", "replay_x_world_r"):
        if name not in arrays or name not in target._transform_arrays:
            raise NativeOpticalArenaProtocolError("dynamic CUDA arena replay storage is incomplete")
        expected_view = arrays[name][frame_index : frame_index + 1]
        if _tensor_view_identity(target._transform_arrays[name]) != _tensor_view_identity(expected_view):
            raise NativeOpticalArenaProtocolError(
                f"CUDA arena replay storage {name!r} is not the approved a2 frame view"
            )


def _require_exact_tensor_objects(
    actual: Mapping[str, object],
    prepared: Mapping[str, object],
    names: tuple[str, ...],
    *,
    label: str,
) -> None:
    if set(actual) != set(names):
        raise NativeOpticalArenaProtocolError(f"CUDA arena {label} storage names are not exact")
    for name in names:
        if prepared.get(name) is not actual[name]:
            raise NativeOpticalArenaProtocolError(
                f"CUDA arena {label} storage {name!r} is not the exact a2 allocation"
            )


def _tensor_view_identity(tensor: object) -> tuple[object, ...]:
    storage = tensor.untyped_storage()
    return (
        int(tensor.device.index),
        str(tensor.dtype),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.data_ptr()) - int(storage.data_ptr()) if storage.nbytes() else 0,
    )


def _canonical_c3_binding(binding: object) -> object:
    from optical_worker.provenance import C3OneShotNativeExecutionBindingEvidence

    if type(binding) is not C3OneShotNativeExecutionBindingEvidence:
        raise TypeError("CUDA arena requires canonical C3 one-shot binding evidence")
    canonical = C3OneShotNativeExecutionBindingEvidence.from_payload(
        C3OneShotNativeExecutionBindingEvidence.to_payload(binding)
    )
    if canonical != binding:
        raise NativeOpticalArenaProtocolError("CUDA arena C3 binding is not canonical")
    return canonical


def _validate_c3_adapter_authority(
    target: CudaArenaPlanTarget,
    binding: object,
    prepared_input: object,
    native_readiness: object,
) -> None:
    from optical_worker.provenance import _assert_cuda_arena_c3_binding_authority

    try:
        _assert_cuda_arena_c3_binding_authority(
            binding,
            target=target,
            prepared_input=prepared_input,
            native_readiness=native_readiness,
        )
    except (TypeError, ValueError) as exc:
        raise NativeOpticalArenaProtocolError(str(exc)) from exc


def _validate_c3_binding_identity(target: CudaArenaPlanTarget, binding: object) -> None:
    spec = target._spec
    expected = {
        "row_id": spec.row_id,
        "case_id": spec.case_id,
        "case_sha256": spec.case_sha256,
        "boundary": spec.boundary,
        "lane": "cuda",
        "renderer_id": spec.renderer_id,
        "acceleration_class": spec.acceleration_class,
        "acceleration_id": spec.acceleration_id,
        "output_profile": spec.output_profile,
        "shadows": spec.shadows,
        "input_contract": spec.input_contract,
        "native_input_manifest_sha256": spec.native_input_manifest_sha256,
        "native_readiness_sha256": spec.native_readiness_sha256,
        "scene_contract_sha256": spec.scene_contract_sha256,
        "transform_replay_sha256": spec.transform_replay_sha256,
        "transform_frame_index": target.evidence.transform_frame_index,
        "fixed_render_fixture_sha256": spec.fixed_render_fixture_sha256,
    }
    for name, expected_value in expected.items():
        if getattr(binding, name, object()) != expected_value:
            raise NativeOpticalArenaProtocolError(
                f"CUDA arena C3 binding {name} does not match its sealed plan"
            )
    stages = tuple(entry.evidence for entry in binding.ordered_stage_evidence)
    runtime_evidence = target._runtime_stage_evidence
    if spec.boundary == "R0":
        expected_stage_names = ("render",)
        runtime_evidence = runtime_evidence[:1]
    else:
        expected_stage_names = ("scene_update", "acceleration", "render")
        runtime_evidence = runtime_evidence[:3]
    if tuple(entry.stage_name for entry in binding.ordered_stage_evidence) != expected_stage_names:
        raise NativeOpticalArenaProtocolError("CUDA arena C3 stage order is not exact")
    expected_operations = tuple(
        spec.renderer_id if index == len(runtime_evidence) - 1 else getattr(evidence, "operation_id", None)
        for index, evidence in enumerate(runtime_evidence)
    )
    if tuple(stage.operation_id for stage in stages) != expected_operations:
        raise NativeOpticalArenaProtocolError("CUDA arena C3 operation identity drifted")
    for stage, runtime_stage_evidence in zip(stages, runtime_evidence, strict=True):
        if stage.stage_name == "render":
            runtime_stage_evidence.validate_render_parameters(binding.bound_render_parameters.to_payload())
            expected_dispatch = _canonical_sha256(_plain_payload(runtime_stage_evidence))
            if stage.output_profile != spec.output_profile or stage.shadows != spec.shadows:
                raise NativeOpticalArenaProtocolError("CUDA arena C3 render identity drifted")
        else:
            expected_dispatch = runtime_stage_evidence.dispatch_plan_sha256
        if stage.dispatch_plan_sha256 != expected_dispatch:
            raise NativeOpticalArenaProtocolError("CUDA arena C3 dispatch evidence drifted")


def _validate_c3_binding_storage(target: CudaArenaPlanTarget, binding: object) -> None:
    actual = _c3_storage_map(target)
    signatures = tuple(binding.runtime_scene_storage) + tuple(binding.derived_fixture_storage)
    for stage in binding.ordered_stage_evidence:
        signatures += tuple(stage.evidence.input_storage) + tuple(stage.evidence.output_storage)
    seen: dict[str, object] = {}
    for signature in signatures:
        prior = seen.get(signature.name)
        if prior is not None and prior != signature:
            raise NativeOpticalArenaProtocolError(
                f"CUDA arena C3 storage {signature.name!r} has inconsistent identities"
            )
        seen[signature.name] = signature
        try:
            tensor, owner_token = actual[signature.name]
        except KeyError as exc:
            raise NativeOpticalArenaProtocolError(
                f"CUDA arena C3 storage {signature.name!r} is not in the sealed graph"
            ) from exc
        _validate_c3_storage_signature(signature, tensor=tensor, owner_token=owner_token)
    render_stage = binding.ordered_stage_evidence[-1].evidence
    expected_outputs = tuple(name for name, _ in target._render_buffers.tensor_items())
    if tuple(value.name for value in render_stage.output_storage) != expected_outputs:
        raise NativeOpticalArenaProtocolError(
            "CUDA arena C3 render outputs must cover the exact ordered buffer ABI"
        )


def _c3_storage_map(
    target: CudaArenaPlanTarget,
) -> dict[str, tuple[object, str]]:
    spec = target._spec
    session_owner = spec.native_input_manifest_sha256
    slot_owner = target.evidence.slot_storage_manifest_sha256
    values: dict[str, tuple[object, str]] = {}

    def add(name: str, tensor: object, owner_token: str) -> None:
        prior = values.get(name)
        if prior is not None and (prior[0] is not tensor or prior[1] != owner_token):
            raise NativeOpticalArenaProtocolError(
                f"CUDA arena logical storage {name!r} is bound more than once"
            )
        values[name] = (tensor, owner_token)

    if spec.boundary == "R0":
        for name, tensor in target._runtime_scene_arrays.items():
            add(name, tensor, session_owner)
        for name, tensor in target._derived_fixture_arrays.items():
            add(name, tensor, session_owner)
    else:
        for name, tensor in target._runtime_scene_arrays.items():
            add(name, tensor, session_owner)
        for name, tensor in target._transform_arrays.items():
            add(name, tensor, session_owner)
        for name, tensor in target._scene_output_arrays.items():
            add(name, tensor, slot_owner)
        for name, tensor in target._acceleration_output_arrays.items():
            add(name, tensor, slot_owner)
    for name, tensor in target._render_buffers.tensor_items():
        add(name, tensor, slot_owner)
    return values


def _validate_c3_storage_signature(
    signature: object,
    *,
    tensor: object,
    owner_token: str,
) -> None:
    storage = tensor.untyped_storage()
    storage_nbytes = int(storage.nbytes())
    logical_nbytes = int(tensor.numel()) * int(tensor.element_size())
    base_pointer = int(storage.data_ptr()) if storage_nbytes else 0
    active_start = int(tensor.data_ptr()) - base_pointer if storage_nbytes else 0
    expected_token = (
        f"cuda:0x{base_pointer:x}" if storage_nbytes else f"cuda:empty:{signature.name}:{owner_token}"
    )
    expected_dtype = {
        "torch.bool": "|b1",
        "torch.uint8": "|u1",
        "torch.int32": "<i4",
        "torch.int64": "<i8",
        "torch.float32": "<f4",
        "torch.float64": "<f8",
    }.get(str(tensor.dtype))
    if expected_dtype is None:
        raise NativeOpticalArenaProtocolError("CUDA arena C3 storage dtype is unsupported")
    observed = (
        signature.backend,
        signature.device_ordinal,
        signature.storage_token,
        signature.dtype,
        tuple(signature.shape),
        signature.nbytes,
        signature.active_start,
        signature.active_count,
        signature.owner_token,
    )
    expected = (
        "cuda",
        int(tensor.device.index),
        expected_token,
        expected_dtype,
        tuple(int(value) for value in tensor.shape),
        storage_nbytes,
        active_start,
        logical_nbytes,
        owner_token,
    )
    if observed != expected:
        raise NativeOpticalArenaProtocolError(
            f"CUDA arena C3 storage {signature.name!r} does not match its exact allocation"
        )


def _tensor_resource_records(
    scope: str, tensors: Sequence[tuple[str, object]]
) -> list[CudaArenaResourceRecord]:
    grouped: dict[tuple[int, int, int, int], dict[str, object]] = {}
    for logical_name, tensor in tensors:
        device = getattr(tensor, "device", None)
        if getattr(device, "type", None) != "cuda" or getattr(device, "index", None) is None:
            raise ValueError("CUDA arena resource manifest accepts only indexed CUDA tensors")
        storage = tensor.untyped_storage()
        key = (
            int(device.index),
            int(getattr(storage, "_cdata", 0)),
            int(storage.data_ptr()),
            int(storage.nbytes()),
        )
        record = grouped.get(key)
        alias = f"{scope}.{logical_name}"
        base_pointer = int(storage.data_ptr()) if storage.nbytes() else 0
        view = CudaArenaLogicalResourceView(
            logical_name=alias,
            dtype=str(tensor.dtype),
            shape=tuple(int(value) for value in tensor.shape),
            stride=tuple(int(value) for value in tensor.stride()),
            active_offset_bytes=(int(tensor.data_ptr()) - base_pointer if storage.nbytes() else 0),
            active_nbytes=int(tensor.numel()) * int(tensor.element_size()),
        )
        if record is None:
            token = (
                f"cuda-storage:{int(device.index)}:"
                f"{int(getattr(storage, '_cdata', 0)):x}:{int(storage.data_ptr()):x}"
            )
            grouped[key] = {
                "logical_views": [view],
                "storage_token": token,
                "nbytes": int(storage.nbytes()),
            }
        else:
            record["logical_views"].append(view)
    return [
        CudaArenaResourceRecord(
            schema_version=CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION,
            resource_kind="cuda_storage",
            device_ordinal=key[0],
            storage_token=str(grouped[key]["storage_token"]),
            logical_views=tuple(sorted(grouped[key]["logical_views"], key=lambda value: value.logical_name)),
            nbytes=int(grouped[key]["nbytes"]),
        )
        for key in sorted(grouped)
    ]


def _validate_global_storage_disjointness(
    session_records: Sequence[CudaArenaResourceRecord],
    slot_records: Sequence[Sequence[CudaArenaResourceRecord]],
) -> None:
    all_records = list(session_records)
    for records in slot_records:
        all_records.extend(records)
    tokens = [record.storage_token for record in all_records]
    if len(tokens) != len(set(tokens)):
        raise ValueError("CUDA arena physical resource tokens must be globally disjoint")
    storage_keys = [
        tuple(record.storage_token.rsplit(":", 3)[-3:])
        for record in all_records
        if record.resource_kind == "cuda_storage" and record.nbytes > 0
    ]
    if len(storage_keys) != len(set(storage_keys)):
        raise ValueError("CUDA arena mutable storage cannot be shared across resource scopes")
    event_handles = [
        record.storage_token.rsplit(":", 1)[-1]
        for record in all_records
        if record.resource_kind == "cuda_event"
    ]
    if len(event_handles) != len(set(event_handles)):
        raise ValueError("CUDA arena events must be identity-unique across slots")


def _event_token(event: object, *, slot_id: int, name: str) -> str:
    handle = getattr(event, "cuda_event", None)
    device = getattr(event, "device", None)
    if not isinstance(handle, int) or handle == 0 or getattr(device, "index", None) is None:
        raise ValueError("CUDA arena events must be initialized before seal")
    return f"torch-event:{int(device.index)}:{slot_id}:{name}:{handle}"


def _event_resource_record(
    event: object,
    *,
    token: str,
    logical_name: str,
) -> CudaArenaResourceRecord:
    device = getattr(event, "device", None)
    if getattr(device, "index", None) is None:
        raise ValueError("CUDA arena event resource requires an indexed device")
    return CudaArenaResourceRecord(
        schema_version=CUDA_NATIVE_ARENA_RESOURCE_SCHEMA_VERSION,
        resource_kind="cuda_event",
        device_ordinal=int(device.index),
        storage_token=token,
        logical_views=(
            CudaArenaLogicalResourceView(
                logical_name=logical_name,
                dtype="cuda_event",
                shape=(),
                stride=(),
                active_offset_bytes=0,
                active_nbytes=0,
            ),
        ),
        nbytes=0,
    )


def _render_buffer_items(
    buffers: CudaDirectLightBuffers,
) -> tuple[tuple[str, object], ...]:
    return tuple((f"render.{name}", value) for name, value in buffers.tensor_items())


def _tensor_identity_tuple(tensors: Sequence[object]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            id(value),
            int(value.data_ptr()),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
        )
        for value in tensors
    )


def _tensor_sequence_sha256(tensors: Sequence[object]) -> str:
    return _canonical_sha256(
        [
            {
                "data_ptr": int(value.data_ptr()),
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
            for value in tensors
        ]
    )


def _dataclass_payload(value: object) -> dict[str, object]:
    return {name: _plain_payload(item) for name, item in asdict(value).items()}


def _plain_payload(value: object) -> object:
    if is_dataclass(value):
        return {name: _plain_payload(item) for name, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(name): _plain_payload(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, name: str) -> None:
    if not (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
