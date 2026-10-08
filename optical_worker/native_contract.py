"""CPU-safe native-input ownership shared by setup and benchmark adapters.

This module deliberately imports no benchmark package and no CUDA-capable
runtime.  The closed worker can therefore construct prepared-input wrappers
without crossing its reviewed repository-source allowlist.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .native_manifest import NATIVE_INPUT_SCHEMA_VERSION

NativeBackend = Literal["warp", "cuda"]



def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class NativeReadinessEvidence:
    """CPU-safe, canonical evidence for one exact prepared native input."""

    schema_version: str
    backend: NativeBackend
    device_ordinal: int
    manifest_sha256: str
    content_sha256: str
    topology_sha256: str
    input_contract: str | None
    scene_contract_schema_version: str | None
    scene_contract_sha256: str | None
    scene_id: str | None
    transform_replay_sha256: str | None
    transform_frame_count: int
    fixed_render_fixture_sha256: str | None
    producer_ready_event_complete: bool
    direct_validation_complete: bool
    allocation_complete: bool
    compile_complete: bool
    warmup_frames_completed: int

    @classmethod
    def from_payload(cls, payload: object) -> "NativeReadinessEvidence":
        expected_fields = {
            "schema_version", "backend", "device_ordinal", "manifest_sha256",
            "content_sha256", "topology_sha256", "input_contract",
            "scene_contract_schema_version", "scene_contract_sha256", "scene_id",
            "transform_replay_sha256", "transform_frame_count",
            "fixed_render_fixture_sha256", "producer_ready_event_complete",
            "direct_validation_complete", "allocation_complete", "compile_complete",
            "warmup_frames_completed",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected_fields:
            raise ValueError("native readiness evidence fields are invalid")
        backend = payload.get("backend")
        if backend not in {"warp", "cuda"}:
            raise ValueError("native readiness backend is invalid")
        for field in (
            "producer_ready_event_complete", "direct_validation_complete",
            "allocation_complete", "compile_complete",
        ):
            if payload.get(field) is not True:
                raise ValueError(f"native readiness {field} must be true")
        for field in ("device_ordinal", "warmup_frames_completed", "transform_frame_count"):
            value = payload.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"native readiness {field} must be a non-negative integer")
        for field in ("manifest_sha256", "content_sha256", "topology_sha256"):
            if not _sha256_text(payload.get(field)):
                raise ValueError(f"native readiness {field} must be a lowercase SHA-256 digest")
        for field in (
            "scene_contract_sha256", "transform_replay_sha256",
            "fixed_render_fixture_sha256",
        ):
            value = payload.get(field)
            if value is not None and not _sha256_text(value):
                raise ValueError(f"native readiness {field} must be null or a SHA-256 digest")
        for field in ("input_contract", "scene_contract_schema_version", "scene_id"):
            value = payload.get(field)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"native readiness {field} must be null or non-empty")
        if (payload.get("scene_contract_sha256") is None) != (payload.get("input_contract") is None):
            raise ValueError("native readiness result-capable provenance is incomplete")
        if payload.get("scene_contract_sha256") is not None and (
            not payload.get("scene_contract_schema_version") or not payload.get("scene_id")
        ):
            raise ValueError("native readiness scene provenance is incomplete")
        if payload.get("schema_version") != NATIVE_INPUT_SCHEMA_VERSION:
            raise ValueError("native readiness schema version is invalid")
        return cls(
            schema_version=NATIVE_INPUT_SCHEMA_VERSION,
            backend=backend,
            device_ordinal=payload["device_ordinal"],
            manifest_sha256=payload["manifest_sha256"],
            content_sha256=payload["content_sha256"],
            topology_sha256=payload["topology_sha256"],
            input_contract=payload["input_contract"],
            scene_contract_schema_version=payload["scene_contract_schema_version"],
            scene_contract_sha256=payload["scene_contract_sha256"],
            scene_id=payload["scene_id"],
            transform_replay_sha256=payload["transform_replay_sha256"],
            transform_frame_count=payload["transform_frame_count"],
            fixed_render_fixture_sha256=payload["fixed_render_fixture_sha256"],
            producer_ready_event_complete=True,
            direct_validation_complete=True,
            allocation_complete=True,
            compile_complete=True,
            warmup_frames_completed=payload["warmup_frames_completed"],
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "device_ordinal": self.device_ordinal,
            "manifest_sha256": self.manifest_sha256,
            "content_sha256": self.content_sha256,
            "topology_sha256": self.topology_sha256,
            "input_contract": self.input_contract,
            "scene_contract_schema_version": self.scene_contract_schema_version,
            "scene_contract_sha256": self.scene_contract_sha256,
            "scene_id": self.scene_id,
            "transform_replay_sha256": self.transform_replay_sha256,
            "transform_frame_count": self.transform_frame_count,
            "fixed_render_fixture_sha256": self.fixed_render_fixture_sha256,
            "producer_ready_event_complete": self.producer_ready_event_complete,
            "direct_validation_complete": self.direct_validation_complete,
            "allocation_complete": self.allocation_complete,
            "compile_complete": self.compile_complete,
            "warmup_frames_completed": self.warmup_frames_completed,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_payload(), allow_nan=False, ensure_ascii=True,
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class _PhysicalStorageSignature:
    """Untimed snapshot of a live allocation, not a caller-supplied digest."""

    device_ordinal: int
    base_pointer: int
    storage_identity: int
    storage_nbytes: int
    active_start: int
    active_count: int
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]


@dataclass(frozen=True)
class _RegisteredArray:
    name: str
    array: object
    signature: _PhysicalStorageSignature
    storage_owner: object | None


def _physical_storage_signature(
    backend: NativeBackend, value: object, device_ordinal: int,
) -> tuple[_PhysicalStorageSignature, object | None]:
    """Inspect the backend's physical storage only during untimed setup/preflight."""

    device = value.device
    if backend == "cuda":
        if device.type != "cuda" or device.index is None:
            raise ValueError("prepared Torch input requires an indexed CUDA device")
        storage = value.untyped_storage()
        storage_nbytes = int(storage.nbytes())
        base_pointer = int(storage.data_ptr()) if storage_nbytes else 0
        active_pointer = int(value.data_ptr()) if storage_nbytes else 0
        active_start = active_pointer - base_pointer
        active_count = int(value.numel()) * int(value.element_size())
        storage_identity = int(storage._cdata)
        strides = tuple(int(item) for item in value.stride())
        contiguous = bool(value.is_contiguous())
        owner = storage
        ordinal = int(device.index)
    else:
        if not device.is_cuda:
            raise ValueError("prepared Warp input requires a CUDA device")
        base_pointer = int(value.ptr or 0)
        storage_nbytes = int(value.capacity)
        active_start = 0
        active_count = storage_nbytes
        storage_identity = base_pointer
        strides = tuple(int(item) for item in value.strides)
        contiguous = bool(value.is_contiguous)
        owner = None
        ordinal = int(device.ordinal)
    shape = tuple(int(item) for item in value.shape)
    if (
        ordinal != device_ordinal
        or not contiguous
        or storage_nbytes < 0
        or active_start < 0
        or active_count < 0
        or active_start + active_count > storage_nbytes
        or (storage_nbytes > 0 and base_pointer <= 0)
        or (storage_nbytes > 0 and storage_identity <= 0)
    ):
        raise ValueError("prepared native array has invalid physical CUDA storage")
    return (
        _PhysicalStorageSignature(
            device_ordinal=ordinal,
            base_pointer=base_pointer,
            storage_identity=storage_identity,
            storage_nbytes=storage_nbytes,
            active_start=active_start,
            active_count=active_count,
            dtype=str(value.dtype),
            shape=shape,
            strides=strides,
        ),
        owner,
    )


_PREPARED_AUTHORITIES: dict[int, tuple[weakref.ReferenceType[object], weakref.ReferenceType[object]]] = {}
_READINESS_AUTHORITIES: dict[int, tuple[weakref.ReferenceType[object], weakref.ReferenceType[object]]] = {}
_PREPARED_STORAGE: dict[
    int, tuple[weakref.ReferenceType[object], object, object, tuple[_RegisteredArray, ...]]
] = {}


def _register_identity_authority(
    registry: dict[int, tuple[weakref.ReferenceType[object], weakref.ReferenceType[object]]],
    subject: object,
    owner: object,
) -> None:
    key = id(subject)
    current = registry.get(key)
    if current is not None and current[0]() is subject:
        raise ValueError("native identity authority cannot be registered twice")

    def discard(reference: weakref.ReferenceType[object]) -> None:
        current = registry.get(key)
        if current is not None and current[0] is reference:
            registry.pop(key, None)

    registry[key] = (weakref.ref(subject, discard), weakref.ref(owner))


def _require_identity_authority(
    registry: dict[int, tuple[weakref.ReferenceType[object], weakref.ReferenceType[object]]],
    subject: object,
    owner: object,
    *,
    label: str,
) -> None:
    registered = registry.get(id(subject))
    if registered is None or registered[0]() is not subject or registered[1]() is not owner:
        raise ValueError(f"{label} was not issued by the canonical a2 preparation path")


def register_prepared_native_input(subject: object, source: object) -> None:
    """Bind one exact prepared wrapper to its exact canonical host source."""

    if type(subject) not in {WarpPreparedNativeInput, CudaPreparedNativeInput}:
        raise TypeError("native prepared-input registration requires an exact production type")
    existing = _PREPARED_AUTHORITIES.get(id(subject))
    if existing is not None and existing[0]() is subject:
        raise ValueError("native identity authority cannot be registered twice")
    if subject.source is not source or subject.manifest is not source.manifest:
        raise ValueError("native prepared input does not own its exact canonical source")
    if not isinstance(subject.arrays, Mapping):
        raise TypeError("native prepared arrays must be a mapping")
    if any(not isinstance(name, str) for name in subject.arrays):
        raise TypeError("native prepared array names must be strings")
    manifest_arrays = {entry.name: entry for entry in subject.manifest.arrays}
    if set(subject.arrays) != set(manifest_arrays):
        raise ValueError("native prepared arrays do not match the canonical manifest")
    arrays = []
    for name in sorted(subject.arrays):
        value = subject.arrays[name]
        signature, owner = _physical_storage_signature(
            subject.backend, value, subject.device_ordinal,
        )
        if (
            signature.shape != tuple(manifest_arrays[name].shape)
            or signature.active_count != int(manifest_arrays[name].nbytes)
        ):
            raise ValueError(f"native prepared array {name!r} does not match its manifest extent")
        arrays.append(_RegisteredArray(name, value, signature, owner))
    _register_identity_authority(_PREPARED_AUTHORITIES, subject, source)
    key = id(subject)

    def discard(reference: weakref.ReferenceType[object]) -> None:
        current = _PREPARED_STORAGE.get(key)
        if current is not None and current[0] is reference:
            _PREPARED_STORAGE.pop(key, None)

    _PREPARED_STORAGE[key] = (
        weakref.ref(subject, discard), subject.arrays, subject.ready_event, tuple(arrays),
    )


def register_native_readiness(subject: object, prepared: object) -> None:
    """Bind readiness evidence to the exact prepared-input wrapper."""

    if type(subject) is not NativeReadinessEvidence:
        raise TypeError("native readiness registration requires the exact production type")
    if type(prepared) not in {WarpPreparedNativeInput, CudaPreparedNativeInput}:
        raise TypeError("native readiness registration requires an exact prepared input")
    _register_identity_authority(_READINESS_AUTHORITIES, subject, prepared)


def _assert_registered_prepared_storage(
    prepared: object, *, backend: NativeBackend, prepared_type: type,
) -> tuple[_RegisteredArray, ...]:
    """Untimed upload/storage check; does not certify execution readiness."""

    if type(prepared) is not prepared_type:
        raise TypeError(f"{backend} a2 authority requires the exact prepared input type")
    source = prepared.source
    _require_identity_authority(
        _PREPARED_AUTHORITIES,
        prepared,
        source,
        label=f"{backend} prepared native input",
    )
    if source is None or prepared.manifest is not source.manifest:
        raise ValueError(f"{backend} prepared native input lost its canonical source identity")
    registered_storage = _PREPARED_STORAGE.get(id(prepared))
    if (
        registered_storage is None
        or registered_storage[0]() is not prepared
        or registered_storage[1] is not prepared.arrays
        or registered_storage[2] is not prepared.ready_event
        or tuple(sorted(prepared.arrays)) != tuple(entry.name for entry in registered_storage[3])
        or any(prepared.arrays[entry.name] is not entry.array for entry in registered_storage[3])
    ):
        raise ValueError(f"{backend} prepared native storage lost its registered object identities")
    for entry in registered_storage[3]:
        current, _ = _physical_storage_signature(
            backend, entry.array, prepared.device_ordinal,
        )
        if current != entry.signature:
            raise ValueError(
                f"{backend} prepared native storage {entry.name!r} changed its physical allocation"
            )
    return registered_storage[3]


def assert_prepared_storage_registered(prepared: object) -> None:
    """Recheck exact registered a2 allocations without fabricating readiness.

    This alone never permits C3/C4 binding or a result-capable execution path.
    Those continue to require their existing full prepared/readiness authority.
    """
    observe_prepared_storage(prepared)


def observe_prepared_storage(prepared: object) -> tuple[_RegisteredArray, ...]:
    """Return the held registry entries after the sole existing storage audit.

    The custody observer projects these CPU signatures and exact object
    references immediately. No second device inspection or registration occurs,
    and the returned tuple grants no prepared/readiness authority.
    """
    if type(prepared) is CudaPreparedNativeInput:
        return _assert_registered_prepared_storage(
            prepared, backend="cuda", prepared_type=CudaPreparedNativeInput,
        )
    if type(prepared) is WarpPreparedNativeInput:
        return _assert_registered_prepared_storage(
            prepared, backend="warp", prepared_type=WarpPreparedNativeInput,
        )
    raise TypeError("storage validation requires an exact production prepared input")


def _assert_prepared_authority(
    prepared: object, readiness: object, *, backend: NativeBackend,
    prepared_type: type,
) -> None:
    """Validate exact live a2 ownership plus frozen execution readiness."""
    if type(prepared) is not prepared_type:
        raise TypeError(f"{backend} a2 authority requires the exact prepared input type")
    if type(readiness) is not NativeReadinessEvidence:
        raise TypeError(f"{backend} a2 authority requires NativeReadinessEvidence")
    _assert_registered_prepared_storage(prepared, backend=backend, prepared_type=prepared_type)
    _require_identity_authority(
        _READINESS_AUTHORITIES,
        readiness,
        prepared,
        label=f"{backend} native readiness",
    )
    canonical = NativeReadinessEvidence.from_payload(readiness.to_payload())
    if canonical != readiness or canonical.sha256 != readiness.sha256:
        raise ValueError(f"{backend} native readiness evidence is not canonical")
    manifest = prepared.manifest
    expected = {
        "backend": backend,
        "device_ordinal": prepared.device_ordinal,
        "manifest_sha256": manifest.sha256,
        "content_sha256": manifest.content_sha256,
        "topology_sha256": manifest.topology_sha256,
        "input_contract": manifest.input_contract,
        "scene_contract_schema_version": manifest.scene_contract_schema_version,
        "scene_contract_sha256": manifest.scene_contract_sha256,
        "scene_id": manifest.scene_id,
        "transform_replay_sha256": manifest.transform_replay_sha256,
        "transform_frame_count": manifest.transform_frame_count,
        "fixed_render_fixture_sha256": manifest.fixed_render_fixture_sha256,
    }
    for name, value in expected.items():
        if getattr(readiness, name) != value:
            raise ValueError(f"{backend} native readiness {name} does not match its prepared input")


def assert_cuda_prepared_authority(prepared: object, readiness: object) -> None:
    _assert_prepared_authority(
        prepared, readiness, backend="cuda", prepared_type=CudaPreparedNativeInput,
    )


def assert_warp_prepared_authority(prepared: object, readiness: object) -> None:
    _assert_prepared_authority(
        prepared, readiness, backend="warp", prepared_type=WarpPreparedNativeInput,
    )


def warp_dtype_from_host_dtype(wp: Any, dtype: object) -> Any:
    """Map a NumPy-like host dtype without importing NumPy in this module."""

    kind = getattr(dtype, "kind", None)
    itemsize = getattr(dtype, "itemsize", None)
    mapping = {
        ("b", 1): wp.bool,
        ("i", 1): wp.int8,
        ("i", 2): wp.int16,
        ("i", 4): wp.int32,
        ("i", 8): wp.int64,
        ("u", 1): wp.uint8,
        ("u", 2): wp.uint16,
        ("u", 4): wp.uint32,
        ("u", 8): wp.uint64,
        ("f", 2): wp.float16,
        ("f", 4): wp.float32,
        ("f", 8): wp.float64,
    }
    try:
        return mapping[(kind, itemsize)]
    except KeyError as exc:
        raise ValueError(f"unsupported Warp native input dtype kind={kind!r}, itemsize={itemsize!r}") from exc


@dataclass(frozen=True)
class WarpPreparedNativeInput:
    """Warp-owned CUDA allocations at the native-ready boundary."""

    manifest: Any
    device_ordinal: int
    arrays: Mapping[str, Any]
    ready_event: Any
    source: Any | None = None

    @property
    def backend(self) -> NativeBackend:
        return "warp"


@dataclass(frozen=True)
class CudaPreparedNativeInput:
    """Torch-owned CUDA allocations at the native-ready boundary."""

    manifest: Any
    device_ordinal: int
    arrays: Mapping[str, Any]
    ready_event: Any
    source: Any | None = None

    @property
    def backend(self) -> NativeBackend:
        return "cuda"


PreparedNativeInput = WarpPreparedNativeInput | CudaPreparedNativeInput
