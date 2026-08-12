"""Independent Warp/Torch native-input preparation for optical benchmarks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .cases import get_case
from .timing import assert_operation_allowed, assert_outside_timed_boundary

NATIVE_INPUT_SCHEMA_VERSION = "q54.p12.4.native-input.v1"
NativeBackend = Literal["warp", "cuda"]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _array_content_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "nbytes": int(array.nbytes),
            }
        )
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class NativeArrayManifest:
    """Stable identity and layout of one canonical native input array."""

    name: str
    semantic: str
    dtype: str
    shape: tuple[int, ...]
    c_contiguous: bool
    nbytes: int
    content_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "semantic": self.semantic,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "c_contiguous": self.c_contiguous,
            "nbytes": self.nbytes,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class NativeInputManifest:
    """Device-independent evidence shared by independently prepared lanes."""

    schema_version: str
    case_id: str
    case_sha256: str
    materializer_id: str
    semantics: Mapping[str, object]
    arrays: tuple[NativeArrayManifest, ...]
    content_sha256: str
    topology_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "case_sha256": self.case_sha256,
            "materializer_id": self.materializer_id,
            "semantics": dict(self.semantics),
            "arrays": [array.to_payload() for array in self.arrays],
            "content_sha256": self.content_sha256,
            "topology_sha256": self.topology_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256_payload(self.to_payload())


@dataclass(frozen=True)
class CanonicalNativeInput:
    """Immutable host-side logical input from which both lanes allocate.

    This object is not a timed backend input.  It is the common logical source
    used before timing to create two independent, runtime-owned device inputs.
    """

    manifest: NativeInputManifest
    arrays: Mapping[str, np.ndarray]

    @classmethod
    def from_arrays(
        cls,
        *,
        case_id: str,
        materializer_id: str,
        arrays: Mapping[str, np.ndarray],
        array_semantics: Mapping[str, str],
        topology_arrays: Sequence[str],
        semantics: Mapping[str, object],
    ) -> "CanonicalNativeInput":
        assert_outside_timed_boundary("canonical native input construction")
        case = get_case(case_id)
        if not materializer_id:
            raise ValueError("materializer_id must be non-empty")
        names = set(arrays)
        if not names:
            raise ValueError("canonical native input must contain at least one array")
        if names != set(array_semantics):
            raise ValueError("array_semantics must describe every and only input array")
        topology_names = tuple(topology_arrays)
        if not topology_names or len(set(topology_names)) != len(topology_names):
            raise ValueError("topology_arrays must be non-empty and unique")
        missing_topology = set(topology_names).difference(names)
        if missing_topology:
            raise ValueError(f"topology arrays are missing: {sorted(missing_topology)}")

        frozen: dict[str, np.ndarray] = {}
        manifests: list[NativeArrayManifest] = []
        for name in sorted(names):
            if not isinstance(name, str) or not name:
                raise ValueError("native input array names must be non-empty strings")
            semantic = array_semantics[name]
            if not isinstance(semantic, str) or not semantic:
                raise ValueError(f"native input semantic for {name!r} must be non-empty")
            source = np.asarray(arrays[name])
            if source.dtype.hasobject:
                raise ValueError(f"native input {name!r} cannot use object dtype")
            value = np.array(source, copy=True, order="C")
            value.setflags(write=False)
            frozen[name] = value
            manifests.append(
                NativeArrayManifest(
                    name=name,
                    semantic=semantic,
                    dtype=value.dtype.str,
                    shape=tuple(int(dimension) for dimension in value.shape),
                    c_contiguous=True,
                    nbytes=int(value.nbytes),
                    content_sha256=_array_content_sha256(value),
                )
            )

        semantic_payload = json.loads(_canonical_json_bytes(semantics).decode("utf-8"))
        array_payload = [manifest.to_payload() for manifest in manifests]
        content_sha256 = _sha256_payload(array_payload)
        manifests_by_name = {manifest.name: manifest for manifest in manifests}
        topology_sha256 = _sha256_payload([manifests_by_name[name].to_payload() for name in topology_names])
        manifest = NativeInputManifest(
            schema_version=NATIVE_INPUT_SCHEMA_VERSION,
            case_id=case.case_id,
            case_sha256=case.sha256,
            materializer_id=materializer_id,
            semantics=MappingProxyType(dict(semantic_payload)),
            arrays=tuple(manifests),
            content_sha256=content_sha256,
            topology_sha256=topology_sha256,
        )
        return cls(manifest=manifest, arrays=MappingProxyType(frozen))


@dataclass(frozen=True)
class WarpPreparedNativeInput:
    """Warp-owned CUDA allocations at the native-ready boundary."""

    manifest: NativeInputManifest
    device_ordinal: int
    arrays: Mapping[str, Any]
    ready_event: Any

    @property
    def backend(self) -> NativeBackend:
        return "warp"


@dataclass(frozen=True)
class CudaPreparedNativeInput:
    """Torch-owned CUDA allocations at the native-ready boundary."""

    manifest: NativeInputManifest
    device_ordinal: int
    arrays: Mapping[str, Any]
    ready_event: Any

    @property
    def backend(self) -> NativeBackend:
        return "cuda"


PreparedNativeInput = WarpPreparedNativeInput | CudaPreparedNativeInput


@dataclass(frozen=True)
class NativeInputValidationReport:
    """Direct, untimed parity evidence for one prepared native input."""

    backend: NativeBackend
    manifest_sha256: str
    content_sha256: str
    topology_sha256: str
    validated_arrays: tuple[str, ...]


@dataclass(frozen=True)
class NativeReadinessEvidence:
    """Lane evidence that preparation, validation, compilation, and warmup ended."""

    schema_version: str
    backend: NativeBackend
    device_ordinal: int
    manifest_sha256: str
    content_sha256: str
    topology_sha256: str
    producer_ready_event_complete: bool
    direct_validation_complete: bool
    allocation_complete: bool
    compile_complete: bool
    warmup_frames_completed: int

    @classmethod
    def from_payload(cls, payload: object) -> "NativeReadinessEvidence":
        expected_fields = {
            "schema_version",
            "backend",
            "device_ordinal",
            "manifest_sha256",
            "content_sha256",
            "topology_sha256",
            "producer_ready_event_complete",
            "direct_validation_complete",
            "allocation_complete",
            "compile_complete",
            "warmup_frames_completed",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected_fields:
            raise ValueError("native readiness evidence fields are invalid")
        backend = payload.get("backend")
        if backend not in {"warp", "cuda"}:
            raise ValueError("native readiness backend is invalid")
        for field in (
            "producer_ready_event_complete",
            "direct_validation_complete",
            "allocation_complete",
            "compile_complete",
        ):
            if payload.get(field) is not True:
                raise ValueError(f"native readiness {field} must be true")
        for field in ("device_ordinal", "warmup_frames_completed"):
            value = payload.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"native readiness {field} must be a non-negative integer")
        for field in ("manifest_sha256", "content_sha256", "topology_sha256"):
            value = payload.get(field)
            if not _sha256_text(value):
                raise ValueError(f"native readiness {field} must be a lowercase SHA-256 digest")
        if payload.get("schema_version") != NATIVE_INPUT_SCHEMA_VERSION:
            raise ValueError("native readiness schema version is invalid")
        return cls(
            schema_version=NATIVE_INPUT_SCHEMA_VERSION,
            backend=backend,
            device_ordinal=payload["device_ordinal"],
            manifest_sha256=payload["manifest_sha256"],
            content_sha256=payload["content_sha256"],
            topology_sha256=payload["topology_sha256"],
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
            "producer_ready_event_complete": self.producer_ready_event_complete,
            "direct_validation_complete": self.direct_validation_complete,
            "allocation_complete": self.allocation_complete,
            "compile_complete": self.compile_complete,
            "warmup_frames_completed": self.warmup_frames_completed,
        }

    @property
    def sha256(self) -> str:
        return _sha256_payload(self.to_payload())


def certify_native_readiness(
    prepared: PreparedNativeInput,
    validation: NativeInputValidationReport,
    *,
    compile_complete: bool,
    warmup_frames_completed: int,
) -> NativeReadinessEvidence:
    """Create evidence only after a lane finishes its untimed setup phases."""

    assert_outside_timed_boundary("native readiness certification")
    if validation.backend != prepared.backend:
        raise ValueError("native validation backend does not match prepared input")
    if validation.manifest_sha256 != prepared.manifest.sha256:
        raise ValueError("native validation manifest does not match prepared input")
    if validation.content_sha256 != prepared.manifest.content_sha256:
        raise ValueError("native validation content does not match prepared input")
    if validation.topology_sha256 != prepared.manifest.topology_sha256:
        raise ValueError("native validation topology does not match prepared input")
    expected_arrays = tuple(manifest.name for manifest in prepared.manifest.arrays)
    if validation.validated_arrays != expected_arrays:
        raise ValueError("native validation did not cover every prepared input array")
    if compile_complete is not True:
        raise ValueError("lane compilation must complete before native readiness")
    if (
        not isinstance(warmup_frames_completed, int)
        or isinstance(warmup_frames_completed, bool)
        or warmup_frames_completed < 0
    ):
        raise ValueError("warmup_frames_completed must be a non-negative integer")
    return NativeReadinessEvidence(
        schema_version=NATIVE_INPUT_SCHEMA_VERSION,
        backend=prepared.backend,
        device_ordinal=prepared.device_ordinal,
        manifest_sha256=prepared.manifest.sha256,
        content_sha256=prepared.manifest.content_sha256,
        topology_sha256=prepared.manifest.topology_sha256,
        producer_ready_event_complete=True,
        direct_validation_complete=True,
        allocation_complete=True,
        compile_complete=True,
        warmup_frames_completed=warmup_frames_completed,
    )


def prepare_warp_native_input(
    source: CanonicalNativeInput,
    *,
    device: str | int | object = "cuda:0",
) -> WarpPreparedNativeInput:
    """Allocate Warp arrays directly from the canonical host source."""

    assert_outside_timed_boundary("Warp native input preparation")
    assert_operation_allowed("host_upload")
    import warp as wp

    target = wp.get_device(device)
    if not target.is_cuda:
        raise ValueError("Warp native benchmark input requires a CUDA device")
    prepared: dict[str, Any] = {}
    with wp.ScopedDevice(target):
        stream = wp.get_stream(target)
        for manifest in source.manifest.arrays:
            value = source.arrays[manifest.name]
            prepared[manifest.name] = wp.array(
                np.array(value, copy=True, order="C"),
                dtype=_warp_dtype(wp, value.dtype),
                device=target,
            )
        ready_event = stream.record_event()
        wp.synchronize_event(ready_event)
    return WarpPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=int(target.ordinal),
        arrays=MappingProxyType(prepared),
        ready_event=ready_event,
    )


def prepare_cuda_native_input(
    source: CanonicalNativeInput,
    *,
    device: str | int | object = "cuda:0",
) -> CudaPreparedNativeInput:
    """Allocate Torch CUDA tensors directly from the canonical host source."""

    assert_outside_timed_boundary("CUDA native input preparation")
    assert_operation_allowed("host_upload")
    import torch

    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("CUDA native benchmark input requires a CUDA device")
    ordinal = torch.cuda.current_device() if target.index is None else target.index
    target = torch.device("cuda", ordinal)
    prepared: dict[str, Any] = {}
    with torch.cuda.device(target):
        stream = torch.cuda.current_stream(target)
        for manifest in source.manifest.arrays:
            value = source.arrays[manifest.name]
            prepared[manifest.name] = torch.tensor(
                np.array(value, copy=True, order="C"),
                device=target,
            ).contiguous()
        ready_event = torch.cuda.Event(enable_timing=False)
        ready_event.record(stream)
        ready_event.synchronize()
    return CudaPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=int(ordinal),
        arrays=MappingProxyType(prepared),
        ready_event=ready_event,
    )


def validate_prepared_native_input(
    source: CanonicalNativeInput,
    prepared: PreparedNativeInput,
) -> NativeInputValidationReport:
    """Directly compare native arrays to their logical source outside timing."""

    assert_outside_timed_boundary("native input parity validation")
    assert_operation_allowed("host_materialization")
    if prepared.manifest.sha256 != source.manifest.sha256:
        raise ValueError("prepared native input manifest does not match its logical source")
    expected_names = tuple(manifest.name for manifest in source.manifest.arrays)
    if set(prepared.arrays) != set(expected_names):
        raise ValueError("prepared native input arrays do not match the source manifest")

    for manifest in source.manifest.arrays:
        actual = _native_array_to_numpy(prepared, prepared.arrays[manifest.name])
        expected = source.arrays[manifest.name]
        if actual.dtype.str != manifest.dtype:
            raise ValueError(f"native input {manifest.name!r} dtype does not match manifest")
        if tuple(actual.shape) != manifest.shape:
            raise ValueError(f"native input {manifest.name!r} shape does not match manifest")
        if not actual.flags.c_contiguous:
            raise ValueError(f"native input {manifest.name!r} is not C-contiguous")
        if not np.array_equal(actual, expected, equal_nan=True):
            raise ValueError(f"native input {manifest.name!r} values do not match logical source")
        if _array_content_sha256(actual) != manifest.content_sha256:
            raise ValueError(f"native input {manifest.name!r} content hash does not match")

    return NativeInputValidationReport(
        backend=prepared.backend,
        manifest_sha256=prepared.manifest.sha256,
        content_sha256=prepared.manifest.content_sha256,
        topology_sha256=prepared.manifest.topology_sha256,
        validated_arrays=expected_names,
    )


def assert_independent_native_inputs(
    warp_input: WarpPreparedNativeInput,
    cuda_input: CudaPreparedNativeInput,
) -> None:
    """Prove the two lanes have equal manifests but distinct allocations."""

    assert_outside_timed_boundary("native allocation independence validation")
    if warp_input.manifest.sha256 != cuda_input.manifest.sha256:
        raise ValueError("Warp and CUDA lanes were not prepared from the same manifest")
    if warp_input.device_ordinal != cuda_input.device_ordinal:
        raise ValueError("Warp and CUDA native inputs target different device ordinals")
    if set(warp_input.arrays) != set(cuda_input.arrays):
        raise ValueError("Warp and CUDA native input names differ")
    for name in warp_input.arrays:
        warp_pointer = int(warp_input.arrays[name].ptr)
        torch_pointer = int(cuda_input.arrays[name].data_ptr())
        if warp_pointer == 0 or torch_pointer == 0:
            raise ValueError(f"native input {name!r} has a null device allocation")
        if warp_pointer == torch_pointer:
            raise ValueError(f"native input {name!r} shares a cross-runtime allocation")


def _native_array_to_numpy(prepared: PreparedNativeInput, value: Any) -> np.ndarray:
    if prepared.backend == "warp":
        return np.asarray(value.numpy())
    return np.asarray(value.detach().cpu().numpy())


def _warp_dtype(wp: Any, dtype: np.dtype[Any]) -> Any:
    mapping = {
        np.dtype(np.bool_): wp.bool,
        np.dtype(np.int8): wp.int8,
        np.dtype(np.int16): wp.int16,
        np.dtype(np.int32): wp.int32,
        np.dtype(np.int64): wp.int64,
        np.dtype(np.uint8): wp.uint8,
        np.dtype(np.uint16): wp.uint16,
        np.dtype(np.uint32): wp.uint32,
        np.dtype(np.uint64): wp.uint64,
        np.dtype(np.float16): wp.float16,
        np.dtype(np.float32): wp.float32,
        np.dtype(np.float64): wp.float64,
    }
    try:
        return mapping[np.dtype(dtype)]
    except KeyError as exc:
        raise ValueError(f"unsupported Warp native input dtype {np.dtype(dtype)}") from exc


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
