"""CPU-safe native-input ownership shared by setup and benchmark adapters.

This module deliberately imports no benchmark package and no CUDA-capable
runtime.  The closed worker can therefore construct prepared-input wrappers
without crossing its reviewed repository-source allowlist.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

NativeBackend = Literal["warp", "cuda"]

_PREPARED_AUTHORITIES: dict[
    int, tuple[weakref.ReferenceType[object], weakref.ReferenceType[object]]
] = {}
_READINESS_AUTHORITIES: dict[
    int, tuple[weakref.ReferenceType[object], weakref.ReferenceType[object]]
] = {}


def _register_identity_authority(
    registry: dict[int, tuple[weakref.ReferenceType[object], weakref.ReferenceType[object]]],
    subject: object,
    owner: object,
) -> None:
    key = id(subject)

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

    _register_identity_authority(_PREPARED_AUTHORITIES, subject, source)


def register_native_readiness(subject: object, prepared: object) -> None:
    """Bind readiness evidence to the exact prepared-input wrapper."""

    _register_identity_authority(_READINESS_AUTHORITIES, subject, prepared)


def assert_cuda_prepared_authority(prepared: object, readiness: object) -> None:
    """Validate the live a2 allocation/readiness identity and frozen facts."""

    if type(prepared) is not CudaPreparedNativeInput:
        raise TypeError("CUDA arena a2 authority requires CudaPreparedNativeInput")
    source = prepared.source
    _require_identity_authority(
        _PREPARED_AUTHORITIES,
        prepared,
        source,
        label="CUDA prepared native input",
    )
    if source is None or prepared.manifest is not source.manifest:
        raise ValueError("CUDA prepared native input lost its canonical source identity")
    _require_identity_authority(
        _READINESS_AUTHORITIES,
        readiness,
        prepared,
        label="CUDA native readiness",
    )
    from_payload = getattr(type(readiness), "from_payload", None)
    to_payload = getattr(readiness, "to_payload", None)
    if not callable(from_payload) or not callable(to_payload):
        raise TypeError("CUDA arena a2 authority requires canonical readiness evidence")
    canonical = from_payload(to_payload())
    if canonical != readiness:
        raise ValueError("CUDA native readiness evidence is not canonical")
    manifest = prepared.manifest
    expected = {
        "backend": "cuda",
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
        if getattr(readiness, name, object()) != value:
            raise ValueError(f"CUDA native readiness {name} does not match its prepared input")


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

    def _assert_cuda_arena_a2_authority(self, readiness: object) -> None:
        assert_cuda_prepared_authority(self, readiness)


PreparedNativeInput = WarpPreparedNativeInput | CudaPreparedNativeInput
