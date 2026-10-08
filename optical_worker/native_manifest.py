"""CPU-only native input descriptors; parsing never grants live authority."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import prod
from types import MappingProxyType

from .lane_contract import CANONICAL_NATIVE_REPLAY_INPUT, FIXED_NATIVE_SCENE_INPUT

NATIVE_INPUT_SCHEMA_VERSION = "q54.p12.4.native-input.v2"

# The canonical builders currently emit these three scalar dtypes. Generic
# NativeInputManifest construction remains compatible with other host inputs.
CANONICAL_DTYPE_METADATA = MappingProxyType({
    "<f4": (4, "torch.float32", "<class 'warp._src.types.float32'>"),
    "<i4": (4, "torch.int32", "<class 'warp._src.types.int32'>"),
    "<i8": (8, "torch.int64", "<class 'warp._src.types.int64'>"),
})
CANONICAL_DTYPE_ITEMSIZES = MappingProxyType({
    name: metadata[0] for name, metadata in CANONICAL_DTYPE_METADATA.items()
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fields(payload: object, cls: type) -> dict:
    _require(type(payload) is dict and set(payload) == {f.name for f in fields(cls)},
             "invalid native manifest fields")
    return payload


def _text(value: object) -> None:
    _require(type(value) is str and bool(value), "invalid native manifest text")


def _digest(value: object) -> None:
    _require(type(value) is str and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "invalid native manifest digest")


def _count(value: object) -> None:
    _require(type(value) is int and value >= 0, "invalid native manifest count")


def _names(value: object) -> tuple[str, ...]:
    _require(type(value) is list, "native manifest names must be a list")
    for name in value:
        _text(name)
    _require(value == sorted(set(value)), "native manifest names must be sorted and unique")
    return tuple(value)

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

    @classmethod
    def from_payload(cls, payload: object) -> NativeArrayManifest:
        """Parse the narrower canonical C2d3 descriptor, without array access."""
        p = _fields(payload, cls)
        _text(p["name"])
        _text(p["semantic"])
        _require(type(p["dtype"]) is str and p["dtype"] in CANONICAL_DTYPE_ITEMSIZES,
                 "unsupported canonical native dtype")
        _require(type(p["shape"]) is list, "native array shape must be a list")
        for dimension in p["shape"]:
            _count(dimension)
        _count(p["nbytes"])
        _require(p["c_contiguous"] is True, "canonical native array must be contiguous")
        _require(p["nbytes"] == prod(p["shape"]) * CANONICAL_DTYPE_ITEMSIZES[p["dtype"]],
                 "native array byte extent differs from shape")
        _digest(p["content_sha256"])
        return cls(**dict(p, shape=tuple(p["shape"])))

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
    input_contract: str | None
    scene_contract_schema_version: str | None
    scene_contract_sha256: str | None
    scene_id: str | None
    scene_array_names: tuple[str, ...]
    transform_replay_sha256: str | None
    transform_frame_count: int
    fixed_render_fixture_sha256: str | None
    fixed_render_fixture_arrays: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: object) -> NativeInputManifest:
        """Parse canonical fixed/replay evidence, never execution readiness.

        The composite content hash is a descriptor consistency check. Actual
        array bytes and topology still require the existing shared validator.
        """
        p = _fields(payload, cls)
        _require(p["schema_version"] == NATIVE_INPUT_SCHEMA_VERSION, "native manifest version")
        for name in ("case_id", "materializer_id", "scene_id", "scene_contract_schema_version"):
            _text(p[name])
        for name in ("case_sha256", "content_sha256", "topology_sha256", "scene_contract_sha256"):
            _digest(p[name])
        for name in ("transform_replay_sha256", "fixed_render_fixture_sha256"):
            if p[name] is not None:
                _digest(p[name])
        _count(p["transform_frame_count"])
        _require(type(p["semantics"]) is dict and p["semantics"] == {
            name: p[name] for name in ("input_contract", "scene_contract_sha256", "scene_id")
        }, "native manifest semantics differ from canonical contract")
        _require(type(p["arrays"]) is list, "native manifest arrays must be a list")
        arrays = tuple(NativeArrayManifest.from_payload(row) for row in p["arrays"])
        names = _names([row.name for row in arrays])
        scene = _names(p["scene_array_names"])
        fixture = _names(p["fixed_render_fixture_arrays"])
        _require(bool(scene) and set(scene) <= set(names), "native scene array subset")
        _require(not set(scene) & set(fixture), "native scene/fixture arrays overlap")
        contract = p["input_contract"]
        if contract == FIXED_NATIVE_SCENE_INPUT:
            _require(p["transform_replay_sha256"] is None and p["transform_frame_count"] == 0
                     and p["fixed_render_fixture_sha256"] is not None and bool(fixture)
                     and set(names) == set(scene) | set(fixture), "invalid fixed native manifest")
        elif contract == CANONICAL_NATIVE_REPLAY_INPUT:
            replay = {"replay_x_world_R", "replay_x_world_r"}
            _require(p["fixed_render_fixture_sha256"] is None and not fixture
                     and p["transform_replay_sha256"] is not None and p["transform_frame_count"] > 0
                     and not set(scene) & replay and set(names) == set(scene) | replay,
                     "invalid replay native manifest")
            by_name = {row.name: row for row in arrays}
            rotation, translation = (by_name[name] for name in ("replay_x_world_R", "replay_x_world_r"))
            _require(rotation.dtype == translation.dtype == "<f4"
                     and len(rotation.shape) == 4 and rotation.shape[2:] == (3, 3)
                     and len(translation.shape) == 3 and translation.shape[2:] == (3,)
                     and rotation.shape[:2] == translation.shape[:2]
                     and rotation.shape[0] == p["transform_frame_count"], "invalid replay array layout")
        else:
            raise ValueError("C2d3 manifest requires canonical fixed or replay input")
        _require(_sha256_payload([row.to_payload() for row in arrays]) == p["content_sha256"],
                 "native composite content digest differs from array descriptors")
        return cls(**dict(p, semantics=MappingProxyType(dict(p["semantics"])), arrays=arrays,
                          scene_array_names=scene, fixed_render_fixture_arrays=fixture))

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
            "input_contract": self.input_contract,
            "scene_contract_schema_version": self.scene_contract_schema_version,
            "scene_contract_sha256": self.scene_contract_sha256,
            "scene_id": self.scene_id,
            "scene_array_names": list(self.scene_array_names),
            "transform_replay_sha256": self.transform_replay_sha256,
            "transform_frame_count": self.transform_frame_count,
            "fixed_render_fixture_sha256": self.fixed_render_fixture_sha256,
            "fixed_render_fixture_arrays": list(self.fixed_render_fixture_arrays),
        }

    @property
    def result_capable(self) -> bool:
        return self.scene_contract_sha256 is not None and self.input_contract is not None

    @property
    def sha256(self) -> str:
        return _sha256_payload(self.to_payload())
