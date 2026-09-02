"""Independent Warp/Torch native-input preparation for optical benchmarks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from optics.device_bvh import build_host_median_bvh
from optics.scene_contract import SCENE_CONTRACT_SCHEMA_VERSION, PackedOpticalSceneHost

from .cases import get_case
from .eligibility import (
    CANONICAL_NATIVE_REPLAY_INPUT,
    FIXED_NATIVE_SCENE_INPUT,
    GPU_ENGINE_PUBLISHED_FRAME_INPUT,
)
from .receipts import NativeStorageSignature
from .timing import assert_operation_allowed, assert_outside_timed_boundary
from .workload_contract import frozen_workload_content

NATIVE_INPUT_SCHEMA_VERSION = "q54.p12.4.native-input.v2"
TRANSFORM_REPLAY_SCHEMA_VERSION = "q54.p12.4.transform-replay.v1"
FIXED_RENDER_FIXTURE_SCHEMA_VERSION = "q54.p12.4.fixed-render-fixture.v1"
NativeBackend = Literal["warp", "cuda"]

_REPLAY_SEMANTICS = MappingProxyType(
    {
        "replay_x_world_R": "transform_replay.x_world_R",
        "replay_x_world_r": "transform_replay.x_world_r",
    }
)

_FIXED_FIXTURE_SEMANTICS = MappingProxyType(
    {
        "triangle_v0_world": "fixed_fixture.triangle.v0_world",
        "triangle_e1_world": "fixed_fixture.triangle.e1_world",
        "triangle_e2_world": "fixed_fixture.triangle.e2_world",
        "triangle_normal_world": "fixed_fixture.triangle.normal_world",
        "triangle_aabb_min": "fixed_fixture.triangle.aabb_min",
        "triangle_aabb_max": "fixed_fixture.triangle.aabb_max",
        "plane_normal_world": "fixed_fixture.plane.normal_world",
        "plane_point_world": "fixed_fixture.plane.point_world",
        "bvh_bounds_min": "fixed_fixture.bvh.bounds_min",
        "bvh_bounds_max": "fixed_fixture.bvh.bounds_max",
        "bvh_left": "fixed_fixture.bvh.left",
        "bvh_right": "fixed_fixture.bvh.right",
        "bvh_start": "fixed_fixture.bvh.start",
        "bvh_count": "fixed_fixture.bvh.count",
        "bvh_node_depth": "fixed_fixture.bvh.node_depth",
        "bvh_level_ranges": "fixed_fixture.bvh.level_ranges",
        "bvh_prim_ids": "fixed_fixture.bvh.primitive_ids",
    }
)

_FIXED_FIXTURE_TOPOLOGY_ARRAYS = (
    "bvh_left",
    "bvh_right",
    "bvh_start",
    "bvh_count",
    "bvh_node_depth",
    "bvh_level_ranges",
    "bvh_prim_ids",
)


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
    input_contract: str | None
    scene_contract_schema_version: str | None
    scene_contract_sha256: str | None
    scene_id: str | None
    scene_array_names: tuple[str, ...]
    transform_replay_sha256: str | None
    transform_frame_count: int
    fixed_render_fixture_sha256: str | None
    fixed_render_fixture_arrays: tuple[str, ...]

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


@dataclass(frozen=True)
class CanonicalTransformReplayHost:
    """Immutable transform sequence uploaded independently by each candidate."""

    case_id: str
    case_sha256: str
    transform_sequence: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]
    array_semantics: Mapping[str, str]
    frame_count: int
    schema_version: str
    transform_replay_sha256: str

    @classmethod
    def from_arrays(
        cls,
        *,
        case_id: str,
        x_world_R: object,
        x_world_r: object,
    ) -> "CanonicalTransformReplayHost":
        assert_outside_timed_boundary("canonical transform replay construction")
        case = get_case(case_id)
        rotation = np.array(x_world_R, dtype=np.float32, copy=True, order="C")
        translation = np.array(x_world_r, dtype=np.float32, copy=True, order="C")
        if rotation.ndim != 4 or rotation.shape[2:] != (3, 3):
            raise ValueError("x_world_R must have shape (frames, bodies, 3, 3)")
        if translation.ndim != 3 or translation.shape[2:] != (3,):
            raise ValueError("x_world_r must have shape (frames, bodies, 3)")
        if rotation.shape[:2] != translation.shape[:2]:
            raise ValueError("replay rotation and translation shapes do not agree")
        frame_count = int(rotation.shape[0])
        expected_frames = int(case.transform_sequence.get("frames", 0))
        if frame_count <= 0 or frame_count != expected_frames:
            raise ValueError("replay frame count does not match the canonical case")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        arrays = MappingProxyType(
            {"replay_x_world_R": rotation, "replay_x_world_r": translation}
        )
        transform_sequence = json.loads(
            _canonical_json_bytes(case.transform_sequence).decode("utf-8")
        )
        payload = {
            "schema_version": TRANSFORM_REPLAY_SCHEMA_VERSION,
            "case_id": case.case_id,
            "case_sha256": case.sha256,
            "transform_sequence": transform_sequence,
            "arrays": _array_manifest_payload(arrays, _REPLAY_SEMANTICS),
            "frame_count": frame_count,
        }
        return cls(
            case_id=case.case_id,
            case_sha256=case.sha256,
            transform_sequence=MappingProxyType(dict(transform_sequence)),
            arrays=arrays,
            array_semantics=_REPLAY_SEMANTICS,
            frame_count=frame_count,
            schema_version=TRANSFORM_REPLAY_SCHEMA_VERSION,
            transform_replay_sha256=_sha256_payload(payload),
        )


@dataclass(frozen=True, init=False)
class CanonicalFixedRenderFixtureHost:
    """Derived-only fixed world geometry and normalized R0 BVH."""

    packed_scene_owner: PackedOpticalSceneHost
    case_id: str
    case_sha256: str
    fixed_transform_id: str
    fixed_transform_sha256: str
    arrays: Mapping[str, np.ndarray]
    array_semantics: Mapping[str, str]
    topology_array_names: tuple[str, ...]
    topology_sha256: str
    schema_version: str
    fixed_render_fixture_sha256: str

    @classmethod
    def from_scene_contract(
        cls,
        packed_scene: PackedOpticalSceneHost,
        *,
        fixed_transform_id: str,
        x_world_R: object,
        x_world_r: object,
    ) -> "CanonicalFixedRenderFixtureHost":
        assert_outside_timed_boundary("canonical fixed render fixture construction")
        case = _validate_packed_scene_case(packed_scene)
        if case.geometry_mode != "static":
            raise ValueError("fixed render fixtures require a static canonical case")
        if not isinstance(fixed_transform_id, str) or not fixed_transform_id:
            raise ValueError("fixed_transform_id must be non-empty")
        rotation = np.array(x_world_R, dtype=np.float32, copy=True, order="C")
        translation = np.array(x_world_r, dtype=np.float32, copy=True, order="C")
        if rotation.ndim != 3 or rotation.shape[1:] != (3, 3):
            raise ValueError("fixed x_world_R must have shape (bodies, 3, 3)")
        if translation.ndim != 2 or translation.shape[1:] != (3,):
            raise ValueError("fixed x_world_r must have shape (bodies, 3)")
        if rotation.shape[0] != translation.shape[0]:
            raise ValueError("fixed rotation and translation body counts do not agree")
        if packed_scene.max_body_index >= rotation.shape[0]:
            raise ValueError("fixed transforms do not cover the packed scene body indices")

        derived = _derive_fixed_fixture_arrays(packed_scene, rotation, translation)
        frozen = _freeze_array_mapping(derived)
        topology_payload = [
            _array_manifest(frozen[name], name=name, semantic=_FIXED_FIXTURE_SEMANTICS[name])
            for name in _FIXED_FIXTURE_TOPOLOGY_ARRAYS
        ]
        topology_sha256 = _sha256_payload(topology_payload)
        transform_payload = {
            "fixed_transform_id": fixed_transform_id,
            "x_world_R": _array_manifest(rotation, name="x_world_R", semantic="fixed_transform.R"),
            "x_world_r": _array_manifest(
                translation, name="x_world_r", semantic="fixed_transform.r"
            ),
        }
        fixed_transform_sha256 = _sha256_payload(transform_payload)
        fixture_payload = {
            "schema_version": FIXED_RENDER_FIXTURE_SCHEMA_VERSION,
            "case_id": case.case_id,
            "case_sha256": case.sha256,
            "scene_contract_sha256": packed_scene.scene_contract_sha256,
            "fixed_transform_id": fixed_transform_id,
            "fixed_transform_sha256": fixed_transform_sha256,
            "derived_arrays": _array_manifest_payload(frozen, _FIXED_FIXTURE_SEMANTICS),
            "topology_sha256": topology_sha256,
            "num_triangles": packed_scene.num_triangles,
            "num_planes": packed_scene.num_planes,
        }
        instance = object.__new__(cls)
        for name, value in {
            "packed_scene_owner": packed_scene,
            "case_id": case.case_id,
            "case_sha256": case.sha256,
            "fixed_transform_id": fixed_transform_id,
            "fixed_transform_sha256": fixed_transform_sha256,
            "arrays": MappingProxyType(frozen),
            "array_semantics": _FIXED_FIXTURE_SEMANTICS,
            "topology_array_names": _FIXED_FIXTURE_TOPOLOGY_ARRAYS,
            "topology_sha256": topology_sha256,
            "schema_version": FIXED_RENDER_FIXTURE_SCHEMA_VERSION,
            "fixed_render_fixture_sha256": _sha256_payload(fixture_payload),
        }.items():
            object.__setattr__(instance, name, value)
        return instance


@dataclass(frozen=True)
class CanonicalNativeInput:
    """Immutable host-side logical input from which both lanes allocate.

    This object is not a timed backend input.  It is the common logical source
    used before timing to create two independent, runtime-owned device inputs.
    """

    manifest: NativeInputManifest
    arrays: Mapping[str, np.ndarray]
    packed_scene_owner: PackedOpticalSceneHost | None = None
    transform_replay_owner: CanonicalTransformReplayHost | None = None
    fixed_fixture_owner: CanonicalFixedRenderFixtureHost | None = None

    @classmethod
    def from_scene_contract(
        cls,
        packed_scene: PackedOpticalSceneHost,
        *,
        input_contract: str,
        replay: CanonicalTransformReplayHost | None = None,
        fixed_fixture: CanonicalFixedRenderFixtureHost | None = None,
    ) -> "CanonicalNativeInput":
        """Bind one result-capable native input to the canonical scene object."""

        assert_outside_timed_boundary("canonical native scene input construction")
        case = _validate_packed_scene_case(packed_scene)
        frozen_content = frozen_workload_content(case.case_id)
        if packed_scene.scene_contract_sha256 != frozen_content.scene_contract_sha256:
            raise ValueError(
                "packed scene content does not match the frozen canonical workload"
            )
        if input_contract not in {
            FIXED_NATIVE_SCENE_INPUT,
            CANONICAL_NATIVE_REPLAY_INPUT,
            GPU_ENGINE_PUBLISHED_FRAME_INPUT,
        }:
            raise ValueError("input_contract is not a supported native input contract")

        arrays: dict[str, np.ndarray] = dict(packed_scene.arrays)
        array_semantics: dict[str, str] = dict(packed_scene.array_semantics)
        scene_names = tuple(sorted(packed_scene.arrays))
        replay_sha256: str | None = None
        replay_frames = 0
        fixture_sha256: str | None = None
        fixture_names: tuple[str, ...] = ()

        if input_contract == FIXED_NATIVE_SCENE_INPUT:
            if replay is not None:
                raise ValueError("fixed native scene input forbids transform replay")
            if fixed_fixture is None:
                raise ValueError("fixed native scene input requires a fixed render fixture")
            if fixed_fixture.packed_scene_owner is not packed_scene:
                raise ValueError("fixed fixture must retain the exact packed scene object")
            if fixed_fixture.case_id != case.case_id or fixed_fixture.case_sha256 != case.sha256:
                raise ValueError("fixed fixture case identity does not match the packed scene")
            if (
                frozen_content.fixed_render_fixture_sha256 is None
                or fixed_fixture.fixed_render_fixture_sha256
                != frozen_content.fixed_render_fixture_sha256
            ):
                raise ValueError(
                    "fixed render fixture does not match the frozen canonical workload"
                )
            _insert_disjoint_arrays(
                arrays,
                array_semantics,
                fixed_fixture.arrays,
                fixed_fixture.array_semantics,
                label="fixed fixture",
            )
            fixture_sha256 = fixed_fixture.fixed_render_fixture_sha256
            fixture_names = tuple(sorted(fixed_fixture.arrays))
            topology_sha256 = fixed_fixture.topology_sha256
        elif input_contract == CANONICAL_NATIVE_REPLAY_INPUT:
            if fixed_fixture is not None:
                raise ValueError("native replay input forbids a fixed render fixture")
            if replay is None:
                raise ValueError("native replay input requires a canonical transform replay")
            if replay.case_id != case.case_id or replay.case_sha256 != case.sha256:
                raise ValueError("transform replay case identity does not match the packed scene")
            if (
                frozen_content.transform_replay_sha256 is None
                or replay.transform_replay_sha256
                != frozen_content.transform_replay_sha256
            ):
                raise ValueError(
                    "transform replay does not match the frozen canonical workload"
                )
            _insert_disjoint_arrays(
                arrays,
                array_semantics,
                replay.arrays,
                replay.array_semantics,
                label="transform replay",
            )
            replay_sha256 = replay.transform_replay_sha256
            replay_frames = replay.frame_count
            topology_sha256 = _scene_topology_source_sha256(packed_scene)
        else:
            if replay is not None or fixed_fixture is not None:
                raise ValueError("physics-published input forbids replay and fixed fixture data")
            topology_sha256 = _scene_topology_source_sha256(packed_scene)

        manifests = tuple(
            NativeArrayManifest(
                name=name,
                semantic=array_semantics[name],
                dtype=arrays[name].dtype.str,
                shape=tuple(int(dimension) for dimension in arrays[name].shape),
                c_contiguous=bool(arrays[name].flags.c_contiguous),
                nbytes=int(arrays[name].nbytes),
                content_sha256=_array_content_sha256(arrays[name]),
            )
            for name in sorted(arrays)
        )
        if any(not manifest.c_contiguous for manifest in manifests):
            raise ValueError("scene-contract native arrays must be C-contiguous")
        content_sha256 = _sha256_payload([manifest.to_payload() for manifest in manifests])
        semantics = MappingProxyType(
            {
                "input_contract": input_contract,
                "scene_id": packed_scene.scene_id,
                "scene_contract_sha256": packed_scene.scene_contract_sha256,
            }
        )
        manifest = NativeInputManifest(
            schema_version=NATIVE_INPUT_SCHEMA_VERSION,
            case_id=case.case_id,
            case_sha256=case.sha256,
            materializer_id=packed_scene.materializer_id,
            semantics=semantics,
            arrays=manifests,
            content_sha256=content_sha256,
            topology_sha256=topology_sha256,
            input_contract=input_contract,
            scene_contract_schema_version=packed_scene.scene_contract_schema_version,
            scene_contract_sha256=packed_scene.scene_contract_sha256,
            scene_id=packed_scene.scene_id,
            scene_array_names=scene_names,
            transform_replay_sha256=replay_sha256,
            transform_frame_count=replay_frames,
            fixed_render_fixture_sha256=fixture_sha256,
            fixed_render_fixture_arrays=fixture_names,
        )
        return cls(
            manifest=manifest,
            arrays=MappingProxyType(arrays),
            packed_scene_owner=packed_scene,
            transform_replay_owner=replay,
            fixed_fixture_owner=fixed_fixture,
        )

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
            input_contract=None,
            scene_contract_schema_version=None,
            scene_contract_sha256=None,
            scene_id=None,
            scene_array_names=(),
            transform_replay_sha256=None,
            transform_frame_count=0,
            fixed_render_fixture_sha256=None,
            fixed_render_fixture_arrays=(),
        )
        return cls(manifest=manifest, arrays=MappingProxyType(frozen))


@dataclass(frozen=True)
class WarpPreparedNativeInput:
    """Warp-owned CUDA allocations at the native-ready boundary."""

    manifest: NativeInputManifest
    device_ordinal: int
    arrays: Mapping[str, Any]
    ready_event: Any
    source: CanonicalNativeInput | None = None

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
    source: CanonicalNativeInput | None = None

    @property
    def backend(self) -> NativeBackend:
        return "cuda"


PreparedNativeInput = WarpPreparedNativeInput | CudaPreparedNativeInput


@dataclass(frozen=True)
class PreparedOpticalLocalScene:
    """Allocation-free typed view of the exact a2 prepared scene arrays."""

    backend: NativeBackend
    prepared_owner: PreparedNativeInput
    arrays: Mapping[str, Any]
    scene_contract_sha256: str


@dataclass(frozen=True)
class PreparedFixedRenderInputs:
    """R0 composition of disjoint packed-scene and derived fixture storage."""

    backend: NativeBackend
    prepared_owner: PreparedNativeInput
    scene_arrays: Mapping[str, Any]
    derived_fixture_arrays: Mapping[str, Any]
    scene_contract_sha256: str
    fixed_render_fixture_sha256: str


def wrap_prepared_optical_local_scene(prepared: PreparedNativeInput) -> PreparedOpticalLocalScene:
    assert_outside_timed_boundary("prepared optical local-scene wrapping")
    manifest = prepared.manifest
    if not manifest.result_capable or manifest.scene_contract_sha256 is None:
        raise ValueError("prepared local scene requires result-capable scene provenance")
    names = manifest.scene_array_names
    if not names or any(name not in prepared.arrays for name in names):
        raise ValueError("prepared input does not contain every manifest scene array")
    selected = MappingProxyType({name: prepared.arrays[name] for name in names})
    return PreparedOpticalLocalScene(
        backend=prepared.backend,
        prepared_owner=prepared,
        arrays=selected,
        scene_contract_sha256=manifest.scene_contract_sha256,
    )


def wrap_prepared_fixed_render_inputs(
    prepared: PreparedNativeInput,
) -> PreparedFixedRenderInputs:
    assert_outside_timed_boundary("prepared fixed-render input wrapping")
    manifest = prepared.manifest
    if manifest.input_contract != FIXED_NATIVE_SCENE_INPUT:
        raise ValueError("fixed render inputs require fixed_native_scene_v1")
    if manifest.scene_contract_sha256 is None or manifest.fixed_render_fixture_sha256 is None:
        raise ValueError("fixed render provenance is incomplete")
    scene_names = manifest.scene_array_names
    fixture_names = manifest.fixed_render_fixture_arrays
    if not scene_names or not fixture_names or set(scene_names).intersection(fixture_names):
        raise ValueError("fixed render scene and derived fixture names must be disjoint")
    if any(name not in prepared.arrays for name in (*scene_names, *fixture_names)):
        raise ValueError("prepared input does not contain every fixed render array")
    return PreparedFixedRenderInputs(
        backend=prepared.backend,
        prepared_owner=prepared,
        scene_arrays=MappingProxyType({name: prepared.arrays[name] for name in scene_names}),
        derived_fixture_arrays=MappingProxyType(
            {name: prepared.arrays[name] for name in fixture_names}
        ),
        scene_contract_sha256=manifest.scene_contract_sha256,
        fixed_render_fixture_sha256=manifest.fixed_render_fixture_sha256,
    )


def prepared_storage_signatures(
    prepared: PreparedNativeInput,
    names: Sequence[str],
) -> tuple[NativeStorageSignature, ...]:
    """Describe exact prepared allocations for pre-enqueue provenance checks."""

    assert_outside_timed_boundary("prepared storage signature construction")
    manifests = {value.name: value for value in prepared.manifest.arrays}
    signatures: list[NativeStorageSignature] = []
    for name in names:
        if name not in prepared.arrays or name not in manifests:
            raise ValueError(f"prepared storage {name!r} is not manifest-owned")
        value = prepared.arrays[name]
        manifest = manifests[name]
        if manifest.nbytes == 0:
            pointer = 0
        elif prepared.backend == "warp":
            pointer = int(value.ptr)
        else:
            pointer = int(value.data_ptr())
        if pointer <= 0 and manifest.nbytes > 0:
            raise ValueError(f"prepared storage {name!r} has a null pointer")
        storage_token = (
            f"{prepared.backend}:0x{pointer:x}"
            if manifest.nbytes > 0
            else f"{prepared.backend}:empty:{name}:{prepared.manifest.sha256}"
        )
        signatures.append(
            NativeStorageSignature(
                name=name,
                backend=prepared.backend,
                device_ordinal=prepared.device_ordinal,
                storage_token=storage_token,
                dtype=manifest.dtype,
                shape=manifest.shape,
                nbytes=manifest.nbytes,
                active_start=0,
                active_count=manifest.nbytes,
                owner_token=prepared.manifest.sha256,
            )
        )
    return tuple(signatures)


def prepared_transform_frame_signatures(
    prepared: PreparedNativeInput,
    frame_index: int,
) -> tuple[NativeStorageSignature, ...]:
    """Bind one replay frame to exact prepared allocations and byte ranges."""

    assert_outside_timed_boundary("prepared transform-frame signature construction")
    manifest = prepared.manifest
    if manifest.input_contract != CANONICAL_NATIVE_REPLAY_INPUT:
        raise ValueError("transform-frame storage requires canonical_native_replay_v1")
    if (
        manifest.transform_replay_sha256 is None
        or manifest.transform_frame_count <= 0
        or frame_index < 0
        or frame_index >= manifest.transform_frame_count
    ):
        raise ValueError("transform replay frame index is out of range")
    signatures: list[NativeStorageSignature] = []
    for base in prepared_storage_signatures(prepared, tuple(_REPLAY_SEMANTICS)):
        if base.shape[0] != manifest.transform_frame_count:
            raise ValueError("prepared transform replay shape does not match frame count")
        if base.nbytes % manifest.transform_frame_count:
            raise ValueError("prepared transform replay bytes are not frame-addressable")
        frame_nbytes = base.nbytes // manifest.transform_frame_count
        signatures.append(
            NativeStorageSignature(
                name=base.name,
                backend=base.backend,
                device_ordinal=base.device_ordinal,
                storage_token=base.storage_token,
                dtype=base.dtype,
                shape=base.shape[1:],
                nbytes=base.nbytes,
                active_start=frame_index * frame_nbytes,
                active_count=frame_nbytes,
                owner_token=base.owner_token,
            )
        )
    return tuple(signatures)


def build_prepared_warp_local_scene(prepared: WarpPreparedNativeInput):
    """Create a production Warp scene wrapper over exact a2 allocations."""

    assert_outside_timed_boundary("prepared Warp local-scene construction")
    if prepared.backend != "warp" or prepared.source is None:
        raise ValueError("prepared Warp local scene requires its canonical source owner")
    packed = prepared.source.packed_scene_owner
    if packed is None:
        raise ValueError("prepared Warp local scene lacks a packed scene owner")
    selected = {name: prepared.arrays[name] for name in prepared.manifest.scene_array_names}
    from optics.device_scene import device_optical_scene_from_arrays

    return device_optical_scene_from_arrays(
        packed,
        selected,
        device=f"cuda:{prepared.device_ordinal}",
        resources=(prepared,),
    )


def build_prepared_warp_transform_inputs(
    prepared: WarpPreparedNativeInput,
    frame_index: int,
):
    """Bind one exact a2 replay allocation range to the Warp scene factory."""

    assert_outside_timed_boundary("prepared Warp transform-frame construction")
    manifest = prepared.manifest
    if prepared.backend != "warp" or manifest.input_contract != CANONICAL_NATIVE_REPLAY_INPUT:
        raise ValueError("prepared Warp transform inputs require canonical_native_replay_v1")
    if manifest.transform_replay_sha256 is None:
        raise ValueError("prepared Warp transform replay provenance is incomplete")
    if frame_index < 0 or frame_index >= manifest.transform_frame_count:
        raise ValueError("prepared Warp transform frame index is out of range")
    from optics.device_scene import warp_replay_transform_inputs

    return warp_replay_transform_inputs(
        prepared.arrays["replay_x_world_R"],
        prepared.arrays["replay_x_world_r"],
        frame_index=frame_index,
        input_contract=manifest.input_contract,
        transform_replay_sha256=manifest.transform_replay_sha256,
        resources=(prepared,),
    )


def build_prepared_warp_scene_update_factory(
    prepared: WarpPreparedNativeInput,
    *,
    stream: object,
):
    """Bind the Warp scene factory to the exact prepared manifest and scene."""

    assert_outside_timed_boundary("prepared Warp scene-command factory construction")
    if not prepared.manifest.result_capable or prepared.manifest.input_contract is None:
        raise ValueError("prepared Warp scene factory requires result-capable provenance")
    from optics.device_scene import WarpSceneUpdateCommandFactory

    return WarpSceneUpdateCommandFactory(
        build_prepared_warp_local_scene(prepared),
        stream=stream,
        native_input_manifest_sha256=prepared.manifest.sha256,
        input_contract=prepared.manifest.input_contract,
    )


def build_prepared_warp_bvh_refit_factory(
    prepared: WarpPreparedNativeInput,
    bvh: object,
    *,
    stream: object,
):
    """Bind one actual Warp topology to the exact prepared scene provenance."""

    assert_outside_timed_boundary("prepared Warp BVH-refit factory construction")
    manifest = prepared.manifest
    if not manifest.result_capable or manifest.input_contract is None:
        raise ValueError("prepared Warp refit factory requires result-capable provenance")
    packed = prepared.source.packed_scene_owner if prepared.source is not None else None
    if packed is None:
        raise ValueError("prepared Warp refit factory lacks its packed scene owner")
    if int(getattr(bvh, "num_primitives", -1)) != packed.num_triangles:
        raise ValueError("prepared Warp refit topology primitive count does not match its scene")
    if int(getattr(getattr(bvh, "device", None), "ordinal", -1)) != prepared.device_ordinal:
        raise ValueError("prepared Warp refit topology is on the wrong device")
    from optics.device_bvh import WarpBvhRefitCommandFactory

    return WarpBvhRefitCommandFactory(
        bvh,
        stream=stream,
        native_input_manifest_sha256=manifest.sha256,
        native_input_topology_sha256=manifest.topology_sha256,
        scene_contract_sha256=manifest.scene_contract_sha256,
        input_contract=manifest.input_contract,
    )


def build_prepared_cuda_local_scene(prepared: CudaPreparedNativeInput):
    """Create a production CUDA scene wrapper over exact a2 allocations."""

    assert_outside_timed_boundary("prepared CUDA local-scene construction")
    if prepared.backend != "cuda" or prepared.source is None:
        raise ValueError("prepared CUDA local scene requires its canonical source owner")
    packed = prepared.source.packed_scene_owner
    if packed is None:
        raise ValueError("prepared CUDA local scene lacks a packed scene owner")
    selected = {name: prepared.arrays[name] for name in prepared.manifest.scene_array_names}
    from optics.cuda_scene import cuda_optical_scene_from_arrays

    return cuda_optical_scene_from_arrays(
        packed,
        selected,
        device_ordinal=prepared.device_ordinal,
        ready_event=prepared.ready_event,
        owner=(prepared,),
    )


@dataclass(frozen=True)
class NativeInputValidationReport:
    """Direct, untimed parity evidence for one prepared native input."""

    backend: NativeBackend
    manifest_sha256: str
    content_sha256: str
    topology_sha256: str
    validated_arrays: tuple[str, ...]
    input_contract: str | None
    scene_contract_sha256: str | None
    transform_replay_sha256: str | None
    transform_frame_count: int
    fixed_render_fixture_sha256: str | None


@dataclass(frozen=True)
class NativeReadinessEvidence:
    """Lane evidence that preparation, validation, compilation, and warmup ended."""

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
            "schema_version",
            "backend",
            "device_ordinal",
            "manifest_sha256",
            "content_sha256",
            "topology_sha256",
            "input_contract",
            "scene_contract_schema_version",
            "scene_contract_sha256",
            "scene_id",
            "transform_replay_sha256",
            "transform_frame_count",
            "fixed_render_fixture_sha256",
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
        for field in ("device_ordinal", "warmup_frames_completed", "transform_frame_count"):
            value = payload.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"native readiness {field} must be a non-negative integer")
        for field in ("manifest_sha256", "content_sha256", "topology_sha256"):
            value = payload.get(field)
            if not _sha256_text(value):
                raise ValueError(f"native readiness {field} must be a lowercase SHA-256 digest")
        for field in (
            "scene_contract_sha256",
            "transform_replay_sha256",
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
            payload.get("scene_contract_schema_version") != SCENE_CONTRACT_SCHEMA_VERSION
            or not payload.get("scene_id")
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
    for name in (
        "input_contract",
        "scene_contract_sha256",
        "transform_replay_sha256",
        "transform_frame_count",
        "fixed_render_fixture_sha256",
    ):
        if getattr(validation, name) != getattr(prepared.manifest, name):
            raise ValueError(f"native validation {name} does not match prepared input")
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
        input_contract=prepared.manifest.input_contract,
        scene_contract_schema_version=prepared.manifest.scene_contract_schema_version,
        scene_contract_sha256=prepared.manifest.scene_contract_sha256,
        scene_id=prepared.manifest.scene_id,
        transform_replay_sha256=prepared.manifest.transform_replay_sha256,
        transform_frame_count=prepared.manifest.transform_frame_count,
        fixed_render_fixture_sha256=prepared.manifest.fixed_render_fixture_sha256,
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
        source=source,
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
        source=source,
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
        input_contract=prepared.manifest.input_contract,
        scene_contract_sha256=prepared.manifest.scene_contract_sha256,
        transform_replay_sha256=prepared.manifest.transform_replay_sha256,
        transform_frame_count=prepared.manifest.transform_frame_count,
        fixed_render_fixture_sha256=prepared.manifest.fixed_render_fixture_sha256,
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
    manifests = {value.name: value for value in warp_input.manifest.arrays}
    for name in warp_input.arrays:
        if manifests[name].nbytes == 0:
            continue
        warp_pointer = int(warp_input.arrays[name].ptr)
        torch_pointer = int(cuda_input.arrays[name].data_ptr())
        if warp_pointer == 0 or torch_pointer == 0:
            raise ValueError(f"native input {name!r} has a null device allocation")
        if warp_pointer == torch_pointer:
            raise ValueError(f"native input {name!r} shares a cross-runtime allocation")


def _validate_packed_scene_case(packed_scene: PackedOpticalSceneHost):
    if not isinstance(packed_scene, PackedOpticalSceneHost):
        raise TypeError("packed_scene must be a PackedOpticalSceneHost")
    case = get_case(packed_scene.source_contract_id)
    if packed_scene.source_contract_sha256 != case.sha256:
        raise ValueError("packed scene source hash does not match the canonical case")
    if packed_scene.scene_id != case.scene_id:
        raise ValueError("packed scene scene_id does not match the canonical case")
    if packed_scene.scene_contract_schema_version != SCENE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("packed scene schema version is not current")
    return case


def _insert_disjoint_arrays(
    destination: dict[str, np.ndarray],
    destination_semantics: dict[str, str],
    source: Mapping[str, np.ndarray],
    source_semantics: Mapping[str, str],
    *,
    label: str,
) -> None:
    names = set(source)
    if names != set(source_semantics):
        raise ValueError(f"{label} semantics must describe every and only array")
    duplicate_names = names.intersection(destination)
    if duplicate_names:
        raise ValueError(f"{label} duplicates native array names: {sorted(duplicate_names)}")
    duplicate_semantics = set(source_semantics.values()).intersection(destination_semantics.values())
    if duplicate_semantics:
        raise ValueError(f"{label} duplicates native semantics: {sorted(duplicate_semantics)}")
    for name in sorted(names):
        value = source[name]
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ValueError(f"{label} array {name!r} must be a non-object NumPy array")
        if not value.flags.c_contiguous or value.flags.writeable:
            raise ValueError(f"{label} array {name!r} must be C-contiguous and read-only")
        destination[name] = value
        destination_semantics[name] = source_semantics[name]


def _scene_topology_source_sha256(packed_scene: PackedOpticalSceneHost) -> str:
    names = ("triangle_primitive_global_id", "triangle_source_order_key")
    return _sha256_payload(
        [
            _array_manifest(
                packed_scene.arrays[name],
                name=name,
                semantic=packed_scene.array_semantics[name],
            )
            for name in names
        ]
    )


def _derive_fixed_fixture_arrays(
    packed_scene: PackedOpticalSceneHost,
    x_world_R: np.ndarray,
    x_world_r: np.ndarray,
) -> dict[str, np.ndarray]:
    packed = packed_scene.arrays
    triangle_vertices = np.asarray(packed["triangle_vertices_local"], dtype=np.float32).reshape(
        -1, 3, 3
    )
    triangle_instance = packed["triangle_instance_index"]
    triangle_v0 = np.empty((packed_scene.num_triangles, 3), dtype=np.float32)
    triangle_e1 = np.empty_like(triangle_v0)
    triangle_e2 = np.empty_like(triangle_v0)
    triangle_normal = np.empty_like(triangle_v0)
    triangle_aabb_min = np.empty_like(triangle_v0)
    triangle_aabb_max = np.empty_like(triangle_v0)

    for primitive in range(packed_scene.num_triangles):
        R_world_geometry, r_world_geometry = _fixed_instance_transform(
            packed_scene,
            int(triangle_instance[primitive]),
            x_world_R,
            x_world_r,
        )
        world = triangle_vertices[primitive] @ R_world_geometry.T + r_world_geometry
        edge1 = world[1] - world[0]
        edge2 = world[2] - world[0]
        normal = np.cross(edge1, edge2)
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-8:
            raise ValueError("fixed fixture unexpectedly produced a degenerate triangle")
        triangle_v0[primitive] = world[0]
        triangle_e1[primitive] = edge1
        triangle_e2[primitive] = edge2
        triangle_normal[primitive] = normal / norm
        triangle_aabb_min[primitive] = np.min(world, axis=0)
        triangle_aabb_max[primitive] = np.max(world, axis=0)

    plane_normal = np.empty((packed_scene.num_planes, 3), dtype=np.float32)
    plane_point = np.empty_like(plane_normal)
    for plane in range(packed_scene.num_planes):
        R_world_geometry, r_world_geometry = _fixed_instance_transform(
            packed_scene,
            int(packed["plane_instance_index"][plane]),
            x_world_R,
            x_world_r,
        )
        normal = R_world_geometry @ packed["plane_normal_local"][plane]
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-8:
            raise ValueError("fixed fixture unexpectedly produced a degenerate plane normal")
        plane_normal[plane] = normal / norm
        plane_point[plane] = (
            R_world_geometry @ packed["plane_point_local"][plane] + r_world_geometry
        )

    host_bvh = build_host_median_bvh(
        triangle_aabb_min,
        triangle_aabb_max,
        packed["triangle_source_order_key"],
        leaf_size=4,
        split_strategy="sort",
    )
    return {
        "triangle_v0_world": triangle_v0,
        "triangle_e1_world": triangle_e1,
        "triangle_e2_world": triangle_e2,
        "triangle_normal_world": triangle_normal,
        "triangle_aabb_min": triangle_aabb_min,
        "triangle_aabb_max": triangle_aabb_max,
        "plane_normal_world": plane_normal,
        "plane_point_world": plane_point,
        "bvh_bounds_min": host_bvh["bounds_min"],
        "bvh_bounds_max": host_bvh["bounds_max"],
        "bvh_left": host_bvh["left"],
        "bvh_right": host_bvh["right"],
        "bvh_start": host_bvh["start"],
        "bvh_count": host_bvh["count"],
        "bvh_node_depth": host_bvh["node_depth"],
        "bvh_level_ranges": host_bvh["level_ranges"],
        "bvh_prim_ids": host_bvh["prim_ids"],
    }


def _fixed_instance_transform(
    packed_scene: PackedOpticalSceneHost,
    instance_index: int,
    x_world_R: np.ndarray,
    x_world_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    packed = packed_scene.arrays
    body_index = int(packed["instance_body_index"][instance_index])
    if body_index < 0:
        R_world_body = np.eye(3, dtype=np.float32)
        r_world_body = np.zeros(3, dtype=np.float32)
    else:
        R_world_body = x_world_R[body_index]
        r_world_body = x_world_r[body_index]
    R_body_geometry = packed["instance_X_body_geometry_R"][instance_index].reshape(3, 3)
    r_body_geometry = packed["instance_X_body_geometry_r"][instance_index]
    return (
        np.asarray(R_world_body @ R_body_geometry, dtype=np.float32),
        np.asarray(r_world_body + R_world_body @ r_body_geometry, dtype=np.float32),
    )


def _freeze_array_mapping(arrays: Mapping[str, object]) -> dict[str, np.ndarray]:
    if set(arrays) != set(_FIXED_FIXTURE_SEMANTICS):
        raise ValueError("fixed fixture arrays do not match the derived semantic registry")
    frozen: dict[str, np.ndarray] = {}
    for name in sorted(arrays):
        value = np.array(arrays[name], copy=True, order="C")
        if value.dtype.hasobject:
            raise ValueError(f"fixed fixture array {name!r} cannot use object dtype")
        value.setflags(write=False)
        frozen[name] = value
    return frozen


def _array_manifest_payload(
    arrays: Mapping[str, np.ndarray],
    semantics: Mapping[str, str],
) -> list[dict[str, object]]:
    return [
        _array_manifest(arrays[name], name=name, semantic=semantics[name])
        for name in sorted(arrays)
    ]


def _array_manifest(array: np.ndarray, *, name: str, semantic: str) -> dict[str, object]:
    return {
        "name": name,
        "semantic": semantic,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "c_contiguous": bool(array.flags.c_contiguous),
        "nbytes": int(array.nbytes),
        "content_sha256": _array_content_sha256(array),
    }


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
