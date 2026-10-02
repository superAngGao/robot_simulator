"""CPU render parameter records. No GPU, readiness or execution authority.

The sole materializer computes a sequence for reproduction; admission compares
that computation to frozen canonical bytes. Native stage/receipt adoption is a
separate integration step, not something a public dataclass/hash can authorize.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from optics.camera_presets import build_lab_video_camera, build_model_bounds_camera
from optics.scene_contract import PackedOpticalSceneHost

from .assets import AssetManifestEvidence
from .camera_contract import resolve_frozen_camera_contract
from .cases import canonical_json_bytes, get_case
from .native_input import _derive_fixed_fixture_arrays
from .render_sequence_contract import RESOLVED_RENDER_SEQUENCE_SHA256
from .timing import assert_outside_timed_boundary
from .workload_contract import frozen_workload_content
from .workloads import _materialize_canonical_scene, build_canonical_transform_replay

SEQUENCE_SCHEMA = "q54.p12.4.resolved-render-sequence.v1"
BOUND_SCHEMA = "q54.p12.4.bound-render-parameters.v1"
_ASYMMETRIC = "asymmetric_synthetic_7x3_v1"
_DYNAMIC = "synthetic_dynamic_rigid_960x640_v1"
_GO2 = ("go2_static_960x640_v1", "go2_static_1920x1080_v1")
_CASE_IDS = (_ASYMMETRIC, _DYNAMIC, *_GO2)
_PROFILES = ("direct_light_full", "rgb_preview", "render_only")
_PASS_DIAGNOSTICS = {"primary_unprofiled": False, "diagnostic_profiled": True}
_SEQUENCE_FIELDS = {
    "schema_version",
    "case_id",
    "case_sha256",
    "scene_contract_sha256",
    "bounds_source_id",
    "bounds_f64",
    "camera_resolution_ids",
    "frames",
    "ambient_f64",
    "background_f64",
    "shadow_bias_f64",
}
_FRAME_FIELDS = {
    "camera_index",
    "width",
    "height",
    "intrinsics_f64",
    "origin_f64",
    "rotation_f64",
    "max_distance_f64",
    "sensor_role",
}
NATIVE_SCALAR_ORDER = (
    "fx",
    "fy",
    "cx",
    "cy",
    "origin_x",
    "origin_y",
    "origin_z",
    "rotation_00",
    "rotation_01",
    "rotation_02",
    "rotation_10",
    "rotation_11",
    "rotation_12",
    "rotation_20",
    "rotation_21",
    "rotation_22",
    "max_distance",
    "ambient_r",
    "ambient_g",
    "ambient_b",
    "background_r",
    "background_g",
    "background_b",
    "shadow_bias",
)


def _case(case_id):
    if type(case_id) is not str or case_id not in _CASE_IDS:
        raise ValueError("no resolved native render sequence for this case; shipping remains closed")
    return get_case(case_id)


def _hex_values(values):
    return [float(value).hex() for value in values]


def _f64(value):
    if type(value) is not str:
        raise ValueError("binary64 must be canonical finite hex text")
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise ValueError("binary64 must be canonical finite hex text") from exc
    if not math.isfinite(number) or number.hex() != value:
        raise ValueError("binary64 must be canonical finite hex text")
    return number


def _vector(value, size):
    if type(value) is not list or len(value) != size:
        raise ValueError("wrong render scalar shape")
    return tuple(_f64(item) for item in value)


def _fields(payload, names):
    if type(payload) is not dict or any(type(key) is not str for key in payload) or set(payload) != names:
        raise ValueError("render record has unexpected fields")


def _index(value):
    if type(value) is not int or value < 0:
        raise ValueError("sample index must be an exact nonnegative integer")
    return value


def _resolution_ids(case_id):
    if case_id == _ASYMMETRIC:
        return ["asymmetric_resolved_camera_v1"]
    if case_id == _DYNAMIC:
        return ["model_bounds_front_frame0_v1"]
    return ["lab_video_camera_orbit_120_30_v1"]


def _validate_sequence(payload):
    _fields(payload, _SEQUENCE_FIELDS)
    case = _case(payload["case_id"])
    if (
        payload["schema_version"] != SEQUENCE_SCHEMA
        or payload["case_sha256"] != case.sha256
        or payload["scene_contract_sha256"] != frozen_workload_content(case.case_id).scene_contract_sha256
        or payload["camera_resolution_ids"] != _resolution_ids(case.case_id)
        or payload["bounds_source_id"]
        != (
            "canonical_replay_frame0_bounds_v1"
            if case.case_id == _DYNAMIC
            else "canonical_fixed_world_bounds_v1"
        )
    ):
        raise ValueError("render sequence source identity mismatch")
    bounds = payload["bounds_f64"]
    if type(bounds) is not list or len(bounds) != 2:
        raise ValueError("render bounds must be 2x3")
    lower, upper = (_vector(row, 3) for row in bounds)
    if any(a > b for a, b in zip(lower, upper, strict=True)):
        raise ValueError("render bounds are inverted")
    _vector(payload["ambient_f64"], 3)
    _vector(payload["background_f64"], 3)
    if _f64(payload["shadow_bias_f64"]) <= 0:
        raise ValueError("shadow bias must be positive")
    frames = payload["frames"]
    if type(frames) is not list or len(frames) != (1 if case.case_id == _ASYMMETRIC else 120):
        raise ValueError("wrong render sequence frame count")
    for index, frame in enumerate(frames):
        _fields(frame, _FRAME_FIELDS)
        if (
            _index(frame["camera_index"]) != index
            or type(frame["width"]) is not int
            or frame["width"] != case.width
            or type(frame["height"]) is not int
            or frame["height"] != case.height
            or frame["sensor_role"] != case.render_contract["sensor_role"]
        ):
            raise ValueError("render frame metadata mismatch")
        intrinsics = _vector(frame["intrinsics_f64"], 4)
        _vector(frame["origin_f64"], 3)
        _vector(frame["rotation_f64"], 9)
        if min(intrinsics[:2]) <= 0 or _f64(frame["max_distance_f64"]) <= 0:
            raise ValueError("focal lengths and range must be positive")


@dataclass(frozen=True, slots=True)
class ResolvedRenderSequence:
    """Immutable serialized descriptive data, not canonical-source authority."""

    canonical_bytes: bytes

    def __post_init__(self):
        self.to_payload()

    @classmethod
    def from_payload(cls, payload):
        assert_outside_timed_boundary("resolved render sequence construction")
        if cls is not ResolvedRenderSequence:
            raise TypeError("exact render sequence type required")
        _validate_sequence(payload)
        return cls(canonical_json_bytes(payload))

    def to_payload(self):
        if type(self) is not ResolvedRenderSequence or type(self.canonical_bytes) is not bytes:
            raise TypeError("exact render sequence and immutable bytes required")
        payload = json.loads(self.canonical_bytes)
        _validate_sequence(payload)
        if canonical_json_bytes(payload) != self.canonical_bytes:
            raise ValueError("render sequence bytes are not canonical")
        return payload

    @property
    def sha256(self):
        self.to_payload()
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalRenderSource:
    """Same materialization's scene and computed sequence; untimed CPU only.

    No consumer may mistake this generic record for prepared/readiness authority.
    Frozen artifact validation and later exact-allocation binding are separate.
    """

    packed_scene: PackedOpticalSceneHost
    sequence: ResolvedRenderSequence


def load_frozen_render_sequence(case_id: str) -> ResolvedRenderSequence:
    """Read only frozen descriptive bytes; this does not validate current assets."""
    assert_outside_timed_boundary("frozen render sequence loading")
    _case(case_id)
    path = Path(__file__).parent / "data" / "render_sequences" / "v1" / (case_id + ".jsonl")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("sequence artifact requires exact one-line JSONL framing")
    sequence = ResolvedRenderSequence(raw[:-1])
    if sequence.sha256 != RESOLVED_RENDER_SEQUENCE_SHA256[case_id]:
        raise ValueError("sequence artifact digest differs from frozen content")
    return sequence


def resolve_render_source(
    case_id: str,
    *,
    go2_model_dir: Path | None = None,
    asset_evidence: AssetManifestEvidence | None = None,
) -> CanonicalRenderSource:
    """Verify actual source/bounds/formula, then use the common artifact bytes.

    A lane may not silently retain different host/libm rounding. Neither this
    CPU record nor its digest substitutes for the a2 physical storage registry.
    """
    reproduced = materialize_render_source(
        case_id,
        go2_model_dir=go2_model_dir,
        asset_evidence=asset_evidence,
    )
    frozen = load_frozen_render_sequence(case_id)
    if reproduced.sequence.canonical_bytes != frozen.canonical_bytes:
        raise ValueError("actual source/bounds/camera formula differs from frozen render sequence")
    return CanonicalRenderSource(reproduced.packed_scene, frozen)


def materialize_render_source(
    case_id: str,
    *,
    go2_model_dir: Path | None = None,
    asset_evidence: AssetManifestEvidence | None = None,
) -> CanonicalRenderSource:
    """Reproduce parameters with the sole existing scene/camera math.

    Go2 is captured/imported exactly once: packed scene and float64 visual
    bounds come from that import. Synthetic bounds reuse CPU world derivation,
    including body/instance transforms at replay frame0, not a copied formula.
    """
    assert_outside_timed_boundary("canonical render source materialization")
    case = _case(case_id)
    packed, bounds = _materialize_canonical_scene(
        case_id, go2_model_dir=go2_model_dir, asset_evidence=asset_evidence
    )
    if bounds is None:
        if case_id == _DYNAMIC:
            replay = build_canonical_transform_replay(case_id)
            rotation = replay.arrays["replay_x_world_R"][0]
            translation = replay.arrays["replay_x_world_r"][0]
        else:
            rotation = np.empty((0, 3, 3), dtype=np.float32)
            translation = np.empty((0, 3), dtype=np.float32)
        world = _derive_fixed_fixture_arrays(packed, rotation, translation)
        bounds = (
            tuple(float(v) for v in world["triangle_aabb_min"].min(axis=0)),
            tuple(float(v) for v in world["triangle_aabb_max"].max(axis=0)),
        )
    lower, upper = (np.asarray(bound, dtype=np.float64) for bound in bounds)
    frames = []
    if case_id == _ASYMMETRIC:
        camera = resolve_frozen_camera_contract(case_id)
        intrinsics, origin, rotation, distance = camera.scalar_values()
        frames.append(
            {
                "camera_index": 0,
                "width": case.width,
                "height": case.height,
                "intrinsics_f64": _hex_values(intrinsics),
                "origin_f64": _hex_values(origin),
                "rotation_f64": _hex_values(rotation),
                "max_distance_f64": distance.hex(),
                "sensor_role": camera.sensor_role,
            }
        )
    else:
        fixed = None
        if case_id == _DYNAMIC:
            fixed = build_model_bounds_camera(
                bounds_min=lower,
                bounds_max=upper,
                width=case.width,
                height=case.height,
                frame_id=1,
                sim_time=0.0,
                view="front",
            )
        for index in range(120):
            # Adapter-owned frozen inputs, never an IPC/caller Namespace.
            camera = (
                fixed
                if fixed is not None
                else build_lab_video_camera(
                    SimpleNamespace(
                        bounds_min=lower,
                        bounds_max=upper,
                        frame=SimpleNamespace(frame_id=index + 1, sim_time=index / 30.0),
                    ),
                    SimpleNamespace(
                        width=case.width, height=case.height, video_mode="camera_orbit", video_frames=120
                    ),
                    index,
                )
            )
            frames.append(
                {
                    "camera_index": index,
                    "width": camera.width,
                    "height": camera.height,
                    "intrinsics_f64": _hex_values((camera.fx, camera.fy, camera.cx, camera.cy)),
                    "origin_f64": _hex_values(camera.X_world_camera.r),
                    "rotation_f64": _hex_values(camera.X_world_camera.R.reshape(-1)),
                    "max_distance_f64": float(camera.max_distance).hex(),
                    "sensor_role": camera.sensor_role,
                }
            )
    payload = {
        "schema_version": SEQUENCE_SCHEMA,
        "case_id": case_id,
        "case_sha256": case.sha256,
        "scene_contract_sha256": packed.scene_contract_sha256,
        "bounds_source_id": "canonical_replay_frame0_bounds_v1"
        if case_id == _DYNAMIC
        else "canonical_fixed_world_bounds_v1",
        "bounds_f64": [_hex_values(bound) for bound in bounds],
        "camera_resolution_ids": _resolution_ids(case_id),
        "frames": frames,
        "ambient_f64": _hex_values((0.03, 0.05, 0.07) if case_id == _ASYMMETRIC else (0.08, 0.085, 0.09)),
        "background_f64": _hex_values((0.0, 0.0, 0.0) if case_id == _ASYMMETRIC else (0.025, 0.03, 0.038)),
        "shadow_bias_f64": (1.0e-4).hex(),
    }
    return CanonicalRenderSource(packed, ResolvedRenderSequence.from_payload(payload))


def _bound_payload(sequence, *, sample_index, profile, shadows, pass_mode, role_to_mask):
    assert_outside_timed_boundary("bound render parameter validation")
    if type(sequence) is not ResolvedRenderSequence:
        raise TypeError("exact render sequence type required")
    source = sequence.to_payload()
    sample_index = _index(sample_index)
    if type(profile) is not str or profile not in _PROFILES or type(shadows) is not bool:
        raise ValueError("invalid profile/shadow parameters")
    if type(pass_mode) is not str or pass_mode not in _PASS_DIAGNOSTICS:
        raise ValueError("invalid diagnostic pass mode")
    frame = source["frames"][sample_index % len(source["frames"])]
    role_mask = role_to_mask[frame["sensor_role"]]
    if type(role_mask) is not int or role_mask <= 0 or role_mask >= (1 << 63):
        raise ValueError("sensor role must have a valid scene role mask")
    scalars = (
        *frame["intrinsics_f64"],
        *frame["origin_f64"],
        *frame["rotation_f64"],
        frame["max_distance_f64"],
        *source["ambient_f64"],
        *source["background_f64"],
        source["shadow_bias_f64"],
    )
    # struct implements one nearest-even binary64->binary32 conversion.
    encoded = [struct.pack("<f", _f64(value)).hex() for value in scalars]
    if any(not math.isfinite(struct.unpack("<f", bytes.fromhex(value))[0]) for value in encoded):
        raise ValueError("native scalar conversion overflow")
    return {
        "schema_version": BOUND_SCHEMA,
        "sequence_sha256": sequence.sha256,
        "camera_index": frame["camera_index"],
        "frame_id": sample_index + 1,
        "sim_time_f64": (sample_index / 30.0).hex(),
        "env_idx": 0,
        "sensor_id": "q54:" + source["case_id"],
        "profile": profile,
        "shadows": shadows,
        "diagnostics": _PASS_DIAGNOSTICS[pass_mode],
        "width": frame["width"],
        "height": frame["height"],
        "sensor_role_mask": role_mask,
        "native_scalars_f32": encoded,
    }


@dataclass(frozen=True, slots=True)
class BoundRenderParameters:
    """Descriptive exact scalar ABI record, validated against a source on use."""

    canonical_bytes: bytes

    def validate(self, sequence, *, sample_index, profile, shadows, pass_mode, role_to_mask):
        expected = _bound_payload(
            sequence,
            sample_index=sample_index,
            profile=profile,
            shadows=shadows,
            pass_mode=pass_mode,
            role_to_mask=role_to_mask,
        )
        if (
            type(self) is not BoundRenderParameters
            or type(self.canonical_bytes) is not bytes
            or self.canonical_bytes != canonical_json_bytes(expected)
        ):
            raise ValueError("bound render parameters differ from actual source/request")
        return expected

    @property
    def sha256(self):
        # A hash is descriptive, not evidence that validate or a GPU call occurred.
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def bind_render_parameters(sequence, *, sample_index, profile, shadows, pass_mode, role_to_mask):
    return BoundRenderParameters(
        canonical_json_bytes(
            _bound_payload(
                sequence,
                sample_index=sample_index,
                profile=profile,
                shadows=shadows,
                pass_mode=pass_mode,
                role_to_mask=role_to_mask,
            )
        )
    )
