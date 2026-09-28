"""Exact CPU-only camera data; never a live execution/readiness authority.

The v1 contract resolves only the previously symbolic asymmetric camera.
Runtime adoption must additionally bind actual camera/ABI values to its plan.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .cases import canonical_json_bytes, get_case

RESOLVED_CAMERA_SCHEMA_VERSION = "q54.p12.4.resolved-camera.v1"
ASYMMETRIC_CAMERA_CASE_ID = "asymmetric_synthetic_7x3_v1"
ASYMMETRIC_CAMERA_CASE_SHA256 = "3f6ce0ac1d59d0c0d0e5ea57f497c308a9d5c4b2a5c19462c900627fd99d0181"
ASYMMETRIC_CAMERA_SHA256 = "04ab7bc2571b8396f06d36b7baea66224868001894c4ce7a5ea12e4faecda1ef"
_ROTATION_ID = "non_identity_compound_xyz_v1"
_ROTATION_F64 = (
    "0x1.e966c0cb6338ep-1",
    "-0x1.1fb2169a99746p-3",
    "-0x1.08406f872195ep-2",
    "0x1.488c96e936144p-3",
    "0x1.f88017b67564dp-1",
    "0x1.d9c950df370bfp-5",
    "0x1.f81f81f81f820p-3",
    "-0x1.8c01c4944e606p-4",
    "0x1.edc5678284ecfp-1",
)
_DATA_FIELDS = (
    "case_id",
    "width",
    "height",
    "intrinsics_f64",
    "origin_world_f64",
    "rotation_world_from_camera_f64",
    "max_distance_f64",
    "sensor_role",
)
_TUPLE_FIELDS = ("intrinsics_f64", "origin_world_f64", "rotation_world_from_camera_f64")


def _frozen_payload() -> dict:
    case = get_case(ASYMMETRIC_CAMERA_CASE_ID)
    if case.sha256 != ASYMMETRIC_CAMERA_CASE_SHA256 or case.camera.get("rotation") != _ROTATION_ID:
        raise ValueError("resolved camera source case or rotation identity drifted")
    payload = {
        "schema_version": RESOLVED_CAMERA_SCHEMA_VERSION,
        "case_id": case.case_id,
        "case_sha256": case.sha256,
        "rotation_id": _ROTATION_ID,
        "convention": "active_camera_to_world_rz_ry_rx_column_vector_v1",
        "ray_layout": "integer_pixel_center_py_times_width_plus_px_v1",
        "scalar_encoding": "ieee754_binary64_canonical_hex_v1",
        "native_scalar_encoding": "binary64_to_binary32_nearest_even_v1",
        "width": case.width,
        "height": case.height,
        "intrinsics_f64": [float(case.camera[name]).hex() for name in ("fx", "fy", "cx", "cy")],
        "origin_world_f64": [float(value).hex() for value in case.camera["translation_world"]],
        "rotation_world_from_camera_f64": list(_ROTATION_F64),
        "max_distance_f64": float(case.camera["max_distance_m"]).hex(),
        "sensor_role": case.render_contract["sensor_role"],
    }
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != ASYMMETRIC_CAMERA_SHA256:
        raise ValueError("resolved camera definition drifted")
    return payload


def _require_exact(actual: object, expected: object) -> None:
    if type(actual) is not type(expected):
        raise ValueError("resolved camera payload type differs from frozen contract")
    if type(expected) is dict:
        if any(type(name) is not str for name in actual):
            raise ValueError("resolved camera payload keys must be exact strings")
        if actual.keys() != expected.keys():
            raise ValueError("resolved camera payload fields differ from frozen contract")
        for name in expected:
            _require_exact(actual[name], expected[name])
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError("resolved camera payload shape differs from frozen contract")
        for value, reference in zip(actual, expected, strict=True):
            _require_exact(value, reference)
    elif actual != expected:
        raise ValueError("resolved camera payload value differs from frozen contract")


@dataclass(frozen=True, slots=True)
class ResolvedPinholeCameraContract:
    """Immutable descriptive scalars, checked against the sole frozen definition."""

    case_id: str
    width: int
    height: int
    intrinsics_f64: tuple[str, ...]
    origin_world_f64: tuple[str, ...]
    rotation_world_from_camera_f64: tuple[str, ...]
    max_distance_f64: str
    sensor_role: str

    def __post_init__(self) -> None:
        self.to_payload()

    @classmethod
    def from_payload(cls, payload: object) -> ResolvedPinholeCameraContract:
        if cls is not ResolvedPinholeCameraContract:
            raise TypeError("exact resolved camera contract type required")
        expected = _frozen_payload()
        _require_exact(payload, expected)
        return cls(
            **{
                name: tuple(expected[name]) if name in _TUPLE_FIELDS else expected[name]
                for name in _DATA_FIELDS
            }
        )

    def to_payload(self) -> dict:
        if type(self) is not ResolvedPinholeCameraContract:
            raise TypeError("exact resolved camera contract type required")
        expected = _frozen_payload()
        actual = dict(expected)
        for name in _DATA_FIELDS:
            value = getattr(self, name)
            if name in _TUPLE_FIELDS:
                if type(value) is not tuple:
                    raise ValueError("resolved camera requires immutable scalar tuples")
                value = list(value)
            actual[name] = value
        _require_exact(actual, expected)
        return actual

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_payload())).hexdigest()

    def scalar_values(self) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float]:
        """Validated binary64 intrinsics, origin, row-major R, maximum distance.

        Untimed CPU preparation only. Native conversion and actual ABI binding
        belong to the separately reviewed runtime integration, not this record.
        """
        payload = self.to_payload()
        return (
            tuple(float.fromhex(value) for value in payload["intrinsics_f64"]),
            tuple(float.fromhex(value) for value in payload["origin_world_f64"]),
            tuple(float.fromhex(value) for value in payload["rotation_world_from_camera_f64"]),
            float.fromhex(payload["max_distance_f64"]),
        )


def resolve_frozen_camera_contract(case_id: str) -> ResolvedPinholeCameraContract:
    if type(case_id) is not str or case_id != ASYMMETRIC_CAMERA_CASE_ID:
        raise ValueError("no frozen resolved camera for this case")
    return ResolvedPinholeCameraContract.from_payload(_frozen_payload())
