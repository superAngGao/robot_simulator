"""Implementation-neutral optical output and host-payload contracts.

This module is deliberately CPU-safe.  Executors, host staging, delivery, and
benchmark schemas all consume these definitions instead of maintaining local
copies of channel names, dtypes, shapes, or byte calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

PRIMARY_DIAGNOSTIC_CHANNEL_NAMES = (
    "bvh_stack_overflow_count",
    "shadow_stack_overflow_count",
    "bvh_max_stack_depth",
    "shadow_max_stack_depth",
)

SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES = (
    "shadow_traversal_ray_count",
    "shadow_traversal_directional_ray_count",
    "shadow_traversal_point_ray_count",
    "shadow_traversal_occluded_count",
    "shadow_traversal_unoccluded_count",
    "shadow_traversal_node_visit_count",
    "shadow_traversal_leaf_visit_count",
    "shadow_traversal_triangle_test_count",
    "shadow_traversal_plane_test_count",
)


@dataclass(frozen=True)
class CanonicalHostChannelSpec:
    """Canonical host dtype and shape for one optical payload channel."""

    name: str
    kind: Literal["per_ray", "scalar"]
    dtype: str
    trailing_shape: tuple[int, ...] = ()
    scalar_shape: tuple[int, ...] = ()

    def expected_shape(self, num_rays: int) -> tuple[int, ...]:
        if self.kind == "per_ray":
            return (int(num_rays), *self.trailing_shape)
        return self.scalar_shape

    def expected_nbytes(self, num_rays: int) -> int:
        shape = self.expected_shape(num_rays)
        elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
        return elements * np.dtype(self.dtype).itemsize

    def schema_payload(self) -> dict[str, object]:
        if self.kind == "per_ray":
            return {"dtype": self.dtype, "trailing_shape": list(self.trailing_shape)}
        return {"dtype": self.dtype, "shape": list(self.scalar_shape)}


@dataclass(frozen=True)
class HostPayloadContract:
    """The canonical host-visible channels for one delivery payload."""

    contract_id: str
    per_ray: tuple[CanonicalHostChannelSpec, ...] = ()
    scalar: tuple[CanonicalHostChannelSpec, ...] = ()

    def expected_observation(
        self, num_rays: int
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
        per_ray = [_channel_observation(spec, num_rays) for spec in self.per_ray]
        scalar = [_channel_observation(spec, num_rays) for spec in self.scalar]
        total = sum(int(channel["nbytes"]) for channel in (*per_ray, *scalar))
        return per_ray, scalar, total

    def schema_payload(self) -> dict[str, dict[str, dict[str, object]]]:
        return {
            "per_ray": {spec.name: spec.schema_payload() for spec in self.per_ray},
            "scalar": {spec.name: spec.schema_payload() for spec in self.scalar},
        }


def _channel_observation(spec: CanonicalHostChannelSpec, num_rays: int) -> dict[str, object]:
    return {
        "name": spec.name,
        "dtype": spec.dtype,
        "shape": list(spec.expected_shape(num_rays)),
        "nbytes": spec.expected_nbytes(num_rays),
    }


DIRECT_LIGHT_FULL_HOST_PER_RAY_CHANNEL_SPECS = (
    CanonicalHostChannelSpec("hit_mask", "per_ray", "bool"),
    CanonicalHostChannelSpec("range_m", "per_ray", "float64"),
    CanonicalHostChannelSpec("position_world", "per_ray", "float64", trailing_shape=(3,)),
    CanonicalHostChannelSpec("normal_world", "per_ray", "float64", trailing_shape=(3,)),
    CanonicalHostChannelSpec("numeric_instance_id", "per_ray", "int64"),
    CanonicalHostChannelSpec("material_index", "per_ray", "int32"),
    CanonicalHostChannelSpec("rgb", "per_ray", "float64", trailing_shape=(3,)),
    CanonicalHostChannelSpec("intensity", "per_ray", "float64"),
)

PRIMARY_HOST_SCALAR_CHANNEL_SPECS = tuple(
    CanonicalHostChannelSpec(name, "scalar", "int32", scalar_shape=(1,))
    for name in PRIMARY_DIAGNOSTIC_CHANNEL_NAMES
)

SHADOW_TRAVERSAL_HOST_SCALAR_CHANNEL_SPECS = tuple(
    CanonicalHostChannelSpec(name, "scalar", "int32", scalar_shape=(1,))
    for name in SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES
)

DIRECT_LIGHT_FULL_PRIMARY_HOST_CHANNEL_SPECS = (
    *DIRECT_LIGHT_FULL_HOST_PER_RAY_CHANNEL_SPECS,
    *PRIMARY_HOST_SCALAR_CHANNEL_SPECS,
)

OUTPUT_PROFILE_REQUIRED_CHANNELS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "geometry_full": frozenset(
            {
                "hit_mask",
                "range_m",
                "position_world",
                "normal_world",
                "numeric_instance_id",
            }
        ),
        "direct_light_full": frozenset(spec.name for spec in DIRECT_LIGHT_FULL_PRIMARY_HOST_CHANNEL_SPECS),
        "rgb_preview": frozenset(
            {
                "hit_mask",
                "rgb",
                *PRIMARY_DIAGNOSTIC_CHANNEL_NAMES,
            }
        ),
        "render_only": frozenset(PRIMARY_DIAGNOSTIC_CHANNEL_NAMES),
    }
)

HOST_PAYLOAD_CONTRACTS: Mapping[str, HostPayloadContract] = MappingProxyType(
    {
        "none_device_v1": HostPayloadContract("none_device_v1"),
        "rgb_native_host_v1": HostPayloadContract(
            "rgb_native_host_v1",
            per_ray=(CanonicalHostChannelSpec("rgb", "per_ray", "float32", trailing_shape=(3,)),),
            scalar=PRIMARY_HOST_SCALAR_CHANNEL_SPECS,
        ),
        "rgb8_native_host_v1": HostPayloadContract(
            "rgb8_native_host_v1",
            per_ray=(CanonicalHostChannelSpec("rgb8", "per_ray", "uint8", trailing_shape=(3,)),),
            scalar=PRIMARY_HOST_SCALAR_CHANNEL_SPECS,
        ),
        "full_canonical_host_v2": HostPayloadContract(
            "full_canonical_host_v2",
            per_ray=DIRECT_LIGHT_FULL_HOST_PER_RAY_CHANNEL_SPECS,
            scalar=PRIMARY_HOST_SCALAR_CHANNEL_SPECS,
        ),
    }
)

HOST_DIAGNOSTIC_OVERLAY_CONTRACTS: Mapping[str, HostPayloadContract] = MappingProxyType(
    {
        "shadow_traversal_counters_v1": HostPayloadContract(
            "shadow_traversal_counters_v1",
            scalar=SHADOW_TRAVERSAL_HOST_SCALAR_CHANNEL_SPECS,
        )
    }
)


def host_payload_contract_schema() -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    """Return the JSON-safe frozen-gate representation of every host payload."""

    return {
        contract_id: contract.schema_payload() for contract_id, contract in HOST_PAYLOAD_CONTRACTS.items()
    }


def host_diagnostic_overlay_contract_schema() -> dict[
    str, dict[str, dict[str, dict[str, object]]]
]:
    """Return the JSON-safe schemas for opt-in diagnostic delivery overlays."""

    return {
        contract_id: contract.schema_payload()
        for contract_id, contract in HOST_DIAGNOSTIC_OVERLAY_CONTRACTS.items()
    }
