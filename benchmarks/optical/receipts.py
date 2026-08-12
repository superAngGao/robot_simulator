"""Lane-owned acceleration and shipping publication receipt contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from .eligibility import GPU_ENGINE_PUBLISHED_FRAME_INPUT

ACCELERATION_RECEIPT_SCHEMA_VERSION = "q54.p12.4.acceleration-receipt.v1"
SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION = "q54.p12.4.shipping-publication-receipt.v1"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AccelerationReceipt:
    """Actual scene-update and acceleration operations observed by a lane."""

    acceleration_schema_version: str
    acceleration_id: str
    workload_class: str
    topology_origin: str
    topology_lifetime: str
    topology_validation_sha256: str
    timed_scene_update_operation: str
    timed_acceleration_operation: str
    operation_classification: str
    scene_update_count: int
    refit_count: int
    rebuild_count: int
    primitive_count: int
    node_count: int
    leaf_count: int
    scene_update_kernel_launch_count: int
    acceleration_kernel_launch_count: int
    device_allocation_count: int
    host_materialization_count: int
    host_upload_bytes: int
    device_to_host_bytes: int
    global_synchronize_count: int
    bridge_creation_count: int
    completion_event_recorded: bool
    instrumentation_trace_sha256: str

    @classmethod
    def from_payload(cls, payload: object) -> "AccelerationReceipt":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("acceleration receipt fields are invalid")
        if payload.get("acceleration_schema_version") != ACCELERATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("acceleration receipt schema version is invalid")
        for name in (
            "acceleration_id",
            "workload_class",
            "topology_origin",
            "topology_lifetime",
            "timed_scene_update_operation",
            "timed_acceleration_operation",
        ):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise ValueError(f"acceleration receipt {name} must be non-empty")
        if payload.get("operation_classification") not in {"none", "refit", "rebuild"}:
            raise ValueError("acceleration receipt operation_classification is invalid")
        for name in (
            "scene_update_count",
            "refit_count",
            "rebuild_count",
            "primitive_count",
            "node_count",
            "leaf_count",
            "scene_update_kernel_launch_count",
            "acceleration_kernel_launch_count",
            "device_allocation_count",
            "host_materialization_count",
            "host_upload_bytes",
            "device_to_host_bytes",
            "global_synchronize_count",
            "bridge_creation_count",
        ):
            value = payload.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"acceleration receipt {name} must be a non-negative integer")
        if not isinstance(payload.get("completion_event_recorded"), bool):
            raise ValueError("acceleration receipt completion_event_recorded must be boolean")
        for name in ("topology_validation_sha256", "instrumentation_trace_sha256"):
            if not _sha256_text(payload.get(name)):
                raise ValueError(f"acceleration receipt {name} must be a lowercase SHA-256 digest")
        return cls(**{name: payload[name] for name in expected})

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


@dataclass(frozen=True)
class ShippingPublicationReceipt:
    """Actual physics producer/consumer handoff and reclaim evidence for S."""

    publication_schema_version: str
    input_contract: str
    physics_run_id: str
    physics_step_index: int
    physics_frame_id: int
    physics_slot_id: int
    physics_device_ordinal: int
    physics_step_count: int
    publish_event_token: str
    publish_event_recorded: bool
    producer_publish_completion_recorded: bool
    consumer_binding_id: str
    consumer_stream_token: str
    borrow_count: int
    consumer_stream_wait_enqueued: bool
    zero_copy_transform_binding: bool
    last_physics_buffer_reader_operation: str
    consumer_done_event_token: str
    consumer_completion_recorded: bool
    slot_logically_reclaimable: bool
    producer_reuse_wait_enqueued_when_reused: bool
    done_event_retained_until_reuse_dependency: bool
    physical_reclaim_evidence: str
    final_payload_drain_complete: bool
    final_reclaim_complete: bool
    replay_source_count: int
    host_poll_count: int
    host_synchronize_count: int
    publication_trace_sha256: str

    @classmethod
    def from_payload(cls, payload: object) -> "ShippingPublicationReceipt":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("shipping publication receipt fields are invalid")
        if payload.get("publication_schema_version") != SHIPPING_PUBLICATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("shipping publication receipt schema version is invalid")
        if payload.get("input_contract") != GPU_ENGINE_PUBLISHED_FRAME_INPUT:
            raise ValueError("shipping publication receipt input contract is invalid")
        for name in (
            "physics_run_id",
            "publish_event_token",
            "consumer_binding_id",
            "consumer_stream_token",
            "last_physics_buffer_reader_operation",
            "consumer_done_event_token",
        ):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise ValueError(f"shipping publication receipt {name} must be non-empty")
        for name in (
            "physics_step_index",
            "physics_frame_id",
            "physics_slot_id",
            "physics_device_ordinal",
            "physics_step_count",
            "borrow_count",
            "replay_source_count",
            "host_poll_count",
            "host_synchronize_count",
        ):
            value = payload.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"shipping publication receipt {name} must be a non-negative integer")
        for name in (
            "publish_event_recorded",
            "producer_publish_completion_recorded",
            "consumer_stream_wait_enqueued",
            "zero_copy_transform_binding",
            "consumer_completion_recorded",
            "slot_logically_reclaimable",
            "producer_reuse_wait_enqueued_when_reused",
            "done_event_retained_until_reuse_dependency",
            "final_payload_drain_complete",
            "final_reclaim_complete",
        ):
            if not isinstance(payload.get(name), bool):
                raise ValueError(f"shipping publication receipt {name} must be boolean")
        physical = payload.get("physical_reclaim_evidence")
        if physical not in {"producer_wait_enqueued_before_reuse", "done_event_retained_no_reuse"}:
            raise ValueError("shipping publication physical_reclaim_evidence is invalid")
        if physical == "producer_wait_enqueued_before_reuse" and not payload.get(
            "producer_reuse_wait_enqueued_when_reused"
        ):
            raise ValueError("shipping publication producer reuse wait evidence is inconsistent")
        if not payload.get("done_event_retained_until_reuse_dependency"):
            raise ValueError("shipping publication done event retention must be proven")
        if not _sha256_text(payload.get("publication_trace_sha256")):
            raise ValueError("shipping publication trace digest is invalid")
        return cls(**{name: payload[name] for name in expected})

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_payload())


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
