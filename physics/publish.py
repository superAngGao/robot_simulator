"""
Publish/control-plane types for frame-oriented physics export.

This module intentionally stays light-weight and CPU-only: it defines the
shared contract that both CPU and GPU execution paths can implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Callable, Generic, Literal, TypeVar

QoSMode = Literal["best_effort", "lossless"]
AccessMode = Literal["borrow", "snapshot"]
AckPoint = Literal["none", "on_borrow_complete", "on_snapshot_staged"]
DetailLevel = Literal["low", "default", "high"]
SlotState = Literal["free", "writing", "ready"]
OnRingFull = Literal["raise", "skip", "block"]

T = TypeVar("T")


@dataclass(frozen=True)
class ViewPolicy:
    enabled: bool = False
    period_steps: int = 1
    env_selector: object | None = None
    detail_level: DetailLevel = "default"
    max_items: int | None = None

    def __post_init__(self) -> None:
        if self.period_steps <= 0:
            raise ValueError("period_steps must be >= 1")

    def should_materialize(self, frame_id: int) -> bool:
        if not self.enabled:
            return False
        return frame_id % self.period_steps == 0


@dataclass(frozen=True)
class PublishPolicy:
    publish_core_every_step: bool = True
    publish_every_n_steps: int = 1
    on_ring_full: OnRingFull = "raise"
    realtime_render: ViewPolicy = field(default_factory=ViewPolicy)
    render_backed_sensing: ViewPolicy = field(default_factory=ViewPolicy)
    debug_export: ViewPolicy = field(default_factory=ViewPolicy)
    publish_rigid_block: bool = False
    publish_telemetry_block: bool = True

    def __post_init__(self) -> None:
        if self.publish_every_n_steps <= 0:
            raise ValueError("publish_every_n_steps must be >= 1")


@dataclass(frozen=True)
class PublishPlan:
    """Per-step publish decision.

    `frame_id` is interpreted as the monotonic physics-step timeline, not the
    count of materialized published slots. When `do_publish_core` is false, the
    engine still advances its internal frame counter for that physics step, and
    the next materialized frame will therefore have a larger `frame_id`.
    """

    do_publish_core: bool

    do_realtime_render: bool
    realtime_env_ids: object | None
    realtime_variant: DetailLevel | None

    do_render_backed_sensing: bool
    render_backed_sensing_env_ids: object | None
    render_backed_sensing_variant: DetailLevel | None

    do_debug_export: bool
    debug_env_ids: object | None
    debug_host_copy: bool

    do_rigid_block_write: bool = False
    do_telemetry_block_write: bool = True

    @classmethod
    def from_policy(cls, frame_id: int, policy: PublishPolicy) -> "PublishPlan":
        realtime = policy.realtime_render.should_materialize(frame_id)
        render_backed_sensing = policy.render_backed_sensing.should_materialize(frame_id)
        debug = policy.debug_export.should_materialize(frame_id)
        return cls(
            do_publish_core=policy.publish_core_every_step and frame_id % policy.publish_every_n_steps == 0,
            do_realtime_render=realtime,
            realtime_env_ids=policy.realtime_render.env_selector if realtime else None,
            realtime_variant=policy.realtime_render.detail_level if realtime else None,
            do_render_backed_sensing=render_backed_sensing,
            render_backed_sensing_env_ids=(
                policy.render_backed_sensing.env_selector if render_backed_sensing else None
            ),
            render_backed_sensing_variant=(
                policy.render_backed_sensing.detail_level if render_backed_sensing else None
            ),
            do_debug_export=debug,
            debug_env_ids=policy.debug_export.env_selector if debug else None,
            debug_host_copy=debug,
            do_rigid_block_write=policy.publish_rigid_block,
            do_telemetry_block_write=policy.publish_telemetry_block,
        )


@dataclass
class PublishedFrameCore:
    frame_id: int
    sim_time: float
    step_index: int
    env_mask: object | None

    state_ref: object
    kinematics_ref: object
    contact_count_ref: object | None
    contacts_ref: object | None
    telemetry_ref: object | None

    ready_flag: object | None = None
    completion_event: object | None = None


@dataclass
class CpuPublishedFrame:
    frame_id: int
    sim_time: float
    step_index: int
    env_mask: object | None

    q: object
    qdot: object
    X_world: object
    v_bodies: object

    contact_count: object | None
    contacts: object | None
    telemetry: object | None
    contact_mask: object | None = None


@dataclass
class GpuPublishedFrame:
    slot_id: int
    frame_id: int
    sim_time: float
    step_index: int
    env_mask_wp: object | None

    q_wp: object
    qdot_wp: object
    x_world_R_wp: object
    x_world_r_wp: object
    v_bodies_wp: object

    contact_count_wp: object | None
    contact_cache_ref: object | None
    telemetry_ref: object | None
    contact_mask_wp: object | None = None

    ready_event: object | None = None
    slot_meta: PublishedSlotMeta | None = None

    def __getattribute__(self, name: str):
        guarded = {
            "q_wp",
            "qdot_wp",
            "x_world_R_wp",
            "x_world_r_wp",
            "v_bodies_wp",
            "contact_count_wp",
            "contact_mask_wp",
            "contact_cache_ref",
            "telemetry_ref",
            "ready_event",
        }
        if name in guarded:
            slot_meta = object.__getattribute__(self, "slot_meta")
            frame_id = object.__getattribute__(self, "frame_id")
            if slot_meta is not None and (slot_meta.invalidated or slot_meta.frame_id != frame_id):
                raise SlotReclaimedError(
                    f"GpuPublishedFrame slot {slot_meta.slot_id} for frame {frame_id} has been reclaimed"
                )
        return object.__getattribute__(self, name)


class LeaseExpiredError(RuntimeError):
    """Raised when code attempts to use a borrowed frame after lease expiry."""


class SlotReclaimedError(RuntimeError):
    """Raised when code accesses a GPU published frame after slot reclaim."""


class BorrowedFrameLease(Generic[T]):
    """Context-managed ephemeral lease for borrowed frames."""

    def __init__(self, frame: T | None, on_release: Callable[[T], None] | None = None) -> None:
        self._frame = frame
        self._active = frame is not None
        self._on_release = on_release

    def __enter__(self) -> "BorrowedFrameLease[T]":
        self._require_active()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.invalidate()

    @property
    def active(self) -> bool:
        return self._active

    def invalidate(self) -> None:
        if self._active and self._frame is not None and self._on_release is not None:
            self._on_release(self._frame)
        self._frame = None
        self._active = False

    def get(self) -> T:
        self._require_active()
        return self._frame

    def __getattr__(self, name: str):
        return getattr(self.get(), name)

    def __getitem__(self, key):
        return self.get()[key]

    def _require_active(self) -> None:
        if not self._active or self._frame is None:
            raise LeaseExpiredError("Borrowed frame lease is no longer active")


@dataclass
class SnapshotHandle(Generic[T]):
    """Phase-1 synchronous snapshot handle with a future-compatible surface."""

    _result: T
    frame_id: int | None = None
    staged: bool = True
    is_ready: bool = True

    def result(self) -> T:
        return self._result


@dataclass(frozen=True)
class HostSnapshotSpec:
    fields: frozenset[str]
    env_ids: object | None = None


@dataclass(frozen=True)
class DeviceSnapshotSpec:
    fields: frozenset[str]
    env_ids: object | None = None


@dataclass
class ConsumerState:
    consumer_id: str
    consumer_kind: str
    qos_mode: QoSMode
    access_mode: AccessMode
    latest_seen_frame_id: int = -1
    acked_frame_id: int = -1
    enabled: bool = True
    max_lag_frames: int | None = None

    @property
    def is_lossless(self) -> bool:
        return self.enabled and self.qos_mode == "lossless"


@dataclass(frozen=True)
class AckPolicy:
    consumer_id: str
    qos_mode: QoSMode
    access_mode: AccessMode
    ack_point: AckPoint

    @classmethod
    def default_for(cls, consumer: ConsumerState) -> "AckPolicy":
        if consumer.qos_mode == "best_effort":
            return cls(
                consumer_id=consumer.consumer_id,
                qos_mode=consumer.qos_mode,
                access_mode=consumer.access_mode,
                ack_point="none",
            )
        ack_point: AckPoint = (
            "on_borrow_complete" if consumer.access_mode == "borrow" else "on_snapshot_staged"
        )
        return cls(
            consumer_id=consumer.consumer_id,
            qos_mode=consumer.qos_mode,
            access_mode=consumer.access_mode,
            ack_point=ack_point,
        )


@dataclass
class PublishedSlotMeta:
    slot_id: int
    frame_id: int = -1
    step_index: int = -1
    sim_time: float = 0.0
    state: SlotState = "free"
    publish_event: object | None = None
    host_export_queued: bool = False
    invalidated: bool = False


@dataclass(frozen=True)
class RingPressureStats:
    min_lossless_acked_frame_id: float
    enabled_lossless_consumers: tuple[str, ...]
    blocking_consumer_ids: tuple[str, ...]


class SlotReclaimer:
    def __init__(self, consumers: list[ConsumerState]) -> None:
        self._consumers = consumers

    @property
    def consumers(self) -> list[ConsumerState]:
        return self._consumers

    def min_lossless_acked_frame_id(self) -> float:
        lossless = [c.acked_frame_id for c in self._consumers if c.is_lossless]
        if not lossless:
            return inf
        return min(lossless)

    def reclaimable(self, slot: PublishedSlotMeta) -> bool:
        return slot.frame_id <= self.min_lossless_acked_frame_id()

    def ring_pressure_stats(self, target_slot: PublishedSlotMeta | None = None) -> RingPressureStats:
        enabled_lossless = tuple(c.consumer_id for c in self._consumers if c.is_lossless)
        min_ack = self.min_lossless_acked_frame_id()
        if target_slot is None or target_slot.frame_id <= min_ack:
            blockers: tuple[str, ...] = ()
        else:
            blockers = tuple(
                c.consumer_id
                for c in self._consumers
                if c.is_lossless and c.acked_frame_id < target_slot.frame_id
            )
        return RingPressureStats(
            min_lossless_acked_frame_id=min_ack,
            enabled_lossless_consumers=enabled_lossless,
            blocking_consumer_ids=blockers,
        )


class PublishedRing:
    """Control-plane ring for published frame slots.

    The ring owns slot metadata, consumer state, reclaim decisions, and the
    latest-frame pointer. It only holds references to slot payload buffers; the
    execution backend remains responsible for allocating and writing those
    buffers.
    """

    def __init__(
        self,
        *,
        slot_buffers: list[object],
        consumers: list[ConsumerState] | None = None,
        policy: PublishPolicy | None = None,
    ) -> None:
        if not slot_buffers:
            raise ValueError("PublishedRing requires at least one slot buffer")
        self._slot_buffers = slot_buffers
        self._slot_meta = [PublishedSlotMeta(slot_id=i) for i in range(len(slot_buffers))]
        self._consumers = consumers if consumers is not None else []
        self._reclaimer = SlotReclaimer(self._consumers)
        self._policy = policy or PublishPolicy()
        self._latest_frame = None

    @property
    def ring_size(self) -> int:
        return len(self._slot_buffers)

    @property
    def policy(self) -> PublishPolicy:
        return self._policy

    @property
    def consumers(self) -> list[ConsumerState]:
        return self._consumers

    @property
    def slot_meta(self) -> list[PublishedSlotMeta]:
        return self._slot_meta

    @property
    def slot_buffers(self) -> list[object]:
        return self._slot_buffers

    @property
    def latest_frame(self):
        return self._latest_frame

    def set_policy(self, policy: PublishPolicy) -> None:
        self._policy = policy

    def set_latest_frame(self, frame) -> None:
        self._latest_frame = frame

    def register_consumer(self, consumer: ConsumerState) -> None:
        for idx, existing in enumerate(self._consumers):
            if existing.consumer_id == consumer.consumer_id:
                self._consumers[idx] = consumer
                return
        self._consumers.append(consumer)

    def unregister_consumer(self, consumer_id: str) -> None:
        self._consumers[:] = [consumer for consumer in self._consumers if consumer.consumer_id != consumer_id]

    def ring_pressure_stats(self, next_frame_id: int | None = None) -> RingPressureStats:
        target = None if next_frame_id is None else self._target_meta(next_frame_id)
        return self._reclaimer.ring_pressure_stats(target)

    def acquire(
        self,
        *,
        frame_id: int,
    ) -> tuple[int, object, PublishedSlotMeta] | None:
        """Acquire the slot for ``frame_id`` or apply the ring-full policy."""

        slot_id = self._target_slot_id(frame_id)
        meta = self._slot_meta[slot_id]
        if meta.state == "ready" and not self._reclaimer.reclaimable(meta):
            action = self._policy.on_ring_full
            blockers = self._reclaimer.ring_pressure_stats(meta).blocking_consumer_ids
            if action == "skip":
                return None
            if action == "block":
                raise NotImplementedError(
                    "PublishPolicy(on_ring_full='block') is not implemented in phase-1; "
                    "use 'raise' or 'skip' until async staging/reclaim lands."
                )
            raise RuntimeError(
                "Published ring backpressure: slot is pinned by lossless consumer(s): " + ", ".join(blockers)
            )

        meta.invalidated = True
        meta.state = "writing"
        return slot_id, self._slot_buffers[slot_id], meta

    def mark_ready(
        self,
        *,
        slot_id: int,
        frame_id: int,
        step_index: int,
        sim_time: float,
        publish_event: object | None,
    ) -> PublishedSlotMeta:
        meta = self._slot_meta[slot_id]
        meta.frame_id = frame_id
        meta.step_index = step_index
        meta.sim_time = sim_time
        meta.state = "ready"
        meta.publish_event = publish_event
        meta.host_export_queued = False
        meta.invalidated = False
        return meta

    def find_frame(self, frame_id: int) -> tuple[PublishedSlotMeta, object] | None:
        for meta, slot in zip(self._slot_meta, self._slot_buffers):
            if meta.state == "ready" and meta.frame_id == frame_id:
                return meta, slot
        return None

    def reset(self) -> None:
        self._latest_frame = None
        for meta in self._slot_meta:
            meta.frame_id = -1
            meta.step_index = -1
            meta.sim_time = 0.0
            meta.state = "free"
            meta.publish_event = None
            meta.host_export_queued = False
            meta.invalidated = False

    def _target_slot_id(self, frame_id: int) -> int:
        return frame_id % self.ring_size

    def _target_meta(self, frame_id: int) -> PublishedSlotMeta:
        return self._slot_meta[self._target_slot_id(frame_id)]


__all__ = [
    "AckPoint",
    "AckPolicy",
    "AccessMode",
    "BorrowedFrameLease",
    "CpuPublishedFrame",
    "ConsumerState",
    "DetailLevel",
    "DeviceSnapshotSpec",
    "GpuPublishedFrame",
    "HostSnapshotSpec",
    "LeaseExpiredError",
    "OnRingFull",
    "PublishPlan",
    "PublishPolicy",
    "PublishedFrameCore",
    "PublishedRing",
    "PublishedSlotMeta",
    "QoSMode",
    "RingPressureStats",
    "SnapshotHandle",
    "SlotReclaimedError",
    "SlotReclaimer",
    "SlotState",
    "ViewPolicy",
]
