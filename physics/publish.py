"""
Publish/control-plane types for frame-oriented physics export.

This module intentionally stays light-weight and CPU-only: it defines the
shared contract that both CPU and GPU execution paths can implement.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from math import inf
from threading import Condition, Lock
from typing import Callable, Generic, Literal, TypeVar

QoSMode = Literal["best_effort", "lossless"]
AccessMode = Literal["borrow", "snapshot"]
AckPoint = Literal["none", "on_borrow_complete", "on_snapshot_staged"]
ConsumerLocation = Literal["host", "device"]
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


class SnapshotHandle(Generic[T]):
    """Handle for a host/device snapshot that may still be staging."""

    def __init__(
        self,
        result: T | None = None,
        *,
        frame_id: int | None = None,
        staged: bool = True,
        is_ready: bool = True,
        future: Future[T] | None = None,
        on_staged: Callable[[T], None] | None = None,
    ) -> None:
        self._result = result
        self.frame_id = frame_id
        self._future = future
        self._on_staged = on_staged
        self._lock = Lock()
        self._staged = staged and future is None
        self._is_ready = is_ready and future is None
        if future is not None:
            self._staged = False
            self._is_ready = False
            future.add_done_callback(self._complete_future)
        elif self._staged and result is not None:
            self._notify_staged(result)

    @classmethod
    def from_future(
        cls,
        future: Future[T],
        *,
        frame_id: int | None = None,
        on_staged: Callable[[T], None] | None = None,
    ) -> "SnapshotHandle[T]":
        return cls(None, frame_id=frame_id, staged=False, is_ready=False, future=future, on_staged=on_staged)

    @property
    def staged(self) -> bool:
        self._refresh_if_done()
        return self._staged

    @property
    def is_ready(self) -> bool:
        self._refresh_if_done()
        return self._is_ready

    def result(self) -> T:
        if self._future is not None and not self._staged:
            self._complete_future(self._future)
        if not self._staged:
            raise RuntimeError("Snapshot is not staged yet")
        return self._result

    def _refresh_if_done(self) -> None:
        if self._future is not None and self._future.done() and not self._staged:
            self._complete_future(self._future)

    def _complete_future(self, future: Future[T]) -> None:
        result = future.result()
        with self._lock:
            if self._staged:
                return
            self._result = result
            self._staged = True
            self._is_ready = True
        self._notify_staged(result)

    def _notify_staged(self, result: T) -> None:
        callback = self._on_staged
        if callback is None:
            return
        self._on_staged = None
        callback(result)


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
    """Runtime state for one publish consumer.

    `max_lag_frames=None` disables stall detection and preserves unbounded
    lossless semantics. A finite value enables fail-fast protection for
    lossless device consumers. `max_lag_frames=0` is valid but strict: because
    a never-completed consumer starts at reclaim frame `-1`, it can stall as
    soon as the first pinned-slot backpressure check observes lag.
    """

    consumer_id: str
    consumer_kind: str
    qos_mode: QoSMode
    access_mode: AccessMode
    latest_seen_frame_id: int = -1
    acked_frame_id: int = -1
    enabled: bool = True
    max_lag_frames: int | None = None
    consumer_location: ConsumerLocation = "host"
    device_completed_frame_id: int = -1
    device_done_event: object | None = None

    def __post_init__(self) -> None:
        if self.max_lag_frames is not None and self.max_lag_frames < 0:
            raise ValueError("max_lag_frames must be >= 0 when set")

    @property
    def is_lossless(self) -> bool:
        return self.enabled and self.qos_mode == "lossless"

    @property
    def reclaim_frame_id(self) -> int:
        if self.consumer_location == "device":
            return self.device_completed_frame_id
        return self.acked_frame_id

    def lag_frames(self, producer_frame_id: int) -> int:
        return max(0, producer_frame_id - self.reclaim_frame_id)


class DeviceConsumerStalledError(RuntimeError):
    """Raised when a lossless device consumer exceeds its frame-lag budget."""

    def __init__(
        self,
        *,
        consumer_id: str,
        producer_frame_id: int,
        reclaim_frame_id: int,
        lag_frames: int,
        max_lag_frames: int,
        target_slot_id: int,
        target_slot_frame_id: int,
    ) -> None:
        self.consumer_id = consumer_id
        self.producer_frame_id = producer_frame_id
        self.reclaim_frame_id = reclaim_frame_id
        self.lag_frames = lag_frames
        self.max_lag_frames = max_lag_frames
        self.target_slot_id = target_slot_id
        self.target_slot_frame_id = target_slot_frame_id
        super().__init__(
            f"Device publish consumer {consumer_id!r} stalled: "
            f"lag_frames={lag_frames} exceeds max_lag_frames={max_lag_frames} "
            f"(producer_frame_id={producer_frame_id}, reclaim_frame_id={reclaim_frame_id}, "
            f"target_slot_id={target_slot_id}, target_slot_frame_id={target_slot_frame_id})"
        )


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


@dataclass(frozen=True)
class PublishedSlotStats:
    slot_id: int
    frame_id: int
    step_index: int
    sim_time: float
    state: SlotState
    invalidated: bool
    pinned_by_consumer_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConsumerPublishStats:
    consumer_id: str
    consumer_kind: str
    consumer_location: ConsumerLocation
    qos_mode: QoSMode
    access_mode: AccessMode
    enabled: bool
    latest_seen_frame_id: int
    acked_frame_id: int
    device_completed_frame_id: int
    reclaim_frame_id: int
    lag_frames: int
    max_lag_frames: int | None
    is_blocking_target_slot: bool
    is_stalled: bool


@dataclass(frozen=True)
class PublishRuntimeStats:
    ring_size: int
    next_frame_id: int
    latest_frame_id: int | None
    target_slot_id: int
    min_lossless_reclaim_frame_id: float
    blocking_consumer_ids: tuple[str, ...]
    stalled_consumer_ids: tuple[str, ...]
    slots: tuple[PublishedSlotStats, ...]
    consumers: tuple[ConsumerPublishStats, ...]
    backpressure_count: int
    skip_count: int
    block_wait_count: int
    stall_count: int
    raise_count: int
    materialized_publish_count: int
    rolling_publish_window_size: int
    rolling_publish_sample_count: int
    last_publish_host_time_s: float | None
    rolling_publish_interval_s: float | None
    rolling_publish_fps: float | None


class SlotReclaimer:
    def __init__(self, consumers: list[ConsumerState]) -> None:
        self._consumers = consumers

    @property
    def consumers(self) -> list[ConsumerState]:
        return self._consumers

    def min_lossless_acked_frame_id(self) -> float:
        lossless = [c.reclaim_frame_id for c in self._consumers if c.is_lossless]
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
                if c.is_lossless and c.reclaim_frame_id < target_slot.frame_id
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
        clock: Callable[[], float] = time.perf_counter,
        stats_window_size: int = 64,
    ) -> None:
        if not slot_buffers:
            raise ValueError("PublishedRing requires at least one slot buffer")
        if stats_window_size <= 0:
            raise ValueError("stats_window_size must be >= 1")
        self._slot_buffers = slot_buffers
        self._slot_meta = [PublishedSlotMeta(slot_id=i) for i in range(len(slot_buffers))]
        self._consumers = consumers if consumers is not None else []
        self._reclaimer = SlotReclaimer(self._consumers)
        self._policy = policy or PublishPolicy()
        self._latest_frame = None
        self._condition = Condition()
        self._clock = clock
        self._publish_host_times: deque[float] = deque(maxlen=stats_window_size)
        self._materialized_publish_count = 0
        self._backpressure_count = 0
        self._skip_count = 0
        self._block_wait_count = 0
        self._stall_count = 0
        self._raise_count = 0

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
        with self._condition:
            self._policy = policy
            self._condition.notify_all()

    def set_latest_frame(self, frame) -> None:
        self._latest_frame = frame

    def register_consumer(self, consumer: ConsumerState) -> None:
        with self._condition:
            for idx, existing in enumerate(self._consumers):
                if existing.consumer_id == consumer.consumer_id:
                    self._consumers[idx] = consumer
                    self._condition.notify_all()
                    return
            self._consumers.append(consumer)
            self._condition.notify_all()

    def unregister_consumer(self, consumer_id: str) -> None:
        with self._condition:
            self._consumers[:] = [
                consumer for consumer in self._consumers if consumer.consumer_id != consumer_id
            ]
            self._condition.notify_all()

    def acknowledge_consumer(self, consumer: ConsumerState, frame_id: int) -> None:
        with self._condition:
            consumer.acked_frame_id = max(consumer.acked_frame_id, frame_id)
            self._condition.notify_all()

    def mark_device_consumer_complete(
        self,
        consumer: ConsumerState,
        frame_id: int,
        *,
        done_event: object | None = None,
    ) -> None:
        if consumer.consumer_location != "device":
            raise ValueError("mark_device_consumer_complete requires a device consumer")
        with self._condition:
            consumer.device_completed_frame_id = max(consumer.device_completed_frame_id, frame_id)
            consumer.device_done_event = done_event
            self._condition.notify_all()

    def ring_pressure_stats(self, next_frame_id: int | None = None) -> RingPressureStats:
        target = None if next_frame_id is None else self._target_meta(next_frame_id)
        return self._reclaimer.ring_pressure_stats(target)

    def publish_stats(
        self,
        *,
        next_frame_id: int,
        latest_frame_id: int | None = None,
    ) -> PublishRuntimeStats:
        """Return a metadata-only publish runtime snapshot.

        `ConsumerPublishStats.is_stalled` is a static lag-budget observation for
        every configured consumer. It does not mean the next `acquire(...)` call
        must raise `DeviceConsumerStalledError`: acquire raises only when that
        stalled device consumer is also blocking the target slot being reused.

        Rolling FPS uses host-observed `mark_ready(...)` timestamps only. It is
        useful for producer/publish cadence monitoring, but it is not a GPU
        kernel-completion measurement and does not synchronize device work.
        """
        target = self._target_meta(next_frame_id)
        target_blockers = self._blocking_consumers(target)
        target_blocker_ids = tuple(consumer.consumer_id for consumer in target_blockers)
        consumer_stats = tuple(
            self._consumer_stats(
                consumer,
                producer_frame_id=next_frame_id,
                target_blocker_ids=target_blocker_ids,
            )
            for consumer in self._consumers
        )
        stalled_ids = tuple(stats.consumer_id for stats in consumer_stats if stats.is_stalled)
        rolling_interval_s = self._rolling_publish_interval_s()
        rolling_fps = None if rolling_interval_s is None else 1.0 / rolling_interval_s
        slot_stats = tuple(
            PublishedSlotStats(
                slot_id=meta.slot_id,
                frame_id=meta.frame_id,
                step_index=meta.step_index,
                sim_time=meta.sim_time,
                state=meta.state,
                invalidated=meta.invalidated,
                pinned_by_consumer_ids=tuple(
                    consumer.consumer_id
                    for consumer in self._consumers
                    if consumer.is_lossless and consumer.reclaim_frame_id < meta.frame_id
                ),
            )
            for meta in self._slot_meta
        )
        return PublishRuntimeStats(
            ring_size=self.ring_size,
            next_frame_id=next_frame_id,
            latest_frame_id=latest_frame_id,
            target_slot_id=target.slot_id,
            min_lossless_reclaim_frame_id=self._reclaimer.min_lossless_acked_frame_id(),
            blocking_consumer_ids=target_blocker_ids,
            stalled_consumer_ids=stalled_ids,
            slots=slot_stats,
            consumers=consumer_stats,
            backpressure_count=self._backpressure_count,
            skip_count=self._skip_count,
            block_wait_count=self._block_wait_count,
            stall_count=self._stall_count,
            raise_count=self._raise_count,
            materialized_publish_count=self._materialized_publish_count,
            rolling_publish_window_size=self._publish_host_times.maxlen,
            rolling_publish_sample_count=len(self._publish_host_times),
            last_publish_host_time_s=(self._publish_host_times[-1] if self._publish_host_times else None),
            rolling_publish_interval_s=rolling_interval_s,
            rolling_publish_fps=rolling_fps,
        )

    def acquire(
        self,
        *,
        frame_id: int,
    ) -> tuple[int, object, PublishedSlotMeta] | None:
        """Acquire the slot for ``frame_id`` or apply the ring-full policy."""

        with self._condition:
            slot_id = self._target_slot_id(frame_id)
            meta = self._slot_meta[slot_id]
            saw_backpressure = False
            while meta.state == "ready" and not self._reclaimer.reclaimable(meta):
                if not saw_backpressure:
                    self._backpressure_count += 1
                    saw_backpressure = True
                action = self._policy.on_ring_full
                blockers = self._blocking_consumers(meta)
                self._raise_if_stalled_device_consumers(
                    blockers,
                    producer_frame_id=frame_id,
                    target_slot=meta,
                )
                blocker_ids = tuple(consumer.consumer_id for consumer in blockers)
                if action == "skip":
                    self._skip_count += 1
                    return None
                if action == "block":
                    device_blockers = tuple(
                        consumer.consumer_id
                        for consumer in blockers
                        if consumer.consumer_location == "device"
                    )
                    if device_blockers:
                        raise NotImplementedError(
                            "PublishPolicy(on_ring_full='block') still requires pending "
                            "device lossless consumers to complete before slot reuse: "
                            + ", ".join(device_blockers)
                        )
                    self._block_wait_count += 1
                    self._condition.wait()
                    continue
                self._raise_count += 1
                raise RuntimeError(
                    "Published ring backpressure: slot is pinned by lossless consumer(s): "
                    + ", ".join(blocker_ids)
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
        self._materialized_publish_count += 1
        self._publish_host_times.append(self._clock())
        return meta

    def find_frame(self, frame_id: int) -> tuple[PublishedSlotMeta, object] | None:
        for meta, slot in zip(self._slot_meta, self._slot_buffers):
            if meta.state == "ready" and meta.frame_id == frame_id:
                return meta, slot
        return None

    def reset(self) -> None:
        self._latest_frame = None
        self._publish_host_times.clear()
        self._materialized_publish_count = 0
        self._backpressure_count = 0
        self._skip_count = 0
        self._block_wait_count = 0
        self._stall_count = 0
        self._raise_count = 0
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

    def _blocking_consumers(self, meta: PublishedSlotMeta) -> tuple[ConsumerState, ...]:
        return tuple(
            consumer
            for consumer in self._consumers
            if consumer.is_lossless and consumer.reclaim_frame_id < meta.frame_id
        )

    def _rolling_publish_interval_s(self) -> float | None:
        if len(self._publish_host_times) < 2:
            return None
        # n timestamps span n - 1 intervals.
        elapsed = self._publish_host_times[-1] - self._publish_host_times[0]
        if elapsed <= 0.0:
            return None
        return elapsed / (len(self._publish_host_times) - 1)

    def _consumer_stats(
        self,
        consumer: ConsumerState,
        *,
        producer_frame_id: int,
        target_blocker_ids: tuple[str, ...],
    ) -> ConsumerPublishStats:
        lag = consumer.lag_frames(producer_frame_id)
        is_stalled = (
            consumer.is_lossless
            and consumer.consumer_location == "device"
            and consumer.max_lag_frames is not None
            and lag > consumer.max_lag_frames
        )
        return ConsumerPublishStats(
            consumer_id=consumer.consumer_id,
            consumer_kind=consumer.consumer_kind,
            consumer_location=consumer.consumer_location,
            qos_mode=consumer.qos_mode,
            access_mode=consumer.access_mode,
            enabled=consumer.enabled,
            latest_seen_frame_id=consumer.latest_seen_frame_id,
            acked_frame_id=consumer.acked_frame_id,
            device_completed_frame_id=consumer.device_completed_frame_id,
            reclaim_frame_id=consumer.reclaim_frame_id,
            lag_frames=lag,
            max_lag_frames=consumer.max_lag_frames,
            is_blocking_target_slot=consumer.consumer_id in target_blocker_ids,
            is_stalled=is_stalled,
        )

    def _raise_if_stalled_device_consumers(
        self,
        blockers: tuple[ConsumerState, ...],
        *,
        producer_frame_id: int,
        target_slot: PublishedSlotMeta,
    ) -> None:
        for consumer in blockers:
            if consumer.consumer_location != "device" or consumer.max_lag_frames is None:
                continue
            lag = consumer.lag_frames(producer_frame_id)
            if lag <= consumer.max_lag_frames:
                continue
            self._stall_count += 1
            raise DeviceConsumerStalledError(
                consumer_id=consumer.consumer_id,
                producer_frame_id=producer_frame_id,
                reclaim_frame_id=consumer.reclaim_frame_id,
                lag_frames=lag,
                max_lag_frames=consumer.max_lag_frames,
                target_slot_id=target_slot.slot_id,
                target_slot_frame_id=target_slot.frame_id,
            )


__all__ = [
    "AckPoint",
    "AckPolicy",
    "AccessMode",
    "BorrowedFrameLease",
    "CpuPublishedFrame",
    "ConsumerState",
    "ConsumerLocation",
    "DetailLevel",
    "DeviceConsumerStalledError",
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
    "PublishedSlotStats",
    "PublishRuntimeStats",
    "QoSMode",
    "ConsumerPublishStats",
    "RingPressureStats",
    "SnapshotHandle",
    "SlotReclaimedError",
    "SlotReclaimer",
    "SlotState",
    "ViewPolicy",
]
