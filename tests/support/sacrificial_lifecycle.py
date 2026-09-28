"""CPU-only one-shot render-validation lifetime authority (not product API).

This composes the production whole-graph guardian and device-scope descriptor.
It deliberately does not change C4c0's seal-before-barrier setup protocol. The
C4c1 disposable driver instead seals after a supervisor-acknowledged barrier.
Callbacks are in-process fixed driver operations, never IPC-supplied callables.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum

from optical_worker.lifecycle import (
    HOST_OWNER,
    DeviceScopeDescriptor,
    NativeWorkerResourceGuardian,
)


class LifetimeError(RuntimeError):
    pass


class State(str, Enum):
    SETUP = "SETUP"
    SETUP_QUIESCENT = "SETUP_QUIESCENT"
    SEALED = "SEALED"
    SUBMITTING = "SUBMITTING"
    VALIDATED = "VALIDATED"
    INSPECTED = "INSPECTED"
    POISONED = "POISONED"
    SYNCED = "SYNCED"
    QUARANTINED = "QUARANTINED"
    RELEASED = "RELEASED"


@dataclass(frozen=True, slots=True)
class LeafPermit:
    sequence: int
    descriptor: DeviceScopeDescriptor


class SacrificialRenderGuardian:
    """Single-threaded trusted-child authority; no arbitrary-code capability claim.

    A leaf permit is consumed by identity, not serializable evidence. Caller
    returns every leaf owner through ``finish_setup_leaf`` before any next leaf.
    Failure of an owner-producing leaf before transfer poisons ownership forever.
    Only the driver can supply the supervised wait; production cannot import this.
    """

    def __init__(self, *, session_id: str, target_ordinal: int):
        if type(target_ordinal) is not int or not 0 <= target_ordinal <= 1023:
            raise LifetimeError("target ordinal must be CPU metadata")
        self._guardian = NativeWorkerResourceGuardian(session_identity=session_id)
        self._authority = self._guardian._control_authority()
        self._target = target_ordinal
        self._state = State.SETUP
        self._dirty: set[int] = set()
        self._unknown_device = False
        self._unknown_owner = False
        self._active: LeafPermit | None = None
        self._sequence = 0
        self._attempted = 0
        self._returned = 0
        self._command_count: int | None = None
        self._schedule: tuple[tuple[str, object], ...] = ()
        self._schedule_index = 0
        self._auxiliary_attempted = 0
        self._auxiliary_returned = 0
        self._launch_active = False
        self._blocking_entered: str | None = None
        self._setup_quiescent = False
        self._device_quiescent = False
        self._trace: list[dict] = []
        self._record("guardian_created")

    @property
    def state(self) -> State:
        return self._state

    @property
    def resource_count(self) -> int:
        return self._guardian.resource_count

    def snapshot(self) -> dict:
        # Diagnostic only; snapshots cannot supply permission to an operation.
        return dict(
            state=self._state.value,
            possible_dirty_device_ordinals=sorted(self._dirty),
            unknown_device_scope=self._unknown_device,
            ownership_scope_unknown=self._unknown_owner,
            possible_submission_latched=self._attempted > 0,
            attempted_launch_count=self._attempted,
            confirmed_launch_return_count=self._returned,
            attempted_auxiliary_count=self._auxiliary_attempted,
            confirmed_auxiliary_return_count=self._auxiliary_returned,
            scheduled_leaf_count=len(self._schedule),
            completed_leaf_count=self._schedule_index,
            setup_quiescence_proven=self._setup_quiescent,
            device_quiescence_proven=self._device_quiescent,
            blocking_role_entered=self._blocking_entered,
            guardian=self._guardian.coverage_payload(),
            trace=copy.deepcopy(self._trace),
        )

    def _record(self, operation: str, **facts) -> None:
        self._trace.append(dict(sequence=len(self._trace), operation=operation, **facts))

    def _require(self, *states: State) -> None:
        if self._state not in states:
            raise LifetimeError(f"operation forbidden in {self._state.value}")

    def _exact_scope(self) -> bool:
        return self._dirty == {self._target} and not self._unknown_device and not self._unknown_owner

    def adopt_host(self, owner, *, role: str) -> None:
        self._require(State.SETUP)
        if self._active is not None:
            raise LifetimeError("active leaf must transfer its owners first")
        self._adopt(owner, role, HOST_OWNER, "host_setup")

    def _adopt(self, owner, role, scope, operation):
        if not self._guardian.contains_exact_owner(owner):
            self._guardian.adopt(owner, role=role, device_scope=scope, operation_id=operation)
            self._record("owner_adopted", role=role, device_scope=scope)

    def begin_setup_leaf(self, descriptor: DeviceScopeDescriptor) -> LeafPermit:
        self._require(State.SETUP)
        if self._active is not None:
            raise LifetimeError("previous leaf owners have not transferred")
        if type(descriptor) is not DeviceScopeDescriptor:
            raise LifetimeError("exact CPU device descriptor required")
        descriptor.validated()
        self._dirty.update(descriptor.device_ordinals)
        self._unknown_device |= not descriptor.device_ordinals
        self._device_quiescent = False
        self._record("setup_leaf_declared", scope=descriptor.to_payload())
        if self._unknown_device or self._dirty != {self._target}:
            self._state = State.POISONED
            raise LifetimeError("foreign or unknown device scope is quarantine-only")
        permit = LeafPermit(self._sequence, descriptor)
        self._sequence += 1
        self._active = permit
        return permit

    def finish_setup_leaf(self, permit: LeafPermit, *, owners: tuple[object, ...]) -> None:
        self._require(State.SETUP)
        if permit is not self._active or type(owners) is not tuple:
            raise LifetimeError("wrong or stale setup permit/owner tuple")
        if permit.descriptor.unreturned_resource_possible and not owners:
            self.fail_setup_leaf(permit)
            raise LifetimeError("owner-producing leaf returned no owner")
        # The entire reachable result graph is in custody before inspecting
        # its members. An invalid middle member must not strand later owners.
        self._adopt(owners, "owner_transfer_capsule", HOST_OWNER, permit.descriptor.operation_id)
        try:
            for owner in owners:
                self._adopt(owner, "setup_owner", self._target, permit.descriptor.operation_id)
        except BaseException:
            self._unknown_owner = True
            self._state = State.POISONED
            self._active = None
            self._record("owner_transfer_failed", leaf_sequence=permit.sequence)
            raise
        self._active = None
        self._record("setup_leaf_completed", leaf_sequence=permit.sequence)

    def fail_setup_leaf(self, permit: LeafPermit, *, partial_owners: tuple[object, ...] = ()) -> None:
        self._require(State.SETUP)
        if permit is not self._active or type(partial_owners) is not tuple:
            raise LifetimeError("wrong setup failure permit")
        self._unknown_owner |= permit.descriptor.unreturned_resource_possible
        self._state = State.POISONED
        self._adopt(partial_owners, "partial_owner_capsule", HOST_OWNER, permit.descriptor.operation_id)
        self._active = None
        self._record("setup_leaf_failed", leaf_sequence=permit.sequence)
        try:
            for owner in partial_owners:
                self._adopt(owner, "partial_owner", self._target, permit.descriptor.operation_id)
        except BaseException:
            self._unknown_owner = True
            self._record("owner_transfer_failed", leaf_sequence=permit.sequence)
            raise

    def _wait(self, role, handshake, wait):
        # ACK is checked by the real test-only supervised channel before return.
        # An arming failure never reaches the blocking callable.
        try:
            ticket = handshake.arm(role)
        except BaseException:
            self.quarantine("watchdog_arm_failed")
            raise
        self._record("watchdog_acked", role=role)
        self._blocking_entered = role
        self._record("blocking_entered", role=role)
        try:
            wait()
            if role == "S":
                handshake.complete(ticket)
        except BaseException:
            self.quarantine("blocking_operation_failed")
            raise
        self._device_quiescent = True
        self._record("blocking_completed", role=role)

    def setup_barrier(self, handshake, wait) -> None:
        self._require(State.SETUP)
        if self._active is not None or not self._exact_scope():
            raise LifetimeError("setup barrier requires complete target-only ownership")
        self._wait("S", handshake, wait)
        self._setup_quiescent = True
        self._blocking_entered = None
        self._state = State.SETUP_QUIESCENT
        self._record("setup_quiescence_proven")

    def seal(self, *, command_count: int, repetitions: int = 1, auxiliary_plan: tuple[str, ...] = ()) -> None:
        self._require(State.SETUP_QUIESCENT)
        if type(command_count) is not int or command_count not in (2, 4, 5):
            raise LifetimeError("fixed render command count must be 2, 4 or 5")
        if type(repetitions) is not int or repetitions not in (1, 2) or type(auxiliary_plan) is not tuple:
            raise LifetimeError("fixed one/two-pass schedule required")
        if repetitions == 1 and auxiliary_plan:
            raise LifetimeError("one-pass validation has no auxiliary leaves")
        if repetitions == 2:
            half = len(auxiliary_plan) // 2
            names = tuple(
                value.removeprefix("snapshot:") for value in auxiliary_plan[:half] if type(value) is str
            )
            if (
                not half
                or len(auxiliary_plan) != 2 * half
                or len(names) != half
                or len(set(names)) != half
                or any(not name.isidentifier() for name in names)
                or auxiliary_plan
                != tuple(f"snapshot:{name}" for name in names) + tuple(f"sentinel:{name}" for name in names)
            ):
                raise LifetimeError("two-pass auxiliary schedule must snapshot then fill each exact output")
        schedule = tuple(("render", index) for index in range(command_count))
        if repetitions == 2:
            schedule += tuple(("auxiliary", name) for name in auxiliary_plan)
            schedule += tuple(("render", index) for index in range(command_count, 2 * command_count))
        self._guardian.seal(self._authority)
        self._command_count = command_count * repetitions
        self._schedule = schedule
        self._state = State.SEALED
        self._record(
            "guardian_sealed", command_count=command_count, repetitions=repetitions, schedule=schedule
        )

    def before_launch(self, position: int) -> None:
        self._require(State.SEALED, State.SUBMITTING)
        if (
            type(position) is not int
            or self._launch_active
            or position != self._attempted
            or self._attempted >= self._command_count
            or self._schedule_index >= len(self._schedule)
            or self._schedule[self._schedule_index] != ("render", position)
        ):
            raise LifetimeError("recorded launch order is invalid")
        self._state = State.SUBMITTING
        self._attempted += 1
        self._launch_active = True
        self._device_quiescent = False
        self._record("possible_submission_latched", position=position)

    def launch_returned(self, position: int) -> None:
        self._require(State.SUBMITTING)
        if (
            type(position) is not int
            or not self._launch_active
            or position != self._returned
            or self._schedule[self._schedule_index] != ("render", position)
        ):
            raise LifetimeError("launch return is not the active leaf")
        self._returned += 1
        self._launch_active = False
        self._schedule_index += 1
        self._record("launch_returned", position=position)

    def before_auxiliary(self, name: str) -> None:
        self._require(State.SUBMITTING)
        if (
            type(name) is not str
            or self._launch_active
            or self._schedule_index >= len(self._schedule)
            or self._schedule[self._schedule_index] != ("auxiliary", name)
        ):
            raise LifetimeError("auxiliary leaf is not the next frozen operation")
        self._auxiliary_attempted += 1
        self._launch_active = True
        self._device_quiescent = False
        self._record("auxiliary_submission_latched", name=name)

    def auxiliary_returned(self, name: str) -> None:
        self._require(State.SUBMITTING)
        if (
            type(name) is not str
            or not self._launch_active
            or self._schedule[self._schedule_index] != ("auxiliary", name)
        ):
            raise LifetimeError("auxiliary return is not the active frozen leaf")
        self._auxiliary_returned += 1
        self._launch_active = False
        self._schedule_index += 1
        self._record("auxiliary_returned", name=name)

    def validation_drain(self, handshake, wait) -> None:
        self._require(State.SUBMITTING)
        if (
            self._launch_active
            or self._returned != self._command_count
            or self._schedule_index != len(self._schedule)
        ):
            raise LifetimeError("validation drain requires all launch returns")
        self._wait("V", handshake, wait)
        self._state = State.VALIDATED
        self._record("validation_quiescence_proven")

    def inspection_completed(self) -> None:
        self._require(State.VALIDATED)
        self._state = State.INSPECTED
        self._record("result_inspected")

    def poison(self, reason: str) -> None:
        self._require(State.SETUP, State.SETUP_QUIESCENT, State.SEALED, State.SUBMITTING)
        if self._active is not None:
            self._unknown_owner |= self._active.descriptor.unreturned_resource_possible
        self._state = State.POISONED
        self._record("poisoned", reason=reason)

    def emergency_drain(self, handshake, wait) -> None:
        self._require(State.POISONED)
        if not self._exact_scope() or self._blocking_entered is not None:
            self.quarantine("unsafe_second_sync_or_unknown_scope")
            raise LifetimeError("emergency sync cannot prove complete graph safety")
        self._wait("E", handshake, wait)
        self._state = State.SYNCED
        self._record("emergency_quiescence_proven")

    def quarantine(self, reason: str) -> None:
        if self._state is State.QUARANTINED:
            return
        self._require(*[state for state in State if state not in (State.RELEASED, State.QUARANTINED)])
        self._guardian.freeze_partial(self._authority)
        self._guardian.transfer_to_quarantine(self._authority)
        self._state = State.QUARANTINED
        self._record("quarantined", reason=reason)

    def release(self) -> None:
        self._require(State.INSPECTED, State.SYNCED, State.SEALED)
        if not self._device_quiescent or not self._exact_scope():
            raise LifetimeError("release requires full graph device quiescence")
        if self._state is State.SEALED and self._attempted:
            raise LifetimeError("submitted graph cannot be cancelled")
        if self._guardian.state == "CONSTRUCTING":
            self._guardian.freeze_partial(self._authority)
        self._guardian.authorize_release(self._authority)
        self._record("resource_release_authorized")
        self._guardian.release_whole_graph(self._authority)
        self._state = State.RELEASED
        self._record("resources_released")
