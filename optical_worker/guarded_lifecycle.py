"""C2d1 guarded setup custody. Not yet a product worker preparation path.

The fixed trusted driver must consume every terminal disposition with hard
process exit. No finalizer, record parser or CPU test grants GPU admission.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager

from .guarded_evidence import (
    LIFECYCLE_VERSION,
    GuardedLifecycleEvidence,
    GuardedTerminationEvidence,
    replay_lifecycle,
)
from .lifecycle import (
    _FAULT_KINDS,
    DeviceScopeDescriptor,
    NativeWorkerSetupLifecycle,
    WorkerSetupProtocolError,
    _valid_identifier,
)
from .wait_protocol import GuardConfiguration, GuardProtocolError, GuardWaitClient


def _forbidden_legacy_notifier(_ordinal):
    raise WorkerSetupProtocolError("guarded custody cannot fall back to the C0 notifier")


# Append-only process-terminal custody. Never reset/unregister in this process.
_TERMINAL_CUSTODY = []


class NativeGuardedSetupLifecycle(NativeWorkerSetupLifecycle):
    """One CPU-created controller owning the actual guard and inherited guardian."""

    def __init__(self, *, lane_id, channel, configuration):
        if type(configuration) is not GuardConfiguration:
            raise GuardProtocolError("guarded setup requires an exact configuration")
        self._guard_client = GuardWaitClient(channel, configuration)
        self._guard_channel = channel
        self._owner_thread = threading.current_thread()
        self._event_lock = threading.RLock()
        self._guard_events = []
        self._guard_reference = None
        self._guard_ticket = None
        self._context_active = False
        self._terminating = False
        self._terminal_fault = False
        self._failure_holds = []
        self._termination_evidence = None
        self._replay = None
        super().__init__(
            lane_id=lane_id,
            target_device_ordinal=configuration.device_ordinal,
            session_identity=configuration.session,
            emergency_sync_notifier=_forbidden_legacy_notifier,
        )

    def _record(self, operation, details):
        with self._event_lock:
            epoch = (
                1
                if operation == "descriptor_observed"
                else (0 if self._replay is None else self._replay.epoch)
            )
            event = {
                "index": len(self._guard_events),
                "epoch": epoch,
                "operation": operation,
                "guard": None if self._guard_reference is None else dict(self._guard_reference),
                "details": dict(details),
            }
            candidate = [*self._guard_events, event]
            replay = replay_lifecycle(self._guard_client.configuration.to_payload(), self._lane_id, candidate)
            self._guard_events.append(event)
            self._replay = replay

    def _append(self, operation, details):
        # Inherited facts are attached at occurrence, not zipped after setup.
        super()._append(operation, details)
        self._record(operation, details)

    def _hold_failure(self, exception):
        if exception is not None and not any(item is exception for item in self._failure_holds):
            self._failure_holds.append(exception)

    def _fail_guard(self, kind, exception=None):
        with self._event_lock:
            self._hold_failure(exception)
            if self._replay.controller == "TERMINAL_INTENT":
                return
            if self._terminating:
                self._terminal_fault = True
                return
            if not self._replay.failed:
                self._record("guard_failed", {"failure_kind": kind})
            if threading.current_thread() is self._owner_thread:
                self._guard_client.fail()

    def _check_owner(self):
        if threading.current_thread() is not self._owner_thread:
            error = WorkerSetupProtocolError("guarded setup belongs to another thread")
            self._fail_guard("thread", error)
            raise error

    def _check_running(self):
        self._check_owner()
        if self._terminating or self._replay.controller != "RUNNING" or self._replay.failed:
            if self._terminating and self._replay.controller != "TERMINAL_INTENT":
                self._terminal_fault = True
            raise WorkerSetupProtocolError("guarded setup is failed or terminal")

    def _require_setup(self):
        self._check_running()
        if not self._context_active or self._guard_ticket is None:
            raise WorkerSetupProtocolError("setup leaf has no armed segment")
        self._guard_client.require_live_ticket(self._guard_ticket, role="setup")

    @contextmanager
    def segment(self, role="setup"):
        self._check_running()
        if self._context_active or role not in {"setup", "resolve"}:
            error = WorkerSetupProtocolError("nested or invalid guarded segment")
            self._fail_guard("reentrant" if self._context_active else "protocol", error)
            raise error
        self._context_active = True
        try:
            sequence = self._guard_client.next_guard_sequence
            self._guard_reference = {"sequence": sequence, "role": role}
            self._record("normal_arm_started", {})
            try:
                ticket = self._guard_client.arm(role)
                self._guard_client.require_live_ticket(ticket, role=role)
                self._guard_ticket = ticket
                self._record("guard_armed", {})
            except BaseException as error:
                self._fail_guard("arm_ack", error)
                raise
            yield
            self._check_running()
            self._guard_client.require_live_ticket(ticket, role=role)
            self._record("normal_body_returned", {})
            self._record("normal_complete_started", {})
            try:
                self._guard_client.complete(ticket)
                self._guard_client.require_completed_ticket(ticket)
                self._record("normal_disarmed", {})
                self._guard_reference, self._guard_ticket = None, None
            except BaseException as error:
                self._fail_guard("complete_ack", error)
                raise
        except BaseException as error:
            self._hold_failure(error)
            if self._termination_evidence is None and not self._terminating:
                self._fail_guard("operation", error)
                self.terminate(
                    fault="setup_failure",
                    detection_phase="guarded_segment",
                    emergency_target_synchronize=None,
                )
            raise
        finally:
            self._context_active = False

    def begin_device_leaf(self, descriptor):
        self._require_setup()
        if type(descriptor) is not DeviceScopeDescriptor:
            raise WorkerSetupProtocolError("setup descriptor must have exact production type")
        descriptor.validated()
        self._record("descriptor_observed", {"descriptor": descriptor.to_payload()})
        return super().begin_device_leaf(descriptor)

    def enter_device_leaf(self, permit):
        self._require_setup()
        return super().enter_device_leaf(permit)

    def adopt_host_owner(self, owner, *, role, operation_id):
        self._check_running()
        return super().adopt_host_owner(owner, role=role, operation_id=operation_id)

    def adopt_returned_device_owner(self, owner, *, actual_device_ordinal, role, operation_id):
        self._require_setup()
        return super().adopt_returned_device_owner(
            owner, actual_device_ordinal=actual_device_ordinal, role=role, operation_id=operation_id
        )

    def mark_unknown_device_scope(self, *, source, reason):
        self._check_running()
        return super().mark_unknown_device_scope(source=source, reason=reason)

    def seal_guardian(self):
        self._check_running()
        if self._pending_leaf_permit is not None:
            raise WorkerSetupProtocolError("seal cannot discard an unentered setup permit")
        return super().seal_guardian()

    def commit_target_barrier(self, **_kwargs):
        raise WorkerSetupProtocolError("final graph/barrier admission is not implemented in C2d1")

    def latch_measured_sample_submission(self):
        raise WorkerSetupProtocolError("measured submission remains closed in C2d1")

    def commit_clean_close(self):
        raise WorkerSetupProtocolError("guarded cleanup requires supervised terminal custody")

    def evidence(self):
        with self._event_lock:
            return GuardedLifecycleEvidence.from_payload(
                {
                    "schema_version": LIFECYCLE_VERSION,
                    "configuration": self._guard_client.configuration.to_payload(),
                    "lane_id": self._lane_id,
                    "record_kind": "terminal" if self._replay.disposition else "prefix",
                    "transition_trace": list(self._guard_events),
                    "guardian_coverage": self.guardian.coverage_payload(),
                    "summary": self._replay.summary(),
                }
            )

    @property
    def termination_evidence(self):
        return self._termination_evidence

    def _quarantine(self):
        self.guardian.transfer_to_quarantine(self._guardian_authority)
        self._record("guardian_quarantine_transferred", {"coverage_sha256": self.guardian.coverage_sha256})
        self._record("quarantine_custody_committed", {"extent": "full"})
        return (
            "multi_or_unknown_device_quarantine_exit"
            if self._replay.compromised
            else "quarantined_process_exit"
        )

    def _terminal_arm(self, purpose):
        self._guard_reference = {"sequence": self._guard_client.next_guard_sequence, "role": "emergency"}
        self._record("terminal_arm_started", {"purpose": purpose})
        try:
            ticket = self._guard_client.arm("emergency")
            self._guard_client.require_live_ticket(ticket, role="emergency")
            self._guard_ticket = ticket
            self._record("guard_armed", {})
        except BaseException as error:
            self._hold_failure(error)
            self._record("terminal_arm_failed", {"failure_kind": "arm_ack"})
            self._guard_client.close()
            return False
        return True

    def _release(self, purpose):
        try:
            self._guard_client.require_live_ticket(self._guard_ticket, role="emergency")
            self.guardian.authorize_release(self._guardian_authority)
            self._record(
                "resource_release_authorized",
                {
                    "reason": "cpu_cancel" if purpose == "cpu_cancel" else "emergency_sync",
                },
            )
            if self._terminal_fault:
                raise WorkerSetupProtocolError("reentrant terminal operation invalidated cleanup")
            self._record("ordinary_cleanup_started", {})
            self.guardian.release_whole_graph(self._guardian_authority)
            if self._terminal_fault:
                raise WorkerSetupProtocolError("reentrant cleanup invalidated success")
            self._record("guardian_released", {})
        except BaseException as error:
            self._hold_failure(error)
            if self.guardian.state not in {"RELEASE_AUTHORIZED", "RELEASED"}:
                # No ordinary release began. Keep the entire graph terminal.
                raise
            self._record("cleanup_failed", {"guardian_state": self.guardian.state})
            self._record("quarantine_custody_committed", {"extent": "remaining"})
            return "cleanup_failed_exit"
        return "cancelled_before_submission_exit" if purpose == "cpu_cancel" else "aborted_synced_exit"

    def terminate(self, *, fault, detection_phase, emergency_target_synchronize):
        self._check_owner()
        if self._terminating:
            if self._termination_evidence is None:
                self._terminal_fault = True
            raise WorkerSetupProtocolError("guarded termination is single-use")
        if fault not in _FAULT_KINDS or not _valid_identifier(detection_phase):
            raise WorkerSetupProtocolError("invalid terminal fault/phase")
        self._hold_failure(sys.exception())
        if self._guard_client.state != "IDLE" or self._replay.failed:
            self._fail_guard("operation")
        self._record("termination_started", {"fault": fault, "detection_phase": detection_phase})
        self._terminating = True
        self._state = "POISONED"
        self._pending_leaf_permit = None
        _TERMINAL_CUSTODY.append(self)
        self._record("terminal_custody_retained", {})
        if self.guardian.state in {"CONSTRUCTING", "SEALED"}:
            self.guardian.freeze_partial(self._guardian_authority)
            self._record("guardian_partial_frozen", {"coverage_sha256": self.guardian.coverage_sha256})
        if self._replay.compromised or self._replay.failed:
            disposition = self._quarantine()
        else:
            purpose = (
                "cpu_cancel" if not self._replay.dirty and self._replay.descriptor is None else "target_sync"
            )
            if not self._terminal_arm(purpose):
                disposition = self._quarantine()
            elif purpose == "cpu_cancel":
                disposition = self._release(purpose)
            else:
                target = self.target_device_ordinal
                self._guard_client.require_live_ticket(self._guard_ticket, role="emergency")
                self._record("emergency_sync_started", {"device_ordinal": target})
                try:
                    if emergency_target_synchronize is None:
                        raise WorkerSetupProtocolError("no exact runtime target sync selected")
                    emergency_target_synchronize()
                    if self._terminal_fault:
                        raise WorkerSetupProtocolError("reentrant terminal sync invalidated success")
                except BaseException as error:
                    self._hold_failure(error)
                    self._record("emergency_sync_failed", {"device_ordinal": target})
                    disposition = self._quarantine()
                else:
                    self._record("emergency_sync_succeeded", {"device_ordinal": target})
                    disposition = self._release(purpose)
        self._record("terminal_intent", {"disposition": disposition})
        self._termination_evidence = GuardedTerminationEvidence(self.evidence())
        return self._termination_evidence
