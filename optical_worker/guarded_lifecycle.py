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
    FINAL_LIFECYCLE_VERSION,
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


class _GuardedSegment:
    """Single-use context, including explicit/duplicate context protocol calls."""

    def __init__(self, controller, role):
        self._controller = controller
        self._context = controller._segment_context(role)
        self._entered = False
        self._exited = False

    def _reject(self):
        error = WorkerSetupProtocolError("guarded segment context is single-use and ordered")
        self._controller._fail_guard("reentrant", error)
        if threading.current_thread() is self._controller._owner_thread and not self._controller._terminating:
            self._controller.terminate(
                fault="setup_failure",
                detection_phase="segment_context_misuse",
                emergency_target_synchronize=None,
            )
        raise error

    def __enter__(self):
        if self._entered or self._exited:
            self._reject()
        self._entered = True
        try:
            return self._context.__enter__()
        except BaseException:
            self._exited = True
            raise

    def __exit__(self, kind, value, traceback):
        if not self._entered or self._exited:
            self._reject()
        self._exited = True
        return self._context.__exit__(kind, value, traceback)


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
        self._failure_holds = []
        self._termination_evidence = None
        self._replay = None
        self._evidence_version = LIFECYCLE_VERSION
        self._final_graph = None
        self._final_sealed_coverage = None
        self._final_barrier_ticket = None
        self._final_admission = None
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
            replay = replay_lifecycle(self._guard_client.configuration.to_payload(), self._lane_id, candidate,
                                      version=self._evidence_version)
            self._guard_events.append(event)
            self._replay = replay

    def _append(self, operation, details):
        # Inherited facts are attached at occurrence, not zipped after setup.
        super()._append(operation, details)
        self._record(operation, details)

    def _hold_failure(self, exception):
        if exception is not None and not any(item is exception for item in self._failure_holds):
            self._failure_holds.append(exception)

    def _retain_controller(self):
        # A wrong-thread or out-of-context failure must not rely on later
        # owner-thread unwinding to keep potentially dirty resources alive.
        if not any(controller is self for controller in _TERMINAL_CUSTODY):
            _TERMINAL_CUSTODY.append(self)

    def _fail_guard(self, kind, exception=None):
        with self._event_lock:
            self._hold_failure(exception)
            self._retain_controller()
            if self._replay.controller == "TERMINAL_INTENT":
                return
            if self._terminating:
                if not self._replay.terminal_failed:
                    self._record("terminal_operation_failed", {"failure_kind": kind})
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
        with self._event_lock:
            if not self._terminating and self._replay.compromised:
                error = WorkerSetupProtocolError("compromised scope forbids further normal work")
                self._fail_guard("operation", error)
                raise error
            if self._terminating or self._replay.controller != "RUNNING" or self._replay.failed:
                error = WorkerSetupProtocolError("guarded setup is failed or terminal")
                if self._terminating:
                    self._fail_guard("reentrant", error)
                raise error

    @property
    def _terminal_fault(self):
        return self._replay.terminal_failed

    def _require_setup(self):
        self._check_running()
        if not self._context_active or self._guard_ticket is None:
            raise WorkerSetupProtocolError("setup leaf has no armed segment")
        self._guard_client.require_live_ticket(self._guard_ticket, role="setup")

    @contextmanager
    def _observe_public_operation(self):
        # Catching a controller-owned API rejection inside an active operation
        # cannot turn that operation into success. Healthy pre-guard validation
        # still rejects without claiming an in-flight failure.
        try:
            yield
        except BaseException as error:
            with self._event_lock:
                if self._context_active or self._terminating:
                    self._fail_guard("operation", error)
            raise

    def segment(self, role="setup"):
        return _GuardedSegment(self, role)

    @contextmanager
    def _segment_context(self, role):
        self._check_running()
        if self._context_active or type(role) is not str or role not in {"setup", "resolve"}:
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
        # Scope declarations and their inherited notifications are one CPU-only
        # operation. A concurrent fault must not split that bookkeeping chain.
        with self._observe_public_operation(), self._event_lock:
            self._require_setup()
            if type(descriptor) is not DeviceScopeDescriptor:
                raise WorkerSetupProtocolError("setup descriptor must have exact production type")
            descriptor.validated()
            self._record("descriptor_observed", {"descriptor": descriptor.to_payload()})
            return super().begin_device_leaf(descriptor)

    def enter_device_leaf(self, permit):
        with self._observe_public_operation(), self._event_lock:
            self._require_setup()
            return super().enter_device_leaf(permit)

    def adopt_host_owner(self, owner, *, role, operation_id):
        with self._observe_public_operation():
            self._check_owner()
            # native_setup._abort must retain its original exception before
            # terminate(), even when the failing public leaf already poisoned
            # normal work. This is retention only, never renewed admission.
            with self._event_lock:
                if (
                    not self._terminating
                    and (self._replay.failed or self._replay.compromised)
                    and isinstance(owner, BaseException)
                    and role == "setup_exception_graph"
                    and operation_id == "setup_exception"
                ):
                    self._fail_guard("operation", owner)
                    return super().adopt_host_owner(owner, role=role, operation_id=operation_id)
            self._check_running()
            return super().adopt_host_owner(owner, role=role, operation_id=operation_id)

    def adopt_returned_device_owner(self, owner, *, actual_device_ordinal, role, operation_id):
        with self._observe_public_operation(), self._event_lock:
            self._require_setup()
            return super().adopt_returned_device_owner(
                owner, actual_device_ordinal=actual_device_ordinal, role=role, operation_id=operation_id
            )

    def mark_unknown_device_scope(self, *, source, reason):
        with self._observe_public_operation():
            self._check_owner()
            with self._event_lock:
                if self._terminating or self._replay.controller != "RUNNING":
                    self._check_running()
                if not _valid_identifier(source) or not _valid_identifier(reason):
                    raise WorkerSetupProtocolError("unknown scope needs valid source/reason")
                # Failure metadata only tightens custody. In particular, an
                # import may fail after another error has already closed normal
                # work; native_setup must still record unknown scope then abort.
                return super().mark_unknown_device_scope(source=source, reason=reason)

    def require_adopted(self, *owners):
        # This inherited public rejection path must observe terminal/reentrant
        # misuse too; it never gains submission or release authority.
        with self._observe_public_operation():
            self._check_running()
            return super().require_adopted(*owners)

    def seal_guardian(self):
        with self._observe_public_operation():
            self._check_running()
            if self._pending_leaf_permit is not None:
                raise WorkerSetupProtocolError("seal cannot discard an unentered setup permit")
            return super().seal_guardian()

    def _reject_closed_entry(self, message):
        self._check_owner()
        error = WorkerSetupProtocolError(message)
        with self._event_lock:
            if self._terminating or self._context_active:
                self._fail_guard("protocol", error)
        raise error

    def observe_terminal_completion(self, *, owner, transaction, event, stream, target, runtime=None):
        """Observe one native terminal under the existing resolve guard.

        Callers validate their sealed transaction before and after this wait.
        This method never commits a lease or drops any resource hold.
        """
        ordinal = target.index if self._lane_id == "cuda" else target.ordinal
        if (self.target_device_ordinal != ordinal or self.unknown_device_scope
                or self.possible_dirty_device_ordinals != (ordinal,)
                or stream.device != target or event.device != target):
            raise ValueError("terminal observation differs from its retained target scope")
        self.require_adopted(owner, transaction, stream, event)
        if self._lane_id == "warp":
            if runtime is None:
                raise TypeError("Warp observation needs its retained runtime")
            self.require_adopted(runtime)
        elif self._lane_id != "cuda" or runtime is not None:
            raise ValueError("invalid terminal observation runtime")
        with self.segment("resolve"):
            ticket = self._guard_ticket
            if self._lane_id == "cuda":
                event.synchronize()
            else:
                runtime.synchronize_event(event)
        self._guard_client.require_completed_ticket(ticket)
        return ticket

    def commit_target_barrier(self, *, _source_session=None, _graph=None, **_kwargs):
        """Seal one locally registered graph and observe its final target barrier."""
        from .native_setup import _require_final_graph
        from .protocol import canonical_json_bytes, canonical_sha256

        if _kwargs or _source_session is None or _graph is None:
            self._reject_closed_entry("final barrier requires its registered source graph")
        self._check_running()
        if (self._context_active or self._guard_client.state != "IDLE" or self._final_graph is not None
                or self.guardian.state != "CONSTRUCTING"):
            self._reject_closed_entry("final barrier is one-shot from healthy unsealed setup")
        graph = _require_final_graph(_source_session, _graph)
        if _source_session.bound_controller() is not self:
            self._reject_closed_entry("final graph belongs to another guarded controller")
        ordinal = graph.target.index if self._lane_id == "cuda" else graph.target.ordinal
        if (ordinal != self.target_device_ordinal or graph.stream.device != graph.target
                or self.unknown_device_scope or self.possible_dirty_device_ordinals != (ordinal,)):
            self._reject_closed_entry("final barrier requires the retained exact target")
        # All graph owners exist before sealing. The guard client/channel and
        # controller retain later CPU observations without adding GPU resources.
        self._final_graph = graph
        try:
            graph_sha256 = graph.sha256
            readiness_sha256 = graph.origin.readiness.sha256
            self._evidence_version = FINAL_LIFECYCLE_VERSION
            self.seal_guardian()
            self._final_sealed_coverage = canonical_json_bytes(self.guardian.coverage_payload())
            coverage_sha256 = self.guardian.coverage_sha256
            self._record("final_graph_sealed", {"graph_sha256": graph_sha256,
                "readiness_sha256": readiness_sha256, "coverage_sha256": coverage_sha256})
            with self.segment("setup"):
                ticket = self._guard_ticket
                self._final_barrier_ticket = ticket
                self._record("final_target_barrier_started", {})
                if self._lane_id == "cuda":
                    with graph.runtime.cuda.device(graph.target), graph.runtime.cuda.stream(graph.stream):
                        graph.runtime.cuda.synchronize(graph.target)
                else:
                    with graph.runtime.ScopedDevice(graph.target), graph.runtime.ScopedStream(graph.stream):
                        graph.runtime.synchronize_device(graph.target)
                self._record("final_target_barrier_returned", {})
            self._guard_client.require_completed_ticket(ticket)
            _require_final_graph(_source_session, graph)
            if (graph.sha256 != graph_sha256 or graph.origin.readiness.sha256 != readiness_sha256
                    or canonical_json_bytes(self.guardian.coverage_payload()) != self._final_sealed_coverage
                    or canonical_sha256(self.guardian.coverage_payload()) != coverage_sha256
                    or graph.stream.device != graph.target):
                raise WorkerSetupProtocolError("final graph or sealed coverage changed during its barrier")
            evidence = super().commit_target_barrier(a2_readiness_sha256=readiness_sha256,
                                                     prepared_graph_sha256=graph_sha256)
            self._final_admission = graph
            return evidence
        except BaseException as error:
            self._hold_failure(error)
            if self.termination_evidence is None and not self._terminating:
                self._fail_guard("operation", error)
                self.terminate(fault="setup_failure", detection_phase="final_target_barrier",
                               emergency_target_synchronize=None)
            raise

    def latch_measured_sample_submission(self):
        self._reject_closed_entry("measured submission remains closed in C2d1")

    def commit_clean_close(self):
        self._reject_closed_entry("guarded cleanup requires supervised terminal custody")

    def evidence(self):
        with self._event_lock:
            return GuardedLifecycleEvidence.from_payload(
                {
                    "schema_version": self._evidence_version,
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
        with self._event_lock:
            if self._terminal_fault:
                return False
            self._guard_reference = {"sequence": self._guard_client.next_guard_sequence, "role": "emergency"}
            self._record("terminal_arm_started", {"purpose": purpose})
        try:
            ticket = self._guard_client.arm("emergency")
            self._guard_client.require_live_ticket(ticket, role="emergency")
            with self._event_lock:
                self._guard_ticket = ticket
                self._record("guard_armed", {})
        except BaseException as error:
            with self._event_lock:
                self._hold_failure(error)
                self._record("terminal_arm_failed", {"failure_kind": "arm_ack"})
            self._guard_client.close()
            return False
        return True

    def _cleanup_failed(self):
        # Called with the controller lock, only after real release admission.
        self._record("cleanup_failed", {"guardian_state": self.guardian.state})
        self._record("quarantine_custody_committed", {"extent": "remaining"})
        return "cleanup_failed_exit"

    def _release(self, purpose):
        with self._event_lock:
            if self._terminal_fault:
                return self._quarantine()
            self._guard_client.require_live_ticket(self._guard_ticket, role="emergency")
            self.guardian.authorize_release(self._guardian_authority)
            self._record(
                "resource_release_authorized",
                {"reason": "cpu_cancel" if purpose == "cpu_cancel" else "emergency_sync"},
            )
            self._record("ordinary_cleanup_started", {})
        try:
            # Destructors may report another failure. Never hold the controller
            # lock here, including when they use another thread.
            self.guardian.release_whole_graph(self._guardian_authority)
        except BaseException as error:
            with self._event_lock:
                self._hold_failure(error)
                return self._cleanup_failed()
        with self._event_lock:
            if self._terminal_fault:
                return self._cleanup_failed()
            self._record("guardian_released", {})
            return "cancelled_before_submission_exit" if purpose == "cpu_cancel" else "aborted_synced_exit"

    def _begin_termination(self, fault, detection_phase):
        self._check_owner()
        with self._event_lock:
            if self._terminating:
                error = WorkerSetupProtocolError("guarded termination is single-use")
                self._fail_guard("reentrant", error)
                raise error
            if fault not in _FAULT_KINDS or not _valid_identifier(detection_phase):
                raise WorkerSetupProtocolError("invalid terminal fault/phase")
            self._hold_failure(sys.exception())
            if self._guard_client.state != "IDLE" or self._replay.failed:
                self._fail_guard("operation")
            self._record("termination_started", {"fault": fault, "detection_phase": detection_phase})
            self._terminating = True
            self._state = "POISONED"
            self._pending_leaf_permit = None
            self._retain_controller()
            self._record("terminal_custody_retained", {})
            if self.guardian.state in {"CONSTRUCTING", "SEALED"}:
                self.guardian.freeze_partial(self._guardian_authority)
                self._record("guardian_partial_frozen", {"coverage_sha256": self.guardian.coverage_sha256})

    def _synchronize(self, synchronize):
        target = self.target_device_ordinal
        with self._event_lock:
            if self._terminal_fault:
                return self._quarantine()
            self._guard_client.require_live_ticket(self._guard_ticket, role="emergency")
            self._record("emergency_sync_started", {"device_ordinal": target})
        try:
            if synchronize is None:
                raise WorkerSetupProtocolError("no exact runtime target sync selected")
            synchronize()
        except BaseException as error:
            with self._event_lock:
                self._hold_failure(error)
                self._record("emergency_sync_failed", {"device_ordinal": target})
                return self._quarantine()
        with self._event_lock:
            if self._terminal_fault:
                self._record("emergency_sync_failed", {"device_ordinal": target})
                return self._quarantine()
            self._record("emergency_sync_succeeded", {"device_ordinal": target})
        return self._release("target_sync")

    def terminate(self, *, fault, detection_phase, emergency_target_synchronize):
        with self._observe_public_operation():
            self._begin_termination(fault, detection_phase)
        with self._event_lock:
            if self._replay.compromised or self._replay.failed or self._terminal_fault:
                disposition = self._quarantine()
            else:
                disposition = None
                purpose = (
                    "cpu_cancel"
                    if not self._replay.dirty and self._replay.descriptor is None
                    else "target_sync"
                )
        if disposition is None:
            if not self._terminal_arm(purpose):
                with self._event_lock:
                    disposition = self._quarantine()
            elif purpose == "cpu_cancel":
                disposition = self._release(purpose)
            else:
                disposition = self._synchronize(emergency_target_synchronize)
        with self._event_lock:
            if self._terminal_fault and self._replay.release == "completed":
                disposition = self._cleanup_failed()
            self._record("terminal_intent", {"disposition": disposition})
            self._termination_evidence = GuardedTerminationEvidence(self.evidence())
            return self._termination_evidence
