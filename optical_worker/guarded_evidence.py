"""CPU-only C2d1 lifecycle replay; evidence never grants execution authority."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .lifecycle import (
    _FAULT_KINDS,
    GUARDIAN_COVERAGE_SCHEMA_VERSION,
    HOST_OWNER,
    UNKNOWN_DEVICE_OWNER,
    DeviceScopeDescriptor,
    _valid_identifier,
)
from .protocol import canonical_json_bytes, canonical_sha256
from .wait_protocol import (
    MAX_SEQUENCE,
    ROLES,
    GuardConfiguration,
    GuardProtocolError,
    bounded_integer,
    exact_fields,
    next_sequence,
)
from .wait_supervision import GuardSupervisorEvidence

LIFECYCLE_VERSION = "q54.p12.4.setup-lifecycle.v3"
FINAL_LIFECYCLE_VERSION = "q54.p12.4.setup-lifecycle.v4"
WORKER_LIFECYCLE_VERSION = "q54.p12.4.setup-lifecycle.v5"
WORKER_TERMINATION_VERSION = "q54.p12.4.transport-termination.v4"
POST_RECLAIM_IDLE_VERSION = "q54.p12.4.post-reclaim-idle.v1"
TERMINATION_VERSION = "q54.p12.4.transport-termination.v3"
VALIDATION_VERSION = "q54.p12.4.guarded-custody-validation.v1"
FAILURE_KINDS = frozenset(
    {
        "operation",
        "arm_ack",
        "complete_ack",
        "protocol",
        "thread",
        "reentrant",
        "sync",
        "cleanup",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuardProtocolError(message)


def _enum(value: object, choices) -> str:
    _require(type(value) is str and value in choices, "unknown guarded lifecycle enum")
    return value


def _identifier(value: object) -> str:
    _require(type(value) is str and _valid_identifier(value), "invalid lifecycle identifier")
    return value


def _digest(value: object) -> str:
    _require(
        type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "invalid lifecycle SHA-256",
    )
    return value


def _reference(value: object) -> dict | None:
    if value is None:
        return None
    value = exact_fields(value, {"sequence", "role"})
    bounded_integer(value["sequence"], MAX_SEQUENCE)
    _enum(value["role"], ROLES)
    return dict(value)


def _descriptor(value: object) -> DeviceScopeDescriptor:
    value = exact_fields(
        value,
        {
            "schema_version",
            "operation_id",
            "leaf_kind",
            "device_ordinals",
            "scope_source",
            "runtime_identity",
            "unreturned_resource_possible",
        },
    )
    _require(type(value["device_ordinals"]) is list, "descriptor ordinals must be an array")
    _require(type(value["unreturned_resource_possible"]) is bool, "descriptor flag must be bool")
    descriptor = DeviceScopeDescriptor(**dict(value, device_ordinals=tuple(value["device_ordinals"])))
    descriptor.validated()
    return descriptor


class _Replay:
    """One reducer for both live descriptive snapshots and parsed records."""

    def __init__(self, configuration: GuardConfiguration, lane: str, *, version=LIFECYCLE_VERSION):
        self.version = _enum(version, {LIFECYCLE_VERSION, FINAL_LIFECYCLE_VERSION, WORKER_LIFECYCLE_VERSION})
        self.worker_manifest = None
        self.worker_samples = None
        self.active_sample = None
        self.next_sample = 0
        self.cancel_only = False
        self.sample_submission_latched = False
        self.sample_resolve_sequence = None
        self.idle_proof = None
        self.normal_close = False
        self.final_graph = None
        self.final_guard = None
        self.final_barrier_returned = False
        self.final_quiescence = None
        self.last_completed_guard = None
        self.config = configuration
        self.lane = _enum(lane, {"warp", "cuda"})
        self.index = 0
        self.epoch = 0
        self.controller = "RUNNING"
        self.guardian = "CONSTRUCTING"
        self.guard_state = "IDLE"
        self.guard = None
        self.next_sequence = 0
        self.purpose = None
        self.failed = False
        self.terminal_failed = False
        self.dirty = set()
        self.unknown = False
        self.notified_dirty = set()
        self.notified_unknown = False
        self.notifications = []
        self.descriptor = None
        self.descriptor_started = False
        self.permit = None
        self.body_returned = False
        self.records = []
        self.operation_count = 0
        self.entered = 0
        self.work_counts = {}
        self.complete_count = 0
        self.terminal_arms = 0
        self.sync_count = 0
        self.sync_outcome = "not_attempted"
        self.custody = "none"
        self.release = "not_authorized"
        self.disposition = None

    @property
    def compromised(self):
        return self.unknown or any(d != self.config.device_ordinal for d in self.dirty)

    @property
    def final_quiescence_current(self):
        # Final setup is historical after submission or any later normal wait.
        return (self.version in {FINAL_LIFECYCLE_VERSION, WORKER_LIFECYCLE_VERSION}
                and self.final_quiescence is not None
                and self.final_guard == self.last_completed_guard and self.guard_state == "IDLE"
                and self.next_sequence == self.final_guard["sequence"] + 1
                and not self.sample_submission_latched
                and not self.failed and not self.terminal_failed and not self.compromised)

    @property
    def may_cancel_without_sync(self):
        no_setup = self.descriptor is None and not self.dirty and not self.unknown
        return no_setup or self.final_quiescence_current

    def coverage(self, *, guardian_state=None):
        counts = {}
        for record in self.records:
            key = str(record["device_scope"])
            counts[key] = counts.get(key, 0) + 1
        return {
            "schema_version": GUARDIAN_COVERAGE_SCHEMA_VERSION,
            "session_identity": self.config.session,
            "state": self.guardian if guardian_state is None else guardian_state,
            "records": self.records,
            "per_device_resource_counts": dict(sorted(counts.items())),
        }

    def summary(self):
        result = {
            "controller_state": self.controller,
            "guardian_state": self.guardian,
            "guard_state": self.guard_state,
            "guard_ref": self.guard,
            "next_sequence": self.next_sequence,
            "terminal_purpose": self.purpose,
            "guarded_segment_failed": self.failed,
            "terminal_operation_failed": self.terminal_failed,
            "possible_dirty_device_ordinals": sorted(self.dirty),
            "unknown_device_scope": self.unknown,
            "setup_operation_count": self.operation_count,
            "entered_leaf_count": self.entered,
            "setup_work_counts": dict(sorted(self.work_counts.items())),
            "normal_complete_count": self.complete_count,
            "terminal_arm_attempt_count": self.terminal_arms,
            "emergency_sync_attempt_count": self.sync_count,
            "emergency_sync_outcome": self.sync_outcome,
            "custody": self.custody,
            "release_status": self.release,
            "disposition": self.disposition,
        }
        if self.version in {FINAL_LIFECYCLE_VERSION, WORKER_LIFECYCLE_VERSION}:
            result["final_setup"] = {
                "graph": self.final_graph,
                "barrier_guard": self.final_guard,
                "barrier_returned": self.final_barrier_returned,
                "quiescence_sha256": self.final_quiescence,
            }
        if self.version == WORKER_LIFECYCLE_VERSION:
            result["worker"] = {
                "manifest_sha256": self.worker_manifest,
                "samples": self.worker_samples,
                "active_sample": self.active_sample,
                "next_sample": self.next_sample,
                "cancel_only": self.cancel_only,
                "historical_submission_latched": self.sample_submission_latched,
                "post_reclaim_idle_sha256": self.idle_proof,
                "normal_close_ack_committed": self.normal_close,
            }
        return result

    def _active_setup(self):
        _require(
            self.controller == "RUNNING"
            and self.guard_state == "ARMED"
            and self.guard is not None
            and self.guard["role"] == "setup"
            and not self.failed
            and not self.body_returned,
            "setup fact requires an armed normal setup segment",
        )

    def _declare_scope(self, ordinals, unknown, *, source, reason):
        # Validate the complete declaration before changing any live field.
        for ordinal in ordinals:
            bounded_integer(ordinal, 1023)
        for ordinal in ordinals:
            if ordinal not in self.dirty:
                self.dirty.add(ordinal)
                self.notifications.append(("dirty", ordinal, source, reason))
        if unknown and not self.unknown:
            self.unknown = True
            self.notifications.append(
                (
                    "unknown",
                    None,
                    source,
                    "unreturned_resource_possible"
                    if reason == "pre_leaf_descriptor"
                    else "owner_device_unknown",
                )
            )

    def apply(self, event, *, preceding_evidence_sha256=None):
        event = exact_fields(event, {"index", "epoch", "operation", "guard", "details"})
        _require(bounded_integer(event["index"]) == self.index, "noncontiguous lifecycle trace")
        operation = event["operation"]
        _require(
            type(operation) is str and self.controller != "TERMINAL_INTENT", "event after terminal intent"
        )
        guard = _reference(event["guard"])
        details = event["details"]
        _require(type(details) is dict, "event details must be an object")
        expected_epoch = 1 if operation == "descriptor_observed" else self.epoch
        _require(bounded_integer(event["epoch"], 1) == expected_epoch, "setup epoch drift")
        if operation in {"normal_arm_started", "terminal_arm_started"}:
            _require(
                guard is not None and guard["sequence"] == self.next_sequence,
                "guard attempt has wrong sequence",
            )
        else:
            _require(guard == self.guard, "event is attached to the wrong guard")
        if self.index == 0:
            exact_fields(details, {"lane_id", "target_device_ordinal"})
            bounded_integer(details["target_device_ordinal"], 1023)
            _require(
                operation == "guardian_created"
                and details
                == {
                    "lane_id": self.lane,
                    "target_device_ordinal": self.config.device_ordinal,
                }
                and guard is None
                and expected_epoch == 0,
                "trace must begin at the exact guardian",
            )
            self.index += 1
            return
        if operation == "normal_arm_started":
            exact_fields(details, set())
            _require(
                self.controller == "RUNNING"
                and self.guard_state == "IDLE"
                and not self.failed
                and not self.compromised
                and guard["role"] in {"setup", "resolve"},
                "normal ARM predecessor",
            )
            self.guard, self.guard_state = guard, "ARM_PENDING"
            self.body_returned = False
        elif operation == "guard_armed":
            exact_fields(details, set())
            _require(self.guard_state == "ARM_PENDING", "ACK without pending ARM")
            following_sequence = next_sequence(guard["sequence"])
            self.guard_state, self.next_sequence = "ARMED", following_sequence
        elif operation == "normal_body_returned":
            exact_fields(details, set())
            _require(
                self.controller == "RUNNING"
                and self.guard_state == "ARMED"
                and guard["role"] != "emergency"
                and not self.body_returned
                and not self.failed
                and not self.compromised
                and self.permit is None
                and not self.notifications
                and (self.descriptor is None or self.descriptor_started),
                "invalid body success",
            )
            self.body_returned = True
        elif operation == "normal_complete_started":
            exact_fields(details, set())
            _require(
                self.controller == "RUNNING"
                and self.guard_state == "ARMED"
                and self.body_returned
                and not self.failed
                and not self.compromised
                and guard["role"] != "emergency",
                "invalid normal COMPLETE",
            )
            self.guard_state = "COMPLETE_PENDING"
        elif operation == "normal_disarmed":
            exact_fields(details, set())
            _require(
                self.guard_state == "COMPLETE_PENDING" and not self.failed and not self.compromised,
                "unexpected DISARMED",
            )
            self.last_completed_guard = dict(self.guard)
            self.guard_state, self.guard = "IDLE", None
            self.complete_count += 1
        elif operation == "guard_failed":
            exact_fields(details, {"failure_kind"})
            _enum(details["failure_kind"], FAILURE_KINDS)
            _require(
                self.controller == "RUNNING"
                and not self.failed
                and (guard is None or guard["role"] != "emergency"),
                "duplicate/late normal failure",
            )
            self.guard_state, self.failed = "FAILED", True
        elif operation == "descriptor_observed":
            exact_fields(details, {"descriptor"})
            self._active_setup()
            _require(
                self.guardian == "CONSTRUCTING"
                and self.permit is None
                and not self.notifications
                and (self.descriptor is None or self.descriptor_started),
                "descriptor overlap",
            )
            self.descriptor = _descriptor(details["descriptor"])
            self.descriptor_started = False
            self._declare_scope(
                self.descriptor.device_ordinals,
                self.descriptor.unreturned_resource_possible,
                source=self.descriptor.scope_source,
                reason="pre_leaf_descriptor",
            )
        elif operation == "setup_operation_started":
            exact_fields(details, {"descriptor_sha256", "operation_id", "leaf_kind"})
            self._active_setup()
            _require(
                self.descriptor is not None
                and not self.descriptor_started
                and self.guardian == "CONSTRUCTING"
                and self.permit is None,
                "operation lacks its preceding descriptor",
            )
            _require(
                details
                == {
                    "descriptor_sha256": self.descriptor.sha256,
                    "operation_id": self.descriptor.operation_id,
                    "leaf_kind": self.descriptor.leaf_kind,
                },
                "descriptor identity drift",
            )
            self.descriptor_started = True
            self.operation_count += 1
            self.work_counts[self.descriptor.leaf_kind] = (
                self.work_counts.get(self.descriptor.leaf_kind, 0) + 1
            )
            self.permit = (self.descriptor.sha256, self.operation_count)
        elif operation == "dirty_ordinal_added":
            exact_fields(details, {"device_ordinal", "source", "reason"})
            self._active_setup()
            ordinal = bounded_integer(details["device_ordinal"], 1023)
            expected = ("dirty", ordinal, _identifier(details["source"]), _identifier(details["reason"]))
            _require(
                self.notifications
                and self.notifications[0] == expected
                and ordinal not in self.notified_dirty,
                "unexpected/repeated dirty notification",
            )
            self.notifications.pop(0)
            self.notified_dirty.add(ordinal)
        elif operation == "unknown_scope_set":
            exact_fields(details, {"source", "reason"})
            _identifier(details["source"])
            _identifier(details["reason"])
            _require(
                self.controller == "RUNNING" and not self.notified_unknown, "repeated/late unknown scope"
            )
            if self.notifications:
                _require(
                    self.notifications[0] == ("unknown", None, details["source"], details["reason"]),
                    "unknown scope notification order",
                )
                self.notifications.pop(0)
            self.unknown, self.notified_unknown = True, True
        elif operation == "low_level_leaf_entered":
            exact_fields(details, {"descriptor_sha256", "operation_sequence"})
            self._active_setup()
            bounded_integer(details["operation_sequence"], minimum=1)
            _require(
                self.permit == (_digest(details["descriptor_sha256"]), details["operation_sequence"])
                and not self.compromised
                and not self.notifications,
                "invalid/unsafe leaf permit",
            )
            self.permit = None
            self.entered += 1
        elif operation == "owner_adopted":
            exact_fields(details, {"sequence", "role", "owner_type", "device_scope", "operation_id"})
            _require(
                self.controller == "RUNNING"
                and self.guardian == "CONSTRUCTING"
                and bounded_integer(details["sequence"]) == len(self.records),
                "owner adoption predecessor",
            )
            _identifier(details["role"])
            _identifier(details["operation_id"])
            _require(
                type(details["owner_type"]) is str and bool(details["owner_type"]), "invalid owner type label"
            )
            scope = details["device_scope"]
            if scope != HOST_OWNER:
                self._active_setup()
                _require(not self.notifications, "owner adoption interrupts scope notifications")
                if scope == UNKNOWN_DEVICE_OWNER:
                    self._declare_scope((), True, source="returned_owner", reason="returned_owner_device")
                else:
                    bounded_integer(scope, 1023)
                    self._declare_scope(
                        (scope,), False, source="returned_owner", reason="returned_owner_device"
                    )
            self.records.append(dict(details))
        elif operation in {"guardian_sealed", "guardian_partial_frozen", "guardian_quarantine_transferred"}:
            exact_fields(details, {"coverage_sha256"})
            if operation == "guardian_sealed":
                _require(
                    self.controller == "RUNNING"
                    and self.guardian == "CONSTRUCTING"
                    and self.permit is None
                    and not self.compromised
                    and not self.notifications,
                    "invalid guardian seal",
                )
                guardian_state = "SEALED"
            elif operation == "guardian_partial_frozen":
                _require(
                    self.controller == "TERMINATING" and self.guardian in {"CONSTRUCTING", "SEALED"},
                    "invalid guardian partial freeze",
                )
                guardian_state = "PARTIAL_FROZEN"
            else:
                _require(
                    self.controller == "TERMINATING"
                    and self.custody == "retained"
                    and self.guardian in {"PARTIAL_FROZEN", "SEALED"}
                    and self.release == "not_authorized",
                    "invalid guardian quarantine transfer",
                )
                guardian_state = "TRANSFERRED_TO_QUARANTINE"
            _require(
                _digest(details["coverage_sha256"]) == canonical_sha256(
                    self.coverage(guardian_state=guardian_state)),
                "guardian coverage does not replay",
            )
            self.guardian = guardian_state
        elif operation == "final_graph_sealed":
            exact_fields(details, {"graph_sha256", "readiness_sha256", "coverage_sha256"})
            _require(self.version in {FINAL_LIFECYCLE_VERSION, WORKER_LIFECYCLE_VERSION}
                     and self.final_graph is None
                     and self.controller == "RUNNING" and self.guard_state == "IDLE"
                     and self.guardian == "SEALED" and not self.failed and not self.compromised,
                     "invalid final graph seal")
            for value in details.values():
                _digest(value)
            _require(details["coverage_sha256"] == canonical_sha256(self.coverage()),
                     "final graph coverage differs from the immutable seal")
            self.final_graph = dict(details)
        elif operation == "final_target_barrier_started":
            exact_fields(details, set())
            self._active_setup()
            _require(self.version in {FINAL_LIFECYCLE_VERSION, WORKER_LIFECYCLE_VERSION}
                     and self.final_graph is not None
                     and self.final_guard is None and self.guardian == "SEALED"
                     and not self.compromised and self.dirty == {self.config.device_ordinal},
                     "final barrier requires one sealed exact-target graph")
            self.final_guard = dict(guard)
        elif operation == "final_target_barrier_returned":
            exact_fields(details, set())
            self._active_setup()
            _require(self.version in {FINAL_LIFECYCLE_VERSION, WORKER_LIFECYCLE_VERSION}
                     and self.final_guard == guard
                     and not self.final_barrier_returned and self.guardian == "SEALED",
                     "final barrier return has no exact started guard")
            self.final_barrier_returned = True
        elif operation == "setup_barrier_succeeded":
            exact_fields(details, {"evidence_sha256"})
            _require(self.version in {FINAL_LIFECYCLE_VERSION, WORKER_LIFECYCLE_VERSION}
                     and self.final_barrier_returned
                     and self.final_quiescence is None and self.guard_state == "IDLE"
                     and self.guardian == "SEALED" and self.controller == "RUNNING"
                     and not self.failed and not self.compromised
                     and self.last_completed_guard == self.final_guard,
                     "final quiescence requires its exact DISARMED barrier")
            self.final_quiescence = _digest(details["evidence_sha256"])
        elif operation == "worker_bound":
            exact_fields(details, {"runtime_manifest_sha256"})
            _require(self.version == WORKER_LIFECYCLE_VERSION and self.worker_manifest is None
                     and self.controller == "RUNNING" and self.guard_state == "IDLE"
                     and self.guardian == "CONSTRUCTING" and not self.failed and not self.dirty,
                     "worker binding must precede native work")
            self.worker_manifest = _digest(details["runtime_manifest_sha256"])
        elif operation == "worker_prepared":
            exact_fields(details, {"sample_count", "graph_sha256", "token_registry_sha256"})
            _require(self.worker_manifest is not None and self.worker_samples is None
                     and self.final_quiescence_current
                     and details["graph_sha256"] == self.final_graph["graph_sha256"],
                     "worker preparation requires the admitted final graph")
            count = bounded_integer(details["sample_count"], 1_000_000, minimum=1)
            _digest(details["token_registry_sha256"])
            self.worker_samples = ["PREPARED"] * count
        elif operation in {"sample_acquired", "sample_submission_latched", "sample_completion_recorded",
                           "sample_resolved", "sample_reclaimed", "sample_cancelled"}:
            exact_fields(details, {"sample_index", "resolve_sequence"}
                         if operation == "sample_resolved" else {"sample_index"})
            _require(self.version == WORKER_LIFECYCLE_VERSION and self.worker_samples is not None
                     and self.controller == "RUNNING" and self.guard_state == "IDLE"
                     and not self.failed and not self.compromised and self.idle_proof is None,
                     "sample transition requires a healthy prepared worker")
            index = bounded_integer(details["sample_index"], len(self.worker_samples) - 1)
            state = self.worker_samples[index]
            if operation == "sample_acquired":
                _require(state == "PREPARED" and not self.cancel_only and self.active_sample is None
                         and index == self.next_sample, "sample acquire is not next FIFO")
                self.active_sample = index
                self.worker_samples[index] = "ACQUIRED"
            elif operation == "sample_cancelled":
                _require(state == "PREPARED", "only unacquired samples may cancel")
                self.cancel_only = True
                self.worker_samples[index] = "CANCELLED"
            else:
                _require(self.active_sample == index, "sample transition differs from active generation")
                if operation == "sample_submission_latched":
                    _require(state == "ACQUIRED", "duplicate or unacquired submission latch")
                    self.worker_samples[index] = "SUBMISSION_POSSIBLE"
                    self.sample_submission_latched = True
                elif operation == "sample_completion_recorded":
                    _require(state == "SUBMISSION_POSSIBLE", "completion without possible submission")
                    self.worker_samples[index] = "COMPLETION_RECORDED"
                    self.sample_resolve_sequence = self.next_sequence
                    self.next_sample += 1
                elif operation == "sample_resolved":
                    bounded_integer(details["resolve_sequence"], MAX_SEQUENCE)
                    _require(state == "COMPLETION_RECORDED" and self.last_completed_guard is not None
                             and self.last_completed_guard == {
                                 "sequence": self.sample_resolve_sequence, "role": "resolve"}
                             and details["resolve_sequence"] == self.sample_resolve_sequence,
                             "resolution requires an acknowledged terminal wait")
                    self.worker_samples[index] = "RESOLVED_HELD"
                else:
                    _require(state == "RESOLVED_HELD", "reclaim before resolution")
                    self.worker_samples[index] = "RECLAIMED"
                    self.active_sample = None
        elif operation == "post_reclaim_idle":
            exact_fields(details, {"evidence"})
            proof = PostReclaimIdleEvidence.from_payload(details["evidence"]).to_payload()
            _require(preceding_evidence_sha256 is not None
                     and proof["guard_evidence_sha256"] == preceding_evidence_sha256,
                     "idle proof does not bind its actual preceding guarded trace")
            _require(self.version == WORKER_LIFECYCLE_VERSION and self.worker_samples is not None
                     and self.active_sample is None and self.controller == "RUNNING"
                     and self.guard_state == "IDLE" and not self.failed and not self.compromised
                     and self.idle_proof is None
                     and all(state in {"RECLAIMED", "CANCELLED"} for state in self.worker_samples)
                     and proof["session_identity"] == self.config.session and proof["lane"] == self.lane
                     and proof["target"] == self.config.device_ordinal
                     and proof["setup_quiescence_sha256"] == self.final_quiescence
                     and proof["prepared_graph_sha256"] == self.final_graph["graph_sha256"]
                     and proof["submitted_count"] == self.worker_samples.count("RECLAIMED")
                     and proof["cancelled_count"] == self.worker_samples.count("CANCELLED")
                     and proof["historical_submission_latched"] == self.sample_submission_latched,
                     "idle proof differs from actual replayed samples")
            self.idle_proof = canonical_sha256(proof)
        elif operation == "normal_close_ack_committed":
            exact_fields(details, {"evidence_sha256"})
            _require(self.version == WORKER_LIFECYCLE_VERSION and self.idle_proof is not None
                     and details["evidence_sha256"] == self.idle_proof
                     and self.controller == "RUNNING" and self.guard_state == "IDLE"
                     and not self.failed and not self.compromised, "close has no idle proof")
            self.normal_close = True
            self.controller, self.custody = "TERMINATING", "retained"
        elif operation == "normal_close_release_authorized":
            exact_fields(details, set())
            _require(self.normal_close and self.controller == "TERMINATING"
                     and self.guard_state == "IDLE" and self.guardian == "SEALED"
                     and self.release == "not_authorized" and not self.terminal_failed,
                     "normal cleanup must follow the written close ACK")
            self.guardian, self.release = "RELEASE_AUTHORIZED", "authorized"
        elif operation == "normal_close_completed":
            exact_fields(details, set())
            _require(self.normal_close and self.controller == "TERMINATING"
                     and self.guard_state == "IDLE" and self.release == "completed"
                     and self.guardian == "RELEASED" and not self.terminal_failed,
                     "normal close did not complete cleanup")
            self.disposition, self.controller = "clean_reclaimed_exit", "TERMINAL_INTENT"
        elif operation == "termination_started":
            exact_fields(details, {"fault", "detection_phase"})
            _enum(details["fault"], _FAULT_KINDS)
            _identifier(details["detection_phase"])
            _require(
                self.controller == "RUNNING" and self.guard_state in {"IDLE", "FAILED"},
                "termination must fail an active normal guard first",
            )
            self.controller, self.permit = "TERMINATING", None
        elif operation == "terminal_custody_retained":
            exact_fields(details, set())
            _require(self.controller == "TERMINATING" and self.custody == "none", "duplicate custody entry")
            self.custody = "retained"
        elif operation == "terminal_operation_failed":
            exact_fields(details, {"failure_kind"})
            _enum(details["failure_kind"], FAILURE_KINDS)
            _require(
                self.controller == "TERMINATING" and self.custody != "none" and not self.terminal_failed,
                "duplicate or premature terminal fault",
            )
            self.terminal_failed = True
        elif operation == "terminal_arm_started":
            exact_fields(details, {"purpose"})
            purpose = _enum(details["purpose"], {"cpu_cancel", "target_sync"})
            _require(
                self.controller == "TERMINATING"
                and self.custody == "retained"
                and self.guardian in {"PARTIAL_FROZEN", "SEALED"}
                and self.guard_state == "IDLE"
                and not self.failed
                and not self.terminal_failed
                and not self.compromised
                and self.terminal_arms == 0
                and guard["role"] == "emergency",
                "terminal ARM predecessor",
            )
            no_work = self.may_cancel_without_sync
            _require(
                (purpose == "cpu_cancel") == no_work
                and (no_work or self.dirty == {self.config.device_ordinal}),
                "wrong terminal purpose",
            )
            self.purpose, self.terminal_arms = purpose, 1
            self.guard_state, self.guard = "ARM_PENDING", guard
        elif operation == "terminal_arm_failed":
            exact_fields(details, {"failure_kind"})
            _enum(details["failure_kind"], FAILURE_KINDS)
            _require(
                self.controller == "TERMINATING"
                and self.guard_state == "ARM_PENDING"
                and guard["role"] == "emergency",
                "terminal failure without pending ARM",
            )
            self.guard_state = "FAILED"
        elif operation in {"emergency_sync_started", "emergency_sync_succeeded", "emergency_sync_failed"}:
            exact_fields(details, {"device_ordinal"})
            _require(
                bounded_integer(details["device_ordinal"], 1023) == self.config.device_ordinal
                and self.controller == "TERMINATING"
                and self.guard_state == "ARMED"
                and self.purpose == "target_sync"
                and guard["role"] == "emergency"
                and self.custody == "retained"
                and not self.compromised,
                "sync lacks exact terminal target guard",
            )
            if operation == "emergency_sync_started":
                _require(self.sync_count == 0 and not self.terminal_failed, "unsafe emergency sync entry")
                self.sync_count, self.sync_outcome = 1, "started"
            else:
                _require(self.sync_outcome == "started", "sync outcome without entry")
                _require(
                    operation == "emergency_sync_failed" or not self.terminal_failed,
                    "sync success after terminal fault",
                )
                self.sync_outcome = "succeeded" if operation.endswith("succeeded") else "failed"
        elif operation == "resource_release_authorized":
            exact_fields(details, {"reason"})
            reason = _enum(details["reason"], {"cpu_cancel", "emergency_sync"})
            _require(
                self.controller == "TERMINATING"
                and self.custody == "retained"
                and self.guardian in {"PARTIAL_FROZEN", "SEALED"}
                and self.release == "not_authorized"
                and self.guard_state == "ARMED"
                and guard["role"] == "emergency"
                and not self.compromised
                and not self.failed
                and not self.terminal_failed,
                "unsafe release authorization",
            )
            _require(
                (reason == "cpu_cancel" and self.purpose == "cpu_cancel" and self.sync_count == 0)
                or (
                    reason == "emergency_sync"
                    and self.purpose == "target_sync"
                    and self.sync_outcome == "succeeded"
                ),
                "release proof missing",
            )
            self.guardian, self.release = "RELEASE_AUTHORIZED", "authorized"
        elif operation == "ordinary_cleanup_started":
            exact_fields(details, set())
            _require(self.release == "authorized" and not self.terminal_failed, "unsafe cleanup entry")
            self.release = "started"
        elif operation == "guardian_released":
            exact_fields(details, set())
            _require(
                self.release == "started"
                and self.guardian == "RELEASE_AUTHORIZED"
                and not self.terminal_failed,
                "unsafe release success",
            )
            self.guardian, self.release = "RELEASED", "completed"
        elif operation == "cleanup_failed":
            exact_fields(details, {"guardian_state"})
            _enum(details["guardian_state"], {"RELEASE_AUTHORIZED", "RELEASED"})
            _require(
                self.release in {"authorized", "started", "completed"}
                and (self.release != "authorized" or details["guardian_state"] == "RELEASE_AUTHORIZED")
                and (self.release != "completed" or details["guardian_state"] == "RELEASED"),
                "cleanup failure outside authorized release",
            )
            self.guardian, self.release = details["guardian_state"], "failed"
        elif operation == "quarantine_custody_committed":
            exact_fields(details, {"extent"})
            extent = _enum(details["extent"], {"full", "remaining"})
            _require(
                self.controller == "TERMINATING" and self.custody == "retained", "custody already committed"
            )
            _require(
                (
                    extent == "full"
                    and self.guardian == "TRANSFERRED_TO_QUARANTINE"
                    and self.release == "not_authorized"
                )
                or (
                    extent == "remaining"
                    and self.release == "failed"
                    and self.guardian in {"RELEASE_AUTHORIZED", "RELEASED"}
                ),
                "wrong custody extent",
            )
            self.custody = extent + "_quarantine"
        elif operation == "terminal_intent":
            exact_fields(details, {"disposition"})
            _require(self.controller == "TERMINATING", "terminal intent predecessor")
            if self.release == "failed" and self.custody == "remaining_quarantine":
                expected = "cleanup_failed_exit"
            elif self.custody == "full_quarantine":
                _require(
                    self.compromised
                    or self.failed
                    or self.terminal_failed
                    or self.guard_state == "FAILED"
                    or self.sync_outcome == "failed",
                    "quarantine has no recorded cause",
                )
                expected = (
                    "multi_or_unknown_device_quarantine_exit"
                    if self.compromised
                    else "quarantined_process_exit"
                )
            else:
                _require(
                    self.custody == "retained"
                    and self.release == "completed"
                    and self.guardian == "RELEASED"
                    and self.guard_state == "ARMED"
                    and not self.terminal_failed,
                    "successful terminal intent lacks release proof",
                )
                expected = (
                    "cancelled_before_submission_exit"
                    if self.purpose == "cpu_cancel"
                    else "aborted_synced_exit"
                )
            _require(details["disposition"] == expected, "terminal disposition does not replay")
            self.disposition, self.controller = expected, "TERMINAL_INTENT"
        else:
            raise GuardProtocolError("unknown guarded lifecycle transition")
        self.epoch = expected_epoch
        self.index += 1


def replay_lifecycle(configuration, lane, trace, *, version=LIFECYCLE_VERSION):
    """Derive descriptive state only; never register an execution object."""
    _require(type(trace) is list and bool(trace), "empty lifecycle trace")
    replay = _Replay(GuardConfiguration.from_payload(configuration), lane, version=version)
    for event in trace:
        prefix_sha256 = None
        if event.get("operation") == "post_reclaim_idle":
            prefix = {"schema_version": version, "configuration": configuration, "lane_id": lane,
                      "record_kind": "prefix", "transition_trace": trace[:replay.index],
                      "guardian_coverage": replay.coverage(), "summary": replay.summary()}
            prefix_sha256 = canonical_sha256(prefix)
        replay.apply(event, preceding_evidence_sha256=prefix_sha256)
    return replay


@dataclass(frozen=True, slots=True)
class PostReclaimIdleEvidence:
    """Descriptive replay record; only the controller's exact live instance can close."""

    _encoded: bytes

    @classmethod
    def from_payload(cls, value):
        p = exact_fields(value, {
            "schema_version", "session_identity", "lane", "target", "setup_quiescence_sha256",
            "prepared_graph_sha256", "arena_identity", "arena_lifecycle_sha256", "token_registry_sha256",
            "submitted_count", "resolved_count", "reclaimed_count", "cancelled_count",
            "remaining_prepared", "remaining_pending", "remaining_resolved", "active_leases",
            "device_holds", "external_holds", "slot_states", "guard_evidence_sha256",
            "historical_submission_latched",
        })
        _require(p["schema_version"] == POST_RECLAIM_IDLE_VERSION, "wrong post-reclaim idle version")
        _identifier(p["session_identity"])
        arena = p["arena_identity"]
        _require(type(arena) is str and arena.startswith("arena:") and len(arena) == 38
                 and all(c in "0123456789abcdef" for c in arena[6:]), "invalid native arena identity")
        _enum(p["lane"], {"cuda", "warp"})
        bounded_integer(p["target"], 1023)
        for name in ("setup_quiescence_sha256", "prepared_graph_sha256", "arena_lifecycle_sha256",
                     "token_registry_sha256", "guard_evidence_sha256"):
            _digest(p[name])
        for name in ("submitted_count", "resolved_count", "reclaimed_count", "cancelled_count"):
            bounded_integer(p[name], 1_000_000)
        _require(p["submitted_count"] == p["resolved_count"] == p["reclaimed_count"],
                 "idle proof has unresolved or unreclaimed submissions")
        for name in ("remaining_prepared", "remaining_pending", "remaining_resolved",
                     "active_leases", "device_holds", "external_holds"):
            _require(type(p[name]) is int and p[name] == 0, "idle proof contains live samples or holds")
        _require(p["slot_states"] == ["FREE"], "worker idle proof requires its single free slot")
        _require(type(p["historical_submission_latched"]) is bool
                 and p["historical_submission_latched"] == (p["submitted_count"] > 0),
                 "historical submission latch differs from completed samples")
        return cls(canonical_json_bytes(p))

    def to_payload(self):
        return json.loads(self._encoded)

    @property
    def sha256(self):
        return canonical_sha256(self.to_payload())


@dataclass(frozen=True, slots=True)
class GuardedLifecycleEvidence:
    _encoded: bytes

    @classmethod
    def from_payload(cls, value):
        root = exact_fields(
            value,
            {
                "schema_version",
                "configuration",
                "lane_id",
                "record_kind",
                "transition_trace",
                "guardian_coverage",
                "summary",
            },
        )
        _enum(root["schema_version"], {LIFECYCLE_VERSION, FINAL_LIFECYCLE_VERSION, WORKER_LIFECYCLE_VERSION})
        kind = _enum(root["record_kind"], {"prefix", "terminal"})
        replay = replay_lifecycle(root["configuration"], root["lane_id"], root["transition_trace"],
                                  version=root["schema_version"])
        _require((kind == "terminal") == (replay.disposition is not None), "prefix/terminal mismatch")
        _require(
            canonical_json_bytes(root["summary"]) == canonical_json_bytes(replay.summary()),
            "summary mismatch",
        )
        _require(
            canonical_json_bytes(root["guardian_coverage"]) == canonical_json_bytes(replay.coverage()),
            "guardian coverage mismatch",
        )
        return cls(canonical_json_bytes(root))

    def to_payload(self):
        return json.loads(self._encoded)

    @property
    def sha256(self):
        return canonical_sha256(self.to_payload())


@dataclass(frozen=True, slots=True)
class GuardedTerminationEvidence:
    lifecycle: GuardedLifecycleEvidence

    @classmethod
    def from_payload(cls, value):
        root = exact_fields(value, {"schema_version", "lifecycle"})
        lifecycle = GuardedLifecycleEvidence.from_payload(root["lifecycle"])
        expected = (WORKER_TERMINATION_VERSION
                    if root["lifecycle"]["schema_version"] == WORKER_LIFECYCLE_VERSION
                    else TERMINATION_VERSION)
        _require(root["schema_version"] == expected, "wrong termination version")
        _require(
            lifecycle.to_payload()["record_kind"] == "terminal", "termination requires terminal lifecycle"
        )
        return cls(lifecycle)

    def to_payload(self):
        payload = self.lifecycle.to_payload()
        version = (WORKER_TERMINATION_VERSION if payload["schema_version"] == WORKER_LIFECYCLE_VERSION
                   else TERMINATION_VERSION)
        return {"schema_version": version, "lifecycle": payload}

    @property
    def sha256(self):
        return canonical_sha256(self.to_payload())


def validate_custody_artifact(value):
    root = exact_fields(value, {"schema_version", "worker", "supervisor"})
    _require(root["schema_version"] == VALIDATION_VERSION, "wrong custody artifact version")
    supervisor = GuardSupervisorEvidence.from_payload(root["supervisor"], require_final=True).to_payload()
    if root["worker"] is None:
        return
    worker = GuardedLifecycleEvidence.from_payload(root["worker"]).to_payload()
    _require(worker["configuration"] == supervisor["configuration"], "artifact configuration mismatch")
    arms, acknowledgements, completes = set(), set(), set()
    for event in supervisor["trace"]:
        details = event["details"]
        if event["operation"] == "arm_committed":
            arms.add((details["sequence"], details["role"]))
        elif event["operation"] == "ack_written":
            acknowledgements.add((details["operation"], details["sequence"], details["role"]))
        elif event["operation"] == "normal_complete_received":
            completes.add((details["sequence"], details["role"]))
    worker_arms, worker_completes, complete_attempts = set(), set(), set()
    supervisor_roles = dict(arms)
    for event in worker["transition_trace"]:
        operation, guard = event["operation"], event["guard"]
        if operation in {"normal_arm_started", "terminal_arm_started"}:
            worker_arms.add((guard["sequence"], guard["role"]))
            _require(
                guard["sequence"] not in supervisor_roles
                or supervisor_roles[guard["sequence"]] == guard["role"],
                "worker ARM attempt conflicts with supervisor role",
            )
        elif operation == "normal_complete_started":
            complete_attempts.add((guard["sequence"], guard["role"]))
        elif operation in {"guard_armed", "normal_disarmed"}:
            response = "ARMED" if operation == "guard_armed" else "DISARMED"
            identity = (guard["sequence"], guard["role"])
            _require((response, *identity) in acknowledgements, "worker ACK absent from supervisor")
            if response == "DISARMED":
                _require(identity in completes, "worker completion absent from supervisor")
                worker_completes.add(identity)
    _require(
        worker["summary"]["normal_complete_count"] <= supervisor["summary"]["normal_complete_count"],
        "worker completion count exceeds supervisor",
    )
    replay = replay_lifecycle(worker["configuration"], worker["lane_id"], worker["transition_trace"],
                              version=worker["schema_version"])
    bounded_tail = (
        worker["record_kind"] == "terminal"
        or replay.controller == "TERMINATING"
        or replay.failed
        or replay.terminal_failed
        or replay.compromised
    )
    if bounded_tail:
        allowed_arms = set(worker_arms)
        if (
            worker["record_kind"] == "prefix"
            and replay.controller == "TERMINATING"
            and replay.guard_state == "IDLE"
            and not replay.failed
            and not replay.terminal_failed
            and not replay.compromised
            and replay.terminal_arms == 0
            and replay.next_sequence is not None
        ):
            allowed_arms.add((replay.next_sequence, "emergency"))
        _require(arms <= allowed_arms, "supervisor ARM after irreversible worker boundary")
        _require(completes <= complete_attempts, "supervisor COMPLETE was never attempted by worker")
    if worker["summary"]["release_status"] == "completed":
        _require(
            arms == worker_arms and completes == worker_completes,
            "successful release has unmatched guard work",
        )
