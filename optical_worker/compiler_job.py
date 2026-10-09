"""Compiler descendant obligation and canonical replay, independent of worker exit.

These records describe observations. They do not grant process or compilation
authority. The dedicated supervisor supplies actual OS observations.
"""

from __future__ import annotations

from dataclasses import dataclass

from .compilation_contract import COMPILER_JOB_VERSION, CUDA_COMPILE_SEQUENCES
from .protocol import MAX_RESPONSE_BYTES, canonical_json_bytes, canonical_sha256, decode_canonical_message
from .wait_protocol import GuardConfiguration, GuardProtocolError, bounded_integer, checked_add, exact_fields
from .wait_supervision import GuardSupervisorCore, GuardSupervisorEvidence, _State, _transition

_DEADLINE_SOURCES = frozenset({"operation", "ack", "terminal_grace", "clock_fault"})
_ERRORS = frozenset({"missing", "permission", "identity_changed", "proc_unavailable",
                     "resource_exhausted", "other_os"})
_DRAIN_REASONS = frozenset({"worker_exit", "guard_fault", "parent_lost", "deadline_expired",
                           "observation_failed", "clock_failed"})
_CLOCK_ERRORS = frozenset({"backward", "out_of_domain", "deadline_overflow"})
_EVENT_FIELDS = {
    "subreaper_ready": set(), "worker_spawned": {"identity"},
    "deadline_snapshot": {"guard_event_index", "sequence", "source", "deadline_ns"},
    "compile_quiescent": {"sequence"}, "draining": {"reason", "source", "deadline_ns"},
    "deadline_expired": {"source", "deadline_ns"}, "clock_failed": {"reason"},
    "child_adopted": {"identity"}, "signal_attempted": {"identity", "signal"},
    "signal_failed": {"identity", "error_kind"}, "child_reaped": {"identity", "wait_status"},
    "no_children": set(), "pipes_eof": set(), "observation_failed": {"error_kind"}, "closed": set(),
}


def _require(condition, message):
    if not condition:
        raise GuardProtocolError(message)


def _enum(value, choices):
    _require(type(value) is str and value in choices, "unknown compiler job enum")
    return value


def process_identity(value):
    row = exact_fields(value, {"pid", "starttime_ticks", "sid"})
    bounded_integer(row["pid"], minimum=1)
    bounded_integer(row["starttime_ticks"])
    bounded_integer(row["sid"], minimum=1)
    return tuple(row[key] for key in ("pid", "starttime_ticks", "sid"))


def identity_payload(identity):
    row = dict(zip(("pid", "starttime_ticks", "sid"), identity, strict=True))
    process_identity(row)
    return row


def wait_status_returncode(status):
    """Linux waitpid exit status only, independent of the parser host OS."""
    bounded_integer(status, 0xffff)
    low = status & 0x7f
    if low == 0:
        _require(status & 0xff == 0, "invalid normal Linux exit status")
        return status >> 8
    _require(low < 0x7f and status >> 8 == 0, "non-exit Linux wait status")
    return -low


def guard_deadline_snapshot(state, event_index):
    first = state.earliest()
    return {
        "guard_event_index": event_index,
        "sequence": None if state.active is None else state.active[0],
        "source": None if first is None else {"terminal": "terminal_grace"}.get(first[1], first[1]),
        "deadline_ns": None if first is None else first[0],
    }


class _JobReplay:
    """One transition implementation for live accumulation and descriptive replay."""

    def __init__(self, configuration, supervisor, worker, guard_states):
        self.configuration = configuration
        self.supervisor = None if supervisor is None else process_identity(supervisor)
        self.worker = None if worker is None else process_identity(worker)
        if self.worker is not None:
            _require(self.supervisor is not None and self.worker != self.supervisor
                     and self.worker[0] == self.worker[2], "worker lacks its own session")
        self.guard_states = guard_states
        self.state = "NEW"
        self.failed = False
        self.ready = False
        self.expired = False
        self.deadline = None
        self.deadline_source = None
        self.snapshot = None
        self.clock_fault = False
        self.clock_fault_at = None
        self.owned = set()
        self.reaped = set()
        self.attempted = set()
        self.signal_errors = set()
        self.quiescent = []
        self.no_children = False
        self.pipes_eof = False
        self.count = 0
        self.last_ns = 0

    def event(self, raw):
        e = exact_fields(raw, {"index", "at_ns", "operation", "details"})
        _require(self.state != "CLOSED", "event after compiler job closure")
        _require(bounded_integer(e["index"]) == self.count, "compiler event index drift")
        now = bounded_integer(e["at_ns"])
        _require(now >= self.last_ns, "compiler event time went backward")
        op = _enum(e["operation"], _EVENT_FIELDS)
        d = exact_fields(e["details"], _EVENT_FIELDS[op])
        if op == "subreaper_ready":
            _require(self.state == "NEW" and not self.ready and self.supervisor is not None,
                     "subreaper readiness is single-use before spawn")
            self.ready = True
        elif op == "worker_spawned":
            identity = process_identity(d["identity"])
            _require(self.state == "NEW" and self.ready and identity == self.worker,
                     "worker launch lacks policy or exact identity")
            self.owned.add(identity)
            self.state = "LIVE"
        elif op == "deadline_snapshot":
            _require(self.state in {"LIVE", "DRAINING"}, "snapshot before worker launch")
            index = bounded_integer(d["guard_event_index"])
            if d["sequence"] is not None:
                bounded_integer(d["sequence"])
            if d["deadline_ns"] is not None:
                bounded_integer(d["deadline_ns"])
                _enum(d["source"], _DEADLINE_SOURCES)
            else:
                _require(d["source"] is None, "null deadline has a source")
            _require(index in self.guard_states, "snapshot refers to absent guard event")
            expected_index = 0 if self.snapshot is None else self.snapshot["guard_event_index"] + 1
            _require(index == expected_index, "missing or repeated core transition snapshot")
            state, event = self.guard_states[index]
            _require(event["operation"] != "process_exited" and now >= event["at_ns"],
                     "cannot clear obligation using worker exit")
            _require(d == guard_deadline_snapshot(state, index), "deadline differs from actual core transition")
            if d["deadline_ns"] is None:
                _require(index == 0 or (event["operation"] == "ack_written"
                         and event["details"]["operation"] == "DISARMED"),
                         "only healthy DISARMED may clear the live obligation")
                _require(self.state == "LIVE", "draining cannot clear obligation")
            self.snapshot = dict(d)
            if self.state == "LIVE":
                self.deadline, self.deadline_source = d["deadline_ns"], d["source"]
            # DRAINING keeps its already frozen obligation, regardless of later
            # core terminal/kill transitions. Clock failure can only reduce it.
        elif op == "compile_quiescent":
            sequence = bounded_integer(d["sequence"])
            _require(self.state == "LIVE" and not self.failed and self.snapshot is not None,
                     "quiescence outside healthy live worker")
            _require(len(self.quiescent) < len(CUDA_COMPILE_SEQUENCES)
                     and sequence == CUDA_COMPILE_SEQUENCES[len(self.quiescent)],
                     "compilation quiescence skipped or repeated")
            state, _ = self.guard_states[self.snapshot["guard_event_index"]]
            _require(state.state == "ARMED_NORMAL" and state.active == (sequence, "setup")
                     and state.pending is None and self.worker not in self.reaped
                     and self.owned - self.reaped == {self.worker}
                     and self.deadline is not None and now < self.deadline,
                     "compile COMPLETE lacks live matching guard or child quiescence")
            self.quiescent.append(sequence)
        elif op == "draining":
            _require(self.state == "LIVE", "compiler draining is single-use")
            reason = _enum(d["reason"], _DRAIN_REASONS)
            source = _enum(d["source"], _DEADLINE_SOURCES)
            deadline = bounded_integer(d["deadline_ns"])
            if self.clock_fault:
                _require(source == "clock_fault" and deadline <= self.clock_fault_at
                         and (self.deadline is None or deadline <= self.deadline),
                         "clock failure renewed deadline")
            elif self.deadline is not None:
                _require((deadline, source) == (self.deadline, self.deadline_source),
                         "draining changed original obligation")
            else:
                _require(source == "terminal_grace"
                         and deadline == checked_add(now, self.configuration.operation_timeout_ms * 1_000_000),
                         "draining invented terminal grace")
            self.state = "DRAINING"
            self.deadline, self.deadline_source = deadline, source
            if reason != "worker_exit":
                self.failed = True
        elif op == "deadline_expired":
            bounded_integer(d["deadline_ns"])
            _enum(d["source"], _DEADLINE_SOURCES)
            _require(self.state == "DRAINING" and not self.expired and self.deadline is not None
                     and now >= self.deadline and d == {"source": self.deadline_source, "deadline_ns": self.deadline},
                     "compiler expiry is early, repeated or changed")
            self.expired = self.failed = True
        elif op == "clock_failed":
            _enum(d["reason"], _CLOCK_ERRORS)
            _require(self.state in {"LIVE", "DRAINING"}, "clock failure outside active job")
            self.failed = self.clock_fault = True
            self.clock_fault_at = now
            if self.state == "DRAINING" and not self.expired:
                self.deadline = now if self.deadline is None else min(now, self.deadline)
                self.deadline_source = "clock_fault"
        elif op == "child_adopted":
            identity = process_identity(d["identity"])
            _require(self.state in {"LIVE", "DRAINING"} and not self.no_children
                     and identity not in self.owned and identity != self.supervisor
                     and identity[2] == self.worker[2], "invalid adopted child identity")
            _require(all(other[0] != identity[0] or other in self.reaped for other in self.owned),
                     "live PID reused without reaping")
            self.owned.add(identity)
        elif op == "signal_attempted":
            identity = process_identity(d["identity"])
            _require(self.expired and self.state == "DRAINING" and type(d["signal"]) is int
                     and d["signal"] == 9 and identity in self.owned - self.reaped
                     and identity not in self.attempted, "unowned, duplicate or premature termination attempt")
            self.attempted.add(identity)
        elif op == "signal_failed":
            identity = process_identity(d["identity"])
            _enum(d["error_kind"], _ERRORS)
            _require(identity in self.attempted and identity not in self.signal_errors,
                     "signal error without unique attempt")
            self.signal_errors.add(identity)
            self.failed = True
        elif op == "child_reaped":
            identity = process_identity(d["identity"])
            code = wait_status_returncode(d["wait_status"])
            _require(self.state in {"LIVE", "DRAINING"} and identity in self.owned - self.reaped,
                     "unowned or repeated reap")
            if identity == self.worker:
                _require(self.state == "DRAINING", "worker reaped before freezing obligation")
                self.failed = self.failed or code != 0
            self.reaped.add(identity)
        elif op == "no_children":
            _require(self.state == "DRAINING" and not self.no_children
                     and self.worker in self.reaped and self.owned == self.reaped,
                     "ECHILD before owned reaping")
            self.no_children = True
        elif op == "pipes_eof":
            _require(self.state == "DRAINING" and not self.pipes_eof, "repeated or early pipe EOF")
            self.pipes_eof = True
        elif op == "observation_failed":
            _enum(d["error_kind"], _ERRORS)
            self.failed = True
        elif op == "closed":
            _require(self.state == "DRAINING" and self.no_children and self.pipes_eof
                     and self.worker in self.reaped and self.owned == self.reaped,
                     "compiler job closure lacks reaping/ECHILD/EOF")
            self.state = "CLOSED"
        self.count += 1
        self.last_ns = now

    def summary(self):
        return {"state": self.state, "failed": self.failed, "deadline_ns": self.deadline,
                "deadline_expired": self.expired, "worker_reaped": self.worker in self.reaped,
                "no_children": self.no_children, "pipes_eof": self.pipes_eof}


def _guard_states(evidence):
    p = evidence.to_payload()
    config = GuardConfiguration.from_payload(p["configuration"])
    states = {}
    state = _State()
    last = 0
    for index, event in enumerate(p["trace"]):
        state = _transition(state, event, config, expected_index=index, last_ns=last)
        states[index] = state, event
        last = event["at_ns"]
    return config, states


@dataclass(frozen=True, slots=True)
class CompilerJobRecord:
    _encoded: bytes

    @classmethod
    def from_payload(cls, value, *, guard_evidence, require_closed=False):
        _require(type(guard_evidence) is GuardSupervisorEvidence, "exact guard evidence required")
        p = exact_fields(value, {"schema_version", "session", "supervisor", "worker",
                                 "guard_evidence_sha256", "events", "summary"})
        _require(p["schema_version"] == COMPILER_JOB_VERSION, "compiler job schema version")
        guard_payload = guard_evidence.to_payload()
        _require(p["guard_evidence_sha256"] == canonical_sha256(guard_payload), "compiler/guard digest mismatch")
        config, states = _guard_states(guard_evidence)
        _require(p["session"] == config.session, "compiler job session mismatch")
        _require(type(p["events"]) is list, "compiler job events must be an array")
        replay = _JobReplay(config, p["supervisor"], p["worker"], states)
        for event in p["events"]:
            replay.event(event)
        _require(p["summary"] == replay.summary()
                 and canonical_json_bytes(p["summary"]) == canonical_json_bytes(replay.summary()),
                 "compiler job summary differs from replay")
        # Every successful compiler COMPLETE must have a preceding quiescence
        # observation, joined to that exact core guard while still ARMED.
        completes = [e for e in guard_payload["trace"] if e["operation"] == "normal_complete_received"
                     and e["details"]["sequence"] in CUDA_COMPILE_SEQUENCES]
        quiescent = [e for e in p["events"] if e["operation"] == "compile_quiescent"]
        for complete in completes:
            matches = [e for e in quiescent if e["details"]["sequence"] == complete["details"]["sequence"]]
            _require(len(matches) == 1 and matches[0]["at_ns"] <= complete["at_ns"],
                     "compiler COMPLETE preceded quiescence")
        exits = [e for e in guard_payload["trace"] if e["operation"] == "process_exited"]
        if replay.worker in replay.reaped:
            reaped = next(e for e in p["events"] if e["operation"] == "child_reaped"
                          and process_identity(e["details"]["identity"]) == replay.worker)
            _require(len(exits) == 1 and exits[0]["at_ns"] <= reaped["at_ns"]
                     and exits[0]["details"]["returncode"] == wait_status_returncode(reaped["details"]["wait_status"]),
                     "worker reap differs from actual core exit")
        if require_closed:
            _require(replay.state == "CLOSED", "compiler job remains incomplete")
        return cls(canonical_json_bytes(p))

    @classmethod
    def from_bytes(cls, encoded, *, guard_evidence, require_closed=False):
        return cls.from_payload(decode_canonical_message(encoded, max_bytes=MAX_RESPONSE_BYTES), guard_evidence=guard_evidence,
                                require_closed=require_closed)

    def to_payload(self):
        return decode_canonical_message(self._encoded, max_bytes=MAX_RESPONSE_BYTES)

    def to_bytes(self):
        return self._encoded

    @property
    def sha256(self):
        return canonical_sha256(self.to_payload())


class _CompilerObligation:
    """Live event accumulator used only by the private OS job owner.

    No OS operation occurs here. The immutable input to every transition is the
    same guard state later replayed by CompilerJobRecord, including after EXITED.
    """

    def __init__(self, core, supervisor):
        _require(type(core) is GuardSupervisorCore and core.state == "IDLE"
                 and len(core._records) == 1, "compiler obligation needs a fresh exact core")
        core._check_thread()
        self.core = core
        self.supervisor = dict(supervisor)
        self.worker = None
        self.events = []
        self.states = {}
        self.replay = _JobReplay(core.configuration, supervisor, None, self.states)
        self._last_observed_ns = core._last_ns

    def _event(self, operation, details, now):
        self.core._check_thread()
        event = {"index": len(self.events), "at_ns": now, "operation": operation, "details": details}
        self.replay.event(event)
        self.events.append(event)

    def _clock_failure(self, reason):
        now = self._last_observed_ns
        if self.replay.state in {"LIVE", "DRAINING"}:
            self._event("clock_failed", {"reason": reason}, now)
            self.drain("clock_failed", now=now)
        return now

    def now(self):
        self.core._check_thread()
        try:
            now = bounded_integer(self.core._clock())
        except BaseException:
            return self._clock_failure("out_of_domain")
        if now < self._last_observed_ns:
            return self._clock_failure("backward")
        self._last_observed_ns = now
        return now

    def subreaper_ready(self):
        self._event("subreaper_ready", {}, self.now())

    def spawned(self, identity):
        _require(self.worker is None, "worker already registered")
        self.worker = dict(identity)
        self.replay.worker = process_identity(identity)
        _require(self.replay.worker[0] == self.replay.worker[2]
                 and self.replay.worker != self.replay.supervisor, "worker must have its own session")
        self._event("worker_spawned", {"identity": dict(identity)}, self.now())
        self.sync_core()

    def sync_core(self):
        """Capture every transition before another operation can erase its fields."""
        self.core._check_thread()
        start = 0 if self.replay.snapshot is None else self.replay.snapshot["guard_event_index"] + 1
        for index in range(start, len(self.core._records)):
            state, event = self.core._records[index]
            self.states[index] = state, event
            if event["operation"] == "process_exited":
                break
            # Core transitions read the shared clock themselves. Keep the last
            # valid time even if its next call fails or runs backward.
            self._last_observed_ns = max(self._last_observed_ns, event["at_ns"])
            if state.state == "TERMINATING" and self.replay.state == "LIVE":
                if event["operation"] == "clock_rejected":
                    self._event("clock_failed", {"reason": event["details"]["reason"]}, self._last_observed_ns)
                    self.drain("clock_failed")
                else:
                    # Freeze the preceding ACK/operation, before the core's
                    # terminal transition clears pending or changes tie order.
                    self.drain("guard_fault")
            self._event("deadline_snapshot", guard_deadline_snapshot(state, index), self._last_observed_ns)
        if self.core.state == "TERMINATING" and self.replay.state == "LIVE":
            self.drain("guard_fault")

    def drain(self, reason, *, now=None):
        if self.replay.state == "DRAINING":
            return
        _require(self.replay.state == "LIVE", "cannot drain inactive compiler job")
        now = self._last_observed_ns if now is None else now
        deadline, source = self.replay.deadline, self.replay.deadline_source
        if self.replay.clock_fault:
            deadline = now if deadline is None else min(now, deadline)
            source = "clock_fault"
        elif deadline is None:
            try:
                deadline = checked_add(now, self.core.configuration.operation_timeout_ms * 1_000_000)
                source = "terminal_grace"
            except GuardProtocolError:
                self._event("clock_failed", {"reason": "deadline_overflow"}, now)
                deadline, source = now, "clock_fault"
        self._event("draining", {"reason": reason, "source": source, "deadline_ns": deadline}, now)

    def tick(self):
        self.core._check_thread()
        if self.replay.state == "CLOSED":
            return
        if self.core.state != "EXITED":
            self.core.tick()
            self.sync_core()
        now = self.now()
        if self.replay.deadline is not None and now >= self.replay.deadline and not self.replay.expired:
            self.drain("deadline_expired", now=now)
            self._event("deadline_expired", {"source": self.replay.deadline_source,
                                             "deadline_ns": self.replay.deadline}, now)

    def adopted(self, identity):
        self._event("child_adopted", {"identity": dict(identity)}, self.now())

    def reaped(self, identity, status):
        now = self.now()
        if process_identity(identity) == self.replay.worker:
            # Freeze before observe_exit, whose correct worker-only semantics
            # clear the core's active and pending obligations permanently.
            self.sync_core()
            self.drain("worker_exit", now=now)
            self.core.observe_exit(wait_status_returncode(status))
            self._last_observed_ns = max(self._last_observed_ns, self.core._last_ns)
            now = self._last_observed_ns
        self._event("child_reaped", {"identity": dict(identity), "wait_status": status}, now)

    def quiescent(self, sequence):
        self.tick()
        self._event("compile_quiescent", {"sequence": sequence}, self.now())

    def signal_attempted(self, identity):
        self._event("signal_attempted", {"identity": dict(identity), "signal": 9}, self.now())

    def signal_failed(self, identity, error_kind):
        self._event("signal_failed", {"identity": dict(identity), "error_kind": error_kind}, self.now())

    def observation_failed(self, error_kind):
        now = self.now()
        self._event("observation_failed", {"error_kind": error_kind}, now)
        if self.replay.state == "LIVE":
            self.drain("observation_failed", now=now)

    def no_children(self):
        self._event("no_children", {}, self.now())

    def pipes_eof(self):
        self._event("pipes_eof", {}, self.now())

    def close(self):
        self._event("closed", {}, self.now())

    def record(self, *, require_closed=False):
        evidence = self.core.evidence()
        return CompilerJobRecord.from_payload({
            "schema_version": COMPILER_JOB_VERSION, "session": self.core.configuration.session,
            "supervisor": dict(self.supervisor), "worker": None if self.worker is None else dict(self.worker),
            "guard_evidence_sha256": canonical_sha256(evidence.to_payload()),
            "events": list(self.events), "summary": self.replay.summary(),
        }, guard_evidence=evidence, require_closed=require_closed)
