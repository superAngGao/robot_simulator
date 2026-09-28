from __future__ import annotations

import copy
import dis
import inspect
import json
import select
import signal
import socket
import subprocess
import sys
import threading
from functools import partial

import pytest

from optical_worker.protocol import canonical_json_bytes, encode_framed_message
from optical_worker.wait_protocol import (
    I64_MAX,
    MAX_BODY_BYTES,
    MAX_GUARDS,
    MAX_SEQUENCE,
    GuardConfiguration,
    GuardFrameDecoder,
    GuardProtocolError,
    GuardWaitClient,
    GuardWaitTicket,
    bounded_integer,
    checked_add,
    next_sequence,
    validate_message,
)
from optical_worker.wait_supervision import (
    GuardSupervisorCore,
    GuardSupervisorEvidence,
    GuardSupervisorTransport,
)


def config(**kwargs):
    return GuardConfiguration(
        **(dict(session="a" * 32, device_ordinal=1, ack_timeout_ms=5, operation_timeout_ms=20) | kwargs)
    )


def setup_core():
    now = [100]
    settings = config()
    return settings, now, GuardSupervisorCore(settings, clock=lambda: now[0])


def arm(settings, core, sequence=0, role="setup"):
    reply = core.accept(settings.message("ARM", sequence, role))
    assert core.state == "ARM_ACK_PENDING"
    core.acknowledge_written(reply)


def complete(settings, core, sequence=0, role="setup"):
    reply = core.accept(settings.message("COMPLETE", sequence, role))
    assert core.state == "DISARM_ACK_PENDING"
    core.acknowledge_written(reply)


def test_normal_snapshot_golden_and_final_evidence():
    c, _, s = setup_core()
    arm(c, s)
    pending = s.accept(c.message("COMPLETE", 0, "setup"))
    assert s.evidence().to_payload()["summary"] == {
        "state": "DISARM_ACK_PENDING",
        "next_sequence": 1,
        "active_wait": {"sequence": 0, "role": "setup"},
        "pending_ack": {"operation": "DISARMED", "sequence": 0, "role": "setup", "deadline_ns": 5000100},
        "operation_deadline_ns": 20000100,
        "terminal_deadline_ns": None,
        "effective_deadline_ns": 5000100,
        "terminal_reason": None,
        "arm_count": 1,
        "normal_complete_count": 0,
        "deadline_expired": False,
        "kill_requested": False,
        "kill_error_kind": None,
        "observed_returncode": None,
        "complete": False,
    }
    with pytest.raises(GuardProtocolError, match="snapshot"):
        s.evidence(require_final=True)
    s.acknowledge_written(pending)
    arm(c, s, 1, "resolve")
    complete(c, s, 1, "resolve")
    arm(c, s, 2, "emergency")
    assert s.state == "ARMED_TERMINAL"
    s.observe_exit(0)
    payload = s.evidence(require_final=True).to_payload()
    assert payload["summary"]["arm_count"] == 3
    assert payload["summary"]["normal_complete_count"] == 2
    assert GuardSupervisorEvidence.from_payload(payload, require_final=True).to_payload() == payload


@pytest.mark.parametrize(
    "field", ["protocol_version", "session", "device_ordinal", "role", "operation", "extra"]
)
def test_wire_identity_extra_fields_fail_closed(field):
    c, _, s = setup_core()
    message = c.message("ARM", 0, "setup")
    message[field] = "wrong"
    with pytest.raises((GuardProtocolError, TypeError)):
        s.accept(message)
    assert s.state == "TERMINATING"
    assert s.pending_response is None
    assert s.evidence().to_payload()["summary"]["arm_count"] == 0


@pytest.mark.parametrize("bad", [True, -1, MAX_SEQUENCE + 1, float("inf"), 0.0, None])
def test_sequence_domain(bad):
    c = config()
    message = c.message("ARM", 0, "setup")
    message["sequence"] = bad
    with pytest.raises(GuardProtocolError):
        validate_message(message, c)


def test_capacity_arithmetic_and_first_illegal_values():
    assert MAX_GUARDS == 2305843009213693949
    assert MAX_SEQUENCE == 2305843009213693948
    assert 1 + 4 * MAX_GUARDS + 6 == 9223372036854775803
    for value in [MAX_SEQUENCE, MAX_GUARDS, 4 * MAX_GUARDS + 6, I64_MAX]:
        assert bounded_integer(value) == value
        assert canonical_json_bytes({"value": value})
    assert next_sequence(MAX_SEQUENCE) is None
    assert next_sequence(MAX_SEQUENCE - 1) == MAX_SEQUENCE
    with pytest.raises(GuardProtocolError):
        next_sequence(MAX_SEQUENCE + 1)
    for limit in [MAX_SEQUENCE, MAX_GUARDS, I64_MAX]:
        assert checked_add(limit - 1, 1, limit) == limit
        with pytest.raises(GuardProtocolError):
            checked_add(limit, 1, limit)
    budget = (2**31 - 1) * 1_000_000
    assert checked_add(I64_MAX - budget, budget) == I64_MAX
    with pytest.raises(GuardProtocolError):
        checked_add(I64_MAX - budget + 1, budget)


@pytest.mark.parametrize("field", ["device_ordinal", "ack_timeout_ms", "operation_timeout_ms"])
@pytest.mark.parametrize("value", [True, -1, 2**31, None, 1.0])
def test_configuration_boundaries(field, value):
    data = config().to_payload()
    data[field] = value
    with pytest.raises(GuardProtocolError):
        GuardConfiguration.from_payload(data)


def test_codec_fragmentation_duplicate_keys_and_limits():
    message = config().message("ARM", MAX_SEQUENCE, "emergency")
    frame = encode_framed_message(message, max_bytes=MAX_BODY_BYTES)
    for split in range(1, len(frame)):
        parser = GuardFrameDecoder()
        assert parser.feed(frame[:split]) is None
        assert parser.feed(frame[split:]) == message
    for invalid in [b"", b"\0\0\0\0", (1025).to_bytes(4, "big"), frame + frame]:
        with pytest.raises(Exception):
            GuardFrameDecoder().feed(invalid)
    duplicate = b'{"operation":"ARM","operation":"ARM"}'
    with pytest.raises(Exception, match="duplicate"):
        GuardFrameDecoder().feed(len(duplicate).to_bytes(4, "big") + duplicate)


@pytest.mark.parametrize("role", ["setup", "resolve"])
def test_stalled_disarm_keeps_watchdog_and_parent_loss_cannot_clear(role):
    c, now, s = setup_core()
    arm(c, s, role=role)
    now[0] += 19_000_000
    reply = s.accept(c.message("COMPLETE", 0, role))
    assert s.effective_deadline_ns == 20000100
    s.fault("parent_lost")
    assert s.effective_deadline_ns == 20000100
    with pytest.raises(GuardProtocolError):
        s.acknowledge_written(reply)
    before = len(s.evidence().to_payload()["trace"])
    s.fault("input_eof")
    assert len(s.evidence().to_payload()["trace"]) == before
    now[0] = 20000100
    s.tick()
    assert s.kill_required
    s.record_kill_request()
    s.record_kill_failure("permission_denied")
    assert not s.kill_required and s.state == "TERMINATING"
    with pytest.raises(GuardProtocolError):
        s.evidence(require_final=True)
    s.observe_exit(-9)
    result = s.evidence(require_final=True).to_payload()["summary"]
    assert result["terminal_reason"] == "parent_lost"
    assert result["normal_complete_count"] == 0
    assert result["kill_error_kind"] == "permission_denied"


def test_disarm_ack_earlier_deadline_expires_at_equality():
    c, now, s = setup_core()
    arm(c, s)
    reply = s.accept(c.message("COMPLETE", 0, "setup"))
    now[0] = 5000100
    with pytest.raises(GuardProtocolError):
        s.acknowledge_written(reply)
    trace = s.evidence().to_payload()["trace"]
    assert trace[-1]["operation"] == "deadline_expired"
    assert trace[-1]["details"] == {"source": "ack"}
    assert s.pending_response is None and s.kill_required


@pytest.mark.parametrize("fault", ["duplicate_arm", "emergency_complete", "wrong_role", "early_complete"])
def test_ordering_errors_cannot_commit_completion(fault):
    c, _, s = setup_core()
    if fault != "early_complete":
        arm(c, s, role="emergency" if fault == "emergency_complete" else "setup")
    message = c.message(
        "ARM" if fault == "duplicate_arm" else "COMPLETE", 0, "resolve" if fault == "wrong_role" else "setup"
    )
    with pytest.raises(GuardProtocolError):
        s.accept(message)
    assert s.state == "TERMINATING"
    assert s.evidence().to_payload()["summary"]["normal_complete_count"] == 0


@pytest.mark.parametrize("bad_clock", [True, -1, I64_MAX + 1, 50, float("nan")])
def test_bad_clock_single_use_and_immediate_kill(bad_clock):
    _, now, s = setup_core()
    now[0] = 200
    s.tick()  # successful observation without a trace transition
    now[0] = bad_clock
    for _ in range(4):
        s.tick()
    assert s.kill_required
    payload = s.evidence().to_payload()
    assert [x["operation"] for x in payload["trace"]] == ["initialized", "clock_rejected", "deadline_expired"]
    assert payload["trace"][1]["at_ns"] == 200


def test_deadline_overflow_rejects_before_ack():
    c, now, s = setup_core()
    now[0] = I64_MAX
    with pytest.raises(GuardProtocolError):
        s.accept(c.message("ARM", 0, "setup"))
    assert s.pending_response is None and s.kill_required
    assert s.evidence().to_payload()["summary"]["arm_count"] == 0


@pytest.mark.parametrize(
    "mutation", ["operation", "count", "index", "drop_ack", "duplicate", "time", "exit", "bool_count"]
)
def test_evidence_replay_not_self_hash_acceptance(mutation):
    c, _, s = setup_core()
    arm(c, s)
    complete(c, s)
    s.observe_exit(0)
    value = s.evidence().to_payload()
    if mutation == "operation":
        value["trace"][1]["operation"] = "invented"
    elif mutation == "count":
        value["summary"]["arm_count"] = 77
    elif mutation == "bool_count":
        value["summary"]["arm_count"] = True
    elif mutation == "index":
        value["trace"][1]["index"] = 2
    elif mutation == "drop_ack":
        del value["trace"][2]
    elif mutation == "duplicate":
        value["trace"].insert(2, copy.deepcopy(value["trace"][1]))
    elif mutation == "time":
        value["trace"][2]["at_ns"] = 99
    else:
        value["summary"]["observed_returncode"] = -9
    if mutation in {"drop_ack", "duplicate"}:
        for index, event in enumerate(value["trace"]):
            event["index"] = index
    canonical_json_bytes(value)  # re-encoding valid JSON cannot repair state flow
    with pytest.raises(GuardProtocolError):
        GuardSupervisorEvidence.from_payload(value)


def test_buffered_request_before_ack_write_poison_and_no_response():
    c, _, s = setup_core()
    left, right = socket.socketpair()
    try:
        transport = GuardSupervisorTransport(right, s)
        for sequence in (0, 1):
            left.sendall(encode_framed_message(c.message("ARM", sequence, "setup"), max_bytes=1024))
            if sequence == 0:
                transport.service(readable=True, writable=False)
            else:
                with pytest.raises(GuardProtocolError):
                    transport.service(readable=True, writable=True)
        assert s.state == "TERMINATING" and not transport.wants_write
        left.settimeout(1)
        assert left.recv(1024) == b""
    finally:
        left.close()
        right.close()


def test_thread_identity_and_client_forgery_reject_without_wire_write():
    c, _, s = setup_core()
    left, right = socket.socketpair()
    try:
        client = GuardWaitClient(left, c)
        errors = []

        def foreign_thread():
            for call in (s.tick, lambda: client.arm("setup")):
                try:
                    call()
                except GuardProtocolError as error:
                    errors.append(str(error))

        t = threading.Thread(target=foreign_thread)
        t.start()
        t.join()
        assert len(errors) == 2 and s.state == "IDLE" and client.state == "IDLE"
        with pytest.raises(GuardProtocolError):
            client.complete(GuardWaitTicket(c.session, c.device_ordinal, "setup", 0))
        assert client.state == "FAILED"
        right.setblocking(False)
        with pytest.raises(BlockingIOError):
            right.recv(1024)
    finally:
        left.close()
        right.close()


def test_cold_import_has_no_runtime_or_harness_dependency():
    script = """
import builtins,json,sys
original=builtins.__import__
attempts=[]
def checked(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','warp','numpy','benchmarks','tests','tools','examples'}:
        attempts.append(name)
        raise ImportError(name)
    return original(name,*args,**kwargs)
builtins.__import__=checked
import optical_worker.wait_protocol,optical_worker.wait_supervision
print(json.dumps(attempts))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == []


@pytest.fixture
def connected_client():
    settings = config(ack_timeout_ms=1000, operation_timeout_ms=3000)
    left, right = socket.socketpair()
    stop, started = threading.Event(), threading.Event()
    reports = []

    def serve():
        core = GuardSupervisorCore(settings)
        transport = GuardSupervisorTransport(right, core)
        started.set()
        try:
            while not stop.is_set():
                read, write, _ = select.select([right], [right] if transport.wants_write else [], [], 0.01)
                transport.service(readable=bool(read), writable=bool(write))
        finally:
            reports.append(core.evidence().to_payload())

    thread = threading.Thread(target=serve)
    thread.start()
    assert started.wait(2)
    try:
        yield GuardWaitClient(left, settings), reports
    finally:
        stop.set()
        thread.join(2)
        left.close()
        right.close()
        assert not thread.is_alive()


def test_real_client_repeated_exact_tickets_and_readonly_configuration(connected_client):
    client, _ = connected_client
    for index, role in enumerate(["setup", "resolve", "setup"]):
        ticket = client.arm(role)
        assert ticket.sequence == index and client.state == "ARMED"
        client.complete(ticket)
        assert client.state == "IDLE"
    for obj in (client, GuardSupervisorCore(config())):
        with pytest.raises(AttributeError):
            obj.configuration = config(session="b" * 32)


@pytest.mark.parametrize("fault", ["copy", "reconstructed", "foreign", "retired", "nested", "emergency"])
def test_live_ticket_misuse_is_sticky_before_wire(connected_client, fault):
    client, _ = connected_client
    ticket = client.arm("emergency" if fault == "emergency" else "setup")
    if fault == "copy":
        ticket = copy.copy(ticket)
    elif fault in {"reconstructed", "foreign"}:
        ticket = GuardWaitTicket(ticket.session, ticket.device_ordinal, ticket.role, ticket.sequence)
    elif fault == "retired":
        client.complete(ticket)
        client.arm("resolve")
    with pytest.raises(GuardProtocolError):
        client.arm("resolve") if fault == "nested" else client.complete(ticket)
    assert client.state == "FAILED"
    with pytest.raises(GuardProtocolError):
        client.arm("emergency")
    assert client.state == "FAILED"


def test_swallowed_nested_failure_cannot_reopen_client(monkeypatch):
    left, right = socket.socketpair()
    try:
        client = GuardWaitClient(left, config())

        def nested(*unused):
            with pytest.raises(GuardProtocolError):
                client.arm("resolve")

        monkeypatch.setattr(client, "_exchange", nested)
        with pytest.raises(GuardProtocolError, match="nested failure"):
            client.arm("setup")
        assert client.state == "FAILED"
    finally:
        left.close()
        right.close()


def test_clock_reentrancy_is_terminal_without_ack():
    core = None

    def clock():
        if core is not None:
            core.tick()
        return 100

    core = GuardSupervisorCore(config(), clock=clock)
    with pytest.raises(GuardProtocolError, match="reentrant"):
        core.tick()
    assert core.state == "TERMINATING" and core.kill_required
    assert core.pending_response is None


def test_partial_eof_at_every_frame_prefix_is_terminal():
    message = encode_framed_message(config().message("ARM", 0, "setup"), max_bytes=MAX_BODY_BYTES)
    for prefix in range(len(message)):
        c, _, core = setup_core()
        left, right = socket.socketpair()
        try:
            transport = GuardSupervisorTransport(right, core)
            if prefix:
                left.sendall(message[:prefix])
                transport.service(readable=True, writable=False)
            left.shutdown(socket.SHUT_WR)
            for _ in range(3):
                assert select.select([right], [], [], 1)[0]
                transport.service(readable=True, writable=True)
                if core.state == "TERMINATING":
                    break
            assert core.state == "TERMINATING"
            assert core.pending_response is None
            assert core.evidence().to_payload()["summary"]["arm_count"] == 0
        finally:
            left.close()
            right.close()


@pytest.mark.parametrize("event_index", range(6))
def test_each_success_transition_rejects_duplication(event_index):
    settings, _, core = setup_core()
    arm(settings, core)
    complete(settings, core)
    core.observe_exit(0)
    payload = core.evidence().to_payload()
    payload["trace"].insert(event_index, copy.deepcopy(payload["trace"][event_index]))
    for index, event in enumerate(payload["trace"]):
        event["index"] = index
    with pytest.raises(GuardProtocolError):
        GuardSupervisorEvidence.from_payload(payload)


def test_each_summary_field_is_derived_not_accepted():
    settings, _, core = setup_core()
    arm(settings, core)
    for name in core.evidence().to_payload()["summary"]:
        payload = core.evidence().to_payload()
        payload["summary"][name] = {"invented": True}
        with pytest.raises(GuardProtocolError):
            GuardSupervisorEvidence.from_payload(payload)


def test_pending_ack_send_failure_keeps_original_guard():
    settings, _, core = setup_core()
    left, right = socket.socketpair()
    try:
        transport = GuardSupervisorTransport(right, core)
        core.accept(settings.message("ARM", 0, "setup"))
        deadline = core.effective_deadline_ns
        # Exact real socket, OS error at the actual transport send entry.
        right.shutdown(socket.SHUT_WR)
        with pytest.raises(OSError):
            transport.service(readable=False, writable=True)
        assert core.state == "TERMINATING"
        assert core.effective_deadline_ns == deadline
        assert core.pending_response is None and not transport.wants_write
        assert core.evidence().to_payload()["summary"]["terminal_reason"] == "ack_write_failed"
    finally:
        left.close()
        right.close()


def test_swallowed_clock_reentry_cannot_return_arm_or_write_ack():
    settings = config()
    core = None
    nested = []

    def clock():
        if core is not None:
            try:
                core.accept(settings.message("ARM", 0, "resolve"))
            except GuardProtocolError as error:
                nested.append(str(error))
        return 100

    core = GuardSupervisorCore(settings, clock=clock)
    with pytest.raises(GuardProtocolError, match="reentrant"):
        core.accept(settings.message("ARM", 0, "setup"))
    assert nested
    assert core.state == "TERMINATING" and core.kill_required
    assert core.pending_response is None
    evidence = core.evidence().to_payload()
    assert evidence["summary"]["arm_count"] == 0
    assert not any(event["operation"] == "ack_written" for event in evidence["trace"])


@pytest.mark.parametrize("swallow", [False, True])
@pytest.mark.parametrize(
    "boundary",
    [
        "arm",
        "arm_ack",
        "complete",
        "disarm_ack",
        "terminal",
        "expiry",
        "kill",
        "kill_failure",
        "exit",
    ],
)
def test_signal_reentry_at_transition_commit_is_sticky_and_replayable(boundary, swallow):
    settings, now, core = setup_core()
    if boundary == "arm":
        operation = "arm_committed"
        outer = partial(core.accept, settings.message("ARM", 0, "setup"))
    else:
        pending = core.accept(settings.message("ARM", 0, "setup"))
        if boundary == "arm_ack":
            operation, outer = "ack_written", lambda: core.acknowledge_written(pending)
        else:
            core.acknowledge_written(pending)
            if boundary == "complete":
                operation = "normal_complete_received"
                outer = partial(core.accept, settings.message("COMPLETE", 0, "setup"))
            else:
                pending = core.accept(settings.message("COMPLETE", 0, "setup"))
                if boundary == "disarm_ack":
                    operation, outer = "ack_written", lambda: core.acknowledge_written(pending)
                elif boundary == "terminal":
                    operation, outer = "terminal_entered", lambda: core.fault("input_eof")
                elif boundary == "exit":
                    operation, outer = "process_exited", lambda: core.observe_exit(0)
                else:
                    now[0] = core.effective_deadline_ns
                    if boundary == "expiry":
                        operation, outer = "deadline_expired", core.tick
                    else:
                        core.tick()
                        if boundary == "kill":
                            operation, outer = "kill_requested", core.record_kill_request
                        else:
                            core.record_kill_request()
                            operation, outer = (
                                "kill_failed",
                                lambda: core.record_kill_failure("permission_denied"),
                            )
    prior_deadline = core.effective_deadline_ns
    source, start = inspect.getsourcelines(GuardSupervisorCore._append)
    line = start + next(
        index for index, text in enumerate(source) if "self._records.setdefault(index, record)" in text
    )
    injected = []

    def on_signal(*unused):
        injected.append(True)
        try:
            core.fault("parent_lost")
        except GuardProtocolError:
            if not swallow:
                raise

    def schedule(frame, event, arg):
        if (
            event == "line"
            and frame.f_code is GuardSupervisorCore._append.__code__
            and frame.f_locals.get("operation") == operation
            and frame.f_lineno == line
        ):
            sys.settrace(None)
            if hasattr(signal, "SIGUSR1"):
                signal.raise_signal(signal.SIGUSR1)
            else:
                on_signal()
        return schedule

    previous_trace = sys.gettrace()
    previous_handler = signal.signal(signal.SIGUSR1, on_signal) if hasattr(signal, "SIGUSR1") else None
    sys.settrace(schedule)
    try:
        with pytest.raises(GuardProtocolError, match="reentrant"):
            outer()
    finally:
        sys.settrace(previous_trace)
        if previous_handler is not None:
            signal.signal(signal.SIGUSR1, previous_handler)
    assert injected == [True]
    assert core.state == "TERMINATING" and core.pending_response is None
    if prior_deadline is not None:
        assert core.effective_deadline_ns <= prior_deadline
    snapshot = core.evidence().to_payload()
    assert snapshot["summary"]["normal_complete_count"] == 0
    assert snapshot["summary"]["terminal_reason"] in {"parent_lost", "deadline_expired"}
    assert snapshot["summary"]["observed_returncode"] is None
    with pytest.raises(GuardProtocolError):
        core.accept(settings.message("ARM", 1, "resolve"))
    now[0] = core.effective_deadline_ns
    core.tick()
    if core.kill_required:
        core.record_kill_request()
    core.observe_exit(-9)
    assert core.evidence(require_final=True).to_payload()["summary"]["complete"]


def test_swallowed_transport_reentry_prevents_pending_ack_output():
    settings, _, core = setup_core()
    left, right = socket.socketpair()
    try:
        transport = GuardSupervisorTransport(right, core)
        core.accept(settings.message("ARM", 0, "setup"))
        deadline = core.effective_deadline_ns
        source, start = inspect.getsourcelines(GuardSupervisorTransport._service)
        line = start + next(i for i, text in enumerate(source) if "self.core.tick()" in text)
        injected = []

        def schedule(frame, event, arg):
            if (
                event == "line"
                and frame.f_code is GuardSupervisorTransport._service.__code__
                and frame.f_lineno == line
            ):
                sys.settrace(None)
                with pytest.raises(GuardProtocolError, match="reentrant"):
                    transport.service(readable=False, writable=True)
                injected.append(True)
            return schedule

        previous = sys.gettrace()
        sys.settrace(schedule)
        try:
            transport.service(readable=False, writable=True)
        finally:
            sys.settrace(previous)
        assert injected == [True]
        assert core.state == "TERMINATING" and core.pending_response is None
        assert core.effective_deadline_ns == deadline
        left.settimeout(1)
        assert left.recv(1024) == b""
        core.evidence()
    finally:
        left.close()
        right.close()


def test_reentry_at_each_disarm_commit_and_finalizer_line_never_loses_terminal_state():
    """CPU publication/return-window regression, not just one earlier line."""
    functions = (
        GuardSupervisorCore._invoke,
        GuardSupervisorCore._append,
        GuardSupervisorCore._acknowledge_written,
        GuardSupervisorCore._now,
        GuardSupervisorCore._state.fget,
    )
    checked = set()
    for function in functions:
        for line in {line for _, line in dis.findlinestarts(function.__code__)}:
            settings, _, core = setup_core()
            arm(settings, core)
            response = core.accept(settings.message("COMPLETE", 0, "setup"))
            injected = []

            def on_signal(*unused):
                injected.append(True)
                try:
                    core.fault("parent_lost")
                except GuardProtocolError:
                    pass  # the exact missed-finalizer counterexample

            def schedule(frame, event, arg):
                if event == "line" and frame.f_code is function.__code__ and frame.f_lineno == line:
                    sys.settrace(None)
                    if hasattr(signal, "SIGUSR1"):
                        signal.raise_signal(signal.SIGUSR1)
                    else:
                        on_signal()
                return schedule

            previous = sys.gettrace()
            handler = signal.signal(signal.SIGUSR1, on_signal) if hasattr(signal, "SIGUSR1") else None
            sys.settrace(schedule)
            try:
                try:
                    core.acknowledge_written(response)
                except GuardProtocolError:
                    pass
            finally:
                sys.settrace(previous)
                if handler is not None:
                    signal.signal(signal.SIGUSR1, handler)
            if not injected:
                continue  # branches not reached by a normal DISARMED commit
            checked.add((function.__name__, line))
            assert injected == [True]
            assert core.state == "TERMINATING", (function.__name__, line, core.state)
            assert core.pending_response is None and core.effective_deadline_ns is not None
            result = core.evidence().to_payload()
            assert result["summary"]["terminal_reason"] == "parent_lost"
            # Before the single commit point the old guard survives; after it,
            # DISARMED legitimately completed, then parent loss creates a new
            # terminal grace deadline. Neither ordering can end IDLE.
            assert result["summary"]["normal_complete_count"] in (0, 1)
    finalizer, start = inspect.getsourcelines(GuardSupervisorCore._invoke)
    release_line = start + next(
        i for i, value in enumerate(finalizer) if "self._operation_active = False" in value
    )
    assert ("_invoke", release_line) in checked
    assert {name for name, _ in checked} == {function.__name__ for function in functions}
    source, start = inspect.getsourcelines(GuardSupervisorCore._append)
    commit_line = start + next(i for i, value in enumerate(source) if "self._records.setdefault" in value)
    assert ("_append", commit_line) in checked


@pytest.mark.parametrize("phase", ["arm", "complete"])
@pytest.mark.parametrize("boundary", ["publication", "finalizer"])
@pytest.mark.parametrize("action", ["nested", "fail", "close"])
def test_client_failure_cannot_be_overwritten_after_exchange(connected_client, phase, boundary, action):
    client, _ = connected_client
    ticket = client.arm("setup") if phase == "complete" else None
    method = GuardWaitClient._commit if boundary == "publication" else getattr(GuardWaitClient, phase)
    source, start = inspect.getsourcelines(method)
    marker = "self._states.setdefault" if boundary == "publication" else "self._operation_active = False"
    line = start + next(i for i, text in enumerate(source) if marker in text)
    injected = []

    def on_signal(*unused):
        try:
            if action == "nested":
                client.arm("resolve")
            elif action == "fail":
                client.fail()
            else:
                client.close()
        except GuardProtocolError:
            pass
        injected.append(client.state)

    def schedule(frame, event, arg):
        if event == "line" and frame.f_code is method.__code__ and frame.f_lineno == line:
            if boundary == "publication" and frame.f_locals["record"][0] != (
                "ARMED" if phase == "arm" else "IDLE"
            ):
                return schedule
            sys.settrace(None)
            signal.raise_signal(signal.SIGUSR1) if hasattr(signal, "SIGUSR1") else on_signal()
        return schedule

    previous = sys.gettrace()
    handler = signal.signal(signal.SIGUSR1, on_signal) if hasattr(signal, "SIGUSR1") else None
    returned = None
    sys.settrace(schedule)
    try:
        try:
            returned = client.arm("setup") if phase == "arm" else client.complete(ticket)
        except GuardProtocolError:
            pass
    finally:
        sys.settrace(previous)
        if handler is not None:
            signal.signal(signal.SIGUSR1, handler)
    assert injected == ["FAILED"] and client.state == "FAILED"
    if boundary == "publication":
        assert returned is None  # failure won before ticket publication
    if returned is not None:
        # Publication may win first; a later failure retires that ticket and
        # cannot make it usable by any subsequent protocol operation.
        with pytest.raises(GuardProtocolError):
            client.complete(returned)
    with pytest.raises(GuardProtocolError):
        client.arm("resolve")
    assert client.state == "FAILED"


@pytest.mark.parametrize("operation", ["ARMED", "DISARMED"])
@pytest.mark.parametrize("fault", ["nested_transport", "parent_lost"])
def test_terminal_kernel_write_fence_at_actual_send(operation, fault):
    settings, _, core = setup_core()
    left, right = socket.socketpair()
    try:
        transport = GuardSupervisorTransport(right, core)
        response = core.accept(settings.message("ARM", 0, "setup"))
        if operation == "DISARMED":
            core.acknowledge_written(response)
            core.accept(settings.message("COMPLETE", 0, "setup"))
        deadline = core.effective_deadline_ns
        source, start = inspect.getsourcelines(GuardSupervisorTransport._service)
        line = start + next(i for i, text in enumerate(source) if "sent = self.channel.send(" in text)
        injected = []

        def on_signal(*unused):
            try:
                if fault == "parent_lost":
                    core.fault("parent_lost")
                else:
                    transport.service(readable=False, writable=True)
            except GuardProtocolError:
                pass
            injected.append(core.state)

        def schedule(frame, event, arg):
            if (
                event == "line"
                and frame.f_code is GuardSupervisorTransport._service.__code__
                and frame.f_lineno == line
            ):
                sys.settrace(None)
                signal.raise_signal(signal.SIGUSR1) if hasattr(signal, "SIGUSR1") else on_signal()
            return schedule

        previous = sys.gettrace()
        handler = signal.signal(signal.SIGUSR1, on_signal) if hasattr(signal, "SIGUSR1") else None
        sys.settrace(schedule)
        try:
            with pytest.raises(OSError):
                transport.service(readable=False, writable=True)
        finally:
            sys.settrace(previous)
            if handler is not None:
                signal.signal(signal.SIGUSR1, handler)
        left.settimeout(1)
        assert left.recv(1028) == b""  # real peer, zero bytes then EOF
        assert injected == ["TERMINATING"]
        assert core.effective_deadline_ns == deadline
        result = core.evidence().to_payload()
        assert result["summary"]["normal_complete_count"] == 0
        assert [
            event["details"]["operation"] for event in result["trace"] if event["operation"] == "ack_written"
        ] == ([] if operation == "ARMED" else ["ARMED"])
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("operation", ["ARMED", "DISARMED"])
def test_already_written_prefix_survives_but_no_continuation_after_terminal(operation):
    settings, _, core = setup_core()
    left, right = socket.socketpair()
    try:
        transport = GuardSupervisorTransport(right, core)
        response = core.accept(settings.message("ARM", 0, "setup"))
        if operation == "DISARMED":
            core.acknowledge_written(response)
            response = core.accept(settings.message("COMPLETE", 0, "setup"))

        class OneByteWrite:
            # A fault-injection proxy around the real socket send, not evidence
            # that tiny frames naturally short-write on this OS. Real blocked
            # socket writes are covered by the process harness separately.
            def send(self, value):
                return right.send(value[:1])

        transport.channel = OneByteWrite()
        transport.service(readable=False, writable=True)
        assert core.pending_response == response
        original = core.effective_deadline_ns
        core.fault("parent_lost")
        transport.service(readable=False, writable=True)
        left.settimeout(1)
        expected = encode_framed_message(response, max_bytes=MAX_BODY_BYTES)
        assert left.recv(1028) == expected[:1]
        assert left.recv(1028) == b""
        assert core.effective_deadline_ns == original
        result = core.evidence().to_payload()
        assert result["summary"]["normal_complete_count"] == 0
        assert transport._output == b""
    finally:
        left.close()
        right.close()


def test_output_fence_binding_is_single_use_before_any_transition():
    settings, _, core = setup_core()
    left, right = socket.socketpair()
    try:
        GuardSupervisorTransport(right, core)
        with pytest.raises(GuardProtocolError, match="binds once"):
            GuardSupervisorTransport(right, core)
        late = GuardSupervisorCore(settings)
        late.accept(settings.message("ARM", 0, "setup"))
        with pytest.raises(GuardProtocolError, match="binds once"):
            GuardSupervisorTransport(right, late)
    finally:
        left.close()
        right.close()
