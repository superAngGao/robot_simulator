"""Actual framed transport/owner interception with CPU-only OS observations.

The exact owner is constructed without its native constructor; mocked process
facts prove transport ordering only, never subreaper, identity or reaping safety.
"""

import socket

import pytest

from optical_worker.compiler_job import _CompilerObligation
from optical_worker.compiler_supervision import _CompilerJob, _LIVE_JOBS
from optical_worker.protocol import encode_framed_message
from optical_worker.wait_protocol import GuardConfiguration, GuardFrameDecoder, GuardProtocolError, MAX_BODY_BYTES
from optical_worker.wait_supervision import GuardSupervisorCore, GuardSupervisorTransport


@pytest.fixture
def transport(monkeypatch):
    config = GuardConfiguration("b" * 32, 1, 1000, 120_000)
    core = GuardSupervisorCore(config, clock=lambda: 100)
    parent, worker = socket.socketpair()
    worker.settimeout(1)
    job = object.__new__(_CompilerJob)
    job.core = core
    job.obligation = _CompilerObligation(core, {"pid": 100, "sid": 100, "starttime_ticks": 1})
    job.obligation.subreaper_ready()
    identity = {"pid": 200, "sid": 200, "starttime_ticks": 2}
    job.obligation.spawned(identity)
    job.process = object()
    job.handles = {job.obligation.replay.worker: -1}  # No native method may use this CPU sentinel.
    job.observed_members = {}
    _LIVE_JOBS[job] = core
    monkeypatch.setattr(_CompilerJob, "_check_owner", lambda self: None)
    monkeypatch.setattr(_CompilerJob, "_discover", lambda self: None)
    monkeypatch.setattr(_CompilerJob, "_reap_one", lambda self: False)
    monkeypatch.setattr(_CompilerJob, "_sample_session", lambda self: {self.obligation.replay.worker})
    channel = GuardSupervisorTransport(parent, core, compiler_job=job)
    try:
        yield config, core, job, channel, worker
    finally:
        _LIVE_JOBS.pop(job, None)
        parent.close()
        worker.close()


def send(fixture, operation, sequence, role="setup"):
    config, core, job, transport, worker = fixture
    worker.sendall(encode_framed_message(config.message(operation, sequence, role), max_bytes=MAX_BODY_BYTES))
    transport.service(readable=True, writable=True)
    return GuardFrameDecoder().feed(worker.recv(MAX_BODY_BYTES + 4))


def input_prefix(fixture):
    for sequence in range(6):
        assert send(fixture, "ARM", sequence)["operation"] == "ARMED"
        assert send(fixture, "COMPLETE", sequence)["operation"] == "DISARMED"


def test_exact_three_compile_completes_are_intercepted_before_core_accept(transport):
    input_prefix(transport)
    _, core, job, _, _ = transport
    for sequence in (6, 7, 8):
        send(transport, "ARM", sequence)
        before = len(core._records)
        assert send(transport, "COMPLETE", sequence)["operation"] == "DISARMED"
        events = job.obligation.events
        q = next(e for e in events if e["operation"] == "compile_quiescent" and e["details"]["sequence"] == sequence)
        after = next(e for e in events if e["operation"] == "deadline_snapshot" and e["details"]["guard_event_index"] == before)
        assert q["index"] < after["index"]
        assert core._records[before][1]["operation"] == "normal_complete_received"
    assert send(transport, "ARM", 9, "emergency")["operation"] == "ARMED"
    job.obligation.record()


def test_actual_transport_withholds_disarmed_for_residual(transport, monkeypatch):
    input_prefix(transport)
    _, core, job, _, worker = transport
    send(transport, "ARM", 6)
    monkeypatch.setattr(_CompilerJob, "_sample_session", lambda self: {self.obligation.replay.worker, (201, 3, 200)})
    with pytest.raises(GuardProtocolError, match="descendant"):
        send(transport, "COMPLETE", 6)
    assert core.state == "TERMINATING" and core.pending_response is None
    assert not job.obligation.replay.quiescent
    assert not any(e[1]["operation"] == "normal_complete_received" and e[1]["details"]["sequence"] == 6 for e in core._records.values())
    assert worker.recv(1) == b""  # The original output fence, not a fabricated ACK.


@pytest.mark.parametrize("operation,sequence,role", [("ARM", 7, "setup"), ("COMPLETE", 7, "setup"),
                                                     ("ARM", 10, "setup"), ("ARM", 6, "resolve")])
def test_shifted_unknown_and_wrong_role_never_receive_success(transport, operation, sequence, role):
    input_prefix(transport)
    with pytest.raises((ValueError, GuardProtocolError)):
        send(transport, operation, sequence, role)
    assert transport[1].state == "TERMINATING"


def test_transport_rejects_unregistered_arbitrary_callback_owner():
    parent, child = socket.socketpair()
    try:
        core = GuardSupervisorCore(GuardConfiguration("c" * 32, 1, 1000, 120_000))
        with pytest.raises(GuardProtocolError, match="registered"):
            GuardSupervisorTransport(parent, core, compiler_job=lambda: True)
    finally:
        parent.close()
        child.close()


def test_process_fixture_withholds_real_pending_ack_before_worker_exit(transport, tmp_path):
    from tests.support.compiler_process_cases import Observer, read

    input_prefix(transport)
    config, core, job, channel, worker = transport
    observer = object.__new__(Observer)
    observer.job, observer.directory, observer.case = job, tmp_path, "leader_ack_exit"
    observer.exited_core = None
    worker.sendall(encode_framed_message(config.message("ARM", 6, "setup"), max_bytes=MAX_BODY_BYTES))
    channel.service(readable=True, writable=False)
    response = dict(core.pending_response)
    deadline = core.effective_deadline_ns
    assert observer.output_ready(True) is False
    channel.service(readable=False, writable=observer.output_ready(True))
    assert core.pending_response == response and read(tmp_path / 'ack-pending.json') == response
    assert not any(e[1]['operation'] == 'ack_written' and e[1]['details']['sequence'] == 6
                   for e in core._records.values())
    from optical_worker.compiler_job import identity_payload

    job.obligation.reaped(identity_payload(job.obligation.replay.worker), 0)
    observer.observe()
    assert core.state == 'EXITED' and job.obligation.replay.deadline_source == 'ack'
    assert job.obligation.replay.deadline == deadline
    observer.observe()


def test_invalid_process_fixture_rejected_before_native_job(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from tests.support import cuda_compilation_process as driver
    from tests.support.compiler_process_cases import LIBC, SCHEMA

    config = GuardConfiguration('a' * 32, 1, 1000, 120_000)  # Not the OS fixture policy.
    request = {'schema_version': SCHEMA, 'root': str(Path(driver.__file__).resolve().parents[2]),
               'revision': 'a' * 40, 'lane': 'cuda', 'guard_configuration': config.to_payload(),
               'process_case': 'normal'}
    (tmp_path / 'request.json').write_text(json.dumps(request))
    policy = {'schema_version': 'q54.p12.4.compiler-supervisor-launch.v1', 'libc_file': LIBC,
              'worker_environment': {}}
    (tmp_path / 'policy.json').write_text(json.dumps(policy))
    def native_forbidden(*args, **kwargs):
        pytest.fail('malformed fixture reached native policy')
    monkeypatch.setattr(_CompilerJob, '__init__', native_forbidden)
    with pytest.raises(AssertionError):
        driver.supervise(-1, tmp_path / 'request.json', tmp_path, tmp_path / 'policy.json')
