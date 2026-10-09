"""CPU exception ordering in the production process adapter; no OS job proof."""

from pathlib import Path
from types import SimpleNamespace
import errno

import pytest

from optical_worker import compiler_supervision as supervision
from optical_worker.compiler_job import _CompilerObligation
from optical_worker.wait_protocol import GuardConfiguration
from optical_worker.wait_supervision import GuardSupervisorCore


@pytest.mark.parametrize("forked", [False, True])
def test_partial_popen_is_held_before_constructor_can_fail(tmp_path, monkeypatch, forked):
    job = object.__new__(supervision._CompilerJob)
    job.process, job.transport_channel = None, None
    job.launch_errors = []
    failures = []
    job.obligation = SimpleNamespace(observation_failed=failures.append)
    job._check_owner = lambda: None
    job._anchor_worker = lambda: None
    job._prepare_pipes = lambda: None
    original = OSError("injected constructor failure; no real fork")

    def initialize(process, *args, **kwargs):
        assert job.process is process  # The old assignment-after-construction loses it.
        process._child_created, process.returncode = forked, 0
        process.pid = 200
        raise original

    monkeypatch.setattr(supervision.subprocess.Popen, "__init__", initialize)
    request = tmp_path / "request.json"
    request.write_bytes(b"{}")
    arguments = dict(repository_root=Path(__file__).resolve().parents[3], request_path=request,
                     output_directory=tmp_path, environment={})
    try:
        if forked:
            assert job.launch(**arguments) is job.transport_channel
            assert failures == ["other_os"]
        else:
            with pytest.raises(OSError) as raised:
                job.launch(**arguments)
            assert raised.value is original and not failures
        assert job.process is not None and job.launch_errors == [original]
    finally:
        if job.transport_channel is not None:
            job.transport_channel.close()


@pytest.mark.parametrize("close_fails", [False, True])
def test_pidfd_close_failure_retains_draining_owner_and_is_not_retried(monkeypatch, close_fails):
    core = GuardSupervisorCore(GuardConfiguration("a" * 32, 1, 1000, 120_000), clock=lambda: 100)
    worker = {"pid": 200, "starttime_ticks": 2, "sid": 200}
    job = object.__new__(supervision._CompilerJob)
    job.core = core
    job.obligation = _CompilerObligation(core, {"pid": 100, "starttime_ticks": 1, "sid": 100})
    job.obligation.subreaper_ready()
    job.obligation.spawned(worker)
    job.obligation.reaped(worker, 0)
    job.obligation.no_children()
    job.obligation.pipes_eof()
    identity = job.obligation.replay.worker
    job.handles = {identity: 12345}  # Never passed to an OS function.
    job._fd_close_attempted = set()
    job._pipe_setup_complete, job._echild_observed, job._servicing_due = True, True, False
    job.channels = {}
    job._check_owner = job._anchor_worker = job._prepare_pipes = job._discover = lambda: None
    job._reap_one = lambda: False
    attempts = []
    original_close = supervision.os.close

    def close(fd):
        if fd != 12345:
            return original_close(fd)
        attempts.append(fd)
        if close_fails:
            raise OSError("modeled close failure")

    monkeypatch.setattr(supervision.os, "close", close)
    exited_core = core.evidence().to_payload()
    job.service_processes()
    job.service_processes()
    assert attempts == [12345]
    assert core.evidence().to_payload() == exited_core
    if close_fails:
        assert job.handles == {identity: 12345} and job.obligation.replay.state == "DRAINING"
        assert job.obligation.replay.failed
        assert not any(event["operation"] == "closed" for event in job.obligation.events)
    else:
        assert job.handles == {} and job.obligation.replay.state == "CLOSED"
        job.obligation.record(require_closed=True)


@pytest.mark.parametrize('error_number', [errno.ESRCH, errno.EMFILE, errno.EPERM])
def test_fixed_libc_pidfd_error_retains_errno_without_numeric_pid_fallback(monkeypatch, error_number):
    policy = object.__new__(supervision._SubreaperPolicy)
    calls = []
    def failed(pid, flags):
        calls.append((pid, flags))
        return -1
    policy._pidfd_open = failed  # CPU C-return model; no CDLL or syscall.
    monkeypatch.setattr(supervision.ctypes, 'get_errno', lambda: error_number)
    with pytest.raises(OSError) as raised:
        policy.open_pidfd(200)
    assert raised.value.errno == error_number and calls == [(200, 0)]


@pytest.mark.parametrize('pid', [0, -1, 2**31, True, 1.0])
def test_fixed_libc_pidfd_rejects_invalid_pid_before_call(pid):
    policy = object.__new__(supervision._SubreaperPolicy)
    policy._pidfd_open = lambda *args: pytest.fail('invalid PID reached native call')
    with pytest.raises(supervision.GuardProtocolError, match='invalid child PID'):
        policy.open_pidfd(pid)


@pytest.mark.parametrize('after', ['same', 'changed', 'missing'])
def test_pidfd_acquisition_uses_fixed_libc_and_keeps_before_after_identity_checks(monkeypatch, after):
    job = object.__new__(supervision._CompilerJob)
    job.pid, job.handles = 100, {}
    job.policy = object.__new__(supervision._SubreaperPolicy)
    identity = (200, 2, 200)
    calls = []
    reads = 0
    def read(pid):
        nonlocal reads
        reads += 1
        calls.append(('read', pid))
        if reads == 2 and after == 'missing':
            raise FileNotFoundError(errno.ENOENT, 'CPU proc model')
        observed = (200, 3, 200) if reads == 2 and after == 'changed' else identity
        return observed, 100, 'S'
    def native_open(pid, flags):
        calls.append(('open', pid, flags))
        return 12345  # Never used as an actual OS descriptor.
    original_close = supervision.os.close
    def close(fd):
        if fd == 12345:
            calls.append(('close', fd))
        else:
            original_close(fd)
    monkeypatch.delattr(supervision.os, 'pidfd_open', raising=False)
    monkeypatch.setattr(supervision, '_proc_identity', read)
    monkeypatch.setattr(supervision.os, 'close', close)
    job.policy._pidfd_open = native_open
    if after == 'same':
        job._acquire(identity)
        assert job.handles == {identity: 12345}
    else:
        with pytest.raises((supervision.GuardProtocolError, FileNotFoundError)):
            job._acquire(identity)
        assert job.handles == {}
    assert calls == [('read', 200), ('open', 200, 0), ('read', 200)] + ([] if after == 'same' else [('close', 12345)])
