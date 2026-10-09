"""CPU exception ordering in the production process adapter; no OS job proof."""

from pathlib import Path
from types import SimpleNamespace

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
