"""Finite Linux process fixtures for the real compiler owner, never GPU work.

Actual execution requires the exact source/runner/native-ABI preflight. Worker
guards0..5 are empty wire fixtures, not captured input or compilation evidence.
Only the dedicated supervisor constructs the production subreaper/job owner.
"""

import errno
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


SCHEMA = "q54.compiler-process.cpu-request.v0"
CASES = ("normal", "separate_pgid", "leader_exit", "leader_ack_exit", "double_fork",
         "fork_during_cleanup", "pipe_hang", "missing_complete", "residual_complete",
         "proc_failure", "handle_failure", "stale_identity", "disappearing_identity",
         "signal_failure", "parent_lost")
LIBC = {"path": "/usr/lib/x86_64-linux-gnu/libc.so.6", "size": 2125328,
        "sha256": "d8db8739a1633c972cec6a4fe0566bdcec6fd088f98723492ab0361f66238f75"}
ROOT = Path(__file__).resolve().parents[2]


def write(directory, name, value):
    path = directory / name
    # Readers use these tiny files as fixture barriers. Publish whole JSON.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_bytes())


def validate(request):
    assert set(request) == {"schema_version", "root", "revision", "lane", "guard_configuration", "process_case"}
    assert request["schema_version"] == SCHEMA and request["lane"] == "cuda"
    assert request["process_case"] in CASES and Path(request["root"]).resolve(strict=True) == ROOT
    config = request["guard_configuration"]
    # Short, fixed OS-fixture deadlines; no actual CUDA compilation uses them.
    assert config["ack_timeout_ms"] == 1000 and config["operation_timeout_ms"] == 2000
    assert config["device_ordinal"] == 1


def _wait_for(path, predicate=lambda value: True, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = read(path)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        else:
            if predicate(value):
                return value
        time.sleep(0.002)
    raise TimeoutError("fixture barrier: " + str(path))


def _member(directory, role):
    value = {"pid": os.getpid(), "ppid": os.getppid(), "sid": os.getsid(0), "pgid": os.getpgrp(),
             "role": role, "at_ns": time.monotonic_ns()}
    write(directory, "member-" + str(os.getpid()) + ".json", value)
    return value


def _natural_exit(directory):
    # A denied test signal must not leave an immortal fixture. This is a
    # predeclared natural exit, not a second supervisor or cleanup kill policy.
    time.sleep(5)
    write(directory, "natural-exit-" + str(os.getpid()) + ".json", {"at_ns": time.monotonic_ns()})
    os._exit(0)


def _fork_member(directory, channel, case):
    pid = os.fork()
    if pid:
        return pid
    try:
        channel.close()  # A descendant never retains or uses the guard socket.
        os.setpgid(0, 0)
        if case != "pipe_hang":
            os.close(1)
            os.close(2)
        _member(directory, "child")
        if case == "separate_pgid":
            _wait_for(directory / ("sampled-" + str(os.getpid()) + ".json"))
            os._exit(0)
        if case in {"double_fork", "fork_during_cleanup"}:
            if case == "fork_during_cleanup":
                _wait_for(directory / "compiler_job_prefix.json", lambda p: p["summary"]["state"] == "DRAINING")
            grandchild = os.fork()
            if grandchild:
                write(directory, "grandchild.json", {"pid": grandchild, "parent": os.getpid()})
                os._exit(0)
            os.setpgid(0, 0)
            _member(directory, "grandchild")
        _natural_exit(directory)
    except BaseException as error:
        write(directory, "member-failure-" + str(os.getpid()) + ".json",
              {"type": type(error).__name__, "error": str(error)})
        os._exit(75)


def worker(fd, directory, request):
    validate(request)
    sys.path.insert(0, str(ROOT))
    from optical_worker.wait_protocol import GuardConfiguration, GuardWaitClient, MAX_BODY_BYTES
    from optical_worker.protocol import encode_framed_message

    case = request["process_case"]
    channel = socket.socket(fileno=fd)
    client = GuardWaitClient(channel, GuardConfiguration.from_payload(request["guard_configuration"]))
    _member(directory, "worker")
    try:
        for sequence in range(6):
            assert client.next_guard_sequence == sequence
            client.complete(client.arm("setup"))
        if case == "leader_ack_exit":
            # Real ARM, deliberately no successful client ACK. The observer
            # suppresses output readiness and records the actual pending core.
            channel.setblocking(True)
            channel.sendall(encode_framed_message(client.configuration.message("ARM", 6, "setup"), max_bytes=MAX_BODY_BYTES))
            _wait_for(directory / "ack-pending.json")
        else:
            ticket = client.arm("setup")
            assert ticket.sequence == 6
        write(directory, "worker-armed.json", {"pid": os.getpid(), "at_ns": time.monotonic_ns()})
        if case in {"normal", "separate_pgid"}:
            if case == "separate_pgid":
                child = _fork_member(directory, channel, case)
                observed, status = os.waitpid(child, 0)
                assert observed == child and status == 0
                write(directory, "worker-wait.json", {"pid": child, "wait_status": status})
            client.complete(ticket)
            for sequence in (7, 8):
                assert client.next_guard_sequence == sequence
                client.complete(client.arm("setup"))
            assert client.arm("emergency").sequence == 9
            write(directory, "worker-finished.json", {"normal_completions": 9})
            os._exit(0)
        if case in {"leader_exit", "leader_ack_exit", "double_fork", "fork_during_cleanup", "pipe_hang", "residual_complete"}:
            child = _fork_member(directory, channel, case)
            _wait_for(directory / ("member-" + str(child) + ".json"))
            if case == "double_fork":
                _wait_for(directory / "grandchild.json")
                observed, status = os.waitpid(child, 0)
                assert observed == child and status == 0
                write(directory, "worker-wait.json", {"pid": child, "wait_status": status})
            if case == "residual_complete":
                client.complete(ticket)
                raise AssertionError("residual descendant received COMPLETE ACK")
            os._exit(0)
        _natural_exit(directory)
    except BaseException as error:
        write(directory, "worker-fixture-error.json", {"type": type(error).__name__, "error": str(error)})
        os._exit(74)


class Observer:
    """Test-only observation/finite OS-error injection around the real owner."""

    def __init__(self, job, directory, request):
        validate(request)
        from optical_worker import compiler_supervision as production

        self.job, self.directory, self.case = job, directory, request["process_case"]
        self.production, self.injections, self.read_count = production, [], 0
        self.original_identity = production._proc_identity
        self.original_open = job.policy.open_pidfd
        self.original_signal = production.signal.pidfd_send_signal
        self.original_sample = job._sample_session
        self.pidfds = []
        self.exited_core = None
        production._proc_identity = self.identity
        job.policy.open_pidfd = self.open_handle
        production.signal.pidfd_send_signal = self.signal
        job._sample_session = self.sample

    def inject(self, operation):
        self.injections.append({"operation": operation, "at_ns": time.monotonic_ns()})

    def identity(self, pid):
        result = self.original_identity(pid)
        if self.job.process is not None and pid == self.job.process.pid:
            self.read_count += 1
            if self.case in {"stale_identity", "disappearing_identity"} and self.read_count == 3:
                self.inject(self.case)
                if self.case == "disappearing_identity":
                    raise FileNotFoundError(errno.ENOENT, "injected proc disappearance")
                identity, parent, state = result
                return (identity[0], identity[1] + 1, identity[2]), parent, state
            if self.case == "proc_failure" and self.job.obligation.replay.expired and not self.injections:
                self.inject("proc_failure")
                raise PermissionError(errno.EACCES, "injected proc read failure")
        return result

    def open_handle(self, pid):
        if self.case == "handle_failure" and not self.injections:
            self.inject("handle_failure")
            raise OSError(errno.EMFILE, "injected pidfd exhaustion")
        fd = self.original_open(pid)
        self.pidfds.append({"pid": pid, "fd": fd,
                            "fdinfo": Path('/proc/self/fdinfo', str(fd)).read_text()})
        return fd

    def signal(self, fd, signal_number, siginfo=None, flags=0):
        if self.case == "signal_failure" and not self.injections:
            self.inject("signal_failure")
            raise PermissionError(errno.EPERM, "injected pidfd signal denial")
        return self.original_signal(fd, signal_number, siginfo, flags)

    def sample(self):
        members = self.original_sample()
        for identity in members:
            if identity != self.job.obligation.replay.worker:
                write(self.directory, "sampled-" + str(identity[0]) + ".json", {"identity": list(identity)})
        return members

    def output_ready(self, ready):
        pending = self.job.core.pending_response
        suppress = self.case == "leader_ack_exit" and pending is not None and pending["sequence"] == 6
        if suppress:
            write(self.directory, "ack-pending.json", dict(pending))
        return ready and not suppress

    def observe(self):
        if self.job.core.state == "EXITED":
            current = self.job.core.evidence(require_final=True).to_payload()
            if self.exited_core is None:
                self.exited_core = current
            assert current == self.exited_core, "exited worker core was mutated during descendant cleanup"

    def finish(self):
        self.observe()
        assert not self.job.handles and not self.job.channels
        for row in self.pidfds:
            try:
                os.fstat(row['fd'])
            except OSError as error:
                assert error.errno == errno.EBADF
            else:
                raise AssertionError("fixture pidfd remains open after CLOSED")
        self.production._proc_identity = self.original_identity
        self.job.policy.open_pidfd = self.original_open
        self.production.signal.pidfd_send_signal = self.original_signal
        self.job._sample_session = self.original_sample
        return {"case": self.case, "injections": self.injections, "pidfds": self.pidfds,
                "exited_core": self.exited_core,
                "scope": "real OS ownership, empty input guards, no compiler or GPU"}


def _identity(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return {"pid": pid, "starttime_ticks": int(fields[19]), "sid": int(fields[3])}


def parent(directory, case):
    assert sys.platform == "linux" and sys.version_info[:3] == (3, 12, 13)
    assert case in CASES and directory.is_absolute() and directory.resolve() == directory
    sys.path.insert(0, str(ROOT))
    from optical_worker.wait_protocol import GuardConfiguration
    from tests.support.setuptools_dependency_process import _create_run_directory
    import uuid

    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    _create_run_directory(directory)
    config = GuardConfiguration(uuid.uuid4().hex, 1, 1000, 2000)
    request = {"schema_version": SCHEMA, "root": str(ROOT), "revision": revision, "lane": "cuda",
               "guard_configuration": config.to_payload(), "process_case": case}
    write(directory, "request.json", request)
    environment = {"PATH": "/usr/bin:/bin", "HOME": str(directory), "LC_ALL": "C"}
    policy = {"schema_version": "q54.p12.4.compiler-supervisor-launch.v1", "libc_file": LIBC,
              "worker_environment": environment}
    write(directory, "policy.json", policy)
    sentinel_read, sentinel_write = os.pipe()
    sentinel = subprocess.Popen([sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()), "sentinel", str(sentinel_read)],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                pass_fds=(sentinel_read,), start_new_session=True)
    os.close(sentinel_read)
    sentinel_before = _identity(sentinel.pid)
    live_read, live_write = os.pipe()
    process = None
    try:
        with (directory / "supervisor.stdout").open("wb") as out, (directory / "supervisor.stderr").open("wb") as err:
            process = subprocess.Popen([sys.executable, "-I", "-S", "-B", str(ROOT / "tests/support/cuda_compilation_process.py"),
                                        "supervisor", str(live_read), str(directory / "request.json"), str(directory), str(directory / "policy.json")],
                                       stdin=subprocess.DEVNULL, stdout=out, stderr=err, pass_fds=(live_read,), start_new_session=True)
            os.close(live_read)
            live_read = None
            write(directory, "parent-identity.json", {"pid": os.getpid(), "supervisor_pid": process.pid,
                                                      "sentinel": sentinel_before})
            deadline = time.monotonic() + 30
            while True:
                if case == "parent_lost" and live_write is not None and (directory / "worker-armed.json").exists():
                    os.close(live_write)
                    live_write = None
                try:
                    code = process.wait(timeout=0.05)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        write(directory, "incomplete.json", {"supervisor_pid": process.pid,
                                                             "reason": "observation timeout; owner left alive"})
                        raise  # Never kill the wait owner or claim its children reaped.
        assert _identity(sentinel.pid) == sentinel_before and sentinel.poll() is None
        write(directory, "parent.json", {"pid": os.getpid(), "supervisor_pid": process.pid,
                                         "supervisor_exit": code, "sentinel_before": sentinel_before,
                                         "sentinel_after": _identity(sentinel.pid)})
        assert code == 0, "supervisor failed; retain raw artifacts"
        validate_result(directory)
    finally:
        if live_read is not None:
            os.close(live_read)
        if live_write is not None:
            os.close(live_write)
        os.close(sentinel_write)  # EOF is the sentinel's only exit request.
        sentinel_exit = sentinel.wait(timeout=5)
        write(directory, 'sentinel-exit.json', {'pid': sentinel.pid, 'exit': sentinel_exit})
        assert sentinel_exit == 0
    print(json.dumps({"case": case, "directory": str(directory), "supervisor_pid": process.pid, "exit": code}), flush=True)


def validate_result(directory):
    from optical_worker.compiler_job import CompilerJobRecord
    from optical_worker.wait_supervision import GuardSupervisorEvidence

    request = read(directory / "request.json")
    validate(request)
    case = request["process_case"]
    result = read(directory / "supervisor.json")
    guard = GuardSupervisorEvidence.from_payload(result["artifact"]["supervisor"], require_final=True)
    record = CompilerJobRecord.from_payload(result["compiler_job"], guard_evidence=guard, require_closed=True).to_payload()
    events = record["events"]
    assert result["process_fixture"]["case"] == case and result["artifact"]["worker"] is None
    assert result["process_fixture"]["exited_core"] == guard.to_payload()
    assert events[-1]["operation"] == "closed"
    operations = [e["operation"] for e in events]
    assert "no_children" in operations and "pipes_eof" in operations
    attempts = [e["details"]["identity"] for e in events if e["operation"] == "signal_attempted"]
    assert len({tuple(sorted(identity.items())) for identity in attempts}) == len(attempts)
    reaped = [e["details"]["identity"] for e in events if e["operation"] == "child_reaped"]
    assert all(not Path('/proc', str(identity['pid'])).exists() for identity in reaped)
    members = [read(path) for path in directory.glob('member-*.json') if not path.name.startswith('member-failure-')]
    assert not list(directory.glob('member-failure-*.json'))
    assert all(not Path('/proc', str(member['pid'])).exists() for member in members)
    worker = record['worker']
    assert operations.count('subreaper_ready') == 1
    assert any(row['pid'] == worker['pid'] for row in result['process_fixture']['pidfds'])
    for row in result['process_fixture']['pidfds']:
        info = dict(line.split(':', 1) for line in row['fdinfo'].splitlines())
        assert int(info['Pid']) == row['pid']
    assert all(member['sid'] == worker['sid'] for member in members)
    child_members = [member for member in members if member['role'] != 'worker']
    assert all(member['pgid'] != worker['pid'] for member in child_members)
    worker_wait_path = directory / 'worker-wait.json'
    worker_wait = read(worker_wait_path) if worker_wait_path.exists() else None
    reaped_pids = {identity['pid'] for identity in reaped}
    assert len(reaped_pids) == len(reaped)
    expected_pids = {member['pid'] for member in members}
    if worker_wait is not None:
        assert worker_wait['wait_status'] == 0 and worker_wait['pid'] not in reaped_pids
        expected_pids.remove(worker_wait['pid'])
    assert reaped_pids == expected_pids
    adopted_pids = {e['details']['identity']['pid'] for e in events if e['operation'] == 'child_adopted'}
    assert adopted_pids == reaped_pids - {worker['pid']}
    assert all(any(row['pid'] == pid for row in result['process_fixture']['pidfds']) for pid in reaped_pids)
    _validate_case_exits(directory, case, events, members, worker, worker_wait, result)
    if case == 'separate_pgid':
        assert len(child_members) == 1 and worker_wait is not None
        assert any(row['identity']['pid'] == worker_wait['pid'] for row in result['sampled_processes'])
    quiescent = [e['details']['sequence'] for e in events if e['operation'] == 'compile_quiescent']
    if case in {'normal', 'separate_pgid'}:
        assert quiescent == [6, 7, 8] and result['worker_returncode'] == 0
        assert 'signal_attempted' not in operations and 'deadline_expired' not in operations
        assert (directory / 'worker-finished.json').exists()
    else:
        assert quiescent == []
    if case in {'leader_exit', 'leader_ack_exit', 'double_fork', 'fork_during_cleanup', 'pipe_hang'}:
        assert result['worker_returncode'] == 0 and child_members
        assert 'deadline_expired' in operations and attempts
        assert all(identity != worker for identity in attempts)
        drain = next(e for e in events if e['operation'] == 'draining')
        expected_source = 'ack' if case == 'leader_ack_exit' else 'operation'
        assert drain['details']['source'] == expected_source
        previous = [e for e in events[:drain['index']] if e['operation'] == 'deadline_snapshot'][-1]
        assert drain['details']['deadline_ns'] == previous['details']['deadline_ns']
        assert all(e['details']['deadline_ns'] == drain['details']['deadline_ns'] for e in events if e['operation'] == 'deadline_expired')
        if case == 'fork_during_cleanup':
            assert any(m['role'] == 'grandchild' and m['at_ns'] >= drain['at_ns'] for m in members)
        if case == 'pipe_hang':
            worker_reap = next(e for e in events if e['operation'] == 'child_reaped' and e['details']['identity'] == worker)
            child_signal = next(e for e in events if e['operation'] == 'signal_attempted')
            eof = next(e for e in events if e['operation'] == 'pipes_eof')
            assert worker_reap['index'] < child_signal['index'] < eof['index']
    if case in {'proc_failure', 'handle_failure', 'stale_identity', 'disappearing_identity', 'signal_failure'}:
        assert [row['operation'] for row in result['process_fixture']['injections']] == [case]
        assert 'observation_failed' in operations or 'signal_failed' in operations
    if case == 'signal_failure':
        assert len(attempts) == 1 and operations.count('signal_failed') == 1
        assert result['worker_returncode'] == 0 and list(directory.glob('natural-exit-*.json'))
    if case == 'residual_complete':
        assert result['worker_returncode'] == 74 and child_members
        assert any('descendant still live' in error['message'] for error in result['diagnostics'])
    if case == 'missing_complete':
        assert result['worker_returncode'] == -9 and len(attempts) == 1
    if case == 'parent_lost':
        assert result['parent_alive'] is False and result['worker_returncode'] == -9
    write(directory, 'validated.json', {'case': case, 'closed': True, 'member_count': len(members),
                                      'reaped_count': len(reaped), 'attempt_count': len(attempts),
                                      'scope': 'OS fixtures only; no compilation or readiness'})


def _validate_case_exits(directory, case, events, members, worker, worker_wait, result):
    """Fixture exit expectations are stronger than general production replay.

    A legal CLOSED record can follow an ineffective signal and natural exit.
    Such a record must fail fixtures that require real SIGKILL termination.
    """
    from optical_worker.compiler_job import wait_status_returncode

    roles = {role: [m['pid'] for m in members if m['role'] == role]
             for role in ('worker', 'child', 'grandchild')}
    expected_roles = ['worker']
    if case in {'separate_pgid', 'leader_exit', 'leader_ack_exit', 'pipe_hang', 'residual_complete'}:
        expected_roles.append('child')
    elif case in {'double_fork', 'fork_during_cleanup'}:
        expected_roles += ['child', 'grandchild']
    assert sorted(m['role'] for m in members) == sorted(expected_roles), 'fixture process topology changed'
    assert roles['worker'] == [worker['pid']]
    wait_cases = {'separate_pgid', 'double_fork'}
    assert (worker_wait is not None) == (case in wait_cases), 'unexpected worker-owned reap'
    if worker_wait is not None:
        assert roles['child'] == [worker_wait['pid']] and worker_wait['wait_status'] == 0

    worker_exit = {'missing_complete': -9, 'proc_failure': -9, 'parent_lost': -9,
                   'handle_failure': 74, 'stale_identity': 74, 'disappearing_identity': 74,
                   'residual_complete': 74}.get(case, 0)
    expected = {worker['pid']: worker_exit}
    for pid in roles['child']:
        if case not in wait_cases:
            expected[pid] = 0 if case == 'fork_during_cleanup' else -9
    expected.update((pid, -9) for pid in roles['grandchild'])
    reaps = {e['details']['identity']['pid']: e for e in events if e['operation'] == 'child_reaped'}
    actual = {pid: wait_status_returncode(e['details']['wait_status']) for pid, e in reaps.items()}
    assert actual == expected, 'fixture wait status differs from required exit mechanism'
    assert result['worker_returncode'] == worker_exit

    attempts = {e['details']['identity']['pid']: e for e in events if e['operation'] == 'signal_attempted'}
    expected_attempts = {pid for pid, code in expected.items() if code == -9}
    if case == 'signal_failure':
        expected_attempts.add(worker['pid'])
    assert set(attempts) == expected_attempts, 'fixture termination target set changed'
    for pid, attempt in attempts.items():
        assert attempt['index'] < reaps[pid]['index'], 'termination attempt follows reap'
    failures = [e['details']['identity']['pid'] for e in events if e['operation'] == 'signal_failed']
    assert failures == ([worker['pid']] if case == 'signal_failure' else []), 'unexpected signal failure'
    natural = sorted(p.name for p in directory.glob('natural-exit-*.json'))
    assert natural == ([f"natural-exit-{worker['pid']}.json"] if case == 'signal_failure' else []), 'unexpected natural fallback'
    assert (directory / 'worker-fixture-error.json').exists() == (worker_exit == 74)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'sentinel':
        assert os.read(int(sys.argv[2]), 1) == b''
    else:
        assert len(sys.argv) == 4 and sys.argv[1] == 'parent'
        parent(Path(sys.argv[2]), sys.argv[3])
