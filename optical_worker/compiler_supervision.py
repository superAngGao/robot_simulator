"""Dedicated Linux compiler-tree owner. Never imported by the captured worker.

Native policy setup and spawning are explicit: importing this file is inert.
Only the fixed CUDA compilation driver can be launched by this owner.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import platform
import select
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

from .compilation_contract import CUDA_COMPILE_SEQUENCES, cuda_foundation_phase
from .compiler_job import _CompilerObligation, identity_payload, process_identity
from .wait_protocol import GuardProtocolError, exact_fields, validate_message

_LIBC_PATH = "/usr/lib/x86_64-linux-gnu/libc.so.6"
_LIVE_JOBS = {}


def _require(condition, message):
    if not condition:
        raise GuardProtocolError(message)


def _error_kind(error):
    if isinstance(error, ProcessLookupError) or getattr(error, "errno", None) in {errno.ENOENT, errno.ESRCH}:
        return "missing"
    if isinstance(error, PermissionError):
        return "permission"
    if getattr(error, "errno", None) in {errno.EMFILE, errno.ENFILE, errno.ENOMEM}:
        return "resource_exhausted"
    return "other_os"


class _SigAction(ctypes.Structure):
    # Installed Linux x86_64 glibc userspace headers: handler; sigset_t (1024
    # bits); int flags; restorer. This is not the kernel rt_sigaction layout.
    _fields_ = [("handler", ctypes.c_void_p), ("mask", ctypes.c_ulong * 16),
                ("flags", ctypes.c_int), ("restorer", ctypes.c_void_p)]


class _SubreaperPolicy:
    def __init__(self, libc_file):
        row = exact_fields(libc_file, {"path", "size", "sha256"})
        _require(sys.platform == "linux" and platform.machine() == "x86_64"
                 and ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_ulong) == 8
                 and ctypes.sizeof(_SigAction) == 152
                 and tuple(getattr(_SigAction, n).offset for n in ("handler", "mask", "flags", "restorer"))
                 == (0, 8, 136, 144), "unsupported pinned glibc sigaction ABI")
        _require(row["path"] == _LIBC_PATH and type(row["size"]) is int
                 and type(row["sha256"]) is str, "libc must be the frozen system file")
        path = Path(_LIBC_PATH)
        _require(path.resolve(strict=True) == path and path.is_file(), "libc path changed")
        data = path.read_bytes()
        _require(len(data) == row["size"] and hashlib.sha256(data).hexdigest() == row["sha256"],
                 "libc differs from deployment inventory")
        self.library = ctypes.CDLL(_LIBC_PATH, use_errno=True)
        self._prctl = self.library.prctl
        self._prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        self._prctl.restype = ctypes.c_int
        self._sigaction = self.library.sigaction
        self._sigaction.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_SigAction)]
        self._sigaction.restype = ctypes.c_int
        # Pinned Conda Python lacks os.pidfd_open despite Linux/glibc support.
        # Use one fixed binding on every deployment, not a runtime fallback.
        self._pidfd_open = self.library.pidfd_open
        self._pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
        self._pidfd_open.restype = ctypes.c_int
        self._check_sigchld()
        self._check_result(self._prctl(36, 1, 0, 0, 0))
        self.verify()

    @staticmethod
    def _check_result(result):
        if result != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))

    def _check_sigchld(self):
        action = _SigAction()
        self._check_result(self._sigaction(signal.SIGCHLD, None, ctypes.byref(action)))
        _require(action.handler is None and not (action.flags & 2),
                 "SIGCHLD must be SIG_DFL without SA_NOCLDWAIT")

    def verify(self):
        value = ctypes.c_int()
        address = ctypes.cast(ctypes.byref(value), ctypes.c_void_p).value
        self._check_result(self._prctl(37, address, 0, 0, 0))
        _require(value.value == 1, "subreaper readback failed")
        self._check_sigchld()

    def open_pidfd(self, pid):
        _require(type(pid) is int and 0 < pid < 2**31, "invalid child PID for pidfd")
        fd = self._pidfd_open(pid, 0)  # Fixed flags; no arbitrary syscall/function.
        if fd < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        return fd


def _proc_identity(pid):
    text = Path("/proc", str(pid), "stat").read_text(encoding="ascii")
    close = text.rfind(")")
    _require(close > 0 and text[:text.find(" (")] == str(pid), "invalid proc identity")
    fields = text[close + 2:].split()
    _require(len(fields) >= 20, "truncated proc identity")
    return (pid, int(fields[19]), int(fields[3])), int(fields[1]), fields[0]


def _own_children(pid):
    return tuple(int(value) for value in Path("/proc", str(pid), "task", str(pid), "children").read_text().split())


def _registered_compiler_job(job, core):
    _require(type(job) is _CompilerJob and _LIVE_JOBS.get(job) is core and job.core is core,
             "exact registered compiler job required")
    job._check_owner()
    return job


class _CompilerJob:
    """One supervisor, one fixed child tree, one wait owner; no callback hooks."""

    def __init__(self, core, *, libc_file):
        _require(sys.platform == "linux" and threading.current_thread() is threading.main_thread()
                 and len(threading.enumerate()) == 1, "compiler supervisor must be dedicated and single-threaded")
        _require(not _LIVE_JOBS, "only one compiler tree per supervisor")
        _require(hasattr(signal, "pidfd_send_signal")
                 and hasattr(os, "waitid") and hasattr(os, "WNOWAIT"), "required Linux process APIs absent")
        self.pid = os.getpid()
        _require({p.name for p in Path("/proc/self/task").iterdir()} == {str(self.pid)},
                 "supervisor has native threads")
        _require(not _own_children(self.pid), "supervisor already owns unrelated children")
        try:
            unexpected = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            pass
        else:
            raise GuardProtocolError("supervisor lacks initial ECHILD: " + str(unexpected))
        self.policy = _SubreaperPolicy(libc_file)
        identity, _, _ = _proc_identity(self.pid)
        self.core = core
        self.thread = threading.current_thread()
        self.obligation = _CompilerObligation(core, identity_payload(identity))
        self.obligation.subreaper_ready()
        self.process = None
        self.handles = {}
        self.observed_members = {}
        self.channels = {}
        self.output = {"stdout": bytearray(), "stderr": bytearray()}
        self.transport_channel = None
        self._servicing_due = False
        self._echild_observed = False
        self._pipe_setup_complete = False
        self._fd_close_attempted = set()
        self.launch_errors = []
        self._last_process_sample_ns = None
        self.output_truncated = {"stdout": 0, "stderr": 0}
        _LIVE_JOBS[self] = core

    def _check_owner(self):
        _require(os.getpid() == self.pid and threading.current_thread() is self.thread
                 and len(threading.enumerate()) == 1, "compiler wait owner changed")

    def launch(self, *, repository_root, request_path, output_directory, environment):
        self._check_owner()
        _require(self.process is None, "compiler worker launch is single-use")
        root = Path(repository_root).resolve(strict=True)
        driver = root / "tests/support/cuda_compilation_process.py"
        _require(driver.is_file() and driver.resolve(strict=True) == driver, "fixed compilation driver missing or symlinked")
        request = Path(request_path).resolve(strict=True)
        output = Path(output_directory).resolve(strict=True)
        _require(type(environment) is dict and all(type(k) is str and type(v) is str for k, v in environment.items()),
                 "frozen worker environment required")
        parent, child = socket.socketpair()
        self.transport_channel = parent
        # From this point even a partially initialized Popen is strongly held.
        # No Popen status method is used anywhere in this module.
        self.process = object.__new__(subprocess.Popen)
        try:
            subprocess.Popen.__init__(
                self.process,
                [sys.executable, "-I", "-S", "-B", str(driver), "worker", str(child.fileno()), str(request), str(output)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                pass_fds=(child.fileno(),), close_fds=True, start_new_session=True,
                env=dict(environment), cwd=str(root),
            )
        except BaseException as error:
            self.launch_errors.append(error)
            if not getattr(self.process, "_child_created", False):
                raise  # No fork occurred; the NEW record cannot claim CLOSED.
        finally:
            child.close()
        try:
            self._anchor_worker()
            self._prepare_pipes()
            if self.launch_errors:
                self.obligation.observation_failed("other_os")
        except BaseException as error:
            # Keep Popen, channels, PID and any acquired identity. The process
            # service retries observation; no ARM is accepted without its pidfd.
            self.obligation.observation_failed("identity_changed" if isinstance(error, GuardProtocolError)
                                               else _error_kind(error))
        return parent

    def _prepare_pipes(self):
        if self._pipe_setup_complete:
            return
        for name in ("stdout", "stderr"):
            stream = getattr(self.process, name, None)
            _require(stream is not None and not stream.closed, "spawn did not return its held output pipes")
            self.channels[name] = stream
            os.set_blocking(stream.fileno(), False)
        self._pipe_setup_complete = True

    def _anchor_worker(self):
        if self.obligation.replay.worker is None:
            identity, ppid, _ = _proc_identity(self.process.pid)
            _require(ppid == self.pid and identity[2] == identity[0], "worker process identity mismatch")
            self.obligation.spawned(identity_payload(identity))
        identity = self.obligation.replay.worker
        if identity not in self.handles and identity not in self.obligation.replay.reaped:
            self._acquire(identity)
        if self.obligation.replay.failed and self.core.state not in {"TERMINATING", "EXITED"}:
            self.core.fault("process_observation_failed")
            self.obligation.sync_core()

    def _acquire(self, expected):
        before, ppid, _ = _proc_identity(expected[0])
        _require(before == expected and ppid == self.pid, "child identity changed before pidfd")
        fd = self.policy.open_pidfd(expected[0])
        try:
            after, ppid, _ = _proc_identity(expected[0])
            _require(after == before and ppid == self.pid, "child identity changed around pidfd")
        except BaseException:
            os.close(fd)
            raise
        _require(expected not in self.handles, "duplicate owned pidfd")
        self.handles[expected] = fd

    def _discover(self):
        for pid in _own_children(self.pid):
            self._due()
            try:
                identity, ppid, _ = _proc_identity(pid)
                _require(ppid == self.pid, "proc child has a different parent")
                if identity not in self.obligation.replay.owned:
                    self.obligation.adopted(identity_payload(identity))
                if identity not in self.handles:
                    self._acquire(identity)
            except (FileNotFoundError, ProcessLookupError):
                # A disappearance is not reaping or ECHILD. Revisit using the
                # sole wait owner; do not signal a newly reused numeric PID.
                continue
            except BaseException as error:
                self.obligation.observation_failed("identity_changed" if isinstance(error, GuardProtocolError)
                                                   else _error_kind(error))
            self._due()

    def _reap_one(self):
        try:
            result = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            self._echild_observed = True
            return False
        self._echild_observed = False
        if result is None:
            return False
        pid = result.si_pid
        identity, ppid, _ = _proc_identity(pid)
        _require(ppid == self.pid, "waitable child changed parent")
        if identity not in self.obligation.replay.owned:
            self.obligation.adopted(identity_payload(identity))
        # A zombie holds its PID/start identity until this exact waitpid.
        observed_pid, status = os.waitpid(pid, os.WNOHANG)
        _require(observed_pid == pid, "waitable child was consumed by another waiter")
        self.obligation.reaped(identity_payload(identity), status)
        if identity == self.obligation.replay.worker:
            from .compiler_job import wait_status_returncode

            self.process.returncode = wait_status_returncode(status)
        return True

    def _due(self):
        if self._servicing_due or self.obligation.replay.state == "NEW":
            return
        self._servicing_due = True
        try:
            self.obligation.tick()
            if not self.obligation.replay.expired:
                return
            for identity, fd in tuple(self.handles.items()):
                replay = self.obligation.replay
                if identity in replay.reaped or identity in replay.attempted:
                    continue
                try:
                    self._signal_owned(identity, fd)
                except BaseException as error:
                    self.obligation.observation_failed("identity_changed" if isinstance(error, GuardProtocolError)
                                                       else _error_kind(error))
        finally:
            self._servicing_due = False

    def _signal_owned(self, identity, fd):
        replay = self.obligation.replay
        # Observe/reap a winning exit before recording any signal.
        if select.select([fd], [], [], 0)[0]:
            for _ in range(16):
                if not self._reap_one():
                    break
            if identity in replay.reaped:
                return
        if identity == replay.worker and self.core.state == "EXITED":
            return
        current, ppid, _ = _proc_identity(identity[0])
        if current != identity or ppid != self.pid:
            self.obligation.observation_failed("identity_changed")
            return
        if identity == replay.worker:
            if not self.core.kill_required:
                # An outer clock failure must also close the live core
                # before the unique worker attempt is mirrored there.
                self.core.tick()
            _require(self.core.kill_required, "live worker lacks legal core termination")
            self.core.record_kill_request()
            self.obligation.sync_core()
        self.obligation.signal_attempted(identity_payload(identity))
        try:
            signal.pidfd_send_signal(fd, signal.SIGKILL, None, 0)
        except OSError as error:
            self.obligation.signal_failed(identity_payload(identity), _error_kind(error))
            if identity == replay.worker:
                self.core.record_kill_failure({"permission": "permission_denied", "missing": "process_missing"}
                                              .get(_error_kind(error), "other_os_error"))
                self.obligation.sync_core()

    def _sample_session(self):
        members = set()
        for path in Path("/proc").iterdir():
            self._due()
            if not path.name.isdecimal():
                continue
            try:
                identity, _, state = _proc_identity(int(path.name))
                if identity[2] != self.obligation.replay.worker[2]:
                    continue
                members.add(identity)
                # Available observations only: short-lived tool commands may
                # disappear between scans. No exhaustive command-log claim.
                record = {"identity": identity_payload(identity), "state": state}
                try:
                    record["cmdline_hex"] = (path / "cmdline").read_bytes().hex()
                    record["exe"] = os.readlink(path / "exe")
                except (FileNotFoundError, ProcessLookupError):
                    record["disappeared_during_sample"] = True
                self.observed_members[identity] = record
            except (FileNotFoundError, ProcessLookupError):
                continue
        return members

    def before_message(self, message):
        self._check_owner()
        request = validate_message(message, self.core.configuration)
        cuda_foundation_phase(request["sequence"], request["role"])
        _require(request["operation"] in {"ARM", "COMPLETE"}, "compiler supervisor accepts requests only")
        _require(self.process is not None and self.obligation.replay.worker in self.handles,
                 "worker must have an anchored pidfd before ARM")
        if request["operation"] == "COMPLETE" and request["sequence"] in CUDA_COMPILE_SEQUENCES:
            _require(self.core.state == "ARMED_NORMAL"
                     and self.core._state.active == (request["sequence"], "setup"), "unknown compiler COMPLETE")
            self._discover()
            for _ in range(16):
                if not self._reap_one():
                    break
                self._due()
            members = self._sample_session()
            _require(members == {self.obligation.replay.worker}
                     and self.obligation.replay.owned - self.obligation.replay.reaped == {self.obligation.replay.worker},
                     "compiler descendant still live at COMPLETE")
            # Every previously sampled identity must now be absent, including
            # children reaped by their own parent (whose status we never invent).
            for identity in self.observed_members:
                self._due()
                if identity == self.obligation.replay.worker:
                    continue
                try:
                    observed, _, _ = _proc_identity(identity[0])
                except (FileNotFoundError, ProcessLookupError):
                    continue
                _require(observed != identity, "retained compiler still exists at COMPLETE")
            self.obligation.quiescent(request["sequence"])

    def after_core_transition(self):
        self.obligation.sync_core()

    def service_processes(self):
        self._check_owner()
        if self.obligation.replay.state == "CLOSED":
            return
        try:
            self._anchor_worker()
            self._due()
            self._discover()
            for _ in range(16):
                if not self._reap_one():
                    break
                self._due()
            self._due()
            active = self.core._state.active
            if active is not None and active[0] in CUDA_COMPILE_SEQUENCES:
                now = self.obligation.now()
                if self._last_process_sample_ns is None or now - self._last_process_sample_ns >= 50_000_000:
                    self._last_process_sample_ns = now
                    self._sample_session()
            self._prepare_pipes()
            for name, stream in tuple(self.channels.items()):
                self._due()
                try:
                    data = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    continue
                if data:
                    room = max(0, 4 * 1024 * 1024 - len(self.output[name]))
                    self.output[name].extend(data[:room])
                    self.output_truncated[name] += max(0, len(data) - room)
                else:
                    stream.close()
                    del self.channels[name]
            replay = self.obligation.replay
            if replay.worker not in replay.reaped:
                return
            if not replay.no_children and self._echild_observed:
                self.policy.verify()
                try:
                    result = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    # No sole-waiter race can create a new child after ECHILD.
                    # Explicitly verify every held identity is absent too.
                    for identity in replay.owned | self.observed_members.keys():
                        self._due()
                        try:
                            observed, _, _ = _proc_identity(identity[0])
                        except (FileNotFoundError, ProcessLookupError):
                            continue
                        _require(observed != identity, "reaped identity remains present")
                    self.obligation.no_children()
                else:
                    # WNOWAIT already returned ECHILD, so no member of the
                    # trusted tree can create this child. Keep the raw status
                    # as a diagnostic, not an invented identity/reaping proof.
                    self.unexpected_closure_wait = result
                    self.obligation.observation_failed("identity_changed")
            if self._pipe_setup_complete and not self.channels and not replay.pipes_eof:
                self.obligation.pipes_eof()
            if replay.no_children and replay.pipes_eof:
                for identity, fd in tuple(self.handles.items()):
                    if identity in self._fd_close_attempted:
                        continue
                    self._fd_close_attempted.add(identity)
                    os.close(fd)
                    del self.handles[identity]
                if not self.handles:
                    self.obligation.close()
        except BaseException as error:
            self.obligation.observation_failed("proc_unavailable" if isinstance(error, GuardProtocolError)
                                               else _error_kind(error))
            self._due()
