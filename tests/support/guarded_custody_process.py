"""Fixed CPU sacrificial programs for C2d1; never a product callback surface."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import select
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from optical_worker.guarded_evidence import VALIDATION_VERSION, validate_custody_artifact  # noqa: E402
from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle  # noqa: E402
from optical_worker.lifecycle import DeviceScopeDescriptor, NativeWorkerResourceGuardian  # noqa: E402
from optical_worker.native_setup import _abort  # noqa: E402
from optical_worker.protocol import encode_framed_message  # noqa: E402
from optical_worker.wait_protocol import MAX_BODY_BYTES, GuardConfiguration  # noqa: E402
from optical_worker.wait_supervision import GuardSupervisorCore, GuardSupervisorTransport  # noqa: E402

FAULTS = (
    "cpu_success",
    "cpu_cleanup_hang",
    "cpu_exit_hang",
    "normal_success",
    "normal_leaf_raise",
    "inner_abort",
    "pre_leaf_raise",
    "foreign",
    "unknown",
    "sync_raise",
    "sync_hang",
    "cleanup_raise_before",
    "cleanup_raise_after",
    "normal_no_ack",
    "normal_wrong_ack",
    "normal_ack_eof",
    "normal_stalled_disarm",
    "terminal_no_ack",
    "terminal_wrong_ack",
    "terminal_ack_eof",
    "target_cleanup_hang",
    "target_exit_hang",
)


def configuration():
    # Operational CPU-test budgets, not product/performance gates.
    return GuardConfiguration("c" * 32, 1, 400, 1400)


def mark(directory, name, value="1"):
    (directory / name).write_text(value, encoding="utf-8")


def stall():
    while True:
        time.sleep(0.02)


class Owner:
    def __init__(self, directory, fault):
        self.directory, self.fault = directory, fault

    def __del__(self):
        mark(self.directory, "destructor")
        if self.fault in {"cpu_cleanup_hang", "target_cleanup_hang"}:
            stall()


def child(fault, fd, directory):
    mark(directory, "identity.json", json.dumps({"pid": os.getpid(), "ppid": os.getppid()}))
    atexit.register(mark, directory, "atexit")
    lifecycle = NativeGuardedSetupLifecycle(
        lane_id="cuda",
        channel=socket.socket(fileno=fd),
        configuration=configuration(),
    )
    lifecycle.adopt_host_owner(Owner(directory, fault), role="cpu_owner", operation_id="test_owner")
    original_release = NativeWorkerResourceGuardian.release_whole_graph
    if fault in {"cleanup_raise_before", "cleanup_raise_after"}:

        def failing_release(self, authority):
            if fault == "cleanup_raise_after":
                original_release(self, authority)
            raise RuntimeError("fixed release failure")

        NativeWorkerResourceGuardian.release_whole_graph = failing_release

    def sync():
        mark(directory, "sync")
        if fault == "sync_hang":
            stall()
        if fault == "sync_raise":
            raise RuntimeError("fixed sync failure")

    try:
        cpu_only = fault.startswith("cpu_") or fault.startswith("terminal_") or fault.startswith("cleanup_")
        if not cpu_only:
            with lifecycle.segment():
                scope = DeviceScopeDescriptor.exact_target(
                    operation_id="cpu_leaf",
                    leaf_kind="allocation",
                    device_ordinal=0 if fault == "foreign" else 1,
                    scope_source="cpu_fixture",
                    runtime_identity="cpu_fixture",
                )
                if fault == "unknown":
                    scope = DeviceScopeDescriptor.unknown(
                        operation_id="cpu_leaf",
                        leaf_kind="allocation",
                        scope_source="cpu_fixture",
                        runtime_identity="cpu_fixture",
                    )
                permit = lifecycle.begin_device_leaf(scope)
                if fault == "pre_leaf_raise":
                    raise RuntimeError("fixed before-enter failure")
                lifecycle.enter_device_leaf(permit)
                mark(directory, "leaf")
                # Snapshot before a deliberately stalled COMPLETE handshake.
                mark(directory, "prefix.json", json.dumps(lifecycle.evidence().to_payload()))
                if fault in {"normal_leaf_raise", "inner_abort"}:
                    try:
                        raise RuntimeError("fixed submit-then-raise")
                    except RuntimeError as error:
                        if fault == "inner_abort":
                            raise _abort(lifecycle, exception=error, sync=sync) from error
                        raise
        evidence = lifecycle.terminate(
            fault="setup_failure", detection_phase="fixed_cpu_driver", emergency_target_synchronize=sync
        )
        mark(directory, "worker.json", json.dumps(evidence.lifecycle.to_payload()))
        if fault in {"cpu_exit_hang", "target_exit_hang"}:
            stall()
        os._exit(0)
    except BaseException:
        evidence = lifecycle.termination_evidence
        if evidence is not None:
            mark(directory, "worker.json", json.dumps(evidence.lifecycle.to_payload()))
        # Strong terminal registry/guardian remains alive; never ordinary exit.
        os._exit(74)


def supervise(fault, liveness_fd, directory):
    channel, worker_channel = socket.socketpair()
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "child",
            fault,
            str(worker_channel.fileno()),
            str(directory),
        ],
        pass_fds=(worker_channel.fileno(),),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    worker_channel.close()
    core = GuardSupervisorCore(configuration())
    transport = GuardSupervisorTransport(channel, core)
    started = time.monotonic()
    injected = False
    parent_alive = True
    try:
        while True:
            if time.monotonic() - started > 12:
                raise RuntimeError("outer test fail-bound; NOT successful supervisor termination")
            code = process.poll()
            if code is not None:
                core.observe_exit(code)
                break
            core.tick()
            if core.kill_required:
                core.record_kill_request()
                try:
                    process.kill()
                except ProcessLookupError:
                    core.record_kill_failure("process_missing")
                except PermissionError:
                    core.record_kill_failure("permission_denied")
                except OSError:
                    core.record_kill_failure("other_os_error")
            pending = core.pending_response
            if pending is not None and not injected:
                if fault.endswith("wrong_ack"):
                    wrong = dict(pending, session="d" * 32)
                    channel.send(encode_framed_message(wrong, max_bytes=MAX_BODY_BYTES))
                    injected = True
                elif fault.endswith("ack_eof"):
                    channel.shutdown(socket.SHUT_WR)
                    injected = True
                elif fault == "normal_stalled_disarm" and pending["operation"] == "DISARMED":
                    channel.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
                    while True:
                        try:
                            channel.send(b"x" * 4096)
                        except BlockingIOError:
                            break
                    injected = True
            readers = [channel] if core.state not in {"TERMINATING", "EXITED"} else []
            if parent_alive:
                readers.append(liveness_fd)
            writers = [channel] if transport.wants_write else []
            if fault.endswith(("no_ack", "wrong_ack", "ack_eof")):
                writers = []
            readable, writable, _ = select.select(readers, writers, [], 0.01)
            if parent_alive and liveness_fd in readable and os.read(liveness_fd, 1) == b"":
                parent_alive = False
                core.fault("parent_lost")
            try:
                transport.service(readable=channel in readable, writable=channel in writable)
            except (RuntimeError, OSError, ValueError):
                if core.state != "TERMINATING":
                    raise
        process.wait(timeout=1)
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        worker = None
        if (directory / "worker.json").exists():
            worker = json.loads((directory / "worker.json").read_text(encoding="utf-8"))
        elif (directory / "prefix.json").exists():
            worker = json.loads((directory / "prefix.json").read_text(encoding="utf-8"))
        artifact = {
            "schema_version": VALIDATION_VERSION,
            "worker": worker,
            "supervisor": core.evidence(require_final=True).to_payload(),
        }
        validate_custody_artifact(artifact)
        return {
            "artifact": artifact,
            "stderr": stderr,
            "parent_alive": parent_alive,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "worker_pid": process.pid,
            "worker_returncode": process.returncode,
            "injected": injected,
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        process.stderr.close()
        channel.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("child", "supervisor"))
    parser.add_argument("fault", choices=FAULTS)
    parser.add_argument("fd", type=int)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.mode == "child":
        child(args.fault, args.fd, args.directory)
    else:
        print(json.dumps(supervise(args.fault, args.fd, args.directory), sort_keys=True))


if __name__ == "__main__":
    main()
