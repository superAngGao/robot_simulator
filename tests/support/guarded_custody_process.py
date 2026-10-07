"""Fixed CPU sacrificial programs for C2d1; never a product callback surface."""

from __future__ import annotations

import argparse
import atexit
import importlib.abc
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from optical_worker.guarded_evidence import VALIDATION_VERSION, validate_custody_artifact  # noqa: E402
from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle  # noqa: E402
from optical_worker.lifecycle import (  # noqa: E402
    DeviceScopeDescriptor,
    NativeWorkerResourceGuardian,
    WorkerSetupProtocolError,
)
from optical_worker.native_setup import (  # noqa: E402
    NativeWorkerPreparationAborted,
    _abort,
    prepare_cuda_native_input_for_closed_worker,
    prepare_warp_native_input_for_closed_worker,
)
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
    "foreign_inner_abort",
    "unknown_inner_abort",
    "invalid_permit_inner_abort",
    "cuda_import_failure",
    "warp_import_failure",
    "cuda_import_thread_failure",
    "warp_import_thread_failure",
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

SOURCE_FAULTS = tuple(
    "source_" + name
    for name in (
        "success",
        "prelude_raise",
        "module_adopt",
        "compile_raise",
        "body_raise",
        "caught_nested",
        "audit_raise",
        "import_hang",
        "parent_loss",
        "no_ack",
        "worker_exit_no_ack",
        "wrong_ack",
        "ack_eof",
        "stalled_disarm",
        "cleanup_hang",
        "exit_hang",
    )
)

# Fixed sacrificial C2d3 driver only. Not product IPC or an arbitrary callback.
RUNTIME_FAULTS = ("runtime_success", "runtime_terminal_sync_raise", "runtime_terminal_sync_hang")


def configuration():
    # Operational CPU-test budgets, not product/performance gates.
    return GuardConfiguration("c" * 32, 1, 400, 1400)


def mark(directory, name, value="1"):
    (directory / name).write_text(value, encoding="utf-8")


def stall():
    while True:
        time.sleep(0.02)


def fail_actual_import(lifecycle, fault, directory):
    """Fixed CPU fixture at the actual prepare call; never import a runtime."""
    runtime = "warp" if fault.startswith("warp_") else "torch"
    assert runtime not in sys.modules
    original = ImportError("fixed CPU runtime import failure")
    rejected = []

    class ImportFailure(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != runtime:
                return None
            if "_thread_" in fault:

                def misuse():
                    try:
                        lifecycle.require_adopted()
                    except WorkerSetupProtocolError as error:
                        rejected.append(error)

                thread = threading.Thread(target=misuse)
                thread.start()
                thread.join(2)
                assert not thread.is_alive() and len(rejected) == 1
            raise original

    injector = ImportFailure()
    source = object()
    sys.meta_path.insert(0, injector)
    try:
        prepare = (
            prepare_warp_native_input_for_closed_worker
            if runtime == "warp"
            else prepare_cuda_native_input_for_closed_worker
        )
        try:
            prepare(source, lifecycle=lifecycle)
        except NativeWorkerPreparationAborted as aborted:
            assert aborted.__cause__ is original
            assert aborted.evidence is lifecycle.termination_evidence
            assert lifecycle.guardian.contains_exact_owner(source)
            assert lifecycle.guardian.contains_exact_owner(original)
            payload = aborted.evidence.lifecycle.to_payload()
            assert payload["summary"]["unknown_device_scope"]
            assert payload["summary"]["custody"] == "full_quarantine"
            mark(directory, "inner_abort.json", json.dumps(payload))
            raise
        raise AssertionError("fixed import failure did not abort")
    finally:
        sys.meta_path.remove(injector)
        assert runtime not in sys.modules


def _assert_actual_prepare_import_failure(peer, tmp_path, lane, prior_fault, propagate):
    controller = NativeGuardedSetupLifecycle(lane_id=lane, channel=peer.channel, configuration=peer.config)
    before_exit = None
    try:
        with controller.segment():
            try:
                fail_actual_import(
                    controller,
                    lane + ("_import_thread_failure" if prior_fault else "_import_failure"),
                    tmp_path,
                )
            except NativeWorkerPreparationAborted as error:
                assert error.evidence is controller.termination_evidence
                before_exit = error.evidence.lifecycle.to_payload()
                assert before_exit["summary"]["unknown_device_scope"]
                assert before_exit["summary"]["disposition"] == "multi_or_unknown_device_quarantine_exit"
                assert before_exit["summary"]["normal_complete_count"] == 0
                assert before_exit["summary"]["terminal_arm_attempt_count"] == 0
                assert before_exit["summary"]["emergency_sync_attempt_count"] == 0
                assert before_exit["summary"]["release_status"] == "not_authorized"
                assert [e["operation"] for e in before_exit["transition_trace"]].count(
                    "termination_started"
                ) == 1
                if propagate:
                    raise
    except NativeWorkerPreparationAborted:
        assert propagate
    except WorkerSetupProtocolError:
        assert not propagate
    else:
        raise AssertionError("failed preparation exited normally")
    assert before_exit is not None
    assert controller.evidence().to_payload() == before_exit


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
        lane_id="warp" if fault.startswith("warp_") else "cuda",
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
                if "_import_" in fault:
                    fail_actual_import(lifecycle, fault, directory)
                if fault in {"foreign_inner_abort", "unknown_inner_abort", "invalid_permit_inner_abort"}:
                    try:
                        if fault == "invalid_permit_inner_abort":
                            lifecycle.enter_device_leaf(object())
                        elif fault == "foreign_inner_abort":
                            lifecycle.begin_device_leaf(
                                DeviceScopeDescriptor.exact_target(
                                    operation_id="cpu_leaf",
                                    leaf_kind="allocation",
                                    device_ordinal=0,
                                    scope_source="cpu_fixture",
                                    runtime_identity="cpu_fixture",
                                )
                            )
                        else:
                            lifecycle.begin_device_leaf(
                                DeviceScopeDescriptor.unknown(
                                    operation_id="cpu_leaf",
                                    leaf_kind="allocation",
                                    scope_source="cpu_fixture",
                                    runtime_identity="cpu_fixture",
                                )
                            )
                    except BaseException as error:
                        aborted = _abort(lifecycle, exception=error, sync=sync)
                        assert aborted.evidence is lifecycle.termination_evidence
                        assert lifecycle.guardian.contains_exact_owner(error)
                        mark(
                            directory, "inner_abort.json", json.dumps(aborted.evidence.lifecycle.to_payload())
                        )
                        raise aborted from error
                    raise AssertionError("fixed abort fault did not occur")
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
        disposition = evidence.lifecycle.to_payload()["summary"]["disposition"]
        os._exit(0 if disposition in {"cancelled_before_submission_exit", "aborted_synced_exit"} else 74)
    except BaseException:
        evidence = lifecycle.termination_evidence
        if evidence is not None:
            mark(directory, "worker.json", json.dumps(evidence.lifecycle.to_payload()))
        # Strong terminal registry/guardian remains alive; never ordinary exit.
        os._exit(74)


def supervise(fault, liveness_fd, directory):
    source_mode = fault in SOURCE_FAULTS
    runtime_mode = fault in RUNTIME_FAULTS
    if runtime_mode:
        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        config = GuardConfiguration.from_payload(request["guard_configuration"])
        child_script = Path(__file__).with_name("native_runtime_process.py")
    else:
        config = GuardConfiguration("c" * 32, 1, 1000, 5000) if source_mode else configuration()
        child_script = (
            Path(__file__).with_name("source_custody_process.py") if source_mode else Path(__file__)
        )
    # Harness-wide failure bound is distinct from unchanged per-operation ACK/
    # watchdog semantics. Seven fixed stages plus terminal disposal, not a
    # renewable deadline within any native call. It cannot authorize success.
    outer_bound = (
        (10 * config.operation_timeout_ms / 1000 + 30) if runtime_mode else (20 if source_mode else 12)
    )
    channel, worker_channel = socket.socketpair()
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            *(["-S"] if source_mode or runtime_mode else []),
            "-B",
            str(child_script.resolve()),
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
    core = GuardSupervisorCore(config)
    transport = GuardSupervisorTransport(channel, core)
    started = time.monotonic()
    injected = False
    parent_alive = True
    try:
        while True:
            if time.monotonic() - started > outer_bound:
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
                elif (
                    fault in {"normal_stalled_disarm", "source_stalled_disarm"}
                    and pending["operation"] == "DISARMED"
                ):
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
                if fault == "source_worker_exit_no_ack" and not injected and channel in readable:
                    # Fixed test-only transport scheduling: let the client's
                    # unchanged ACK clock precede the supervisor ARM clock.
                    # No source body is admitted without ACK; no deadline is
                    # extended/restarted after commitment. Exercise self-exit
                    # as well as the racing supervisor-killed no-ACK ending.
                    time.sleep(0.4)
                    injected = True
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
    parser.add_argument(
        "mode", choices=("child", "supervisor", "import_probe_caught", "import_probe_propagated")
    )
    parser.add_argument("fault", choices=FAULTS + SOURCE_FAULTS + RUNTIME_FAULTS)
    parser.add_argument("fd", type=int)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.mode.startswith("import_probe_"):
        if args.fault not in {
            "cuda_import_failure",
            "warp_import_failure",
            "cuda_import_thread_failure",
            "warp_import_thread_failure",
        }:
            raise ValueError("not a fixed import probe")
        # Reuse the unit assertion/real socket peer in a cold CPU process. This
        # is a test-only role, never a production worker request or callback.
        from tests.support.guarded_peer import Peer

        peer = Peer()
        try:
            _assert_actual_prepare_import_failure(
                peer,
                args.directory,
                "warp" if args.fault.startswith("warp_") else "cuda",
                "_thread_" in args.fault,
                args.mode == "import_probe_propagated",
            )
        finally:
            peer.close()
        print(json.dumps({"pid": os.getpid(), "ppid": os.getppid()}))
    elif args.mode == "child":
        child(args.fault, args.fd, args.directory)
    else:
        print(json.dumps(supervise(args.fault, args.fd, args.directory), sort_keys=True))


if __name__ == "__main__":
    main()
