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
from tests.support.native_runtime_process import RUNTIME_ACK_FAULTS, RUNTIME_FAULTS, RUNTIME_WORKER_CASES


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


def _native_worker_requests(channel, directory, request, stop, deadline, outcome):
    """Fixed validation client; the child uses the unmodified production codec/loop."""
    from optical_worker.guarded_evidence import PostReclaimIdleEvidence
    from optical_worker.protocol import (
        INTEGRATED_WORKER_PROTOCOL_VERSION, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES,
        ClosedNativeLaneWorkerEvidence, IntegratedWorkerRuntimeManifest, decode_canonical_message,
        decode_frame_header, validate_response_envelope,
    )

    trace = []
    channel.settimeout(0.2)

    def read_exact(count):
        chunks = bytearray()
        while len(chunks) < count:
            if stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("fixed native request driver did not complete")
            try:
                chunk = channel.recv(count - len(chunks))
            except TimeoutError:
                continue
            if not chunk:
                raise EOFError("native worker response channel closed")
            chunks.extend(chunk)
        return bytes(chunks)

    def exchange(operation, arguments):
        sequence = len(trace)
        message = {"protocol_version": INTEGRATED_WORKER_PROTOCOL_VERSION, "sequence": sequence,
                   "operation": operation, "arguments": arguments}
        frame = encode_framed_message(message, max_bytes=MAX_REQUEST_BYTES)
        (directory / f"request_{sequence:04d}.bin").write_bytes(frame)
        record = {"request": message, "response": None}
        trace.append(record)
        channel.sendall(frame)
        header = read_exact(4)
        size = decode_frame_header(header, max_bytes=MAX_RESPONSE_BYTES)
        encoded = read_exact(size)
        (directory / f"response_{sequence:04d}.bin").write_bytes(header + encoded)
        response = validate_response_envelope(
            decode_canonical_message(encoded, max_bytes=MAX_RESPONSE_BYTES),
            expected_version=INTEGRATED_WORKER_PROTOCOL_VERSION,
        )
        record["response"] = response
        if response["sequence"] != sequence or response["status"] != "ok":
            raise RuntimeError("native request did not succeed: " + operation)
        if operation == "CLOSE_SESSION":
            # Timestamp the actual complete successful response separately from
            # proof parsing, cleanup and observed/reaped process exit.
            outcome["close_ack_observed_ns"] = time.monotonic_ns()
        return response["payload"]

    try:
        while not (directory / "worker_ready.json").exists():
            if stop.wait(0.01) or time.monotonic() >= deadline:
                raise TimeoutError("native worker did not become ready")
        manifest = IntegratedWorkerRuntimeManifest.from_payload(
            json.loads((directory / "worker_manifest.json").read_text(encoding="utf8")),
        )
        hello = exchange("HELLO", {"manifest_sha256": manifest.sha256, "lane_id": manifest.lane_id,
                                   "device_ordinal": manifest.device_ordinal})
        # Parsing checks wire consistency; the child's controller retains authority.
        ClosedNativeLaneWorkerEvidence.from_payload(hello["worker_evidence"])
        exchange("OPEN_FROZEN_SESSION", {"gate_sha256": manifest.gate_sha256,
                                        "workload_set_sha256": manifest.workload_set_sha256})
        prepared = exchange("PREPARE_FROZEN_ROW", {key: request[key] for key in (
            "row_id", "case_id", "repeat_index", "sample_count")})
        tokens = prepared["prepared_tokens"]
        if len(tokens) != request["sample_count"]:
            raise RuntimeError("native worker prepared a different sample count")
        for token in tokens:
            pending = exchange("EXECUTE_PREPARED_SAMPLE", {"prepared_token": token})
            resolved = exchange("RESOLVE_PREPARED_SAMPLE", pending)
            exchange("RELEASE_RESOLVED_SAMPLE", resolved)
        close = exchange("CLOSE_SESSION", {})
        proof = PostReclaimIdleEvidence.from_payload(close["post_reclaim_idle"])
        if proof.to_payload()["reclaimed_count"] != len(tokens):
            raise RuntimeError("close did not account for all actual submitted samples")
        outcome.update(completed=True)
    except BaseException as error:
        outcome.update(completed=False, error_type=type(error).__name__, error=str(error))
    finally:
        mark(directory, "worker_request_trace.json", json.dumps(trace, sort_keys=True, separators=(",", ":")))
        try:
            channel.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _observe_worker_client_exit(core, command, stop, outcome, now_ns):
    """Bound post-ACK exit and client failure with the original supervisor.

    This executes on the supervisor thread even if its request client or the
    worker is stalled. Protocol completion and process completion stay separate.
    """
    if "terminal_transport_closed_ns" in outcome:
        return
    acknowledged = outcome.get("close_ack_observed_ns")
    expired = (acknowledged is not None
               and now_ns >= acknowledged + core.configuration.operation_timeout_ms * 1_000_000)
    if not expired and outcome.get("completed") is not False:
        return
    if expired:
        outcome["close_exit_wait_expired_ns"] = now_ns
    stop.set()
    try:
        command.shutdown(socket.SHUT_RDWR)
    except OSError as shutdown_error:
        outcome["terminal_shutdown_error"] = type(shutdown_error).__name__
        try:
            command.close()
        except OSError as close_error:
            outcome["terminal_close_error"] = type(close_error).__name__
    outcome["terminal_transport_closed_ns"] = now_ns
    # fault() retains an earlier operation/emergency deadline and fences guard
    # ACK writes. It cannot grant a worker completion, release or second sync.
    if core.state not in {"TERMINATING", "EXITED"}:
        core.fault("process_observation_failed")


def _read_worker_lifecycle(directory, *, native_worker_mode):
    """Use the actual terminal record when hard exit bypasses the driver tail."""
    if (directory / "worker.json").exists():
        return json.loads((directory / "worker.json").read_text(encoding="utf-8"))
    worker = None
    if native_worker_mode and (directory / "worker_journal.bin").exists():
        from optical_worker.guarded_evidence import GuardedTerminationEvidence
        from optical_worker.protocol import MAX_RESPONSE_BYTES, decode_canonical_message, decode_frame_header

        raw = (directory / "worker_journal.bin").read_bytes()
        while raw:
            length = decode_frame_header(raw[:4], max_bytes=MAX_RESPONSE_BYTES)
            if len(raw) < length + 4:
                raise RuntimeError("native worker terminal journal has a partial frame")
            record = decode_canonical_message(raw[4:4 + length], max_bytes=MAX_RESPONSE_BYTES)
            raw = raw[4 + length:]
            if record["kind"] in {"transport_termination", "native_preparation_aborted"}:
                terminal = GuardedTerminationEvidence.from_payload(record["evidence"])
                if terminal.sha256 != record["evidence_sha256"]:
                    raise RuntimeError("native terminal journal digest mismatch")
                worker = terminal.lifecycle.to_payload()
    if worker is None and (directory / "prefix.json").exists():
        worker = json.loads((directory / "prefix.json").read_text(encoding="utf-8"))
    return worker


def _process_identity(pid):
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = stat[stat.rindex(")") + 2:].split()
    return {"pid": pid, "ppid": int(fields[1]), "start_ticks": int(fields[19])}


def supervise(fault, liveness_fd, directory):
    source_mode = fault in SOURCE_FAULTS
    runtime_mode = fault in RUNTIME_FAULTS
    native_worker_mode = fault in RUNTIME_WORKER_CASES
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
    command, worker_command = socket.socketpair() if native_worker_mode else (None, None)
    extra_args = [str(worker_command.fileno()), str(liveness_fd)] if native_worker_mode else []
    inherited = ((worker_channel.fileno(), worker_command.fileno(), liveness_fd)
                 if native_worker_mode else (worker_channel.fileno(),))
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
            *extra_args,
        ],
        pass_fds=inherited,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    worker_channel.close()
    core = GuardSupervisorCore(config)
    transport = GuardSupervisorTransport(channel, core)
    started = time.monotonic()
    client = None
    client_stop, client_outcome = threading.Event(), {}
    if native_worker_mode:
        worker_command.close()
        client = threading.Thread(target=_native_worker_requests, args=(
            command, directory, request, client_stop, started + outer_bound, client_outcome,
        ))
        client.start()
    injected = False
    parent_alive = True
    try:
        if native_worker_mode:
            mark(directory, "process_identity.json", json.dumps({
                "worker": _process_identity(process.pid), "supervisor": _process_identity(os.getpid()),
                "parent": _process_identity(os.getppid()),
            }))
        while True:
            if time.monotonic() - started > outer_bound:
                raise RuntimeError("outer test fail-bound; NOT successful supervisor termination")
            code = process.poll()
            if code is not None:
                core.observe_exit(code)
                if native_worker_mode:
                    client_outcome["worker_exit_observed_ns"] = time.monotonic_ns()
                break
            if native_worker_mode:
                _observe_worker_client_exit(core, command, client_stop, client_outcome, time.monotonic_ns())
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
            targeted_ack = (fault in RUNTIME_ACK_FAULTS and pending is not None
                            and pending["sequence"] == (6 if fault.startswith("runtime_terminal_") else 3)
                            and pending["operation"] == ("DISARMED" if fault.endswith("stalled_disarm") else "ARMED"))
            if pending is not None and not injected and (fault not in RUNTIME_ACK_FAULTS or targeted_ack):
                if targeted_ack:
                    mark(directory, "protocol_fault_entered.json", json.dumps({"fault": fault, "pending": pending}))
                if fault.endswith("wrong_ack"):
                    wrong = dict(pending, session="d" * 32)
                    channel.send(encode_framed_message(wrong, max_bytes=MAX_BODY_BYTES))
                    injected = True
                elif fault.endswith("ack_eof"):
                    channel.shutdown(socket.SHUT_WR)
                    injected = True
                elif targeted_ack:
                    # Drop exactly the selected real ACK. The original core's
                    # pending-ACK clock remains armed and is never refreshed.
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
            if ((fault not in RUNTIME_ACK_FAULTS and fault.endswith(("no_ack", "wrong_ack", "ack_eof")))
                    or targeted_ack):
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
        if native_worker_mode:
            client_outcome["worker_reaped_ns"] = time.monotonic_ns()
        if client is not None:
            client.join(timeout=1)
            if client.is_alive():
                client_stop.set()
                command.shutdown(socket.SHUT_RDWR)
                client.join(timeout=1)
            if client.is_alive():
                raise RuntimeError("native request client was not joined")
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        worker = _read_worker_lifecycle(directory, native_worker_mode=native_worker_mode)
        artifact = {
            "schema_version": VALIDATION_VERSION,
            "worker": worker,
            "supervisor": core.evidence(require_final=True).to_payload(),
        }
        validate_custody_artifact(artifact)
        result = {
            "artifact": artifact,
            "stderr": stderr,
            "parent_alive": parent_alive,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "worker_pid": process.pid,
            "worker_returncode": process.returncode,
            "injected": injected,
        }
        if native_worker_mode:
            result["worker_protocol"] = client_outcome
        return result
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        process.stderr.close()
        channel.close()
        if client is not None:
            client_stop.set()
            try:
                command.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.join(timeout=1)
            command.close()


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
