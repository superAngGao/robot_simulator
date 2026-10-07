"""Fixed CPU source-custody child; never a worker role, IPC or plugin API.

Only stdlib is loaded before the actual captured-source bootstrap. Fault
injection deliberately instruments real leaves outside the admitted release.
The supervisor is the existing guarded_custody_process production-core driver.
"""

import atexit
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path


def mark(directory, name, value="1"):
    (directory / name).write_text(value, encoding="utf-8")


def stall():
    while True:
        time.sleep(0.02)


def child(fault, fd, directory):
    request = json.loads((directory / "request.json").read_text())
    root = Path(request["root"])
    mark(directory, "identity.json", json.dumps({"pid": os.getpid(), "ppid": os.getppid()}))
    atexit.register(mark, directory, "atexit")
    spec = importlib.util.spec_from_file_location("source_test", root / "optical_worker/native_bootstrap.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.__name__ = "__main__"
    sys.modules["__main__"] = entry
    session = entry._install(root, request["revision"], request["lane"], request["catalog_sha256"])
    entry._verify(session)
    original = RuntimeError("fixed CPU source fault: " + fault)
    entry.original_error = original
    original_compile = entry._CapturedSourceLoader.source_to_code
    original_exec = entry._CapturedSourceLoader.exec_module

    class Sentinel:
        def __del__(self):
            mark(directory, "destructor")

    def compile_source(loader, data, path, *args, **kwargs):
        if fault == "source_prelude_raise" and loader.name == "optical_worker.guarded_lifecycle":
            raise original
        if loader.name in entry._SETUP_DEFINITIONS:
            controller = session.bound_controller()
            module = session.modules[loader.name][0]
            assert controller.guardian.contains_exact_owner(module)
            assert controller._guard_client.state == "ARMED"
            module.fixed_test_sentinel = Sentinel()
            mark(directory, "body_entry")
            mark(directory, "prefix.json", json.dumps(controller.evidence().to_payload()))
            if fault in {"source_import_hang", "source_parent_loss"}:
                stall()
            if fault == "source_compile_raise" or (
                fault == "source_caught_nested" and loader.name.endswith("native_contract")
            ):
                raise original
            if fault == "source_body_raise":
                # Deliberate mutant to reach body failure, NOT frozen success.
                data += b'\nimport sys\nraise sys.modules["__main__"].original_error\n'
            if fault == "source_caught_nested" and loader.name.endswith("native_setup"):
                data = b"try:\n from . import native_contract\nexcept RuntimeError:\n pass\n"
        return original_compile(loader, data, path, *args, **kwargs)

    def execute(loader, module):
        original_exec(loader, module)
        if loader.name == "optical_worker.guarded_lifecycle" and fault == "source_module_adopt":
            kind = module.NativeGuardedSetupLifecycle
            old_adopt = kind.adopt_host_owner

            def adopt(self, owner, **kwargs):
                if kwargs["operation_id"] == "optical_worker_native_setup":
                    owner.fixed_test_sentinel = Sentinel()
                    raise original
                return old_adopt(self, owner, **kwargs)

            kind.adopt_host_owner = adopt

    entry._CapturedSourceLoader.source_to_code = compile_source
    entry._CapturedSourceLoader.exec_module = execute
    controller = None
    try:
        session.bind_guarded_custody(
            socket.socket(fileno=fd),
            {
                "protocol_version": "q54.closed-worker.guarded-wait.v1",
                "session": "c" * 32,
                "device_ordinal": 1,
                "ack_timeout_ms": 1000,
                "operation_timeout_ms": 5000,
            },
        )
        controller = session.bound_controller()
        if fault == "source_audit_raise":
            old_audit = session.audit

            def audit():
                if session.phase == "importing" and len(session.modules) == 12:
                    raise original
                return old_audit()

            session.audit = audit
        session.import_setup_definitions()
        session.audit()
        assert len(session.modules) == 12
        assert all(controller.guardian.contains_exact_owner(x[0]) for x in session.modules.values())
        if fault == "source_cleanup_hang":
            # At the real whole-graph release leaf, after terminal ACK; no
            # replacement lifecycle or cleanup implementation is being tested.
            type(controller.guardian).release_whole_graph = lambda *args: stall()
        mark(directory, "prefix.json", json.dumps(controller.evidence().to_payload()))
        terminal = controller.terminate(
            fault="setup_failure", detection_phase="fixed_source_driver", emergency_target_synchronize=None
        )
        mark(directory, "worker.json", json.dumps(terminal.lifecycle.to_payload()))
        assert terminal.lifecycle.to_payload()["summary"]["disposition"] == "cancelled_before_submission_exit"
        assert not {"torch", "warp", "numpy"}.intersection(sys.modules)
        mark(directory, "no_runtime")
        if fault == "source_exit_hang":
            stall()
        os._exit(0)
    except BaseException:
        if controller is None:
            binding = entry._SOURCE_CUSTODY.get(session)
            controller = None if binding is None else binding[0]
        assert session.failed
        if controller is not None:
            terminal = controller.termination_evidence
            assert terminal is not None
            payload = terminal.lifecycle.to_payload()
            assert payload["summary"]["custody"] == "full_quarantine"
            assert payload["summary"]["release_status"] == "not_authorized"
            if fault in {
                "source_module_adopt",
                "source_compile_raise",
                "source_body_raise",
                "source_caught_nested",
                "source_audit_raise",
            }:
                assert any(error is original for error in controller._failure_holds)
            mark(directory, "worker.json", json.dumps(payload))
        assert not (directory / "destructor").exists()
        assert not {"torch", "warp", "numpy"}.intersection(sys.modules)
        mark(directory, "no_runtime")
        os._exit(74)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "child":
        raise ValueError("fixed sacrificial child invocation required")
    child(sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]))
