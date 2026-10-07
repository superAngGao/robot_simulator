"""Fixed C2d3 captured-runtime sacrificial child; never a product entry.

Guardian/source custody precedes all third-party imports. The existing session
owns canonical materialization, the sole upload loops and explicit readback.
This driver only sequences them and exits under the production guard supervisor.
No user callback, sample execution, final readiness, renderer or result export.
"""

from __future__ import annotations

import atexit
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path


def _write(directory, name, payload):
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    temporary = directory / (name + ".partial")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, directory / name)


def _stall():
    while True:
        time.sleep(0.02)


def child(fault, fd, directory):
    if fault not in {"runtime_success", "runtime_terminal_sync_raise", "runtime_terminal_sync_hang"}:
        raise ValueError("not a fixed runtime fault selector")
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if set(request) != {
        "root",
        "revision",
        "lane",
        "catalog_sha256",
        "guard_configuration",
        "environment_path",
        "environment_sha256",
        "case_id",
        "go2_model_dir",
    }:
        raise ValueError("unexpected fixed runtime configuration field")
    root = Path(request["root"])
    _write(directory, "identity.json", {"pid": os.getpid(), "ppid": os.getppid()})
    atexit.register(_write, directory, "ordinary_atexit.json", {"ran": True})
    spec = importlib.util.spec_from_file_location("source_test", root / "optical_worker/native_bootstrap.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.__name__ = "__main__"
    sys.modules["__main__"] = entry
    session, controller = None, None
    phase = "source_verification"

    def observe():
        # Debug facts, not runtime-custody receipt/authority. Preserve each fixed
        # phase separately; a killed partial write cannot replace a good prefix.
        _write(directory, "prefix.json", controller.evidence().to_payload())
        admission = session.external
        native_metadata = []
        if admission is not None:
            for name, record in admission.records.items():
                if record["kind"] != "file_module":
                    continue
                loader = admission.loaders[name]
                if loader.source is not None:
                    continue
                module, spec = record["module"], admission.specs[name]
                raw = vars(module)
                native_metadata.append({
                    "name": name, "module_name": raw.get("__name__"),
                    "package": raw.get("__package__"), "file": raw.get("__file__"),
                    "spec_name": spec.name, "state": record["state"],
                    "exact_module": sys.modules.get(name) is module,
                    "exact_spec": raw.get("__spec__") is spec,
                    "exact_loader": raw.get("__loader__") is loader and spec.loader is loader,
                    "same_object_keys": sorted(key for key, value in sys.modules.items() if value is module),
                    "create_started": sum(e["name"] == name and e["operation"] == "create_started"
                                          for e in admission.events),
                    "exec_started": sum(e["name"] == name and e["operation"] == "exec_started"
                                        for e in admission.events),
                })
        _write(
            directory,
            phase + ".json",
            {
                "schema_version": "q54.c2d3.driver-observation.v0",
                "phase": phase,
                "external_events": [] if admission is None else list(admission.events),
                "native_module_metadata": native_metadata,
                "repository_trace": list(session.trace),
                "sample_submission_count": 0,
            },
        )

    try:
        session = entry._install(root, request["revision"], request["lane"], request["catalog_sha256"])
        entry._verify(session)
        session.bind_guarded_custody(socket.socket(fileno=fd), request["guard_configuration"])
        controller = session.bound_controller()
        phase = "setup_definitions"
        session.import_setup_definitions()
        observe()
        phase = "external_environment"
        payload = json.loads(Path(request["environment_path"]).read_text(encoding="utf-8"))
        session.configure_external_environment(payload, request["environment_sha256"])
        observe()
        phase = "canonical_materialization"
        session.materialize_native_source(request["case_id"], go2_model_dir=request["go2_model_dir"])
        observe()
        phase = "runtime_initialization"
        binding = session.initialize_native_runtime()
        observe()
        phase = "native_upload"
        session.upload_native_input(binding)
        observe()
        phase = "input_validation"
        report = session.validate_native_input()
        observe()
        _write(
            directory,
            "input_validation_report.json",
            {
                "manifest_sha256": session.prepared_input.manifest.sha256,
                "content_sha256": report.content_sha256,
                "topology_sha256": report.topology_sha256,
                "validated_arrays": list(report.validated_arrays),
                "final_execution_readiness_issued": False,
            },
        )
        phase = "terminal_disposal"

        def synchronize():
            _write(
                directory, "terminal_sync_entered.json", {"device_ordinal": controller.target_device_ordinal}
            )
            if fault == "runtime_terminal_sync_raise":
                raise RuntimeError("fixed terminal sync failure")
            if fault == "runtime_terminal_sync_hang":
                _stall()
            # The consumed binding remains exact/held, revalidated immediately
            # before readback. No import, discovery or new device here.
            if request["lane"] == "cuda":
                binding.module.cuda.synchronize(binding.target)
            else:
                from optical_worker.native_setup import _warp_synchronize_target_exact

                _warp_synchronize_target_exact(binding.module, binding.target)

        terminal = controller.terminate(
            fault="setup_failure",
            detection_phase="fixed_runtime_driver",
            emergency_target_synchronize=synchronize,
        )
        result = terminal.lifecycle.to_payload()
        _write(directory, "worker.json", result)
        os._exit(0 if result["summary"]["disposition"] == "aborted_synced_exit" else 74)
    except BaseException as error:
        # Use the SAME source/controller failure policy. Do not retry sync or
        # return a reusable Python disposition after partially submitted work.
        if session is not None:
            try:
                session.fail(error)
            except BaseException:
                pass  # retained process graph still ends through hard exit
            if controller is None:
                record = entry._SOURCE_CUSTODY.get(session)
                controller = None if record is None else record[0]
        try:
            # Bounded diagnostic chain, never a custody receipt or admission.
            # Keep the original lower-level cause visible when a dependency
            # wraps it as an optional-package/missing-trimesh error.
            causes, seen = [], set()
            cause = error
            while cause is not None and id(cause) not in seen and len(causes) < 8:
                seen.add(id(cause))
                causes.append({"type": type(cause).__name__, "message": str(cause)[:512]})
                cause = cause.__cause__ if cause.__cause__ is not None else cause.__context__
            admission = None if session is None else session.external
            # __init__ may have failed before events exists. Diagnostic output
            # must not replace the original terminal evidence in that case.
            events = None if admission is None else vars(admission).get("events")
            if type(events) is list:
                metadata = []
                partial = vars(admission)
                contexts = partial.get("failure_contexts", [])
                for failed_error, failed_name, native_create in contexts[:8]:
                    record = partial.get("records", {}).get(failed_name)
                    module = None if record is None else record.get("module")
                    raw = {} if module is None else vars(module)
                    spec = partial.get("specs", {}).get(failed_name)
                    loader = partial.get("loaders", {}).get(failed_name)
                    metadata.append({
                        "name": failed_name, "native_create": native_create,
                        "error_type": type(failed_error).__name__, "error": str(failed_error)[:512],
                        "module_name": raw.get("__name__"), "package": raw.get("__package__"),
                        "file": raw.get("__file__"),
                        "exact_spec": raw.get("__spec__") is spec,
                        "exact_loader": raw.get("__loader__") is loader,
                        "spec_loader_is_wrapper": spec is not None and spec.loader is loader,
                    })
                # Preserve the failed prefix as observations, never as an
                # admission rule or a successful runtime-custody receipt.
                _write(
                    directory,
                    "external_failure.json",
                    {
                        "phase": phase,
                        "events": list(events),
                        "failed": admission.failed,
                        "failure_metadata": metadata,
                    },
                )
            _write(
                directory,
                "failure.json",
                {"phase": phase, "type": type(error).__name__, "message": str(error)[:1024],
                 "exception_chain": causes},
            )
            if controller is not None and controller.termination_evidence is not None:
                _write(directory, "worker.json", controller.termination_evidence.lifecycle.to_payload())
        finally:
            os._exit(74)  # even evidence I/O failure must not run ordinary cleanup


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "child":
        raise ValueError("fixed sacrificial child invocation required")
    child(sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]))
