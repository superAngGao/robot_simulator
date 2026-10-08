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


RUNTIME_NATIVE_FAULTS = (
    "runtime_success", "runtime_terminal_sync_raise", "runtime_terminal_sync_hang",
    "runtime_numpy_create_before_raise", "runtime_numpy_create_after_raise",
    "runtime_numpy_exec_before_raise", "runtime_numpy_exec_after_raise", "runtime_numpy_create_hang",
    "runtime_backend_exec_before_raise", "runtime_backend_exec_after_raise",
    "runtime_init_hang", "runtime_stream_return_before_adopt",
    "runtime_allocate_before_raise", "runtime_allocate_after_raise",
    "runtime_copy_before_raise", "runtime_copy_after_raise", "runtime_copy_hang",
    "runtime_event_create_after_raise", "runtime_event_record_after_raise",
    "runtime_ready_wait_before_raise", "runtime_ready_wait_after_raise", "runtime_ready_wait_hang",
    "runtime_owner_register_raise", "runtime_storage_audit_raise", "runtime_readback_copy_after_raise",
    "runtime_terminal_cleanup_hang", "runtime_terminal_exit_hang",
)
RUNTIME_PHASES = ("setup_definitions", "external_environment", "canonical_materialization",
                  "runtime_initialization", "native_upload", "input_validation")
RUNTIME_ACK_FAULTS = tuple("runtime_" + phase + "_" + failure
                          for phase in ("init", "terminal")
                          for failure in ("no_ack", "wrong_ack", "ack_eof")) + ("runtime_init_stalled_disarm",)
RUNTIME_PARENT_FAULTS = tuple("runtime_" + phase + "_parent_eof" for phase in RUNTIME_PHASES)
RUNTIME_FAULTS = RUNTIME_NATIVE_FAULTS + RUNTIME_ACK_FAULTS + RUNTIME_PARENT_FAULTS


def _install_runtime_fault(fault, entry, session, controller, directory):
    """Fault real fixed entrypoints; never substitute an uploader or guard.

    All wrappers and their real return values are retained by the already held
    source session. The production source closure has no fault selector.
    """
    if fault in {"runtime_success", "runtime_terminal_sync_raise", "runtime_terminal_sync_hang"} or fault in RUNTIME_ACK_FAULTS:
        return None
    state = {"fired": False, "returned": [], "originals": []}
    session.runtime_owners.append(state)

    def trip(boundary, returned=None):
        assert not state["fired"]
        state["fired"] = True
        if returned is not None:
            state["returned"].append(returned)
        _write(directory, "fault_entered.json", {
            "fault": fault, "boundary": boundary,
            "guard": controller._guard_reference,
            "lifecycle_event_count": len(controller._guard_events),
            "returned_owner_held": returned is not None,
        })
        if fault.endswith(("hang", "parent_eof")):
            _stall()
        raise RuntimeError("fixed actual runtime fault: " + fault)

    def replace(owner, name, replacement):
        state["originals"].append(getattr(owner, name))
        setattr(owner, name, replacement)

    if fault in RUNTIME_PARENT_FAULTS:
        sequence = RUNTIME_PARENT_FAULTS.index(fault)
        original = controller.segment

        def segment(role="setup"):
            actual = original(role)

            class ObservedSegment:
                def __enter__(self):
                    result = actual.__enter__()
                    if controller._guard_reference["sequence"] == sequence:
                        trip("armed_phase." + RUNTIME_PHASES[sequence])
                    return result

                def __exit__(self, kind, value, traceback):
                    return actual.__exit__(kind, value, traceback)

            return ObservedSegment()

        replace(controller, "segment", segment)
    elif fault.startswith(("runtime_numpy_", "runtime_backend_")):
        numpy = fault.startswith("runtime_numpy_")
        target = "numpy._core._multiarray_umath" if numpy else (
            "torch._C" if session.lane == "cuda" else "warp._src.context")
        create = "_create_" in fault
        # Captured Python bodies execute through super().exec_module on the
        # actual SourceLoader wrapper, not the retained SourceFileLoader
        # delegate (which is used only for pinned resolution/metadata).
        loader_type = (importlib.machinery.ExtensionFileLoader if numpy or session.lane == "cuda"
                       else importlib.abc.SourceLoader)
        name = "create_module" if create else "exec_module"
        original = getattr(loader_type, name)

        def native(loader, subject):
            if loader.name != target:
                return original(loader, subject)
            if "_before_" in fault or fault.endswith("hang"):
                trip(target + "." + name + ".before")
            returned = original(loader, subject)
            trip(target + "." + name + ".after", returned if create else subject)

        replace(loader_type, name, native)
    elif fault == "runtime_init_hang":
        original = session._runtime_leaf

        def leaf(operation, kind):
            result = original(operation, kind)
            if operation == "indexed_runtime_stream":
                trip(operation)
            return result

        replace(session, "_runtime_leaf", leaf)
    elif fault == "runtime_stream_return_before_adopt":
        original = session._observe_runtime_return

        def returned(producer, owner):
            original(producer, owner)
            if producer == "runtime_stream":
                trip(producer + ".returned_before_adopt", owner)

        replace(session, "_observe_runtime_return", returned)
    elif fault == "runtime_terminal_cleanup_hang":
        def cleanup(authority):
            trip("guardian.release_whole_graph")

        replace(controller.guardian, "release_whole_graph", cleanup)
    return state, trip, replace


def _install_upload_fault(fault, session, controller, injection):
    if injection is None:
        return
    state, trip, replace = injection
    setup = session.modules["optical_worker.native_setup"][0]
    prefix = "_torch_" if session.lane == "cuda" else "_warp_"
    names = {
        "runtime_allocate_before_raise": "empty_exact", "runtime_allocate_after_raise": "empty_exact",
        "runtime_copy_before_raise": "copy_exact", "runtime_copy_after_raise": "copy_exact",
        "runtime_copy_hang": "copy_exact", "runtime_readback_copy_after_raise": "copy_exact",
        "runtime_event_create_after_raise": "event_exact", "runtime_event_record_after_raise": "record_event_exact",
        "runtime_ready_wait_before_raise": "wait_event_host_exact",
        "runtime_ready_wait_after_raise": "wait_event_host_exact", "runtime_ready_wait_hang": "wait_event_host_exact",
    }
    if fault in names:
        name = prefix + names[fault]
        original = getattr(setup, name)

        def helper(*args, **kwargs):
            # D2H uses the same original copy helper under the readback phase.
            if fault == "runtime_readback_copy_after_raise" and session.phase != "validating_input":
                return original(*args, **kwargs)
            if "_before_" in fault or fault.endswith("hang"):
                trip(name + ".before")
            returned = original(*args, **kwargs)
            trip(name + ".after", returned)

        replace(setup, name, helper)
    elif fault == "runtime_owner_register_raise":
        original = controller.adopt_returned_device_owner

        def adopt(owner, **kwargs):
            if kwargs["role"] == session.lane + "_a2_device_array":
                trip("array.returned_before_adopt", owner)
            return original(owner, **kwargs)

        replace(controller, "adopt_returned_device_owner", adopt)
    elif fault == "runtime_storage_audit_raise":
        contract = session.modules["optical_worker.native_contract"][0]
        original = contract.observe_prepared_storage

        def audit(prepared):
            original(prepared)
            trip("prepared_storage.after_actual_audit", prepared)

        replace(contract, "observe_prepared_storage", audit)


def _write(directory, name, payload):
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    _write_bytes(directory, name, data)


def _write_bytes(directory, name, data):
    temporary = directory / (name + ".partial")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, directory / name)


def _persist_runtime_custody(directory, record, selection):
    # Ordered atomic writes are deliberately not a two-file transaction.
    # Readers reject the pair if either write failed or the sidecar is stale.
    if record is not None:
        _write_bytes(directory, "runtime_custody.json", record.encoded)
    _write(directory, "runtime_custody_selection.json", selection)


def _persist_final_runtime_custody(directory, session):
    observation = session.runtime_observation
    record, selection = observation.final()
    diagnostic_present = False
    if session.failed or observation.failures:
        try:
            admission = {} if session.external is None else vars(session.external)
            _write(directory, "runtime_custody_failure_diagnostic.json", {
                "schema_version": "q54.c2d3.runtime-custody-diagnostic.v0", "validating": False,
                "journal": observation.journal,
                "raw_native_calls": [{key: frame.get(key) for key in ("sequence", "operation", "phase", "state")}
                                     for frame in admission.get("native_calls", [])],
                "raw_native_global_events": admission.get("native_global_events"),
                "raw_module_replacement_events": admission.get("replacement_events"),
                "raw_interpreter_alias_events": vars(session).get("interpreter_alias_events"),
                "observed_lifecycle_event_count": selection["observed_lifecycle_event_count"],
                "collector_errors": [type(item).__name__ for item in observation.failures],
            })
            diagnostic_present = True
        except BaseException as diagnostic_error:
            observation.failures.append(diagnostic_error)
    _persist_runtime_custody(directory, record, dict(selection, failure_diagnostic_present=diagnostic_present))


def _stall():
    while True:
        time.sleep(0.02)


def _unrecorded_module_observations(admission):
    """Bounded failure diagnostics, not classification or admission authority.

    Includes ordinary stdlib/repository imports: lack of an external record
    alone does not mean forbidden. Observe raw current objects and held-owner
    identity without importing, resolving, adopting, deleting or normalizing.
    """
    partial = vars(admission)
    baseline, records = partial.get("baseline", {}), partial.get("records", {})
    retained = partial.get("retained", [])
    entries = sorted((key, value) for key, value in tuple(sys.modules.items())
                     if key not in baseline and key not in records)
    observations = []
    for key, value in entries[:256]:
        raw = vars(value) if type(value) is type(sys) else {}
        spec = raw.get("__spec__")
        observations.append({
            "key": key, "type": type(value).__name__, "module_name": raw.get("__name__"),
            "package": raw.get("__package__"), "file": raw.get("__file__"),
            "spec_name": None if spec is None else spec.name,
            "spec_origin": None if spec is None else spec.origin,
            "loader_is_none": raw.get("__loader__") is None,
            "held_in_failure_graph": any(owner is value for owner in retained),
            "raw_attribute_names": sorted(raw)[:32],
        })
    return {"total": len(entries), "truncated": len(entries) > 256, "modules": observations}


def child(fault, fd, directory):
    if fault not in RUNTIME_FAULTS:
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
    trusted_inputs = json.loads((directory / "runtime_custody_inputs.json").read_bytes())
    environment_payload = json.loads(Path(request["environment_path"]).read_bytes())
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
                "native_global_observations": (
                    None if admission is None else admission.native_global_observations()
                ),
                "module_replacement_observations": (
                    None if admission is None else admission.replacement_observations()
                ),
                "interpreter_alias_observations": session.interpreter_alias_observations(),
                "native_module_metadata": native_metadata,
                "repository_trace": list(session.trace),
                "sample_submission_count": 0,
            },
        )
        record = session.runtime_observation.last_validated
        count = len(controller._guard_events)
        if record is None or record.to_payload()["observation_cut"]["lifecycle_event_count"] != count:
            raise RuntimeError("normal custody checkpoint does not match current lifecycle")
        _persist_runtime_custody(directory, record, {
            "schema_version": "q54.p12.4.runtime-custody-selection.v1", "selection": "current",
            "runtime_record_sha256": record.sha256, "selected_lifecycle_event_count": count,
            "observed_lifecycle_event_count": count, "failure_diagnostic_present": False,
        })

    try:
        session = entry._install(root, request["revision"], request["lane"], request["catalog_sha256"])
        entry._verify(session)
        session.bind_guarded_custody(socket.socket(fileno=fd), request["guard_configuration"])
        controller = session.bound_controller()
        session.configure_runtime_custody(environment_payload, trusted_inputs)
        observe()
        injection = _install_runtime_fault(fault, entry, session, controller, directory)
        phase = "setup_definitions"
        session.import_setup_definitions()
        observe()
        phase = "external_environment"
        session.configure_external_environment(environment_payload, request["environment_sha256"])
        observe()
        phase = "canonical_materialization"
        session.materialize_native_source(request["case_id"], go2_model_dir=request["go2_model_dir"])
        observe()
        phase = "runtime_initialization"
        binding = session.initialize_native_runtime()
        observe()
        _install_upload_fault(fault, session, controller, injection)
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

        session.runtime_observation.freeze_adoptions()
        terminal = controller.terminate(
            fault="setup_failure",
            detection_phase="fixed_runtime_driver",
            emergency_target_synchronize=synchronize,
        )
        result = terminal.lifecycle.to_payload()
        _write(directory, "worker.json", result)
        _persist_final_runtime_custody(directory, session)
        if fault == "runtime_terminal_exit_hang":
            injection[1]("after_terminal_evidence_before_hard_exit")
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
            if controller is not None and controller.termination_evidence is not None:
                _write(directory, "worker.json", controller.termination_evidence.lifecycle.to_payload())
            if session is not None and controller is not None:
                _persist_final_runtime_custody(directory, session)
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
                        "raw_native_calls": [{key: frame.get(key) for key in ("sequence", "operation", "phase", "state")}
                                             for frame in partial.get("native_calls", [])],
                        "raw_native_global_events": partial.get("native_global_events"),
                        "raw_module_replacement_events": partial.get("replacement_events"),
                        "raw_interpreter_alias_events": vars(session).get("interpreter_alias_events"),
                        "failed": admission.failed,
                        "failure_metadata": metadata,
                        "unrecorded_modules": _unrecorded_module_observations(admission),
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
