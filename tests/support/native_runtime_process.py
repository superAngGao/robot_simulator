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
RUNTIME_SCOPE_FAULTS = ("runtime_foreign_stream_return", "runtime_unknown_stream_raise")
RUNTIME_GLOBAL_FAULTS = ("runtime_native_global_pending_raise", "runtime_native_global_admitted_raise")
RUNTIME_CUDA_ONLY_FAULTS = ("runtime_c1_stream_only_copy", "runtime_module_replacement_pending_raise",
                            "runtime_interpreter_alias_after_raise")
RUNTIME_OBSERVATION_FAULTS = ("runtime_scope_success", "runtime_scope_getter_error",
                            "runtime_scope_after_return_raise", "runtime_scope_after_append_raise",
                            "runtime_scope_missing_warp_scope", "runtime_scope_c1_copy")
RUNTIME_FAULTS = (RUNTIME_NATIVE_FAULTS + RUNTIME_ACK_FAULTS + RUNTIME_PARENT_FAULTS
                  + RUNTIME_SCOPE_FAULTS + RUNTIME_GLOBAL_FAULTS + RUNTIME_CUDA_ONLY_FAULTS
                  + RUNTIME_OBSERVATION_FAULTS)


def _install_runtime_fault(fault, entry, session, controller, directory):
    """Fault real fixed entrypoints; never substitute an uploader or guard.

    All wrappers and their real return values are retained by the already held
    source session. The production source closure has no fault selector.
    """
    if fault in {"runtime_success", "runtime_terminal_sync_raise", "runtime_terminal_sync_hang"} or fault in RUNTIME_ACK_FAULTS + RUNTIME_OBSERVATION_FAULTS:
        return None
    state = {"fired": False, "returned": [], "originals": []}
    session.runtime_owners.append(state)

    def mark_fault(boundary, returned=None):
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

    def trip(boundary, returned=None):
        mark_fault(boundary, returned)
        if fault.endswith(("hang", "parent_eof")):
            _stall()
        raise RuntimeError("fixed actual runtime fault: " + fault)

    def replace(owner, name, replacement):
        state["originals"].append(getattr(owner, name))
        setattr(owner, name, replacement)

    if fault in RUNTIME_GLOBAL_FAULTS or fault in RUNTIME_CUDA_ONLY_FAULTS[1:]:
        original = session.runtime_observation.append

        def append(kind, details):
            result = original(kind, details)
            selected = False
            if kind == "native_global" and fault in RUNTIME_GLOBAL_FAULTS:
                row = session.external.native_global_events[details["event_index"]]
                selected = row["operation"] == ("observed_pending" if "pending" in fault else "admitted")
            elif kind == "module_replacement" and fault == "runtime_module_replacement_pending_raise":
                row = session.external.replacement_events[details["event_index"]]
                selected = row["operation"] == "observed_pending"
            elif kind == "interpreter_alias" and fault == "runtime_interpreter_alias_after_raise":
                selected = True
            if selected and not state["fired"]:
                trip(kind + ".after_original_observation")
            return result

        replace(session.runtime_observation, "append", append)
    elif fault in RUNTIME_PARENT_FAULTS:
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
    elif fault in RUNTIME_SCOPE_FAULTS:
        original = session._runtime_leaf

        def leaf(operation, kind):
            result = original(operation, kind)
            if operation == "indexed_runtime_stream":
                module = sys.modules["torch" if session.lane == "cuda" else "warp"]
                stream_owner = module.cuda if session.lane == "cuda" else module
                stream_name = "Stream" if session.lane == "cuda" else "get_stream"
                make_stream = getattr(stream_owner, stream_name)

                def foreign_stream(*args, **kwargs):
                    # Deliberate negative only: the already-entered real helper
                    # returns/touches device0 against configured target1. This
                    # cannot be used as a normal target-only observation.
                    if session.lane == "cuda":
                        target = module.device("cuda", 0)
                        stream = make_stream(device=target)
                        ordinal = stream.device.index
                        handle = int(stream.cuda_stream)
                        context_present = vars(module._C)["_cuda_hasPrimaryContext"](0)
                        device_uuid = None  # external fixed observer records UUID mapping
                    else:
                        target = module.get_device("cuda:0")
                        stream = make_stream(target)
                        ordinal = stream.device.ordinal
                        handle = int(stream.cuda_stream or 0)
                        context_present = vars(target)["_context"] is not None
                        device_uuid = target.uuid
                    state["returned"].append(target)
                    state["returned"].append(stream)
                    _write(directory, "foreign_work.json", {
                        "lane": session.lane, "configured_target": controller.target_device_ordinal,
                        "actual_stream_device": ordinal, "stream_handle": handle,
                        "foreign_context_present": context_present, "device_uuid": device_uuid,
                        "guard": controller._guard_reference,
                        "scope_establishment": "intentional foreign stream/context after original helper entry",
                    })
                    assert ordinal == 0 and context_present
                    if fault == "runtime_unknown_stream_raise":
                        # The unchanged production initialization exception
                        # handler must establish UNKNOWN; this injector does
                        # not call mark_unknown_device_scope on its behalf.
                        trip("foreign_stream.before_return_raise", stream)
                    mark_fault("foreign_stream.returned_to_production", stream)
                    return stream

                replace(stream_owner, stream_name, foreign_stream)
            return result

        replace(session, "_runtime_leaf", leaf)
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
        def cleanup(guardian, authority):
            assert guardian is controller.guardian
            trip("guardian.release_whole_graph")

        replace(type(controller.guardian), "release_whole_graph", cleanup)
    return state, trip, replace


def _install_upload_fault(fault, session, controller, injection, directory):
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
    if fault == "runtime_c1_stream_only_copy":
        assert session.lane == "cuda"

        def stream_only(module, destination, source, stream):
            native = module._C
            before = {"logical": vars(native)["_cuda_getDevice"](),
                      "contexts": [vars(native)["_cuda_hasPrimaryContext"](i) for i in range(8)]}
            assert before == {"logical": 0, "contexts": [False, True, False, False, False, False, False, False]}
            # The already-known C1 counterexample, deliberately substituted at
            # one helper only. No allocation/copy loop or fallback is added.
            with module.cuda.stream(stream):
                destination.copy_(source, non_blocking=True)
            after = {"logical": vars(native)["_cuda_getDevice"](),
                     "contexts": [vars(native)["_cuda_hasPrimaryContext"](i) for i in range(8)]}
            _write(directory, "c1_scope.json", {"before": before, "after": after,
                    "guard": controller._guard_reference, "lifecycle_event_count": len(controller._guard_events)})
            assert after["logical"] == 0 and after["contexts"][0] is True
            controller.mark_unknown_device_scope(source="fixed_c1_observer", reason="observed_foreign_context")
            trip("c1_stream_only_copy.observed_foreign_context")

        replace(setup, "_torch_copy_exact", stream_only)
    elif fault in names:
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


class _NativeScopeObserver:
    """Fixed sacrificial observations; never an import/resource authority.

    Native calls run only under the original ticket. The production collector
    receives held primitive results and never invokes this object.
    """

    def __init__(self, session, controller, directory, environment, fault):
        self.session, self.controller, self.directory = session, controller, directory
        self.environment, self.fault = environment, fault
        self.held, self.raw, self.originals = [], [], []
        self.module = self.runtime = self.target = self.stream = self.binding = None
        self.library = None
        self.current_stream_baseline = None
        self.target_context = None
        self.scope_depth = 0
        session.runtime_owners.append(self)
        hold, leaf, returned, append = (session._hold_runtime_owner, session._runtime_leaf,
                                       session._observe_runtime_return, session.runtime_observation.append)
        self.originals.extend((hold, leaf, returned, append))

        def observed_hold(owner, role, *, device_ordinal=None):
            result = hold(owner, role, device_ordinal=device_ordinal)
            if role in {"torch_runtime_root", "warp_runtime_root"}:
                self.runtime = owner
            elif role in {"torch_target_descriptor", "warp_runtime_target"}:
                self.target = owner
            elif role == "runtime_setup_binding":
                self.binding = owner
            return result

        def observed_leaf(operation, kind):
            result = leaf(operation, kind)
            if operation == "indexed_runtime_stream":
                self.module = sys.modules["torch" if session.lane == "cuda" else "warp"]
                self._initialize_queries()
                self.stream_before = self.snapshot("runtime_stream", "before", self._latest_leaf())
            return result

        def observed_return(producer, owner):
            result = returned(producer, owner)
            if producer == "runtime_stream":
                self.stream = owner  # already held by production before this hook
                if self.fault == "runtime_scope_after_return_raise":
                    self._fault("after_real_stream_return")
                if session.lane == "warp":
                    assert vars(self.target)["_stream"] is owner
                    self.target_context = vars(self.target)["_context"]
                    assert type(self.target_context) is int and self.target_context > 0
                self.snapshot("runtime_stream", "after", self.stream_before[1], self.stream_before)
            return result

        def observed_append(kind, details):
            result = append(kind, details)
            if kind == "runtime_binding":
                assert self.binding is not None
                self.snapshot("runtime_binding", "point", len(controller._guard_events) - 1)
                if session.lane == "cuda":
                    self._check_cuda_deferred_state()
                self._wrap_helpers()
            return result

        session._hold_runtime_owner = observed_hold
        session._runtime_leaf = observed_leaf
        session._observe_runtime_return = observed_return
        session.runtime_observation.append = observed_append

    def _ticket(self):
        c = self.controller
        c._guard_client.require_live_ticket(c._guard_ticket, role="emergency" if c._terminating else "setup")

    def _fail(self, error):
        self.held.append(error)
        c = self.controller
        if not c._terminating and not c._replay.unknown:
            c.mark_unknown_device_scope(source="native_scope_observer", reason="native_observation_failed")
        try:
            _write(self.directory, "native_scope_diagnostic.json", {
                "schema_version": "q54.c2d3.native-scope-diagnostic.v0", "validating": False,
                "raw": self.raw, "exception_type": type(error).__name__, "message": str(error)})
        except BaseException as diagnostic_error:
            self.held.append(diagnostic_error)

    def _fault(self, boundary):
        _write(self.directory, "fault_entered.json", {
            "fault": self.fault, "boundary": boundary,
            "guard": self.controller._guard_reference,
            "lifecycle_event_count": len(self.controller._guard_events)})
        error = RuntimeError("fixed native observation fault: " + boundary)
        self._fail(error)
        raise error

    def _latest_leaf(self):
        return next(e["index"] for e in reversed(self.controller._guard_events)
                    if e["operation"] == "low_level_leaf_entered")

    def _initialize_queries(self):
        self._ticket()
        try:
            import ctypes
            import hashlib
            self.ctypes = ctypes
            cuda = self.session.lane == "cuda"
            key = ("site", "nvidia/cuda_runtime/lib/libcudart.so.12") if cuda else ("os_native", "libcuda.so.595.71.05")
            descriptor = next(f for f in self.environment["files"] if (f["root"], f["path"]) == key)
            root = next(r["path"] for r in self.environment["roots"] if r["id"] == key[0])
            path = Path(root) / key[1]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["sha256"]
            mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                      if len(line.split()) >= 6 and line.split()[-1].startswith("/")}
            assert str(path.resolve(strict=True)) in mapped
            library = ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
            self.held.append(library)
            self.library = {name: descriptor[name] for name in ("root", "path", "sha256")}

            def function(name, arguments):
                value = getattr(library, name)
                self.held.append(value)
                value.argtypes, value.restype = arguments, ctypes.c_int
                return value

            if cuda:
                self.raw_device = function("cudaGetDevice", [ctypes.POINTER(ctypes.c_int)])
                native = vars(self.module)["_C"]
                self.logical = vars(native)["_cuda_getDevice"]
                self.primary = vars(native)["_cuda_hasPrimaryContext"]
                self.stream_query = vars(native)["_cuda_getCurrentRawStream"]
                self.count = vars(native)["_cuda_getDeviceCount"]()
                state = vars(sys.modules["torch.cuda"])
                assert state["_initialized"] is False
                queued = state["_queued_calls"]
                assert type(queued) is list and len(queued) == 3
                assert [item[0] for item in queued] == [state[name] for name in
                    ("_check_capability", "_check_cubins", "_register_triton_kernels")]
                tracker = state["_lazy_seed_tracker"]
                assert type(tracker) is vars(sys.modules["torch._utils"])["_LazySeedTracker"]
                assert vars(tracker) == {"manual_seed_all_cb": None, "manual_seed_cb": None, "call_order": []}
            else:
                pointer = ctypes.POINTER
                self.device_count = function("cuDeviceGetCount", [pointer(ctypes.c_int)])
                self.device_get = function("cuDeviceGet", [pointer(ctypes.c_int), ctypes.c_int])
                self.primary = function("cuDevicePrimaryCtxGetState", [ctypes.c_int, pointer(ctypes.c_uint), pointer(ctypes.c_int)])
                self.context_get = function("cuCtxGetCurrent", [pointer(ctypes.c_void_p)])
                self.context_device = function("cuCtxGetDevice", [pointer(ctypes.c_int)])
                count = ctypes.c_int(-1)
                self._result("cuDeviceGetCount", self.device_count(ctypes.byref(count)), count.value)
                self.count = count.value
                assert self.count == 8
                self.devices = []
                for ordinal in range(self.count):
                    device = ctypes.c_int(-1)
                    self._result("cuDeviceGet", self.device_get(ctypes.byref(device), ordinal), device.value)
                    self.devices.append(device.value)
                assert len(set(self.devices)) == self.count
                self.stream_query = self.runtime.core.wp_cuda_context_get_stream
            assert self.count == 8
            assert not any(name in os.environ for name in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "LD_LIBRARY_PATH"))
        except BaseException as error:
            self._fail(error)
            raise

    def _result(self, name, result, value):
        self.raw.append({"getter": name, "result": result, "value": value if result == 0 else None})
        if result != 0:
            raise RuntimeError("native observer getter error: " + name + "=" + str(result))
        return value

    def _check_cuda_deferred_state(self):
        """Retain the prior reviewed no-Triton-execution acceptance check."""
        try:
            self._ticket()
            admission = self.session.external
            admission.audit()
            events = [e for e in admission.events if e["name"] == "triton"]
            assert len(events) == 1 and events[0]["operation"] == "file_resolved"
            assert "triton" not in sys.modules and "triton" not in admission.records
            spec, loader = admission.specs["triton"], admission.loaders["triton"]
            loader._check_binding()
            assert spec.loader is loader and spec.name == "triton"
            assert spec.origin == "/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages/triton/__init__.py"
            assert list(spec.submodule_search_locations) == [str(Path(spec.origin).parent)]
            assert type(loader.delegate) is importlib.machinery.SourceFileLoader
            assert loader.delegate.name == "triton" and loader.delegate.path == spec.origin
            operations = vars(vars(self.module)["_TritonLibrary"])["ops_table"]
            assert set(operations) == {("_triton_bsr_dense_mm_out", "SparseCsrCUDA"),
                                       ("_triton_bsr_dense_addmm_out", "SparseCsrCUDA")}
            assert all(vars(kernel)["kernel_invoked"] is False for kernel in operations.values())
            self.raw.append({"triton_events": events, "triton_executed": False})
        except BaseException as error:
            self._fail(error)
            raise

    def _state(self, pristine):
        ctypes = self.ctypes
        held = None if pristine else {"ordinal": int(self.stream.device.index if self.session.lane == "cuda"
                                                    else self.stream.device.ordinal),
                                     "handle": int(self.stream.cuda_stream)}
        if self.session.lane == "cuda":
            raw = ctypes.c_int(-1)
            raw_device = self._result("cudaGetDevice", self.raw_device(ctypes.byref(raw)), raw.value)
            contexts = [{"ordinal": i, "active": self.primary(i)} for i in range(self.count)]
            initialized = vars(sys.modules["torch.cuda"])["_initialized"]
            if pristine:
                # Derived logical0 is conditional. Preserve failed prerequisites
                # as raw facts; do not fabricate a complete derived snapshot.
                self.raw.append({"cold_raw_device": raw_device, "primary_contexts": contexts,
                                 "initialized": initialized})
                assert raw_device == 0 and initialized is False and not any(row["active"] for row in contexts)
            return {"initialized": initialized, "logical_device": 0 if pristine else self.logical(),
                    "logical_basis": "cold_source_derived" if pristine else "native_return",
                    "raw_device": raw_device, "primary_contexts": contexts,
                    "target_current_stream": None if pristine else self.stream_query(self.controller.target_device_ordinal),
                    "held_stream": held}
        context = ctypes.c_void_p()
        current = self._result("cuCtxGetCurrent", self.context_get(ctypes.byref(context)), context.value)
        ordinal = None
        if current is not None:
            device = ctypes.c_int(-1)
            handle = self._result("cuCtxGetDevice", self.context_device(ctypes.byref(device)), device.value)
            ordinal = self.devices.index(handle)
        contexts = []
        for index, device in enumerate(self.devices):
            flags, active = ctypes.c_uint(), ctypes.c_int(-1)
            self._result("cuDevicePrimaryCtxGetState", self.primary(device, ctypes.byref(flags), ctypes.byref(active)),
                         {"ordinal": index, "flags": flags.value, "active": active.value})
            assert active.value in (0, 1)
            contexts.append({"ordinal": index, "flags": flags.value, "active": active.value == 1})
        return {"default_device": self.runtime.default_device.ordinal, "current_context": current,
                "current_context_device": ordinal, "primary_contexts": contexts,
                "target_context": None if pristine else self.target_context,
                "target_current_stream": None if pristine else int(self.stream_query(self.target_context) or 0),
                "held_stream": held}

    def snapshot(self, site, edge, operation, before=None):
        self._ticket()
        try:
            if self.fault == "runtime_scope_getter_error" and site == "empty" and edge == "before":
                self._fault("before_allocation_observer_getter")
            pristine = site == "runtime_stream" and edge == "before"
            state = self._state(pristine)
            observation = self.session.runtime_observation
            self.raw.append({"site": site, "edge": edge, "state": state})
            details = {"site": site, "edge": edge, "before_index": None if before is None else before[0],
                       "operation_event_index": operation, "worker_pid": os.getpid(),
                       "thread_id": int(Path("/proc/thread-self").resolve(strict=True).name),
                       "module_token": observation.token(self.module), "runtime_token": observation.token(self.runtime),
                       "target_token": observation.token(self.target),
                       "stream_token": None if pristine else observation.token(self.stream),
                       "binding_token": None if site == "runtime_stream" else observation.token(self.binding),
                       "libraries": [self.library], "state": state}
            index = len(observation.journal)
            observation.append("native_scope", details)
            if self.fault == "runtime_scope_after_append_raise" and site == "empty" and edge == "after":
                self._fault("after_returned_allocation_snapshot_append")
            expected_contexts = [False] * self.count
            if not pristine:
                expected_contexts[self.controller.target_device_ordinal] = True
            assert [row["active"] for row in state["primary_contexts"]] == expected_contexts
            if self.session.lane == "cuda":
                assert state["logical_device"] == 0 and state["raw_device"] == (0 if pristine else 1)
                assert state["initialized"] is (not pristine)
            else:
                interior = self.scope_depth > 0
                assert state["default_device"] == (1 if interior else 0)
                assert state["current_context"] == (self.target_context if interior else None)
                assert state["current_context_device"] == (1 if interior else None)
            if not pristine:
                assert state["held_stream"]["ordinal"] == 1
                if self.current_stream_baseline is None:
                    self.current_stream_baseline = state["target_current_stream"]
                assert state["target_current_stream"] == self.current_stream_baseline
            return index, operation
        except BaseException as error:
            self._fail(error)
            raise

    def _wrap_helpers(self):
        setup = self.session.modules["optical_worker.native_setup"][0]
        prefix = "_torch_" if self.session.lane == "cuda" else "_warp_"
        for suffix, site in (("empty_exact", "empty"), ("copy_exact", "copy"), ("event_exact", "event_create"),
                             ("record_event_exact", "event_record"), ("wait_event_host_exact", "event_wait")):
            name = prefix + suffix
            original = vars(setup)[name]
            self.originals.append(original)
            if self.fault == "runtime_scope_c1_copy" and site == "copy":
                assert self.session.lane == "cuda"
                def stream_only_copy(module, destination, source, stream):
                    _write(self.directory, "fault_entered.json", {
                        "fault": self.fault, "boundary": "copy_without_outer_indexed_device"})
                    with module.cuda.stream(stream):
                        destination.copy_(source, non_blocking=True)
                original = stream_only_copy
            def wrapper(*args, _original=original, _site=site, **kwargs):
                opening = self.snapshot(_site, "before", self._latest_leaf())
                result = _original(*args, **kwargs)
                if result is not None:
                    self.held.append(result)
                self.snapshot(_site, "after", opening[1], opening)
                return result
            setattr(setup, name, wrapper)
        if self.session.lane == "warp":
            original_scope = self.module.ScopedDevice
            self.originals.append(original_scope)
            observer = self
            class Scope:
                def __init__(self, target):
                    assert target is observer.target
                    self.original = original_scope(target)
                    observer.held.append(self.original)
                    self.site = "upload_scope" if observer.session.phase == "uploading" else "readback_scope"
                    self.omit = observer.fault == "runtime_scope_missing_warp_scope" and self.site == "readback_scope"
                def __enter__(self):
                    self.before = observer.snapshot(self.site, "before", observer._latest_leaf())
                    if self.omit:
                        _write(observer.directory, "fault_entered.json", {"fault": observer.fault,
                               "boundary": "omit_readback_scope_for_negative_control"})
                        return observer.target
                    result = self.original.__enter__()
                    observer.scope_depth += 1
                    return result
                def __exit__(self, exc_type, exc, traceback):
                    result = None if self.omit else self.original.__exit__(exc_type, exc, traceback)
                    if not self.omit:
                        observer.scope_depth -= 1
                    if exc_type is None and not observer.session.failed and not observer.controller._replay.failed:
                        observer.snapshot(self.site, "after", self.before[1], self.before)
                    return result
            def scoped(target):
                if observer.controller._terminating:
                    return original_scope(target)
                result = Scope(target)
                observer.held.append(result)
                return result
            self.module.ScopedDevice = scoped

    def synchronize(self, original):
        operation = self.controller._guard_events[-1]["index"]
        assert self.controller._guard_events[-1]["operation"] == "emergency_sync_started"
        opening = self.snapshot("terminal_sync", "before", operation)
        original()
        self.snapshot("terminal_sync", "after", operation, opening)


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
        scope_observer = (_NativeScopeObserver(session, controller, directory, environment_payload, fault)
                          if fault in RUNTIME_OBSERVATION_FAULTS else None)
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
        _install_upload_fault(fault, session, controller, injection, directory)
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
        def observed_synchronize():
            scope_observer.synchronize(synchronize)

        terminal = controller.terminate(
            fault="setup_failure",
            detection_phase="fixed_runtime_driver",
            emergency_target_synchronize=synchronize if scope_observer is None else observed_synchronize,
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
