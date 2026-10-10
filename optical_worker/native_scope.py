"""Read-only native state for the captured preparation path.

The source session owns this reader. It records facts into the CPU-only custody
journal; it does not create execution, wait or release authority.
"""

from __future__ import annotations

import hashlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class NativeStateReader:
    def __init__(self, session, module, runtime, target):
        controller = session.bound_controller()
        controller._require_setup()
        _require(
            session.phase == "initializing_runtime" and session._native_scope_record is None,
            "native reader construction is ordered and single-use",
        )
        self.session, self.controller = session, controller
        self.module, self.runtime, self.target = module, runtime, target
        self.stream = self.binding = None
        self.environment = session.runtime_observation.environment.to_payload()
        self.held, self.raw = [], []
        self.library = self.target_context = self.current_stream_baseline = None
        self.cache_established = False
        self.scope_depth = 0
        session.runtime_owners.append(self)
        session._native_scope_record = (self, controller, module, runtime, target, None, None)
        controller.adopt_host_owner(self, role="native_state_reader", operation_id="native_state_reader")

    def _ticket(self):
        try:
            c = self.controller
            c._check_owner()
            _require(
                not self.session.failed
                and not c._replay.compromised
                and not c._replay.failed
                and not c._replay.terminal_failed,
                "native query after sticky failure",
            )
            if c._terminating:
                # An emergency ticket deliberately remains armed through cleanup
                # and process exit. Only the original in-flight sync permits queries.
                _require(
                    c._replay.controller == "TERMINATING"
                    and c._replay.sync_outcome == "started"
                    and c._replay.purpose == "target_sync"
                    and c._replay.custody == "retained"
                    and c._replay.release == "not_authorized"
                    and c.guardian.state == "PARTIAL_FROZEN",
                    "native query outside the original terminal sync",
                )
                c._guard_client.require_live_ticket(c._guard_ticket, role="emergency")
            else:
                c._require_setup()
            _require(require_reader(self.session) is self, "native reader identity changed")
        except BaseException as error:
            self._fail(error)
            raise

    def _fail(self, error):
        self.held.append(error)
        c = self.controller
        try:
            if not c._terminating and not c._replay.unknown:
                c.mark_unknown_device_scope(source="native_state_reader", reason="native_observation_failed")
            c._fail_guard("operation", error)
        except BaseException as diagnostic_error:
            self.held.append(diagnostic_error)

    def _latest_leaf(self):
        return next(
            e["index"]
            for e in reversed(self.controller._guard_events)
            if e["operation"] == "low_level_leaf_entered"
        )

    def start_stream(self):
        self._initialize_queries()
        if self.session.lane == "warp":
            self._warp_cache_precondition(before=True)
        return self.snapshot("runtime_stream", "before", self._latest_leaf())

    def stream_returned(self, stream, opening):
        # Caller retained and journaled the real return before entering here.
        with self.controller._observe_public_operation():
            self._ticket()
            _require(self.stream is None, "native reader stream return is single-use")
            record = self.session._native_scope_record
            self.session._native_scope_record = (*record[:5], stream, None)
            self.stream = stream
            if self.session.lane == "warp":
                self._warp_cache_precondition(before=False)
            self.snapshot("runtime_stream", "after", opening[1], opening)

    def binding_returned(self, binding):
        with self.controller._observe_public_operation():
            self._ticket()
            _require(
                self.stream is not None and self.binding is None,
                "native reader binding return is ordered and single-use",
            )
            record = self.session._native_scope_record
            self.session._native_scope_record = (*record[:6], binding)
            self.binding = binding
            self.snapshot("runtime_binding", "point", len(self.controller._guard_events) - 1)
            if self.session.lane == "cuda":
                self.raw.append(self.session._check_cuda_deferred_state(self.module))

    def hold_returned(self, owner):
        self.held.append(owner)

    @contextmanager
    def operation(self, site):
        opening = self.snapshot(site, "before", self._latest_leaf())
        yield
        self.snapshot(site, "after", opening[1], opening)

    @contextmanager
    def outer_scope(self, site):
        opening = self.snapshot(site, "before", self._latest_leaf())
        previous = self.scope_depth
        self.scope_depth += 1
        try:
            yield
        finally:
            self.scope_depth = previous
        self.snapshot(site, "after", opening[1], opening)

    def _initialize_queries(self):
        self._ticket()
        try:
            import ctypes

            _require(
                os.getpid() == int(Path("/proc/thread-self").resolve(strict=True).name),
                "native reader requires the original process main thread",
            )
            _require(self.controller.target_device_ordinal == 1, "native reader deployment target differs")
            _require(
                not any(
                    name in os.environ for name in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "LD_LIBRARY_PATH")
                ),
                "native reader environment remapped",
            )
            self.ctypes = ctypes
            cuda = self.session.lane == "cuda"
            key = (
                ("site", "nvidia/cuda_runtime/lib/libcudart.so.12")
                if cuda
                else ("os_native", "libcuda.so.595.71.05")
            )
            descriptor = next((f for f in self.environment["files"] if (f["root"], f["path"]) == key))
            root = next((r["path"] for r in self.environment["roots"] if r["id"] == key[0]))
            path = Path(root) / key[1]
            expected_sha = (
                "c3a75b33af334a3486d197dbd1584a2985183ba4688d237a2be5f2f679329920"
                if cuda
                else "76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6"
            )
            _require(
                descriptor["sha256"] == expected_sha and path.is_file() and path.resolve(strict=True) == path,
                "native query library differs from the pinned file",
            )
            _require(
                hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["sha256"],
                "native scope prerequisite or state mismatch",
            )
            mapped = {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if len(line.split()) >= 6 and line.split()[-1].startswith("/")
            }
            _require(str(path.resolve(strict=True)) in mapped, "native scope prerequisite or state mismatch")
            library = ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
            self.held.append(library)
            self.library = {name: descriptor[name] for name in ("root", "path", "sha256")}

            def function(name, arguments):
                value = getattr(library, name)
                self.held.append(value)
                value.argtypes, value.restype = (arguments, ctypes.c_int)
                return value

            if cuda:
                self.raw_device = function("cudaGetDevice", [ctypes.POINTER(ctypes.c_int)])
                native = vars(self.module)["_C"]
                self.logical = vars(native)["_cuda_getDevice"]
                self.primary = vars(native)["_cuda_hasPrimaryContext"]
                self.stream_query = vars(native)["_cuda_getCurrentRawStream"]
                self.device_count = vars(native)["_cuda_getDeviceCount"]
                self.count = self._getter("_cuda_getDeviceCount", self.device_count)
                state = vars(sys.modules["torch.cuda"])
                _require(state["_initialized"] is False, "native scope prerequisite or state mismatch")
                queued = state["_queued_calls"]
                _require(
                    type(queued) is list and len(queued) == 3, "native scope prerequisite or state mismatch"
                )
                _require(
                    [item[0] for item in queued]
                    == [
                        state[name]
                        for name in ("_check_capability", "_check_cubins", "_register_triton_kernels")
                    ],
                    "native scope prerequisite or state mismatch",
                )
                tracker = state["_lazy_seed_tracker"]
                _require(
                    type(tracker) is vars(sys.modules["torch._utils"])["_LazySeedTracker"],
                    "native scope prerequisite or state mismatch",
                )
                _require(
                    vars(tracker) == {"manual_seed_all_cb": None, "manual_seed_cb": None, "call_order": []},
                    "native scope prerequisite or state mismatch",
                )
            else:
                pointer = ctypes.POINTER
                self.device_count = function("cuDeviceGetCount", [pointer(ctypes.c_int)])
                self.device_get = function("cuDeviceGet", [pointer(ctypes.c_int), ctypes.c_int])
                self.primary = function(
                    "cuDevicePrimaryCtxGetState",
                    [ctypes.c_int, pointer(ctypes.c_uint), pointer(ctypes.c_int)],
                )
                self.context_get = function("cuCtxGetCurrent", [pointer(ctypes.c_void_p)])
                self.context_device = function("cuCtxGetDevice", [pointer(ctypes.c_int)])
                count = ctypes.c_int(-1)
                self._result("cuDeviceGetCount", self.device_count(ctypes.byref(count)), count.value)
                self.count = count.value
                _require(self.count == 8, "native scope prerequisite or state mismatch")
                self.devices = []
                for ordinal in range(self.count):
                    device = ctypes.c_int(-1)
                    self._result("cuDeviceGet", self.device_get(ctypes.byref(device), ordinal), device.value)
                    self.devices.append(device.value)
                _require(len(set(self.devices)) == self.count, "native scope prerequisite or state mismatch")
                self.stream_query = self.runtime.core.wp_cuda_context_get_stream
                self.held.append(self.stream_query)
                self.error_query = self.runtime.core.wp_get_error_string
                self.held.append(self.error_query)
                _require(
                    self.stream_query.argtypes == [ctypes.c_void_p]
                    and self.stream_query.restype is ctypes.c_void_p,
                    "Warp stream query ABI changed",
                )
                _require(
                    self.error_query.argtypes == [] and self.error_query.restype is ctypes.c_char_p,
                    "native scope prerequisite or state mismatch",
                )
            _require(self.count == 8, "native scope prerequisite or state mismatch")
            _require(
                not any(
                    (name in os.environ for name in ("CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "LD_LIBRARY_PATH"))
                ),
                "native scope prerequisite or state mismatch",
            )
        except BaseException as error:
            self._fail(error)
            raise

    def _result(self, name, result, value):
        self.raw.append({"getter": name, "result": result, "value": value if result == 0 else None})
        if result != 0:
            raise RuntimeError("native observer getter error: " + name + "=" + str(result))
        return value

    def _getter(self, name, function, *arguments):
        """Record each real call once, including the prefix before an exception."""
        try:
            value = function(*arguments)
        except BaseException as error:
            self.held.append(error)
            self.raw.append(
                {
                    "getter": name,
                    "arguments": list(arguments),
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
            )
            raise
        self.raw.append(
            {"getter": name, "arguments": list(arguments), "value_hex": value.hex()}
            if type(value) is bytes
            else {"getter": name, "arguments": list(arguments), "value": value}
        )
        return value

    def _warp_cache_precondition(self, *, before):
        """Prove original cache initialization; never initialize it by observation."""
        self._ticket()
        try:
            raw = vars(self.target)
            context = raw.get("_context")
            stream = raw.get("_stream")
            diagnostic = {
                "cache_precondition": "before" if before else "after",
                "context_present": "_context" in raw,
                "context_type": type(context).__name__,
                "context": context if type(context) in (int, bool, str, float, type(None)) else None,
                "stream_present": "_stream" in raw,
                "stream_is_none": stream is None,
                "cache_established": self.cache_established,
            }
            if not before:
                returned = vars(self.stream)
                handle = returned.get("cuda_stream")
                diagnostic.update(
                    stream_is_returned=stream is self.stream,
                    returned_handle_present="cuda_stream" in returned,
                    returned_handle_type=type(handle).__name__,
                    returned_handle=handle if type(handle) in (int, bool, str, float, type(None)) else None,
                )
            self.raw.append(diagnostic)
            if before:
                _require(
                    diagnostic["context_present"] and context is None,
                    "native scope prerequisite or state mismatch",
                )
                _require(
                    diagnostic["stream_present"] and stream is None,
                    "native scope prerequisite or state mismatch",
                )
                _require(not self.cache_established, "native scope prerequisite or state mismatch")
            else:
                _require(
                    diagnostic["stream_present"] and stream is self.stream,
                    "native scope prerequisite or state mismatch",
                )
                self.target_context = context
                _require(
                    type(self.target_context) is int and self.target_context > 0,
                    "native scope prerequisite or state mismatch",
                )
                _require(type(handle) is int and handle > 0, "native scope prerequisite or state mismatch")
            result = self._getter("wp_get_error_string", self.error_query)
            _require(
                type(result) is bytes and result == b"", "Warp original initialization has a native error"
            )
            if not before:
                self.cache_established = True
        except BaseException as error:
            self._fail(error)
            raise

    def _state(self, pristine):
        self._ticket()
        try:
            ctypes = self.ctypes
            held = (
                None
                if pristine
                else {
                    "ordinal": int(
                        self.stream.device.index
                        if self.session.lane == "cuda"
                        else self.stream.device.ordinal
                    ),
                    "handle": int(self.stream.cuda_stream),
                }
            )
            if self.session.lane == "cuda":
                raw = ctypes.c_int(-1)
                raw_device = self._result("cudaGetDevice", self.raw_device(ctypes.byref(raw)), raw.value)
                contexts = [
                    {"ordinal": i, "active": self._getter("_cuda_hasPrimaryContext", self.primary, i)}
                    for i in range(self.count)
                ]
                initialized = vars(sys.modules["torch.cuda"])["_initialized"]
                if pristine:
                    self.raw.append(
                        {
                            "cold_raw_device": raw_device,
                            "primary_contexts": contexts,
                            "initialized": initialized,
                        }
                    )
                    _require(
                        raw_device == 0
                        and initialized is False
                        and (not any((row["active"] for row in contexts))),
                        "native scope prerequisite or state mismatch",
                    )
                return {
                    "initialized": initialized,
                    "logical_device": 0 if pristine else self._getter("_cuda_getDevice", self.logical),
                    "logical_basis": "cold_source_derived" if pristine else "native_return",
                    "raw_device": raw_device,
                    "primary_contexts": contexts,
                    "target_current_stream": None
                    if pristine
                    else self._getter(
                        "_cuda_getCurrentRawStream", self.stream_query, self.controller.target_device_ordinal
                    ),
                    "held_stream": held,
                }
            context = ctypes.c_void_p()
            current = self._result("cuCtxGetCurrent", self.context_get(ctypes.byref(context)), context.value)
            ordinal = None
            if current is not None:
                device = ctypes.c_int(-1)
                handle = self._result(
                    "cuCtxGetDevice", self.context_device(ctypes.byref(device)), device.value
                )
                ordinal = self.devices.index(handle)
            contexts = []
            for index, device in enumerate(self.devices):
                flags, active = (ctypes.c_uint(), ctypes.c_int(-1))
                self._result(
                    "cuDevicePrimaryCtxGetState",
                    self.primary(device, ctypes.byref(flags), ctypes.byref(active)),
                    {"ordinal": index, "flags": flags.value, "active": active.value},
                )
                _require(active.value in (0, 1), "native scope prerequisite or state mismatch")
                contexts.append({"ordinal": index, "flags": flags.value, "active": active.value == 1})
            stream = None
            if not pristine:
                _require(self.cache_established, "original Warp native cache prerequisite is absent")
                stream = int(
                    self._getter("wp_cuda_context_get_stream", self.stream_query, self.target_context) or 0
                )
            return {
                "default_device": self.runtime.default_device.ordinal,
                "current_context": current,
                "current_context_device": ordinal,
                "primary_contexts": contexts,
                "target_context": None if pristine else self.target_context,
                "target_current_stream": stream,
                "held_stream": held,
            }
        except BaseException as error:
            self._fail(error)
            raise

    def snapshot(self, site, edge, operation, before=None):
        self._ticket()
        try:
            pristine = site == "runtime_stream" and edge == "before"
            state = self._state(pristine)
            observation = self.session.runtime_observation
            self.raw.append({"site": site, "edge": edge, "state": state})
            details = {
                "site": site,
                "edge": edge,
                "before_index": None if before is None else before[0],
                "operation_event_index": operation,
                "worker_pid": os.getpid(),
                "thread_id": int(Path("/proc/thread-self").resolve(strict=True).name),
                "module_token": observation.token(self.module),
                "runtime_token": observation.token(self.runtime),
                "target_token": observation.token(self.target),
                "stream_token": None if pristine else observation.token(self.stream),
                "binding_token": None if site == "runtime_stream" else observation.token(self.binding),
                "libraries": [self.library],
                "state": state,
            }
            index = len(observation.journal)
            observation.append("native_scope", details)
            expected_contexts = [False] * self.count
            if not pristine:
                expected_contexts[self.controller.target_device_ordinal] = True
            _require(
                [row["active"] for row in state["primary_contexts"]] == expected_contexts,
                "native scope prerequisite or state mismatch",
            )
            if self.session.lane == "cuda":
                interior = False
                _require(
                    state["logical_device"] == 0 and state["raw_device"] == (0 if pristine else 1),
                    "native scope prerequisite or state mismatch",
                )
                _require(
                    state["initialized"] is (not pristine), "native scope prerequisite or state mismatch"
                )
            else:
                interior = self.scope_depth > 0
                _require(
                    state["default_device"] == (1 if interior else 0),
                    "native scope prerequisite or state mismatch",
                )
                _require(
                    state["current_context"] == (self.target_context if interior else None),
                    "native scope prerequisite or state mismatch",
                )
                _require(
                    state["current_context_device"] == (1 if interior else None),
                    "native scope prerequisite or state mismatch",
                )
            if not pristine:
                _require(state["held_stream"]["ordinal"] == 1, "native scope prerequisite or state mismatch")
                if self.current_stream_baseline is None:
                    if self.session.lane == "warp":
                        _require(
                            state["target_current_stream"] == state["held_stream"]["handle"],
                            "native scope prerequisite or state mismatch",
                        )
                    self.current_stream_baseline = state["target_current_stream"]
                expected_stream = (
                    state["held_stream"]["handle"]
                    if self.session.lane == "cuda" and interior
                    else self.current_stream_baseline
                )
                _require(
                    state["target_current_stream"] == expected_stream,
                    "native scope prerequisite or state mismatch",
                )
            return (index, operation)
        except BaseException as error:
            self._fail(error)
            raise


def require_reader(session):
    controller = session.bound_controller()
    with controller._observe_public_operation():
        record = session._native_scope_record
        _require(type(record) is tuple and len(record) == 7, "captured native reader is missing")
        reader, original_controller, module, runtime, target, stream, binding = record
        _require(
            type(reader) is NativeStateReader
            and reader.session is session
            and original_controller is controller
            and reader.controller is controller
            and reader.module is module
            and reader.runtime is runtime
            and reader.target is target
            and reader.stream is stream
            and reader.binding is binding
            and controller.guardian.contains_exact_owner(reader),
            "captured native reader binding changed",
        )
        if binding is not None:
            _require(
                binding.session is session
                and binding.controller is controller
                and binding.module is module
                and binding.runtime is runtime
                and binding.target is target
                and binding.stream is stream
                and controller.guardian.contains_exact_owner(binding),
                "native reader runtime binding changed",
            )
        return reader


def reader_for_input(lifecycle, module, target, stream, *, phase, input_entry=False):
    from .guarded_lifecycle import NativeGuardedSetupLifecycle

    if type(lifecycle) is not NativeGuardedSetupLifecycle:
        return None
    with lifecycle._observe_public_operation():
        source = lifecycle._bound_source_session()
        if source is None:
            return None
        if source.phase != phase:
            _require(not input_entry, "captured input reader phase differs")
            return None
        reader = require_reader(source)
        _require(
            reader.controller is lifecycle
            and reader.module is module
            and reader.target is target
            and reader.stream is stream,
            "native input reader differs from its original runtime",
        )
        return reader
