"""CPU failure models of the production reader, never native-scope proof."""
from types import SimpleNamespace

import pytest

from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.lifecycle import DeviceScopeDescriptor, WorkerSetupProtocolError
from optical_worker.native_bootstrap import _RuntimeObservation
from optical_worker.native_scope import NativeStateReader
from tests.support.guarded_peer import Peer


@pytest.fixture
def observer(tmp_path):
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id="warp", channel=peer.channel, configuration=peer.config)
    session = SimpleNamespace(bound_controller=lambda: controller, lane="warp", failed=False)
    session.runtime_observation = _RuntimeObservation(session)
    value = object.__new__(NativeStateReader)
    value.session, value.controller, value.directory = session, controller, tmp_path
    value.raw, value.held = [], []
    value.fault = "runtime_scope_success"
    value.module, value.runtime, value.target, value.stream, value.binding = (object() for _ in range(5))
    value.library = {"root": "os_native", "path": "libcuda.so.595.71.05", "sha256": "a" * 64}
    value.count, value.scope_depth, value.target_context = 8, 0, 9000
    value.originals = []
    value.current_stream_baseline = 123
    # Keep PID/TID observation separate from the CPU state model on Windows.
    value.snapshot_state = {"default_device": 0, "current_context": None, "current_context_device": None,
                            "primary_contexts": [{"ordinal": i, "active": i == 1, "flags": 0} for i in range(8)],
                            "target_context": 9000, "target_current_stream": 123,
                            "held_stream": {"ordinal": 1, "handle": 123}}
    value._state = lambda pristine: value.snapshot_state
    _bind_model_reader(value)
    controller.adopt_host_owner(value, role="reader", operation_id="reader")
    try:
        yield value
    finally:
        if controller.termination_evidence is None:
            controller.terminate(fault="setup_failure", detection_phase="observer_unit_end", emergency_target_synchronize=None)
        peer.close()


def _bind_model_reader(value):
    value.binding = SimpleNamespace(session=value.session, controller=value.controller,
        module=value.module, runtime=value.runtime, target=value.target, stream=value.stream)
    value.controller.adopt_host_owner(value.binding, role="runtime_setup_binding", operation_id="binding")
    _record_model_reader(value)


def _record_model_reader(value):
    value.session._native_scope_record = (value, value.controller, value.module, value.runtime, value.target,
                                         value.stream, value.binding)


def _enter(controller):
    descriptor = DeviceScopeDescriptor.exact_target(operation_id="indexed_validation_scope",
        leaf_kind="runtime_import_or_init", device_ordinal=1, scope_source="closed_worker_allowlist",
        runtime_identity="warp_scoped_device_v1")
    controller.enter_device_leaf(controller.begin_device_leaf(descriptor))
    return controller._guard_events[-1]["index"]


def test_native_result_error_is_never_successful_null(observer):
    assert observer._result("cuCtxGetCurrent", 0, None) is None
    with pytest.raises(RuntimeError, match="getter error"):
        observer._result("cuCtxGetCurrent", 999, None)
    assert observer.raw == [{"getter": "cuCtxGetCurrent", "result": 0, "value": None},
                            {"getter": "cuCtxGetCurrent", "result": 999, "value": None}]


def test_exit_return_with_wrong_context_preserves_row_then_poison(observer, monkeypatch):
    import os
    from pathlib import Path
    original = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda p, **kw: Path(str(os.getpid()))
                        if str(p).replace("\\", "/") == "/proc/thread-self" else original(p, **kw))
    controller = observer.controller
    with pytest.raises(ValueError):
        with controller.segment():
            operation = _enter(controller)
            before = observer.snapshot("readback_scope", "before", operation)
            # CPU model of original __exit__ returning without restoration.
            observer.snapshot_state = dict(observer.snapshot_state, current_context=9000, current_context_device=1)
            observer.snapshot("readback_scope", "after", operation, before)
            raise RuntimeError("normal continuation must not be reached")
    rows = observer.session.runtime_observation.journal
    assert len(rows) == 2 and rows[-1]["details"]["edge"] == "after"
    assert rows[-1]["details"]["state"]["current_context"] == 9000
    summary = controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["unknown_device_scope"] is True
    assert summary["normal_complete_count"] == summary["emergency_sync_attempt_count"] == 0
    assert summary["release_status"] == "not_authorized"


def test_failed_getter_emits_no_complete_snapshot_or_retry(observer):
    calls = []
    def fail(pristine):
        calls.append(pristine)
        observer._result("cuCtxGetCurrent", 999, None)
    observer._state = fail
    with pytest.raises(RuntimeError, match="getter error"):
        with observer.controller.segment():
            operation = _enter(observer.controller)
            observer.snapshot("readback_scope", "before", operation)
    assert calls == [False]
    assert observer.session.runtime_observation.journal == []
    assert observer.controller.termination_evidence.lifecycle.to_payload()["summary"]["unknown_device_scope"]




@pytest.fixture
def warp_cache_model(observer, monkeypatch):
    """Exact pinned Device Python methods, modeled C++ only; no backend import."""
    import ast
    import ctypes
    import hashlib
    import os
    from pathlib import Path
    path = Path(os.environ.get("ROBOT_SIM_WARP_CONTEXT_SOURCE",
        "/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages/warp/_src/context.py"))
    if not path.is_file():
        pytest.skip("pinned Warp source text required for exact-Device CPU model")
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == "eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee"
    device = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Device")
    methods = [n for n in device.body if isinstance(n, ast.FunctionDef)
               and n.name in {"context", "stream", "_init_streams", "set_stream"}]
    model = SimpleNamespace(error=b"", setter_failure=False, cache={}, getter_calls=[], setters=0)
    class Core:
        def wp_cuda_context_get_current(self):
            return None
        def wp_cuda_device_get_primary_context(self, ordinal):
            return 9000
        def wp_cuda_context_set_current(self, context):
            assert context is None
        def wp_cuda_context_set_stream(self, context, stream, sync):
            model.setters += 1
            if model.setter_failure:
                model.error = b"Warp CUDA error 999: modeled original cuCtxGetDevice failure"
                return
            model.cache[context] = stream
        def wp_cuda_context_get_stream(self, context):
            model.getter_calls.append(context)
            # Forbidden branch would initialize native cache. It must be unreachable.
            assert context in model.cache, "observer reached native cache miss"
            return model.cache[context]
    core = Core()
    runtime = SimpleNamespace(core=core, context_map={}, default_device=SimpleNamespace(ordinal=0))
    def Stream(device, **kwargs):
        assert device.context == 9000
        return SimpleNamespace(device=device, cuda_stream=kwargs.get("cuda_stream", 123))
    namespace = {"runtime": runtime, "Stream": Stream}
    definition = ast.ClassDef(name="PinnedDevice", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(module, "<pinned Device CPU model>", "exec"), namespace)
    target = namespace["PinnedDevice"]()
    target.runtime, target.ordinal = runtime, 1
    target._context = target._stream = None
    target.is_primary = target.is_cuda = True
    observer.target, observer.runtime, observer.ctypes = target, runtime, ctypes
    observer.cache_established = False
    observer.error_query = lambda: model.error
    observer.stream_query = core.wp_cuda_context_get_stream
    observer.devices = list(range(8))
    def current(out):
        out._obj.value = None
        return 0
    def primary(device, flags, active):
        flags._obj.value, active._obj.value = 0, int(device == 1)
        return 0
    observer.context_get, observer.primary = current, primary
    del observer._state  # Use the actual implemented query ordering.
    _bind_model_reader(observer)
    observer.binding = None  # Runtime binding follows the actual stream return.
    observer.stream = None
    _record_model_reader(observer)
    return observer, model


@pytest.mark.parametrize("failure", [False, True])
def test_original_void_setter_cache_precondition(warp_cache_model, failure):
    value, model = warp_cache_model
    model.setter_failure = failure
    try:
        with value.controller.segment():
            _enter(value.controller)
            value._warp_cache_precondition(before=True)
            value.stream = value.target.stream
            _record_model_reader(value)
            value.held.append(value.stream)  # original returned owner remains retained
            assert value.target._stream is value.stream and model.setters == 1
            value._warp_cache_precondition(before=False)
            assert not failure
            state = value._state(False)
            assert state["target_current_stream"] == state["held_stream"]["handle"] == 123
            assert model.getter_calls == [9000]
    except ValueError as error:
        if not failure:
            raise
        assert "native error" in str(error)
    if failure:
        assert value.stream in value.held and not value.cache_established
        assert not model.getter_calls and model.cache == {}
        assert value.raw[-1]["value_hex"] == model.error.hex()
        assert value.session.runtime_observation.journal == []
        summary = value.controller.termination_evidence.lifecycle.to_payload()["summary"]
        assert summary["unknown_device_scope"]
        assert summary["normal_complete_count"] == summary["emergency_sync_attempt_count"] == 0
        assert summary["release_status"] == "not_authorized"


@pytest.mark.parametrize("error", [b"stale error", None, "", b"\xff\x80"])
def test_stale_or_invalid_error_buffer_prevents_original_stream(warp_cache_model, error):
    value, model = warp_cache_model
    model.error = error
    with pytest.raises(ValueError, match="native error"):
        with value.controller.segment():
            _enter(value.controller)
            value._warp_cache_precondition(before=True)
            raise RuntimeError("must not enter original stream creation")
    assert model.setters == 0 and not model.getter_calls and not value.cache_established


def test_unproved_python_stream_metadata_cannot_call_native_getter(warp_cache_model):
    value, model = warp_cache_model
    with pytest.raises(ValueError, match="prerequisite is absent"):
        with value.controller.segment():
            _enter(value.controller)
            value.stream = value.target.stream
            _record_model_reader(value)
            value.target_context = value.target._context
            value._state(False)
    assert not model.getter_calls


@pytest.mark.parametrize("handle", [0, 124])
def test_first_warp_native_stream_must_match_actual_return(observer, monkeypatch, handle):
    import os
    from pathlib import Path
    original = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda p, **kw: Path(str(os.getpid()))
                        if str(p).replace("\\", "/") == "/proc/thread-self" else original(p, **kw))
    observer.current_stream_baseline = None
    observer.snapshot_state["target_current_stream"] = handle
    with pytest.raises(ValueError):
        with observer.controller.segment():
            operation = _enter(observer.controller)
            observer.snapshot("runtime_stream", "after", operation)
    assert observer.current_stream_baseline is None
    assert observer.session.runtime_observation.journal[-1]["details"]["state"]["target_current_stream"] == handle
    assert observer.controller.termination_evidence.lifecycle.to_payload()["summary"]["unknown_device_scope"]


@pytest.mark.parametrize("failed", list(range(8)) + ["logical", "stream"])
def test_cuda_getter_failure_preserves_ordered_successful_prefix(observer, monkeypatch, failed):
    import ctypes
    import sys
    value = observer
    value.session.lane = "cuda"
    value.ctypes = ctypes
    value.stream = SimpleNamespace(device=SimpleNamespace(index=1), cuda_stream=123)
    calls = []
    def invoke(name, result):
        calls.append(name)
        if name == failed:
            raise RuntimeError("modeled C10 error")
        return result
    def raw_device(out):
        out._obj.value = 1
        return 0
    value.raw_device = raw_device
    value.primary = lambda ordinal: invoke(ordinal, ordinal == 1)
    value.logical = lambda: invoke("logical", 0)
    value.stream_query = lambda ordinal: invoke("stream", 123)
    monkeypatch.setitem(sys.modules, "torch.cuda", SimpleNamespace(_initialized=True))
    del value._state
    _bind_model_reader(value)
    with pytest.raises(RuntimeError, match="modeled C10 error"):
        with value.controller.segment():
            operation = _enter(value.controller)
            value.snapshot("empty", "before", operation)
    expected = list(range(8)) + ["logical", "stream"]
    expected = expected[:expected.index(failed) + 1]
    assert calls == expected
    raw = value.raw
    assert raw[0] == {"getter": "cudaGetDevice", "result": 0, "value": 1}
    assert len(raw) == 1 + len(expected)
    for actual, name in zip(raw[1:], expected):
        assert actual["getter"] == ("_cuda_hasPrimaryContext" if type(name) is int else
                                    "_cuda_getDevice" if name == "logical" else "_cuda_getCurrentRawStream")
        assert actual["arguments"] == ([name] if type(name) is int else [] if name == "logical" else [1])
        if name == failed:
            assert actual["exception_type"] == "RuntimeError" and "value" not in actual
        else:
            assert actual["value"] == (name == 1 if type(name) is int else 0)
    assert value.session.runtime_observation.journal == []
    summary = value.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["unknown_device_scope"] and summary["emergency_sync_attempt_count"] == 0
    assert summary["release_status"] == "not_authorized"


def test_count_getter_exception_is_held_and_named(observer):
    error = RuntimeError("count failure")
    calls = []
    def count():
        calls.append(1)
        raise error
    with pytest.raises(RuntimeError, match="count failure"):
        observer._getter("_cuda_getDeviceCount", count)
    assert calls == [1] and error in observer.held
    assert observer.raw == [{"getter": "_cuda_getDeviceCount", "arguments": [],
                             "exception_type": "RuntimeError", "message": "count failure"}]


def test_retained_stream_survives_production_reader_cache_failure(warp_cache_model):
    value, model = warp_cache_model
    model.setter_failure = True
    session = value.session
    session.runtime_owners = [value]
    original_returns = []
    session._observe_runtime_return = lambda producer, owner: original_returns.append((producer, owner))
    with pytest.raises(ValueError, match="native error"):
        with value.controller.segment():
            operation = _enter(value.controller)
            value._warp_cache_precondition(before=True)
            returned = value.target.stream
            session.runtime_owners.append(returned)
            session._observe_runtime_return("runtime_stream", returned)
            value.stream_returned(returned, (0, operation))
    assert original_returns == [("runtime_stream", returned)]
    assert value in session.runtime_owners and returned in session.runtime_owners
    assert value.stream is returned and not value.cache_established and not model.getter_calls
    summary = value.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["unknown_device_scope"] and summary["normal_complete_count"] == 0


def _prepare_model_query(value):
    with value.controller.segment():
        _enter(value.controller)
        value._warp_cache_precondition(before=True)
        value.stream = value.target.stream
        _record_model_reader(value)
        value._warp_cache_precondition(before=False)


def test_reader_cannot_query_after_ordinary_release(warp_cache_model):
    value, model = warp_cache_model
    _prepare_model_query(value)
    end = value.controller.terminate(fault="setup_failure", detection_phase="input_validated",
                                     emergency_target_synchronize=lambda: None)
    assert end.lifecycle.to_payload()["summary"]["release_status"] == "completed"
    calls = list(model.getter_calls)
    with pytest.raises(ValueError, match="outside the original terminal sync"):
        value._state(False)
    assert model.getter_calls == calls


@pytest.mark.parametrize("failed", [False, True])
def test_reader_query_is_limited_to_healthy_terminal_sync(warp_cache_model, failed):
    value, model = warp_cache_model
    _prepare_model_query(value)

    def synchronize():
        if failed:
            value._fail(RuntimeError("terminal reader failure"))
        calls = len(model.getter_calls)
        if failed:
            with pytest.raises(ValueError, match="sticky failure"):
                value._state(False)
            assert len(model.getter_calls) == calls
        else:
            value._state(False)
            assert len(model.getter_calls) == calls + 1

    end = value.controller.terminate(fault="setup_failure", detection_phase="input_validated",
                                     emergency_target_synchronize=synchronize)
    assert end.lifecycle.to_payload()["summary"]["release_status"] == (
        "not_authorized" if failed else "completed")


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("failure", ["native_result", "getter_exception"])
def test_caught_raw_query_failure_cannot_resume_or_release(warp_cache_model, terminal, failure):
    from optical_worker import native_scope
    from tests.support.native_runtime_process import _NativeScopeProbe

    value, model = warp_cache_model
    controller = value.controller
    _prepare_model_query(value)
    value.session.modules = {"optical_worker.native_scope": (native_scope,)}
    probe = object.__new__(_NativeScopeProbe)
    probe.session = value.session

    def exercise():
        getter_name = "stream_query" if failure == "getter_exception" else "context_get"
        original = getattr(value, getter_name)
        original_error = RuntimeError("modeled raw getter failure")

        def broken(out):
            if failure == "getter_exception":
                raise original_error
            return 999

        setattr(value, getter_name, broken)
        with pytest.raises(RuntimeError):
            probe._state(False)
        assert controller._replay.terminal_failed if terminal else controller._replay.failed
        if failure == "getter_exception":
            assert any(error is original_error for error in value.held)
        else:
            assert value.raw[-1] == {"getter": "cuCtxGetCurrent", "result": 999, "value": None}
        setattr(value, getter_name, original)
        calls = list(model.getter_calls)
        with pytest.raises(ValueError, match="sticky failure"):
            probe._state(False)
        assert model.getter_calls == calls

    if terminal:
        controller.terminate(fault="setup_failure", detection_phase="input_validated",
                             emergency_target_synchronize=exercise)
    else:
        with pytest.raises(WorkerSetupProtocolError):
            with controller.segment():
                exercise()
    summary = controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["release_status"] == "not_authorized"
    assert summary["normal_complete_count"] == 1  # Only the earlier cache preparation.
    assert summary["emergency_sync_attempt_count"] == int(terminal)


@pytest.mark.parametrize("failure", [None, "before", "after"])
def test_later_helper_diagnostic_uses_same_query_failure_boundary(warp_cache_model, failure):
    from optical_worker import native_scope
    from tests.support.native_runtime_process import _NativeScopeProbe

    value, model = warp_cache_model
    _prepare_model_query(value)
    value.session.modules = {"optical_worker.native_scope": (native_scope,)}
    value.session.phase = "preparing_fixed_render_storage"
    value.session.runtime_owners = []
    value.runtime.default_device.ordinal = 1

    def current(out):
        out._obj.value = 9000
        return 0

    def device(out):
        out._obj.value = 1
        return 0

    value.context_get, value.context_device = current, device
    owner, calls = object(), []

    def helper():
        calls.append(True)
        if failure == "after":
            value.context_get = lambda out: 999
        return owner

    setup = SimpleNamespace(**{"_warp_" + name: helper for name in (
        "empty_exact", "copy_exact", "event_exact", "record_event_exact", "wait_event_host_exact")})
    probe = object.__new__(_NativeScopeProbe)
    probe.session, probe.originals = value.session, []
    probe._install_shared_diagnostics(setup)
    if failure == "before":
        value.context_get = lambda out: 999
    if failure:
        with pytest.raises(WorkerSetupProtocolError):
            with value.controller.segment():
                with pytest.raises(RuntimeError, match="cuCtxGetCurrent=999"):
                    setup._warp_empty_exact()
                assert value.controller._replay.failed
                getter_calls = list(model.getter_calls)
                with pytest.raises(ValueError, match="sticky failure"):
                    setup._warp_empty_exact()
                assert model.getter_calls == getter_calls
        summary = value.controller.termination_evidence.lifecycle.to_payload()["summary"]
        assert summary["release_status"] == "not_authorized"
        assert summary["emergency_sync_attempt_count"] == 0
    else:
        with value.controller.segment():
            assert setup._warp_empty_exact() is owner
        assert value.raw[-1]["shared_helper"] == "empty"
    assert calls == ([] if failure == "before" else [True])
    assert (owner in value.session.runtime_owners) is (failure != "before")
    assert value.session.runtime_observation.journal == []


@pytest.mark.parametrize("boundary", ["after_sync", "failed_sync", "cleanup"])
def test_terminal_boundaries_reject_queries_before_any_getter(warp_cache_model, monkeypatch, boundary):
    value, model = warp_cache_model
    _prepare_model_query(value)
    controller = value.controller
    reached = []

    def rejected_query():
        calls = list(model.getter_calls)
        with pytest.raises(ValueError, match="outside the original terminal sync"):
            value._state(False)
        assert model.getter_calls == calls
        assert controller._replay.terminal_failed
        reached.append(boundary)

    def synchronize():
        if boundary == "failed_sync":
            raise RuntimeError("modeled terminal synchronization failure")

    if boundary == "after_sync":
        original = controller._release

        def release(purpose):
            assert controller._replay.sync_outcome == "succeeded"
            rejected_query()
            return original(purpose)

        monkeypatch.setattr(controller, "_release", release)
    elif boundary == "failed_sync":
        original = controller._quarantine

        def quarantine():
            assert controller._replay.sync_outcome == "failed"
            rejected_query()
            return original()

        monkeypatch.setattr(controller, "_quarantine", quarantine)
    else:
        def cleanup(guardian, authority):
            assert guardian is controller.guardian
            assert guardian.state == "RELEASE_AUTHORIZED" and controller._replay.release == "started"
            rejected_query()
            raise RuntimeError("cleanup probe stopped after rejected query")

        monkeypatch.setattr(type(controller.guardian), "release_whole_graph", cleanup)

    end = controller.terminate(fault="setup_failure", detection_phase="input_validated",
                               emergency_target_synchronize=synchronize)
    assert reached == [boundary]
    summary = end.lifecycle.to_payload()["summary"]
    assert summary["emergency_sync_attempt_count"] == 1
    assert summary["release_status"] == ("failed" if boundary == "cleanup" else "not_authorized")
