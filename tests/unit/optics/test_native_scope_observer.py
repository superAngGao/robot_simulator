"""CPU failure models of the fixed observer, never native-scope proof."""
from types import SimpleNamespace

import pytest

from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.lifecycle import DeviceScopeDescriptor
from optical_worker.native_bootstrap import _RuntimeObservation
from tests.support.guarded_peer import Peer
from tests.support.native_runtime_process import _NativeScopeObserver


@pytest.fixture
def observer(tmp_path):
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id="warp", channel=peer.channel, configuration=peer.config)
    session = SimpleNamespace(bound_controller=lambda: controller, lane="warp", failed=False)
    session.runtime_observation = _RuntimeObservation(session)
    value = object.__new__(_NativeScopeObserver)
    value.session, value.controller, value.directory = session, controller, tmp_path
    value.raw, value.held = [], []
    value.fault = "runtime_scope_success"
    value.module, value.runtime, value.target, value.stream, value.binding = (object() for _ in range(5))
    value.library = {"root": "os_native", "path": "libcuda.so.595.71.05", "sha256": "a" * 64}
    value.count, value.scope_depth, value.target_context = 8, 0, 9000
    value.current_stream_baseline = 123
    # Keep PID/TID observation separate from the CPU state model on Windows.
    value.snapshot_state = {"default_device": 0, "current_context": None, "current_context_device": None,
                            "primary_contexts": [{"ordinal": i, "active": i == 1, "flags": 0} for i in range(8)],
                            "target_context": 9000, "target_current_stream": 123,
                            "held_stream": {"ordinal": 1, "handle": 123}}
    value._state = lambda pristine: value.snapshot_state
    try:
        yield value
    finally:
        if controller.termination_evidence is None:
            controller.terminate(fault="setup_failure", detection_phase="observer_unit_end", emergency_target_synchronize=None)
        peer.close()


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
    with pytest.raises(AssertionError):
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
