"""Shared upload mechanics; CPU tests do not prove real GPU setup safety."""

from __future__ import annotations

import ast
import copy
import inspect
import subprocess
from pathlib import Path

import pytest

from optical_worker import native_setup
from optical_worker.lifecycle import DeviceScopeDescriptor, NativeWorkerSetupLifecycle


class _Owners:
    def __init__(self):
        self.owners = []

    def adopt_host_owner(self, owner, **_):
        self.owners.append(owner)

    def require_adopted(self, *owners):
        if any(not any(held is owner for held in self.owners) for owner in owners):
            raise ValueError("owner missing")


def _context():
    lifecycle = _Owners()
    runtime, target, stream = object(), object(), object()
    context = native_setup._create_upload_context(lifecycle, runtime, target, stream)
    return context, (lifecycle, runtime, target, stream)


def test_context_is_adopted_exactly_and_consumed_once():
    context, owners = _context()
    assert owners[0].owners == [context]
    assert native_setup._consume_upload_context(context) == owners
    assert id(context) not in native_setup._UPLOAD_CONTEXTS
    with pytest.raises(ValueError, match="consumed"):
        native_setup._consume_upload_context(context)


def test_context_accepts_exact_nonweakrefable_production_c0_lifecycle():
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id="cuda",
        target_device_ordinal=1,
        emergency_sync_notifier=lambda _: pytest.fail("CPU-only context synchronized"),
        session_identity="upload_context_unit",
    )
    runtime, target, stream = object(), object(), object()
    context = native_setup._create_upload_context(lifecycle, runtime, target, stream)
    assert native_setup._consume_upload_context(context) == (lifecycle, runtime, target, stream)
    assert id(context) not in native_setup._UPLOAD_CONTEXTS


def test_failed_context_adoption_removes_strong_identity_registration():
    class RejectOwner(_Owners):
        def adopt_host_owner(self, owner, **_):
            self.rejected = owner
            raise RuntimeError("context adoption rejected")

    lifecycle = RejectOwner()
    before = set(native_setup._UPLOAD_CONTEXTS)
    with pytest.raises(RuntimeError, match="adoption rejected"):
        native_setup._create_upload_context(lifecycle, object(), object(), object())
    assert set(native_setup._UPLOAD_CONTEXTS) == before
    with pytest.raises(ValueError, match="unregistered"):
        native_setup._consume_upload_context(lifecycle.rejected)


@pytest.mark.parametrize("field", ["lifecycle", "runtime", "target", "stream"])
def test_same_context_field_rebinding_burns_registration(field):
    context, _ = _context()
    object.__setattr__(context, field, _Owners() if field == "lifecycle" else object())
    with pytest.raises(ValueError, match="changed"):
        native_setup._consume_upload_context(context)
    assert id(context) not in native_setup._UPLOAD_CONTEXTS


def test_copy_or_handmade_context_has_no_registration():
    context, owners = _context()
    for impostor in (copy.copy(context), native_setup._NativeUploadContext(*owners)):
        with pytest.raises(ValueError, match="unregistered"):
            native_setup._consume_upload_context(impostor)
    assert native_setup._consume_upload_context(context) == owners


def test_missing_guardian_hold_is_not_upload_admission():
    context, owners = _context()
    owners[0].owners.clear()
    with pytest.raises(ValueError, match="owner missing"):
        native_setup._consume_upload_context(context)
    with pytest.raises(ValueError, match="unregistered"):
        native_setup._consume_upload_context(context)


@pytest.mark.parametrize("lane", ["cuda", "warp"])
def test_upload_body_matches_approved_pre_extraction_ast(lane):
    """Preserve the extraction except operation IDs and event-scope validation."""
    root = Path(__file__).resolve().parents[3]
    old_text = subprocess.run(
        ["git", "show", "956d1fa8d51b85c25ac17825adef17640f0e69b5:optical_worker/native_setup.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    old = next(
        n
        for n in ast.parse(old_text).body
        if isinstance(n, ast.FunctionDef) and n.name == f"prepare_{lane}_native_input_for_closed_worker"
    )
    old_try = next(n for n in old.body if isinstance(n, ast.Try))
    start = next(
        i for i, n in enumerate(old_try.body) if isinstance(n, ast.Import) and n.names[0].name == "numpy"
    )
    old_body = old_try.body[start:]

    # The original C0 code rejected canonical uppercase names (x_world_R),
    # after earlier copies had already been submitted. Normalize only its five
    # exact per-array operation-ID expressions; retain every other AST node.
    class OperationIds(ast.NodeTransformer):
        changed = 0

        def visit_keyword(self, node):
            if node.arg == "operation_id" and isinstance(node.value, ast.JoinedStr):
                pieces = node.value.values
                assert len(pieces) == 2 and isinstance(pieces[0], ast.Constant)
                assert isinstance(pieces[1], ast.FormattedValue)
                assert ast.dump(pieces[1].value) == ast.dump(ast.parse("manifest.name", mode="eval").body)
                assert pieces[1].conversion == -1 and pieces[1].format_spec is None
                prefix = pieces[0].value
                assert prefix in {
                    f"{lane}_{value}_" for value in ("stage", "host_view", "allocate", "upload")
                }
                node.value = ast.Call(
                    func=ast.Name(id="_array_operation_id", ctx=ast.Load()),
                    args=[ast.Constant(prefix[:-1]), pieces[1].value],
                    keywords=[],
                )
                self.changed += 1
            return node

    original = ast.Module(body=old_body, type_ignores=[])
    ids = OperationIds()
    ids.visit(original)
    assert ids.changed == 5
    new_function = getattr(native_setup, f"_upload_{lane}_native_input")
    new = ast.parse(inspect.getsource(new_function)).body[0]
    new_start = next(
        i for i, n in enumerate(new.body) if isinstance(n, ast.Import) and n.names[0].name == "numpy"
    )
    new_body = new.body[new_start:]
    # Normalize only the four explicit custody/check statements and the one
    # actual-device argument added by the whole-review event repair. Behavioral
    # fault tests below verify those new checks; all other mechanics stay equal.
    cuda = lane == "cuda"
    additions = [ast.dump(ast.parse(code).body[0]) for code in (
        "ready_owners = []",
        'lifecycle.adopt_host_owner(ready_owners, role="ready_event_custody", operation_id="ready_event_custody")',
        "ready_owners.append(ready_event)",
        f"_require_ready_event_target(ready_event, cuda={cuda}, lifecycle=lifecycle)",
    )]
    class EventValidation(ast.NodeTransformer):
        removed = 0
        ordinal = 0

        def visit(self, node):
            if isinstance(node, ast.stmt) and ast.dump(node) in additions:
                additions.remove(ast.dump(node))
                self.removed += 1
                return None
            return super().visit(node)

        def visit_keyword(self, node):
            if node.arg == "actual_device_ordinal" and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "_event_ordinal":
                expected = f"_event_ordinal(ready_event, cuda={cuda}, lifecycle=lifecycle" + (", allow_unbound=True)" if cuda else ")")
                assert ast.dump(node.value) == ast.dump(ast.parse(expected, mode="eval").body)
                node.value = ast.parse("lifecycle.target_device_ordinal" if cuda else "int(target.ordinal)", mode="eval").body
                self.ordinal += 1
            return node
    normalization = EventValidation()
    normalized = normalization.visit(ast.Module(body=new_body, type_ignores=[]))
    assert not additions and normalization.removed == 4 and normalization.ordinal == 1
    assert ast.dump(ast.Module(body=old_body, type_ignores=[]), include_attributes=False) == ast.dump(
        normalized,
        include_attributes=False,
    )
    # C0's wrapper must call the sole core, not retain an allocation loop.
    wrapper = ast.parse(
        inspect.getsource(
            getattr(
                native_setup,
                f"prepare_{lane}_native_input_for_closed_worker",
            )
        )
    ).body[0]
    assert not any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(wrapper))
    assert (
        sum(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == f"_upload_{lane}_native_input"
            for n in ast.walk(wrapper)
        )
        == 1
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "cuda_stage",
        "cuda_host_view",
        "cuda_allocate",
        "cuda_upload",
        "warp_stage",
        "warp_host_view",
        "warp_allocate",
        "warp_upload",
        "validation_host",
        "validation_view",
        "validation_d2h",
    ],
)
def test_array_operation_ids_preserve_case_distinction_and_are_guardian_valid(prefix):
    lifecycle = NativeWorkerSetupLifecycle(
        lane_id="warp",
        target_device_ordinal=1,
        emergency_sync_notifier=lambda _: pytest.fail("CPU operation-ID check synchronized"),
        session_identity="array_ids_unit",
    )
    names = [
        "fixed_x_world_R",
        "fixed_x_world_r",
        "replay_x_world_R",
        "replay_x_world_r",
        "instance_X_body_geometry_R",
        "instance_X_body_geometry_r",
        "A" * 1024,
    ]
    ids = [native_setup._array_operation_id(prefix, name) for name in names]
    assert len(set(ids)) == len(names)
    assert ids == [native_setup._array_operation_id(prefix, name) for name in names]
    for identifier in ids:
        assert len(identifier) <= 128
        lifecycle.adopt_host_owner(object(), role="array_name_test", operation_id=identifier)
        DeviceScopeDescriptor.exact_target(
            operation_id=identifier,
            leaf_kind="allocation",
            device_ordinal=1,
            scope_source="array_name_test",
            runtime_identity="cpu_validation",
        )


@pytest.mark.parametrize(
    "prefix,name", [("INVALID", "a"), ("a" * 64, "a"), (None, "a"), ("valid", ""), ("valid", None)]
)
def test_invalid_array_operation_ids_fail_before_resource_registration(prefix, name):
    with pytest.raises(ValueError):
        native_setup._array_operation_id(prefix, name)


@pytest.fixture(params=[(lane, mode) for lane in ("cuda", "warp") for mode in ("guarded", "c0")])
def upload_event_case(request, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import weakref
    import numpy as np
    from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
    from optical_worker.native_input import CanonicalNativeInput
    from tests.support.guarded_peer import Peer
    from tests.unit.optics.test_guarded_native_readback import _CudaArray, _WarpArray

    lane, mode = request.param
    peer = Peer() if mode == "guarded" else None
    syncs, notices = [], []
    lifecycle = (NativeGuardedSetupLifecycle(lane_id=lane, channel=peer.channel, configuration=peer.config)
                 if peer else NativeWorkerSetupLifecycle(lane_id=lane, target_device_ordinal=1,
                     emergency_sync_notifier=lambda target: notices.append(target), session_identity="event_scope_unit"))
    target = SimpleNamespace(type="cuda", index=1, ordinal=1, is_cuda=True)
    foreign = SimpleNamespace(type="cuda", index=0, ordinal=0, is_cuda=True)
    stream = SimpleNamespace(device=target)
    state = SimpleNamespace(fault=None, event=None, records=0, waits=[], syncs=syncs, notices=notices, terminal=None)
    source = CanonicalNativeInput.from_arrays(case_id="asymmetric_synthetic_7x3_v1",
        materializer_id="upload_event_cpu_double", arrays={"a": np.array([1, 2], dtype=np.float32)},
        array_semantics={"a": "unit.a"}, topology_arrays=("a",), semantics={})

    class Event:
        def __init__(self):
            self.recorded = False
            self.actual = None if lane == "cuda" else target
            if state.fault == "foreign_create":
                self.actual = foreign
            state.event = weakref.ref(self)

        @property
        def device(self):
            # The returned owner must already be held before metadata access,
            # even when that very access throws before ordinary adoption.
            assert any(type(owner) is list and any(item is self for item in owner)
                       for owner in lifecycle.guardian._owners)
            if state.fault == ("unknown_record" if self.recorded else "unknown_create"):
                raise RuntimeError("event metadata unavailable")
            return self.actual

        def record(self, actual_stream):
            assert actual_stream is stream
            self.recorded = True
            state.records += 1
            self.actual = target
            if state.fault == "foreign_record":
                self.actual = foreign
            elif state.fault == "unbound_record":
                self.actual = None
            elif state.fault == "boolean_record":
                self.actual = SimpleNamespace(type="cuda", index=True, ordinal=True, is_cuda=True)

        def synchronize(self):
            state.waits.append(self.actual)

    stream.record_event = lambda event: event.record(stream)
    array_type = _CudaArray if lane == "cuda" else _WarpArray
    runtime = SimpleNamespace(from_numpy=lambda host: host.view(),
        empty=lambda shape, dtype, device: array_type(np.empty(shape, dtype=dtype), device),
        cuda=SimpleNamespace(Event=lambda **kwargs: Event()), ScopedDevice=lambda device: nullcontext(),
        array=lambda host, **kwargs: host.view(), Event=lambda **kwargs: Event(),
        synchronize_event=lambda event: event.synchronize())
    for dtype in ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                  "float16", "float32", "float64"):
        setattr(runtime, dtype, np.dtype(dtype))
    for owner in (source, runtime, target, stream):
        lifecycle.adopt_host_owner(owner, role="unit_upload_owner", operation_id="unit_upload_owner")
    monkeypatch.setattr(native_setup, "_torch_copy_exact", lambda *args: None)
    monkeypatch.setattr(native_setup, "_warp_copy_exact", lambda *args: None)

    def run():
        try:
            with lifecycle.segment("setup") if peer else nullcontext():
                context = native_setup._create_upload_context(lifecycle, runtime, target, stream)
                return getattr(native_setup, "_upload_" + lane + "_native_input")(source, context=context)
        except BaseException as error:
            if not peer:
                state.terminal = native_setup._abort(lifecycle, exception=error, sync=lambda: syncs.append(1)).evidence
            raise
    try:
        yield SimpleNamespace(run=run, lifecycle=lifecycle, state=state, target=target, mode=mode)
    finally:
        if (lifecycle.termination_evidence is None if peer else state.terminal is None):
            lifecycle.terminate(fault="setup_failure", detection_phase="unit_end",
                                emergency_target_synchronize=lambda: syncs.append(1))
        if peer:
            peer.close()


def test_upload_ready_event_actual_target_allows_single_wait(upload_event_case):
    case = upload_event_case
    prepared = case.run()
    assert prepared.ready_event is case.state.event()
    assert case.state.records == 1 and case.state.waits == [case.target]
    assert not case.state.syncs and not case.state.notices
    if case.mode == "guarded":
        assert case.lifecycle.evidence().to_payload()["summary"]["normal_complete_count"] == 1


@pytest.mark.parametrize("fault", ["foreign_create", "foreign_record", "unknown_create", "unknown_record",
                                  "unbound_record", "boolean_record"])
def test_upload_event_scope_failure_never_waits_or_releases(upload_event_case, fault):
    import gc
    case = upload_event_case
    case.state.fault = fault
    with pytest.raises((RuntimeError, ValueError, AttributeError)):
        case.run()
    gc.collect()
    assert case.state.event() is not None
    assert case.state.records == (0 if fault.endswith("create") else 1)
    assert case.state.waits == case.state.syncs == case.state.notices == []
    summary = (case.lifecycle.termination_evidence.lifecycle.to_payload()["summary"]
               if case.mode == "guarded" else case.state.terminal.to_payload())
    if case.mode == "guarded":
        assert summary["release_status"] == "not_authorized"
    else:
        assert summary["resource_release_authorized"] is summary["ordinary_cleanup_started"] is False
    assert summary["emergency_sync_attempt_count"] == 0
    if fault.startswith("foreign"):
        assert 0 in summary["possible_dirty_device_ordinals"]
    else:
        assert summary["unknown_device_scope"] is True
    if case.mode == "guarded":
        assert summary["normal_complete_count"] == summary["terminal_arm_attempt_count"] == 0
