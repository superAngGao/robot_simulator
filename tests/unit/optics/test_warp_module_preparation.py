"""Real lifecycle/entry tests; simulated Warp leaves do not prove GPU behavior."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from optical_worker import native_bootstrap as bootstrap, native_setup, warp_setup
from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.wait_protocol import GuardConfiguration
from tests.support.guarded_peer import Peer

compile_warp_module = warp_setup.compile_module
definition_facts = warp_setup._definition_facts


def test_compilation_inventory_covers_actual_kernel_function_and_struct_declarations():
    root = Path(__file__).resolve().parents[3]
    for module, kernels, functions, structs in warp_setup._DEFINITIONS:
        tree = ast.parse((root / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8"))
        actual = {kind: [] for kind in ("kernel", "func", "struct")}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    if (isinstance(decorator, ast.Attribute) and isinstance(decorator.value, ast.Name)
                            and decorator.value.id == "wp" and decorator.attr in actual):
                        actual[decorator.attr].append(node.name)
        assert actual == {"kernel": list(kernels), "func": list(functions), "struct": list(structs)}


@pytest.fixture
def prepared_session(monkeypatch, request):
    lane = getattr(request, "param", "warp")
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id=lane, channel=peer.channel, configuration=peer.config)
    session = object.__new__(bootstrap._SourceSession)
    session.lane, session.phase = lane, "input_validated"
    session.failed, session.failures, session.runtime_owners = False, [], []
    session.warp_preparation_started = False
    session.canonical_input, session.prepared_input, session.validation_report = object(), object(), object()
    session.runtime_observation = bootstrap._RuntimeObservation(session)
    session.modules = {"optical_worker.native_setup": (native_setup, None, "complete")}
    wp = ModuleType("warp" if lane == "warp" else "torch")
    runtime = object()
    wp.config = SimpleNamespace(quiet=True)
    target = SimpleNamespace(is_cuda=True, ordinal=controller.target_device_ordinal, runtime=runtime)
    stream = SimpleNamespace(device=target, cuda_stream=412)
    session.external = SimpleNamespace(
        audit=lambda: controller._require_setup(), records={"warp": {"module": wp},
            "warp._src.context": {"module": SimpleNamespace(runtime=runtime)},
            "warp._src.types": {"module": SimpleNamespace()}})
    if lane == "cuda":
        target.index, target.type = target.ordinal, "cuda"
        wp._C = runtime
        session.external.records = {"torch": {"module": wp}}
        monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setitem(sys.modules, wp.__name__, wp)
    monkeypatch.setitem(bootstrap._SOURCE_CUSTODY, session,
        (controller, controller.guardian, peer.config, controller._guard_client.configuration,
         peer.channel, NativeGuardedSetupLifecycle, GuardConfiguration))

    def check():
        bootstrap._require(not session.failed, "source session failed")
        bootstrap._require(not controller._terminating and not controller._replay.failed, "terminal source")

    session.check_live = check
    session.audit = lambda: controller._require_setup()
    binding = object.__new__(bootstrap._RuntimeSetupBinding)
    identity = (target.ordinal, 412)
    for name, value in dict(session=session, controller=controller, module=wp, runtime=runtime,
                            target=target, stream=stream, stream_identity=identity,
                            source=session.canonical_input).items():
        object.__setattr__(binding, name, value)
    runtime_record = (binding, controller, session.external, wp, runtime, target, stream,
                      identity, session.canonical_input)
    monkeypatch.setitem(bootstrap._UPLOADED_INPUTS, session, (runtime_record, session.prepared_input))
    for value in (session, wp, runtime, target, stream, binding, session.canonical_input, session.prepared_input):
        controller.adopt_host_owner(value, role="unit_owner", operation_id="unit_owner")
    # Model completed upload: subsequent IDLE failure has outstanding exact-target work.
    with controller.segment("setup"):
        session._runtime_leaf("unit_upload", "allocation")
    syncs = []
    monkeypatch.setattr(native_setup, "_warp_synchronize_target_exact",
                        lambda actual_wp, actual_target: syncs.append((actual_wp, actual_target)))
    if lane == "cuda":
        wp.cuda = SimpleNamespace(device=lambda actual: nullcontext(),
                                  synchronize=lambda actual: syncs.append((wp, actual)))
    checkpoints = []
    monkeypatch.setattr(session.runtime_observation, "checkpoint", lambda: checkpoints.append(True))
    imports = []
    import optics
    for name in ("device_scene", "device_bvh", "warp_native_render"):
        module = ModuleType("optics." + name)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(optics, name, module, raising=False)
        imports.append(module)
    monkeypatch.setattr(warp_setup, "validate_compilation_inputs", lambda actual: controller._require_setup())
    monkeypatch.setattr(warp_setup, "definition_owners", lambda actual, index: (imports[index], imports[index], ()))
    records = []
    cache = SimpleNamespace(records=records)
    controller.adopt_host_owner(cache, role="unit_cache", operation_id="unit_cache")
    monkeypatch.setattr(warp_setup, "validate_warp_cache", lambda actual, actual_wp: cache)

    def compile_one(actual, actual_binding, index):
        assert actual is session and actual_binding is binding and index == len(records)
        controller._require_setup()
        session._runtime_leaf("unit_compile", "runtime_import_or_init")
        record = SimpleNamespace(guard=(controller._guard_reference["sequence"], "setup"))
        session._hold_runtime_owner(record, "unit_compile_result")
        records.append(record)
        return record

    monkeypatch.setattr(warp_setup, "compile_module", compile_one)
    try:
        yield SimpleNamespace(session=session, controller=controller, binding=binding, records=records,
                              syncs=syncs, checkpoints=checkpoints, compile_one=compile_one)
    finally:
        if controller.termination_evidence is None:
            controller.terminate(fault="setup_failure", detection_phase="unit_end",
                                 emergency_target_synchronize=lambda: None)
        warp_setup._MODULE_SETS.pop(session, None)
        peer.close()


def test_import_then_three_nonnested_compile_guards_issue_one_held_set(prepared_session):
    p = prepared_session
    owner = p.session.prepare_warp_modules()
    assert owner.session is p.session and owner.binding is p.binding
    assert owner.modules == tuple(p.records)
    assert p.session.phase == "warp_modules_ready" and p.checkpoints == [True]
    p.controller.require_adopted(owner, *p.records)
    summary = p.controller.evidence().to_payload()["summary"]
    assert summary["normal_complete_count"] == 5  # upload, imports, three compiles
    assert p.syncs == []


def test_preparation_called_in_an_active_guard_keeps_that_guard(prepared_session):
    p = prepared_session
    with pytest.raises(bootstrap.SourceBootstrapError, match="healthy IDLE"):
        with p.controller.segment():
            original_guard = dict(p.controller._guard_reference)
            p.session.prepare_warp_modules()
    failures = [event for event in p.controller._guard_events if event["operation"] == "guard_failed"]
    assert len(failures) == 1 and failures[0]["guard"] == original_guard
    assert not p.records and not p.syncs
    summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["normal_complete_count"] == 1 and summary["emergency_sync_attempt_count"] == 0


@pytest.mark.parametrize("boundary", ["entry", "issuance", "repeat"])
def test_idle_failure_preserves_successful_guards_and_uses_existing_emergency(prepared_session, monkeypatch, boundary):
    p = prepared_session
    failure = RuntimeError("CPU issuance failed")
    if boundary == "entry":
        p.session.validation_report = None
        expected = 1
    elif boundary == "issuance":
        def issue(*args):
            assert p.controller._guard_client.state == "IDLE"
            raise failure
        monkeypatch.setattr(warp_setup, "issue_module_set", issue)
        expected = 5
    else:
        p.session.prepare_warp_modules()
        expected = 5
    with pytest.raises((bootstrap.SourceBootstrapError, RuntimeError)) as caught:
        p.session.prepare_warp_modules()
    if boundary == "issuance":
        assert caught.value is failure
    summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["normal_complete_count"] == expected
    assert summary["emergency_sync_attempt_count"] == 1
    assert summary["disposition"] == "aborted_synced_exit"
    assert p.syncs == [(p.binding.module, p.binding.target)]
    assert not any(e["operation"] == "guard_failed" for e in p.controller._guard_events)
    assert p.session.failed


@pytest.mark.parametrize("compromised", ["unknown", "unknown_valid_report", "missing_owner", "binding_drift"])
def test_idle_failure_with_unprovable_custody_quarantines_without_sync(prepared_session, compromised):
    p = prepared_session
    if compromised.startswith("unknown"):
        p.controller.mark_unknown_device_scope(source="unit_fault", reason="unknown_native_effect")
    elif compromised == "missing_owner":
        owners = p.controller.guardian._owners
        index = next(i for i, value in enumerate(owners) if value is p.binding.stream)
        owners[index] = object()
    else:
        object.__setattr__(p.binding, "stream", object())
    if compromised != "unknown_valid_report":
        p.session.validation_report = None
    with pytest.raises(bootstrap.SourceBootstrapError):
        p.session.prepare_warp_modules()
    summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["emergency_sync_attempt_count"] == 0
    assert p.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE" and p.syncs == []
    assert not any(e["operation"] == "guard_failed" for e in p.controller._guard_events)
    assert summary["normal_complete_count"] == 1 and not p.records


@pytest.mark.parametrize("compromised", ["unknown", "missing_owner"])
def test_actual_idle_issuer_preserves_completed_guards_on_scope_failure(prepared_session, monkeypatch, compromised):
    p = prepared_session
    original = warp_setup.issue_module_set

    def issue(session, binding, records):
        assert p.controller._guard_client.state == "IDLE" and p.controller._guard_reference is None
        if compromised == "unknown":
            p.controller.mark_unknown_device_scope(source="unit_issuance_fault", reason="unknown_device_scope")
        else:
            # Keep all actual roots held, but invalidate required guardian coverage.
            owners = p.controller.guardian._owners
            index = next(i for i, value in enumerate(owners) if value is p.binding.stream)
            owners[index] = object()
            raise RuntimeError("CPU issuance failed with incomplete custody")
        return original(session, binding, records)

    monkeypatch.setattr(warp_setup, "issue_module_set", issue)
    with pytest.raises((ValueError, RuntimeError)):
        p.session.prepare_warp_modules()
    assert len(p.records) == 3 and not p.syncs and not p.checkpoints
    assert p.session not in warp_setup._MODULE_SETS
    assert all(any(owner is record for owner in p.session.runtime_owners) for record in p.records)
    assert not any(e["operation"] == "guard_failed" for e in p.controller._guard_events)
    summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["normal_complete_count"] == 5 and summary["guard_state"] == "IDLE"
    assert summary["emergency_sync_attempt_count"] == 0
    assert p.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"


@pytest.mark.parametrize("stage", ["import", "first_compile", "last_compile"])
def test_guarded_failure_preserves_original_error_and_partial_roots(prepared_session, monkeypatch, stage):
    p = prepared_session
    failure = RuntimeError("partial Warp work")
    if stage == "import":
        def definitions(*args):
            p.controller._require_setup()
            raise failure
        monkeypatch.setattr(warp_setup, "definition_owners", definitions)
        expected = 1
    else:
        selected = 0 if stage == "first_compile" else 2
        def compile_one(*args):
            result = p.compile_one(*args)
            if args[2] == selected:
                raise failure
            return result
        monkeypatch.setattr(warp_setup, "compile_module", compile_one)
        expected = selected + 2
    with pytest.raises(RuntimeError) as caught:
        p.session.prepare_warp_modules()
    assert caught.value is failure
    summary = p.controller.termination_evidence.lifecycle.to_payload()["summary"]
    assert summary["normal_complete_count"] == expected
    assert summary["emergency_sync_attempt_count"] == 0
    assert p.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE" and p.syncs == []
    assert p.checkpoints == [] and p.session not in warp_setup._MODULE_SETS
    assert all(any(owner is record for owner in p.session.runtime_owners) for record in p.records)


@pytest.mark.parametrize("fault", [None, "load_before", "load_after_publication", "adoption", "metadata"])
def test_actual_compile_helper_holds_mapping_and_executable_across_failures(
        prepared_session, monkeypatch, tmp_path, fault):
    p = prepared_session
    failure = RuntimeError("fixed compile failure")
    target, wp = p.binding.target, p.binding.module
    target.arch, target.context = 90, 42
    wp.ScopedDevice = wp.ScopedStream = lambda owner: nullcontext(owner)
    identifier = "wp_optics.device_scene_abc1234"
    module_hash = b"module hash"
    module = SimpleNamespace(
        execs={}, failed_builds={}, get_module_hash=lambda block_dim: module_hash,
        get_module_identifier=lambda: identifier,
        _get_compile_output_name=lambda target: identifier + ".sm90.cubin",
        _get_meta_name=lambda: identifier + ".meta")
    python_module = ModuleType("optics.device_scene")
    loader = SimpleNamespace(relative="optics/device_scene.py")
    p.session.modules[python_module.__name__] = (python_module, loader, "complete")
    p.session.sources = {loader.relative: (b"held source", "a" * 64)}
    cache = SimpleNamespace(path=tmp_path, context=SimpleNamespace(ModuleExec=SimpleNamespace), records=[])
    monkeypatch.setattr(warp_setup, "validate_warp_cache", lambda *args: cache)
    monkeypatch.setattr(warp_setup, "definition_owners", lambda *args: (python_module, module, ()))
    returned = []

    def load(**kwargs):
        # Ownership precedes both publication and the potentially raising call.
        p.controller.require_adopted(module, module.execs)
        assert kwargs == dict(module=module, device=target, recursive=False, block_dim=256, max_workers=0)
        if fault == "load_before":
            raise failure
        executable = SimpleNamespace(device=target, handle=99, module_hash=module_hash,
                                     meta={"value": 2 if fault == "metadata" else 1})
        module.execs[(target.context, 256)] = executable
        returned.append(executable)
        if fault == "load_after_publication":
            raise failure
        directory = tmp_path / identifier
        directory.mkdir()
        (directory / (identifier + ".sm90.cubin")).write_bytes(b"\x7fELF\x02\x01\x01unit binary")
        (directory / (identifier + ".meta")).write_bytes(b'{"value":1}')
        (directory / (identifier + ".cu")).write_bytes(b"generated source")

    wp.load_module = load
    if fault == "adoption":
        def reject_adoption(owner, **kwargs):
            assert any(value is owner for value in p.session.runtime_owners)
            raise failure
        monkeypatch.setattr(p.controller, "adopt_returned_device_owner", reject_adoption)
    if fault is None:
        with p.controller.segment():
            record = compile_warp_module(p.session, p.binding, 0)
        assert record.executable is returned[0] and cache.records == [record]
        assert record.mapping is module.execs and len(record.artifacts) == 3
        p.controller.require_adopted(record, record.executable, module.execs)
    else:
        with pytest.raises((RuntimeError, ValueError)) as caught:
            with p.controller.segment():
                compile_warp_module(p.session, p.binding, 0)
        if fault != "metadata":
            assert caught.value is failure
        assert cache.records == []
        assert any(value is module.execs for value in p.session.runtime_owners)
        if returned:
            assert module.execs[(target.context, 256)] is returned[0]
        assert p.controller.guardian.state == "TRANSFERRED_TO_QUARANTINE"
        assert p.controller.termination_evidence.lifecycle.to_payload()["summary"]["emergency_sync_attempt_count"] == 0


def test_aggregate_cannot_issue_from_copied_compilation_records(prepared_session, monkeypatch):
    p = prepared_session
    original = warp_setup.issue_module_set
    def copied(session, binding, records):
        copies = tuple(SimpleNamespace(guard=record.guard) for record in records)
        return original(session, binding, copies)
    monkeypatch.setattr(warp_setup, "issue_module_set", copied)
    with pytest.raises(ValueError, match="copied"):
        p.session.prepare_warp_modules()
    assert p.session not in warp_setup._MODULE_SETS
    assert p.controller.termination_evidence.lifecycle.to_payload()["summary"]["emergency_sync_attempt_count"] == 1


@pytest.mark.parametrize("change", [None, "dtype", "ndim", "entry_hash", "struct_field", "nested_array"])
def test_later_admission_freezes_abi_values_and_entry_identity(prepared_session, monkeypatch, change):
    p = prepared_session
    class Array:
        def __init__(self):
            self.dtype, self.ndim = float, 1
    class Struct:
        pass
    # Dependency metadata doubles only. The test reaches actual production
    # snapshot, issuance and later validation with a real lifecycle/guardian.
    types = SimpleNamespace(
        get_type_code=lambda value: "array" if isinstance(value, Array) else "struct" if isinstance(value, Struct) else value.__name__,
        is_array=lambda value: isinstance(value, Array),
        type_is_struct=lambda value: isinstance(value, Struct))
    p.session.external.records["warp._src.types"]["module"] = types
    p.binding.target.context = 42
    cache = SimpleNamespace(records=[])
    p.session._hold_runtime_owner(cache, "unit_cache_for_validation")
    monkeypatch.setattr(warp_setup, "validate_warp_cache", lambda *args: cache)
    monkeypatch.setattr(warp_setup, "_artifact_facts", lambda *args: ((), b"{}"))
    definitions = []
    kernels, structs = [], []
    for index in range(3):
        python_module = ModuleType("unit_module_" + str(index))
        module = SimpleNamespace(get_module_hash=lambda block: b"unchanged cached module hash")
        executable = SimpleNamespace(device=p.binding.target, module_hash=module.get_module_hash(256),
                                     handle=100 + index, meta={})
        module.execs = {(42, 256): executable}
        def function():
            pass
        kernel = SimpleNamespace(func=function, adj=SimpleNamespace(args=[SimpleNamespace(label="array", type=Array())]),
                                 options={}, key="unit_kernel", hash=b"compiled kernel hash")
        kernel.get_mangled_name = lambda owner=kernel: owner.key + "_" + owner.hash.hex()[:8]
        struct = Struct()
        struct.cls, struct.module, struct.ctype, struct.instance_type = Struct, module, object(), object()
        struct.vars = {"scalar": SimpleNamespace(type=float), "nested": SimpleNamespace(type=Array())}
        struct.hash, struct.native_name = b"struct hash", "unit_struct"
        owners = (kernel, struct)
        definitions.append((python_module, module, owners))
        kernels.append(kernel)
        structs.append(struct)
        with p.controller.segment():
            record = warp_setup._CompiledModule(
                python_module, module, module.execs, executable, owners, definition_facts(owners, types),
                executable.module_hash, executable.handle, b"{}", (), "a" * 64,
                (p.controller._guard_reference["sequence"], "setup"))
            for value in (module, module.execs, executable, record):
                p.session._hold_runtime_owner(value, "unit_compiled_owner")
            cache.records.append(record)
    monkeypatch.setattr(warp_setup, "definition_owners", lambda session, index: definitions[index])
    records = tuple(cache.records)
    owner = warp_setup.issue_module_set(p.session, p.binding, records)
    with p.controller.segment():
        assert warp_setup.validate_module_set(p.session, owner) is records
    if change is None:
        with p.controller.segment():
            assert warp_setup.validate_module_set(p.session, owner) is records
        return
    if change == "dtype":
        kernels[0].adj.args[0].type.dtype = int
    elif change == "ndim":
        kernels[0].adj.args[0].type.ndim = 2
    elif change == "entry_hash":
        kernels[0].hash = b"another kernel entry"
    elif change == "struct_field":
        structs[0].vars["scalar"].type = int
    else:
        structs[0].vars["nested"].type.ndim = 2
    with pytest.raises(ValueError, match="ABI changed"):
        with p.controller.segment():
            warp_setup.validate_module_set(p.session, owner)
    assert p.controller.termination_evidence.lifecycle.to_payload()["summary"]["emergency_sync_attempt_count"] == 0
