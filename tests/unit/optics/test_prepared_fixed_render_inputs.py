"""Exact fixed-input wrapping with CPU arrays; not GPU execution evidence."""
from dataclasses import replace
from contextlib import contextmanager
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from optical_worker import native_input as native
from optical_worker.native_contract import (
    CudaPreparedNativeInput, WarpPreparedNativeInput, register_prepared_native_input,
)
from optical_worker.workloads import build_canonical_packed_scene, build_canonical_fixed_fixture
from tests.unit.optics.test_guarded_native_readback import _CudaArray, _WarpArray


def test_fixture_preserves_stats_from_its_single_actual_host_build(monkeypatch):
    built = []
    original = native.build_host_median_bvh

    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        built.append(result)
        return result

    monkeypatch.setattr(native, 'build_host_median_bvh', build)
    packed = build_canonical_packed_scene('asymmetric_synthetic_7x3_v1')
    fixture = build_canonical_fixed_fixture(packed)
    assert len(built) == 1
    assert fixture.bvh_stats is built[0]['stats']
    assert fixture.bvh_stats.sah_quality_cost > 0
    assert fixture.bvh_stats.num_nodes == len(fixture.arrays['bvh_left'])
    # build_canonical_fixed_fixture has checked the existing frozen digest.


@pytest.fixture(params=['warp', 'cuda'])
def fixed_inputs(request, monkeypatch):
    from optics import device_scene, cuda_direct_light
    from optical_worker.render_parameters import resolve_render_source

    render = resolve_render_source('asymmetric_synthetic_7x3_v1')
    packed = render.packed_scene
    fixture = build_canonical_fixed_fixture(packed)
    source = native.CanonicalNativeInput.from_scene_contract(
        packed, input_contract='fixed_native_scene_v1', fixed_fixture=fixture)
    device = NS(type='cuda', index=1, ordinal=1, is_cuda=True)
    lane = request.param
    array_type = _WarpArray if lane == 'warp' else _CudaArray
    prepared_type = WarpPreparedNativeInput if lane == 'warp' else CudaPreparedNativeInput
    arrays = {name: array_type(value, device) for name, value in source.arrays.items()}
    prepared = prepared_type(source.manifest, 1, arrays, object(), source)
    register_prepared_native_input(prepared, source)
    # Native allocation/copy methods are deliberately absent. Only native
    # metadata is doubled; real registry, constructors and wrappers execute.
    runtime = NS(**{name: np.dtype(name) for name in (
        'bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
        'float16', 'float32', 'float64')})
    if lane == 'warp':
        runtime.get_device = lambda value: device
        monkeypatch.setattr(device_scene, 'wp', runtime)
        monkeypatch.setattr(device_scene, '_HAS_WARP', True)
        build = native.build_prepared_warp_fixed_render_inputs
    else:
        runtime.Tensor = _CudaArray
        monkeypatch.setitem(sys.modules, 'torch', runtime)

        def build(value):
            return native.build_prepared_cuda_fixed_render_inputs(
                value, sensor_role_mask=1, max_distance=50.0)

    return NS(prepared=prepared, fixture=fixture, lane=lane, build=build,
              runtime=runtime, device=device, render=render)


def test_fixed_render_views_borrow_exact_original_arrays(fixed_inputs):
    p = fixed_inputs
    scene, first, second = p.build(p.prepared)
    arrays = p.prepared.arrays
    if p.lane == 'warp':
        from optics.device_scene import _warp_world_scene_arrays

        for name, array in _warp_world_scene_arrays(first):
            assert array is arrays[name]
        for name in ('bounds_min', 'bounds_max', 'left', 'right', 'start', 'count',
                     'node_depth', 'level_ranges', 'prim_ids'):
            assert getattr(second, name) is arrays['bvh_' + name]
        assert second.stats is p.fixture.bvh_stats
        assert second.primitive_global_id is arrays['triangle_primitive_global_id']
        assert second.ready_event is p.prepared.ready_event
        assert first.resources[0] is p.prepared and second.resources[0] is p.prepared
        assert second.topology_origin_id is None  # Fixed R0 does not grant refit authority.
    else:
        for name, array in first.tensor_items() + second.tensor_items():
            assert array is arrays[name]
        assert first.num_bvh_nodes == p.fixture.bvh_stats.num_nodes
        assert first.num_planes == scene.num_planes and second.num_lights == scene.num_lights
        assert first.sensor_role_mask == 1 and first.max_distance == 50.0
    assert scene.resources[0] is p.prepared


@pytest.mark.parametrize('attack', ['copy', 'array', 'shape', 'device', 'fixture_owner'])
def test_fixed_views_reject_lost_identity_before_scene_construction(fixed_inputs, monkeypatch, attack):
    p = fixed_inputs
    prepared = p.prepared
    if attack == 'copy':
        prepared = replace(prepared)
    elif attack == 'array':
        value = prepared.arrays['bvh_left']
        prepared.arrays['bvh_left'] = type(value)(value.data, value.device)
    elif attack == 'shape':
        prepared.arrays['bvh_left'].shape = (999,)
    elif attack == 'device':
        prepared.arrays['bvh_left'].device = NS(type='cuda', index=0, ordinal=0, is_cuda=True)
    else:
        object.__setattr__(prepared.source, 'fixed_fixture_owner', object())
    monkeypatch.setattr(native, 'build_prepared_' + p.lane + '_local_scene',
                        lambda _: pytest.fail('invalid prepared input reached scene construction'))
    with pytest.raises((ValueError, TypeError)):
        p.build(prepared)


@pytest.fixture
def fixed_storage_session(fixed_inputs, monkeypatch, fail_at):
    from optical_worker.native_bootstrap import _SourceSession, _UPLOADED_INPUTS, SourceBootstrapError
    from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
    from optical_worker import warp_setup
    from optics import cuda_direct_light, warp_native_render
    from tests.support.guarded_peer import Peer

    p = fixed_inputs
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id=p.lane, channel=peer.channel, configuration=peer.config)
    stream = NS(device=p.device, cuda_stream=123)
    state = NS(device=0, stream=None, arrays=[], calls=0)
    failure = RuntimeError('fixed output allocation failure')

    @contextmanager
    def scope(owner, kind):
        previous = getattr(state, kind)
        setattr(state, kind, owner)
        try:
            yield
        finally:
            setattr(state, kind, previous)

    array_type = _WarpArray if p.lane == 'warp' else _CudaArray

    def empty(shape, *, dtype, device):
        controller._require_setup()
        assert device is p.device and state.device is p.device and state.stream is stream
        state.calls += 1
        if state.calls == fail_at:
            raise failure
        value = array_type(np.empty(shape, dtype=dtype), device)
        value.ndim, value.size = value.data.ndim, value.data.size
        value.ctype, value.grad, value._ref, value.deleter = None, None, None, object()
        state.arrays.append(value)
        return value

    def available():
        # Existing CUDA preamble runs after scope entry, not on ambient device.
        controller._require_setup()
        assert state.device is p.device and state.stream is stream
        return True

    p.runtime.empty = empty
    p.runtime.device = lambda value: p.device
    p.runtime.cuda = NS(device=lambda value: scope(value, 'device'),
                        stream=lambda value: scope(value, 'stream'), is_available=available)
    p.runtime.ScopedDevice = lambda value: scope(value, 'device')
    p.runtime.ScopedStream = lambda value: scope(value, 'stream')
    p.runtime.array = array_type
    p.runtime.types = NS(type_size_in_bytes=lambda dtype: dtype.itemsize)
    monkeypatch.setattr(cuda_direct_light, 'torch', p.runtime)
    monkeypatch.setattr(cuda_direct_light, '_HAS_CUDA_DIRECT_LIGHT_DEPS', True)
    monkeypatch.setattr(warp_native_render, 'wp', p.runtime)
    monkeypatch.setattr(warp_native_render, '_HAS_WARP', True)
    # Source/environment/compile intake is doubled here; the actual session
    # method, allocation/view builders, guard and owner handling all execute.
    session = object.__new__(_SourceSession)
    session.lane, session.prepared_input, session.render_source = p.lane, p.prepared, p.render
    session.fixed_row = None
    session.phase = 'warp_modules_ready' if p.lane == 'warp' else 'cuda_factories_ready'
    session.validation_report, session.warp_modules = object(), object()
    session.runtime_owners, session.failures = [], []
    session.bound_controller = lambda: controller
    session.check_live = controller._check_running
    session.audit = lambda: None
    session.external = NS(audit=lambda: None)
    session.runtime_observation = NS(checkpoint=lambda: None)
    session.fail = session.failures.append
    session.cuda_extensions = lambda: object()
    record = (object(),)
    monkeypatch.setitem(_UPLOADED_INPUTS, session, (record, p.prepared))

    def validate_binding(binding, actual):
        assert actual is record and binding is record[0]
        controller._require_setup()
        return p.runtime, p.device, stream

    session._validate_runtime_binding_record = validate_binding

    def compiled(actual, owner):
        assert actual is session and owner is session.warp_modules
        controller._require_setup()

    monkeypatch.setattr(warp_setup, 'validate_module_set', compiled)
    for owner in (session, p.prepared, p.runtime, p.device, stream):
        controller.adopt_host_owner(owner, role='fixture', operation_id='fixture')
    try:
        yield NS(p=p, session=session, controller=controller, stream=stream, state=state, failure=failure)
    finally:
        if controller.termination_evidence is None:
            controller.terminate(fault='setup_failure', detection_phase='fixture_exit',
                                 emergency_target_synchronize=lambda: None)
        peer.close()


@pytest.mark.parametrize('diagnostics', [False, True])
@pytest.mark.parametrize('fail_at', [None, 2])
def test_fixed_storage_connects_real_builders_under_existing_guard(fixed_storage_session, diagnostics, fail_at):
    from optical_worker.native_bootstrap import SourceBootstrapError

    f = fixed_storage_session
    p, session, controller, state, failure = f.p, f.session, f.controller, f.state, f.failure
    def build():
        return session.prepare_fixed_render_storage(traversal_diagnostics=diagnostics)

    if fail_at:
        with pytest.raises(RuntimeError) as caught:
            build()
        assert caught.value is failure and state.calls == fail_at
        assert controller.guardian.state == 'TRANSFERRED_TO_QUARANTINE'
        summary = controller.termination_evidence.lifecycle.to_payload()['summary']
        assert summary['normal_complete_count'] == summary['emergency_sync_attempt_count'] == 0
    else:
        result = build()
        assert session.phase == 'fixed_render_storage_ready'
        assert result is session.fixed_render_storage
        assert len(result) == 4 and result[0].resources[0] is p.prepared
        outputs = result[3]
        assert outputs.origins.shape == (21, 3)
        count = len(outputs.traversal_counters if p.lane == 'warp' else outputs.shadow_traversal_counters)
        assert count == (9 if diagnostics else 0)
        before = state.calls
        with pytest.raises(SourceBootstrapError, match='once'):
            build()
        assert state.calls == before
    assert state.device == 0 and state.stream is None
    for array in state.arrays:
        assert any(type(owner) is list and any(value is array for value in owner)
                   for owner in controller.guardian._owners)


@pytest.mark.parametrize('lane', ['warp', 'cuda'])
@pytest.mark.parametrize('phase', ['input_validated', 'preparing_fixed_render_storage', 'fixed_render_storage_ready',
                                   'preparing_fixed_render_graph', 'fixed_render_graph_ready'])
def test_fixed_storage_source_closure_remains_lane_specific(lane, phase):
    from optical_worker.native_bootstrap import _SourceSession

    session = object.__new__(_SourceSession)
    session.phase, session.lane = phase, lane
    admitted = session.admitted_sources()
    ready = phase != 'input_validated'
    assert ('optics.warp_native_render' in admitted) is (ready and lane == 'warp')
    assert ('optics.cuda_direct_light' in admitted) is (ready and lane == 'cuda')
    graph = phase in {'preparing_fixed_render_graph', 'fixed_render_graph_ready'}
    assert ('optics.execution' in admitted) is (lane == 'cuda' and ready or lane == 'warp' and graph)
    # R0 materialization now uses the original mesh-wide transform functions.
    assert 'optics.scene' in admitted and 'optics.query_numerics' in admitted
    assert not any(name.startswith(('benchmarks', 'tests', 'tools')) for name in admitted)


@pytest.mark.parametrize('fail_at', [None])
@pytest.mark.parametrize('selector,diagnostic_failure', [
    ('success', False), ('diagnostics', False), ('allocate_raise', False), ('adoption_raise', False),
    ('allocate_raise', True), ('adoption_raise', True)])
def test_fixed_storage_native_support_hits_production_boundary(
        fixed_storage_session, selector, diagnostic_failure, tmp_path, monkeypatch):
    import json
    from tests.support.native_runtime_process import _prepare_observed_fixed_storage
    from tests.support.cuda_artifact_runtime_acceptance import _validate_fixed_storage

    f = fixed_storage_session
    p, session, c, stream, state = f.p, f.session, f.controller, f.stream, f.state
    fault = 'runtime_fixed_storage_' + selector
    binding = NS(module=p.runtime, target=p.device, stream=stream)

    def ticket():
        c._guard_client.require_live_ticket(c._guard_ticket, role='setup')

    def native_state(pristine):
        ticket()
        interior = state.device is p.device
        result = {'primary_contexts': [{'ordinal': 0, 'active': False}, {'ordinal': 1, 'active': True}],
                  'target_current_stream': stream.cuda_stream if state.stream is stream else 0,
                  'held_stream': {'ordinal': 1, 'handle': stream.cuda_stream}}
        if p.lane == 'cuda':
            result.update(logical_device=int(interior), raw_device=1)
        else:
            result.update(default_device=int(interior), current_context=123 if interior else None,
                          current_context_device=1 if interior else None)
        return result

    observer = NS(_ticket=ticket, _state=native_state)
    for _ in range(11 if p.lane == 'warp' else 7):
        with c.segment('setup'):
            pass
    original_empty, original_adopt = p.runtime.empty, c.adopt_returned_device_owner
    if diagnostic_failure:
        from tests.support import native_runtime_process as driver
        original_write = driver._write

        def write(directory, name, value):
            if name == 'fixed_storage_failure.json':
                raise OSError('fixed diagnostic write failure')
            return original_write(directory, name, value)

        monkeypatch.setattr(driver, '_write', write)
    successful = selector in {'success', 'diagnostics'}
    if successful:
        _prepare_observed_fixed_storage(session, c, binding, observer, tmp_path, fault)
    else:
        with pytest.raises(RuntimeError, match='fixed render storage failure: ' + fault):
            _prepare_observed_fixed_storage(session, c, binding, observer, tmp_path, fault)
        assert state.calls == (2 if selector == 'adoption_raise' else 1)
        assert c.guardian.state == 'TRANSFERRED_TO_QUARANTINE'
    assert p.runtime.empty is original_empty
    assert c.adopt_returned_device_owner.__func__ is original_adopt.__func__
    assert c.adopt_returned_device_owner.__self__ is original_adopt.__self__
    if diagnostic_failure:
        assert any(type(owner) is OSError and str(owner) == 'fixed diagnostic write failure'
                   for owner in session.runtime_owners)
        assert not (tmp_path / 'fixed_storage_failure.json').exists()
        return
    _validate_fixed_storage(tmp_path, p.lane, 'asymmetric_synthetic_7x3_v1', fault, successful=successful)
    path = tmp_path / ('fixed_storage_binding.json' if successful else 'fixed_storage_failure.json')
    original = json.loads(path.read_bytes())
    for attack in ('retention', 'stream', 'scope') + (('restoration',) if successful else ()):
        data = json.loads(json.dumps(original))
        if attack == 'retention':
            data['retained_ids'].pop()
        elif attack == 'stream':
            data['calls'][0]['state']['target_current_stream'] += 1
        elif attack == 'scope':
            data['calls'][0]['state']['primary_contexts'][0]['active'] = True
        else:
            data['after']['target_current_stream'] += 1
        path.write_text(json.dumps(data), encoding='utf-8')
        with pytest.raises(RuntimeError):
            _validate_fixed_storage(tmp_path, p.lane, 'asymmetric_synthetic_7x3_v1', fault,
                                    successful=successful)
