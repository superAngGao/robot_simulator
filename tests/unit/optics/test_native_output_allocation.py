"""Actual allocator/lifecycle failures; runtime doubles do not prove GPU scope."""
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest

from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.native_setup import _create_upload_context, _output_array_allocation
from tests.support.guarded_peer import Peer


@pytest.fixture(params=['cuda', 'warp'])
def allocation(request):
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id=request.param, channel=peer.channel,
                                           configuration=peer.config)
    target = NS(index=peer.config.device_ordinal, ordinal=peer.config.device_ordinal)
    stream = NS(device=target)
    state = NS(device=0, stream=None, calls=[], arrays=[], fault=None)
    failure = RuntimeError('original allocation failure')

    @contextmanager
    def scope(owner, *, kind):
        previous = getattr(state, kind)
        setattr(state, kind, owner)
        try:
            yield
        finally:
            setattr(state, kind, previous)

    class Array:
        @property
        def device(self):
            if state.fault == 'metadata' and len(state.arrays) == 2:
                raise failure
            return target

    def empty(shape, *, dtype, device):
        assert state.device is target and device is target
        assert state.stream is stream
        state.calls.append((shape, dtype, device))
        if state.fault == 'allocation' and len(state.calls) == 2:
            raise failure
        value = Array()
        state.arrays.append(value)
        return value

    runtime = NS(empty=empty, float32='float32',
                 cuda=NS(device=lambda owner: scope(owner, kind='device'),
                         stream=lambda owner: scope(owner, kind='stream')),
                 ScopedDevice=lambda owner: scope(owner, kind='device'),
                 ScopedStream=lambda owner: scope(owner, kind='stream'))
    for owner in (runtime, target, stream):
        controller.adopt_host_owner(owner, role='fixture_runtime', operation_id='fixture_runtime')
    original = controller.adopt_returned_device_owner

    def adopt(owner, **kwargs):
        # Retention must already precede the fallible adoption, not be repaired
        # by the test's own references or the exception traceback.
        assert any(type(root) is list and any(value is owner for value in root)
                   for root in controller.guardian._owners)
        if state.fault == 'adoption' and len(state.arrays) == 2:
            raise failure
        return original(owner, **kwargs)

    controller.adopt_returned_device_owner = adopt
    try:
        yield NS(controller=controller, runtime=runtime, target=target, stream=stream,
                 state=state, failure=failure, lane=request.param)
    finally:
        if controller.termination_evidence is None:
            controller.terminate(fault='setup_failure', detection_phase='test_end',
                                 emergency_target_synchronize=lambda: None)
        peer.close()


@pytest.mark.parametrize('fault', [None, 'allocation', 'metadata', 'adoption'])
def test_returned_arrays_are_held_before_next_leaf_and_original_failure_survives(allocation, fault):
    p = allocation
    p.state.fault = fault
    saved = []

    def run():
        with p.controller.segment():
            context = _create_upload_context(p.controller, p.runtime, p.target, p.stream)
            with _output_array_allocation(p.runtime, p.target, lane=p.lane, context=context) as empty:
                saved.append(empty)
                for index in range(3):
                    empty('output_' + str(index), (2, 3), p.runtime.float32)

    if fault is None:
        run()
        assert len(p.state.calls) == 3
        p.controller.require_adopted(*p.state.arrays)
    else:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is p.failure
        assert len(p.state.calls) == 2
        assert p.controller.guardian.state == 'TRANSFERRED_TO_QUARANTINE'
        summary = p.controller.termination_evidence.lifecycle.to_payload()['summary']
        assert summary['emergency_sync_attempt_count'] == 0 and summary['normal_complete_count'] == 0
    assert p.state.device == 0 and p.state.stream is None
    for array in p.state.arrays:
        assert any(type(root) is list and any(value is array for value in root)
                   for root in p.controller.guardian._owners)
    with pytest.raises(ValueError, match='original scope'):
        saved[0]('late_array', (1,), p.runtime.float32)


@pytest.mark.parametrize('changed', ['runtime', 'target', 'stream', 'copied_context'])
def test_foreign_or_copied_context_rejects_before_scope_entry(allocation, changed):
    p = allocation
    with pytest.raises((ValueError, TypeError)):
        with p.controller.segment():
            context = _create_upload_context(p.controller, p.runtime, p.target, p.stream)
            runtime, target, stream = p.runtime, p.target, p.stream
            if changed == 'runtime':
                runtime = object()
            elif changed == 'target':
                target = object()
            elif changed == 'stream':
                stream = object()
            else:
                context = type(context)(context.lifecycle, context.runtime, context.target, context.stream)
            with _output_array_allocation(runtime, target, lane=p.lane, stream=stream, context=context):
                pytest.fail('invalid context was admitted')
    assert not p.state.calls and p.state.device == 0 and p.state.stream is None


@pytest.mark.parametrize('fail_at', range(1, 9))
def test_existing_world_builders_keep_every_partial_array(allocation, monkeypatch, fail_at):
    p = allocation
    if p.lane == 'warp':
        from optics import device_scene as module
        monkeypatch.setattr(module, 'wp', p.runtime)
        monkeypatch.setattr(module, '_HAS_WARP', True)
        build = module.allocate_warp_world_scene_buffers
        scene = NS(device=p.target, num_triangles=5, num_planes=2)
    else:
        from optics import cuda_scene as module
        p.runtime.device = lambda *args: p.target
        monkeypatch.setattr(module, '_torch_module', lambda: p.runtime)
        build = module.allocate_cuda_world_scene_buffers
        scene = NS(device_ordinal=p.target.index, num_triangles=5, num_planes=2)
    original = p.runtime.empty

    def empty(*args, **kwargs):
        if len(p.state.calls) + 1 == fail_at:
            raise p.failure
        return original(*args, **kwargs)

    p.runtime.empty = empty
    with pytest.raises(RuntimeError) as caught:
        with p.controller.segment():
            context = _create_upload_context(p.controller, p.runtime, p.target, p.stream)
            build(scene, _context=context)
    assert caught.value is p.failure
    assert len(p.state.calls) == fail_at - 1
    assert all(p.controller.guardian.contains_exact_owner(value) for value in p.state.arrays)
    assert p.controller.guardian.state == 'TRANSFERRED_TO_QUARANTINE'
    assert p.state.device == 0 and p.state.stream is None
