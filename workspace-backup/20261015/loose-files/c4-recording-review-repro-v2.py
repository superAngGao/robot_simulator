"""Independent v2 ownership/preflight attacks; never executes render commands."""
import gc
import inspect
import runpy
import sys
import weakref
from unittest.mock import patch

import warp as wp

ROOT = '/data3/ang.gao/robot_sim_c4_recording_review2'
sys.path.insert(0, ROOT)
helpers = runpy.run_path(ROOT + '/tests/gpu/test_warp_render_recording.py')
from optics.warp_native_render import WarpDirectLightRenderCommandFactory

print('CANDIDATE_MODULE', inspect.getfile(WarpDirectLightRenderCommandFactory))
wp.init()
setup = helpers['_setup'](wp, device='cuda:1')
buffers = helpers['_buffers'](wp, setup)
genuine_owner = wp.Stream(device=setup.scene.device)
borrowed = wp.Stream(device=setup.scene.device, cuda_stream=genuine_owner.cuda_stream)
assert borrowed.owner is False

def prepare(stream):
    return WarpDirectLightRenderCommandFactory.prepare(
        prepared_input=setup.prepared, readiness=setup.readiness,
        local_scene=setup.scene, device=setup.scene.device, stream=stream)

def rejection(call):
    try:
        call()
    except ValueError as error:
        assert 'Warp-owned stream' in str(error), str(error)
    else:
        raise AssertionError('invalid owner accepted')

with patch.object(wp.Launch, 'launch', helpers['_no_execution']):
    with patch.object(wp, 'load_module', helpers['_no_execution']), patch.object(wp, 'launch', helpers['_no_execution']):
        rejection(lambda: prepare(borrowed))
    print('BORROWED_REJECTED_BEFORE_LOAD_AND_RECORD')
    bundle = helpers['_seal'](setup, buffers)
    bundle.validate_bound()
    for value in (False, 0, 1, None, 'true'):
        try:
            setup.stream.owner = value
            with patch.object(wp, 'load_module', helpers['_no_execution']), patch.object(wp, 'launch', helpers['_no_execution']):
                rejection(lambda: prepare(setup.stream))
                rejection(setup.factory.validate_bound)
                rejection(bundle.binding.validate_bound)
                rejection(bundle.validate_bound)
                rejection(lambda: helpers['_seal'](setup, buffers))
        finally:
            setup.stream.owner = True
        bundle.validate_bound()
        print('OWNER_DRIFT_REJECTED_ALL_ENTRIES', repr(value))

    ambient = wp.Stream(device='cuda:0')
    target_ambient = wp.Stream(device='cuda:1')
    with wp.ScopedStream(target_ambient), wp.ScopedStream(ambient):
        before = wp.get_device()
        for failing in (False, True):
            if failing:
                with patch.object(wp, 'load_module', helpers['_no_execution']):
                    try:
                        prepare(setup.stream)
                    except AssertionError:
                        pass
                    else:
                        raise AssertionError('raising load not reached')
            else:
                prepare(setup.stream).validate_bound()
            assert wp.get_device() is before
            assert wp.get_stream('cuda:0') is ambient
            assert wp.get_stream('cuda:1') is target_ambient
        print('PREPARE_SUCCESS_AND_FAILURE_RESTORE_BOTH_STREAMS_AND_DEVICE')
        helpers['_seal'](setup, buffers).validate_bound()
        with patch.object(wp, 'launch', helpers['_no_execution']):
            try:
                helpers['_seal'](setup, buffers)
            except AssertionError:
                pass
            else:
                raise AssertionError('raising recording not reached')
        assert wp.get_device() is before
        assert wp.get_stream('cuda:0') is ambient
        assert wp.get_stream('cuda:1') is target_ambient
        print('RECORD_SUCCESS_AND_FAILURE_RESTORE_BOTH_STREAMS_AND_DEVICE')

    retained_stream = weakref.ref(setup.stream)
    del setup, buffers
    gc.collect()
    assert retained_stream() is bundle.binding.factory.stream
    assert retained_stream().owner is True
    bundle.validate_bound()
    print('BUNDLE_RETAINS_GENUINE_STREAM_OWNER_AFTER_CALLER_REFERENCES_DROP')

assert genuine_owner.owner is True
assert borrowed.owner is False
print('PASS_NO_RECORDED_RENDER_EXECUTION')
