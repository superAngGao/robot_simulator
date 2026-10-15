import runpy
import sys
import warp as wp

sys.path.insert(0, '/data3/ang.gao/robot_sim_c4_recording_review1')
helpers = runpy.run_path('/data3/ang.gao/robot_sim_c4_recording_review1/tests/gpu/test_warp_render_recording.py')
from optics.warp_native_render import WarpDirectLightRenderCommandFactory

wp.init()
setup = helpers['_setup'](wp)
wp.Launch.launch = helpers['_no_execution']
genuine_owner = wp.Stream(device=setup.scene.device)
borrowed = wp.Stream(device=setup.scene.device, cuda_stream=genuine_owner.cuda_stream)
factory = WarpDirectLightRenderCommandFactory.prepare(
    prepared_input=setup.prepared, readiness=setup.readiness,
    local_scene=setup.scene, device=setup.scene.device, stream=borrowed,
)
factory.validate_bound()
setup.factory = factory
bundle = helpers['_seal'](setup, helpers['_buffers'](wp, setup))
bundle.validate_bound()
print('BORROWED_STREAM_ACCEPTED', borrowed.owner, factory.stream is borrowed)
print('BORROWED_WRAPPER_FIELDS', sorted(vars(borrowed)))

setup2 = helpers['_setup'](wp)
bundle2 = helpers['_seal'](setup2, helpers['_buffers'](wp, setup2))
original_owner = setup2.stream.owner
try:
    setup2.stream.owner = False
    bundle2.validate_bound()
    print('STREAM_OWNERSHIP_DRIFT_ACCEPTED')
finally:
    setup2.stream.owner = original_owner

ambient = wp.Stream(device='cuda:0')
target_prior = wp.get_stream(setup2.scene.device)
with wp.ScopedStream(ambient):
    before = wp.get_stream('cuda:0')
    helpers['_seal'](setup2, helpers['_buffers'](wp, setup2))
    assert wp.get_stream('cuda:0') is before
print('NONDEFAULT_AMBIENT_STREAM_RESTORED')

# Keep the genuine owner alive throughout; do not intentionally destroy a
# live stream or execute a recorded render command in this reproduction.
assert genuine_owner.owner
