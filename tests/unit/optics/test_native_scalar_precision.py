"""Exercise the shared scalar codec and CUDA's actual bound argument evidence."""
from dataclasses import replace
from types import SimpleNamespace
import math
import struct

import pytest

from optical_worker.render_parameters import BOUND_SCHEMA, decode_native_scalars, encode_native_scalars
from optics.cuda_direct_light import CudaDirectLightStageEvidence


def _payload():
    # These camera values lose information in binary32, including a physical cutoff.
    return dict(schema_version=BOUND_SCHEMA,
                **encode_native_scalars([1.00000002] * 16 + [0.10000000000000002],
                                        [0.03, 0.05, 0.07, 0., 0., -0., 1e-4]))


def test_camera_values_are_exact_and_shading_rounds_once():
    payload = _payload()
    values = decode_native_scalars(payload)
    assert values[:16] == (1.00000002,) * 16
    assert values[16] == 0.10000000000000002
    assert values[17] == struct.unpack('<f', struct.pack('<f', 0.03))[0]
    assert math.copysign(1., values[22]) == -1.


@pytest.mark.parametrize('mutation', ['old_schema', 'mixed', 'missing', 'f32_camera',
                                      'f64_shading', 'uppercase', 'whitespace', 'nan', 'infinity'])
def test_obsolete_mixed_or_noncanonical_scalar_abi_is_rejected(mutation):
    payload = _payload()
    if mutation == 'old_schema':
        payload['schema_version'] = 'q54.p12.4.bound-render-parameters.v1'
    elif mutation == 'mixed':
        payload['native_scalars_f32'] = ['00000000'] * 24
    elif mutation == 'missing':
        del payload['native_shading_scalars_f32']
    elif mutation == 'f32_camera':
        payload['native_camera_scalars_f64'][0] = struct.pack('<f', 1.).hex()
    elif mutation == 'f64_shading':
        payload['native_shading_scalars_f32'][0] = struct.pack('<d', 0.03).hex()
    elif mutation == 'uppercase':
        payload['native_camera_scalars_f64'][0] = payload['native_camera_scalars_f64'][0].upper()
    elif mutation == 'whitespace':
        payload['native_camera_scalars_f64'][0] = ' ' * 16
    else:
        payload['native_camera_scalars_f64'][0] = struct.pack('<d', float(mutation if mutation == 'nan' else 'inf')).hex()
    with pytest.raises(ValueError):
        decode_native_scalars(payload)


@pytest.mark.parametrize('camera,shade', [([math.inf]*17, [0.]*7), ([0.]*17, [1e40]*7),
                                        ([math.nan]*17, [0.]*7), ([0.]*16, [0.]*7)])
def test_invalid_values_reject_before_binding(camera, shade):
    with pytest.raises(ValueError):
        encode_native_scalars(camera, shade)


def test_cuda_evidence_detects_sub_float32_camera_changes():
    payload = dict(_payload(), width=7, height=3, sensor_role_mask=8,
                   profile='direct_light_full', shadows=True, diagnostics=False)
    record = CudaDirectLightStageEvidence(
        operation='direct_light', ray_source='camera', output_profile=payload['profile'],
        shadows=True, extension_entry='shade_direct_light_with_shadows_out', num_rays=21,
        traversal_diagnostics=False, camera_dimensions=(7, 3), sensor_role_mask=8,
        native_camera_scalars_f64=tuple(payload['native_camera_scalars_f64']),
        native_shading_scalars_f32=tuple(payload['native_shading_scalars_f32']))
    record.validate_render_parameters(payload)
    altered = list(record.native_camera_scalars_f64)
    altered[0] = struct.pack('<d', 1.).hex()
    with pytest.raises(ValueError, match='actual CUDA scalar ABI'):
        replace(record, native_camera_scalars_f64=tuple(altered)).validate_render_parameters(payload)


def test_cuda_stage_evidence_reads_bound_launch_tuples():
    from optics.cuda_direct_light import CudaDirectLightRenderStage

    values = decode_native_scalars(_payload())
    stage = SimpleNamespace(
        _sealed=False, _prepared_evidence=None, camera=object(), shade_inputs=object(),
        renderer=SimpleNamespace(context=object()),
        _validate_cuda_transaction_stage=lambda context: None,
        _raygen_args=(object(), object(), 7, 3, *values[:16]),
        first_hit_inputs=SimpleNamespace(max_distance=values[16], sensor_role_mask=8),
        shadows=True, shadow_bias=values[-1],
        _shade_args=(*([object()]*24), values[-1], *values[17:23]),
        output_profile=SimpleNamespace(value='direct_light_full'),
        _shade_entry_name='shade_direct_light_with_shadows_out',
        buffers=SimpleNamespace(origins=SimpleNamespace(shape=(21, 3)), shadow_traversal_counters=()))
    record = CudaDirectLightRenderStage._build_stage_evidence(stage)
    assert record.native_camera_scalars_f64 == tuple(_payload()['native_camera_scalars_f64'])
    assert record.native_shading_scalars_f32 == tuple(_payload()['native_shading_scalars_f32'])
