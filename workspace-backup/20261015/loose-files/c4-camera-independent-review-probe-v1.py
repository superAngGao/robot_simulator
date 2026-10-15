"""Independent design probe; scratch, not production or a frozen contract."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import hashlib
import json
import math
import struct
from fractions import Fraction as F
import numpy as np
from optical_worker.cases import get_case, canonical_json_bytes
from optical_worker.protocol import decode_canonical_message, ClosedWorkerProtocolError
from benchmarks.optical.workloads import (
    _asymmetric_registry, build_canonical_packed_scene, build_canonical_fixed_fixture,
)
from optics.execution import CpuDirectLightOpticalExecutor, CpuReferenceOpticalExecutor
from optics.scene import OpticalSceneCache, OpticalFrameInputs
from physics.publish import CpuPublishedFrame
from physics.spatial import SpatialTransform
from sensing.optical import OpticalPinholeCameraSpec, build_pinhole_camera_rays

def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def axis(i, t):
    c, s = (1-t*t)/(1+t*t), 2*t/(1+t*t)
    out = [[F(int(r == q)) for q in range(3)] for r in range(3)]
    j, k = (i+1)%3, (i+2)%3
    out[j][j] = out[k][k] = c
    out[j][k], out[k][j] = -s, s
    return out

X, Y, Z = [axis(i, t) for i, t in enumerate((F(-1,20), F(-1,8), F(1,12)))]
exact = mul(mul(Z,Y),X)
rational = [[F(693,725), F(-8168,58145), F(-75024,290725)],
            [F(1512,9425), F(744813,755885), F(218584,3779425)],
            [F(16,65), F(-504,5213), F(25137,26065)]]
hexes = '''0x1.e966c0cb6338ep-1 -0x1.1fb2169a99746p-3 -0x1.08406f872195ep-2
0x1.488c96e936144p-3 0x1.f88017b67564dp-1 0x1.d9c950df370bfp-5
0x1.f81f81f81f820p-3 -0x1.8c01c4944e606p-4 0x1.edc5678284ecfp-1'''.split()
assert exact == rational
assert [float(x).hex() for row in exact for x in row] == hexes
assert mul(list(map(list, zip(*exact))),exact) == [[F(i==j) for j in range(3)] for i in range(3)]
det = sum(exact[0][i] * (exact[1][(i+1)%3]*exact[2][(i+2)%3]-exact[1][(i+2)%3]*exact[2][(i+1)%3]) for i in range(3))
assert det == 1
R = np.array([[float(v) for v in row] for row in exact])
case = get_case('asymmetric_synthetic_7x3_v1')
assert case.sha256 == '3f6ce0ac1d59d0c0d0e5ea57f497c308a9d5c4b2a5c19462c900627fd99d0181'
cam = case.camera
payload = dict(schema_version='q54.p12.4.resolved-camera.v1', case_id=case.case_id,
    case_sha256=case.sha256, rotation_id=cam['rotation'],
    convention='active_camera_to_world_rz_ry_rx_column_vector_v1',
    ray_layout='integer_pixel_center_py_times_width_plus_px_v1',
    scalar_encoding='ieee754_binary64_canonical_hex_v1',
    native_scalar_encoding='binary64_to_binary32_nearest_even_v1',
    width=case.width, height=case.height,
    intrinsics_f64=[cam[k].hex() for k in ('fx','fy','cx','cy')],
    origin_world_f64=[x.hex() for x in cam['translation_world']],
    rotation_world_from_camera_f64=hexes,
    max_distance_f64=cam['max_distance_m'].hex(), sensor_role=case.render_contract['sensor_role'])
assert decode_canonical_message(canonical_json_bytes(payload), max_bytes=65536) == payload
try:
    decode_canonical_message(b'{"width":7,"width":7}', max_bytes=65536)
except ClosedWorkerProtocolError:
    pass
else:
    raise AssertionError('duplicate JSON key accepted')
packed = build_canonical_packed_scene(case.case_id)
fixture = build_canonical_fixed_fixture(packed)
registry = _asymmetric_registry()
frame = CpuPublishedFrame(frame_id=0,sim_time=0.,step_index=0,env_mask=None,
    q=np.zeros(0),qdot=np.zeros(0),X_world=[],v_bodies=[],contact_count=0,contacts=[],telemetry=None)
snapshot = OpticalSceneCache(registry).snapshot_from_frame_inputs(
    OpticalFrameInputs.from_published_frame(frame), acceleration='cpu_bvh')

def quant(v):
    return struct.unpack('<f',struct.pack('<f',float(v)))[0]

def spec(matrix, q=False):
    conv = quant if q else float
    return OpticalPinholeCameraSpec(frame_id=0,sim_time=0.,env_idx=0,sensor_id='review',
        width=case.width,height=case.height,**{k:conv(cam[k]) for k in ('fx','fy','cx','cy')},
        X_world_camera=SpatialTransform(np.array([[conv(v) for v in row] for row in matrix]),
            np.array([conv(v) for v in cam['translation_world']])),
        max_distance=conv(cam['max_distance_m']),sensor_role=case.render_contract['sensor_role'])

def run(matrix,q=False):
    rays = build_pinhole_camera_rays(spec(matrix,q))
    result = CpuDirectLightOpticalExecutor(shadows=True,ambient_rgb=tuple(case.lighting['ambient_rgb'])).execute(snapshot,rays)
    mask = result.channel('hit_mask')
    idx = np.flatnonzero(mask).tolist()
    mats = result.channel('material_index')[mask].tolist()
    for name in ('position_world','normal_world','rgb','intensity'):
        assert np.isfinite(result.channel(name)[mask]).all()
    assert np.isposinf(result.channel('range_m')[~mask]).all()
    for name in ('position_world','normal_world'):
        assert np.isnan(result.channel(name)[~mask]).all()
    assert (result.channel('material_index')[~mask] == 0).all()
    assert (result.channel('rgb')[~mask] == 0).all()
    assert (result.channel('intensity')[~mask] == 0).all()
    ref = CpuReferenceOpticalExecutor().execute(snapshot,rays)
    np.testing.assert_array_equal(ref.channel('hit_mask'),mask)
    np.testing.assert_allclose(ref.channel('range_m'),result.channel('range_m'),rtol=1e-14)
    return rays,result,idx,mats

rays,result,idx,mats = run(R)
assert idx == [4,11,16,17] and mats == [5,12,16,18]
assert run(R,True)[2:] == (idx,mats)
directions=[]
for py in range(3):
    for px in range(7):
        d=[(px-cam['cx'])/cam['fx'],(py-cam['cy'])/cam['fy'],1.]
        w=[sum(float(exact[i][k])*d[k] for k in range(3)) for i in range(3)]
        norm=math.sqrt(sum(x*x for x in w))
        directions.append([x/norm for x in w])
ray_error=float(np.max(np.abs(rays.directions_world-np.array(directions))))
assert ray_error < 5e-16
assert np.unique(rays.directions_world,axis=0).shape[0] == 21
np.testing.assert_array_equal(rays.origins_world,np.tile(cam['translation_world'],(21,1)))
negative={}
for name,matrix in [('transpose',R.T),('identity',np.eye(3)),
                    ('reversed',np.array(mul(mul(X,Y),Z),dtype=float)),
                    ('column_major',R.flatten(order='F').reshape(3,3))]:
    wrong,_,hits,materials=run(matrix)
    assert not np.allclose(wrong.directions_world,rays.directions_world,rtol=1e-7,atol=1e-8)
    negative[name]={'hits':hits,'materials':materials}
assert negative['transpose']['hits'] == [14]
assert negative['identity']['hits'] == [7,8,10,14,15,17]
bary=[]
for i in idx:
    inst=next(v for v in snapshot.instances if v.instance_id == result.channel('instance_id')[i])
    verts=inst.geometry.vertices_local
    a,b,c=verts[np.asarray(inst.geometry.triangles)[0]]
    uv=np.linalg.lstsq(np.stack([b-a,c-a],axis=1),result.channel('position_world')[i]-a,rcond=None)[0]
    bary.append([float(1-sum(uv)),*map(float,uv)])
print(json.dumps({'python':sys.version.split()[0],'numpy':np.__version__,
    'fraction_matrix_and_hex':'PASS','exact_determinant':str(det),
    'orthogonal_error':float(np.max(np.abs(R.T@R-np.eye(3)))),
    'case_sha256':case.sha256,'scene_sha256':packed.scene_contract_sha256,
    'fixture_sha256':fixture.fixed_render_fixture_sha256,
    'resolved_payload_sha256':hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    'hits':idx,'materials':mats,'binary32_inputs_preserve_hits':True,
    'all21_scalar_ray_max_error':ray_error,'barycentric':bary,
    'minimum_barycentric':min(min(v) for v in bary),'negative_controls':negative},sort_keys=True,indent=2))
