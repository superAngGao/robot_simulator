"""Review-only adversarial and independent numerical probe; not candidate code."""
import copy
import hashlib
import json
import sys
from fractions import Fraction as F

sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from optical_worker.camera_contract import ResolvedPinholeCameraContract as Record
from optical_worker.camera_contract import resolve_frozen_camera_contract

CASE = "asymmetric_synthetic_7x3_v1"
record = resolve_frozen_camera_contract(CASE)
payload = record.to_payload()
expected_r = (F(693,725), F(-8168,58145), F(-75024,290725),
              F(1512,9425), F(744813,755885), F(218584,3779425),
              F(16,65), F(-504,5213), F(25137,26065))
assert tuple(float(v).hex() for v in expected_r) == record.rotation_world_from_camera_f64
expected_payload = {
    "schema_version": "q54.p12.4.resolved-camera.v1",
    "case_id": CASE,
    "case_sha256": "3f6ce0ac1d59d0c0d0e5ea57f497c308a9d5c4b2a5c19462c900627fd99d0181",
    "rotation_id": "non_identity_compound_xyz_v1",
    "convention": "active_camera_to_world_rz_ry_rx_column_vector_v1",
    "ray_layout": "integer_pixel_center_py_times_width_plus_px_v1",
    "scalar_encoding": "ieee754_binary64_canonical_hex_v1",
    "native_scalar_encoding": "binary64_to_binary32_nearest_even_v1",
    "width": 7, "height": 3,
    "intrinsics_f64": [v.hex() for v in (5.25,7.75,1.3,0.8)],
    "origin_world_f64": [v.hex() for v in (0.31,-0.27,0.43)],
    "rotation_world_from_camera_f64": [float(v).hex() for v in expected_r],
    "max_distance_f64": 50.0.hex(), "sensor_role": "rgb",
}
assert payload == expected_payload
digest = hashlib.sha256(json.dumps(expected_payload, sort_keys=True, ensure_ascii=True,
                                  separators=(",", ":"), allow_nan=False).encode()).hexdigest()
assert digest == record.sha256 == "04ab7bc2571b8396f06d36b7baea66224868001894c4ce7a5ea12e4faecda1ef"

class StrSubclass(str): pass
class IntSubclass(int): pass
class TupleSubclass(tuple): pass
class ListSubclass(list): pass
class ObjectBomb:
    def __eq__(self, other):
        raise AssertionError("untrusted equality invoked")

rejections = 0
def reject(candidate):
    global rejections
    try:
        Record.from_payload(candidate)
    except ValueError:
        rejections += 1
    else:
        raise AssertionError("mutant accepted")

for field, value in payload.items():
    values = [None, True, False, 0, 1, float("nan"), float("inf"), ObjectBomb(), {}, ()]
    if type(value) is str:
        values.extend((StrSubclass(value), value.upper(), ""))
    elif type(value) is int:
        values.extend((IntSubclass(value), float(value), -1))
    elif type(value) is list:
        values.extend((ListSubclass(value), tuple(value), []))
    for bad in values:
        if type(bad) is type(value) and bad == value:
            continue
        reject({**payload, field: bad})
    if type(value) is list:
        for index, scalar in enumerate(value):
            for bad in (None, True, float.fromhex(scalar), StrSubclass(scalar), scalar.upper(),
                        "nan", "inf", "-inf", "0x0.0p+0", ObjectBomb()):
                candidate = copy.deepcopy(payload)
                candidate[field][index] = bad
                reject(candidate)
    candidate = dict(payload)
    del candidate[field]
    candidate[StrSubclass(field)] = value
    reject(candidate)

for field in ("intrinsics_f64", "origin_world_f64", "rotation_world_from_camera_f64"):
    candidate = record.to_payload()
    candidate[field].clear()
    assert record.to_payload() == payload
    data = {name: getattr(record, name) for name in record.__dataclass_fields__}
    data[field] = TupleSubclass(data[field])
    try:
        Record(**data)
    except ValueError:
        pass
    else:
        raise AssertionError("tuple subclass accepted")

class Derived(Record): pass
try:
    Derived.from_payload(payload)
except TypeError:
    pass
else:
    raise AssertionError("subclass accepted")

for field in record.__dataclass_fields__:
    damaged = resolve_frozen_camera_contract(CASE)
    object.__setattr__(damaged, field, ObjectBomb())
    for operation in (damaged.to_payload, damaged.scalar_values, lambda: damaged.sha256):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("damaged object exported")

assert resolve_frozen_camera_contract(CASE).to_payload() == expected_payload
print(json.dumps({"python":sys.version.split()[0],"digest":digest,
                  "payload_mutations_rejected":rejections,"result":"pass"}))
