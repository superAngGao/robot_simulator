# C4c2 asymmetric camera numerical resolution

Initiative: q54-optical-lab-p12-4c4c2-camera-resolution
Stage: design-request
Author: codex
Version: v1
Date: 2026-09-29
Status: in_review
Related Files: optical_worker/cases.py; sensing/optical.py; physics/spatial.py;
C4c2a design section8; C4c2b design v3 section9; C4c design v7
Owner Summary: Resolve the existing symbolic asymmetric camera without changing
scene/replay/fixture content. This is a new explicit numerical decision, not a
claim to recover an old definition. CPU contract implementation follows design
approval; worker adoption and C4c2 runtime integration remain separate.

Base: ccfeb82673ddefdc58af1dbe4c90e809a4031c4d.
Owner authorization: after the request to propose an explicit camera numerical
definition, independently review it and freeze the common definition for both
backends, the owner replied "没错，开始做吧". This author-authored document records
that authorization; it does not claim the owner personally selected the numbers.

## 1. Facts and scope

`non_identity_compound_xyz_v1` entered benchmarks/optical/cases.py at 231a541
and moved without a resolver to optical_worker/cases.py at 953cbe3. Neither
commit supplies angles or a numerical matrix. The fixed scene's 21 triangles
are already content-hashed. A C1c sacrificial camera uses 31x13 and different
intrinsics/pose; it is not this frozen 7x3 camera and cannot silently replace it.

Deliver one stdlib-only production definition, CPU resolver/validator and tests
for this case only. Do not move materializers/importers or implement runtime
arena, GPU raygen, rendering, IPC adoption, timers, results or baseline. No
existing case ID/hash, scene/replay/fixture digest, row, threshold or gate is
rewritten. The new resolved-camera identity supplements, rather than pretends
to have been present in, the old case identity. Unknown cases/rotation IDs fail;
there is no identity/default-camera fallback.

## 2. Exact geometric convention

Retain all existing camera values: width7, height3, fx5.25, fy7.75, cx1.3,
cy0.8, translation_world=(0.31,-0.27,0.43) metres, maximum range50m, role rgb.
Retain integer pixel centres and flattening `flat = py*width + px` from the
production host ray generator; no added half-pixel offset.

Camera axes are +X right, +Y down, +Z forward as in OpticalPinholeCameraSpec.
R maps a column vector from camera to world, using right-handed active axis
rotations and the same order as SpatialTransform.from_rpy:

```text
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
tan(roll/2) = -1/20
tan(pitch/2) = -1/8
tan(yaw/2) = 1/12
```

Each angle is the principal value `2*atan(t)` in (-pi,pi), in radians.
The half-angle fractions define exact real numbers, not rounded degree inputs.
For each axis cos=(1-t*t)/(1+t*t), sin=2*t/(1+t*t). Exact rational composition
gives the following normative row-major matrix:

```text
 693/725       -8168/58145       -75024/290725
1512/9425     744813/755885      218584/3779425
  16/65        -504/5213         25137/26065
```

Production stores its nearest-even IEEE binary64 entries as these exact hex
strings (parsed with float.fromhex). It never recomputes sin/cos/atan or builds
a second rounded Euler implementation at runtime:

```text
 0x1.e966c0cb6338ep-1  -0x1.1fb2169a99746p-3  -0x1.08406f872195ep-2
 0x1.488c96e936144p-3   0x1.f88017b67564dp-1   0x1.d9c950df370bfp-5
 0x1.f81f81f81f820p-3  -0x1.8c01c4944e606p-4   0x1.edc5678284ecfp-1
```

Host reference rays use binary64 R and existing host normalization. Native
camera scalar inputs for both lanes use nearest-even binary32 conversion of
the same binary64 values, once, outside timing. Matrix storage remains
row-major; no transpose, camera-from-world inverse, or per-lane Euler resolver.
Native ray output is still float32 with existing parity tolerances; this is not
a claim that CPU float64 and GPU float32 arithmetic is bit-identical.

## 3. Single definition and canonical identity

Add optical_worker/camera_contract.py, importing stdlib and CPU-safe cases only.
An immutable ResolvedPinholeCameraContract contains tuples/primitive scalars,
not mutable NumPy/Torch/Warp objects, live allocation authority or a GPU handle.
`resolve_frozen_camera_contract(case_id)` admits only the exact known case ID
and checks that the current source case still has its frozen hash and symbolic
rotation ID. There are no caller-provided pose overrides. `from_payload` accepts
only the exact canonical known record; equal descriptive data is not presented
as resource/execution authority. `to_payload` returns independent lists/dicts.

Canonical payload has exactly these fields:

```text
schema_version = q54.p12.4.resolved-camera.v1
case_id = asymmetric_synthetic_7x3_v1
case_sha256 = 3f6ce0ac1d59d0c0d0e5ea57f497c308a9d5c4b2a5c19462c900627fd99d0181
rotation_id = non_identity_compound_xyz_v1
convention = active_camera_to_world_rz_ry_rx_column_vector_v1
ray_layout = integer_pixel_center_py_times_width_plus_px_v1
scalar_encoding = ieee754_binary64_canonical_hex_v1
native_scalar_encoding = binary64_to_binary32_nearest_even_v1
width = 7
height = 3
intrinsics_f64 = [fx, fy, cx, cy] as canonical float.hex strings
origin_world_f64 = [x, y, z] as canonical float.hex strings
rotation_world_from_camera_f64 = nine row-major canonical float.hex strings
max_distance_f64 = canonical float.hex string
sensor_role = rgb
```

Identity is SHA-256 of cases.canonical_json_bytes(payload), whose sorted-key,
ASCII, compact encoding is already the case contract encoder. The implementation
freezes the expected digest in its test fixture and publishes it in its reviewed
implementation evidence; independent recomputation must agree. Duplicate JSON
keys are rejected by the existing IPC decoder; this CPU API consumes an exact
dict, not raw JSON. Reject missing/extra fields, wrong types (including bool
dimensions), noncanonical hex, reordered matrix/intrinsics and any changed
value, even if a caller supplies a self-consistent hash. Parsed identity never
mints readiness, C3 binding, C4 authority or result capability.

The rational table is the design derivation, not a second production resolver.
Tests independently multiply Fraction axis matrices and round once to verify
the stored hex table. The production camera intrinsics/translation/range/role
are read from the exact existing case, not maintained in a duplicate table.

## 4. Measurable correctness, not a performance selection

The candidate was examined only with the existing CPU canonical scene/executor.
No lane timings, performance ranking, GPU result or baseline was inspected.
The choice provides nonzero compound rotations, off-centre unequal focal axes,
mixed hit/miss rays and different materials without altering scene geometry.

Author CPU exploratory observations (must be independently reproduced):

- hit flat indices [4,11,16,17], material ordinals [5,12,16,18];
- all three image rows represented, three distinct columns, seventeen misses;
- R.T@R maximum error about2.6e-17; determinant positive;
- quantizing camera inputs to float32 preserves those hit/material identities;
- smallest barycentric coordinate of the four hits about0.0703, so these are
  not deliberately selected edge/tie cases;
- R transpose gives only hit[14], and identity rotation gives
  hits[7,8,10,14,15,17], so neither silently passes the known hit pattern.

These are exact-fixture diagnostics, not product coverage or performance gates.
The geometry is not claimed to make all21 pixels hit unique triangles: direct
origin/direction tests cover every pixel, including misses. Lighting/shadow
correctness still requires its separate approved matrices and CPU oracle.

## 5. Implementation acceptance

1. Independent Fraction composition equals every stored binary64 matrix entry;
   proper rotation, explicit sign/order/convention, no runtime trig. Check the
   independent source case hash and all existing logical-content identities
   remain unchanged. Cold importing the contract attempts no NumPy/Torch/Warp,
   benchmark, example, tool, renderer or GPU module import.
2. Exact canonical payload roundtrip/hash; mutate each field; extra/missing
   fields; bool/int/float confusion; null/nonfinite/noncanonical hex; unknown
   case/rotation; copied payload mutation cannot alter the next resolved value.
3. Construct a production OpticalPinholeCameraSpec in CPU tests from the resolved
   values. Compare all21 production host rays with an independent scalar-loop
   formula. Verify flat row boundaries0/6/7/13/14/20, repeated origins, unique
   normalized directions. Negative controls use transpose, identity, reversed
   axis composition and column-major permutation, and must fail ray equality.
4. Existing CpuDirectLightOpticalExecutor and actual canonical registry verify
   the four hit/material identities, finite hit positions/normals/RGB/intensity,
   and the existing miss contract. Binary32 input quantization preserves the
   identities. Independently validate Fraction/CPU observations on Linux too.
5. No changes to gate/runtime manifests, C0/C1 supervision or product entries;
   existing case/content tests pass. Add this common CPU module and its tests
   to cleanup ledger as durable canonical input support while consumed; no
   renderer selector, duplicated per-lane camera math or fallback added.

CPU tests may use the old materializer outside the closed worker. They do not
certify its import closure for product use. Runtime GPU parity/admission and
full R0/R1 integration remain mandatory in the next complete design/slice.

## 6. Mandatory integration handoff, not implicit activation

Before any C4c2 runtime implementation is accepted, its complete design must
bind this resolved-camera digest AND actual frozen camera scalar/ABI values to
the existing case/row, worker-prepared plan, C3 render evidence and later receipt.
Camera resolution cannot be validated independently then replaced in a stage.
Wrong matrix with the correct digest, correct matrix on wrong row and altered
float32 ABI arguments must fail before first enqueue. Every mandatory case
must obtain its own equally explicit camera resolution; this module has no
fallback for unresolved Go2/dynamic cameras. Existing production builders can
be extracted once, not copied into independent lane implementations.

The current gate has no resolved-camera field. This slice deliberately does
not pretend adding a CPU constant makes old gate/receipt records camera-bound.
The complete integration design must version the relevant gate/receipt/schema
extension and require the new provenance before an executable worker plan.
That later reviewed change must preserve all activation=false values. No old
result gains retroactive validity and no baseline is generated here.

This authorization resolves the previously outstanding camera-policy decision
only. It does not waive complete integration design or C4c2/c3/c4/d/B3 review,
GPU setup/post-submit fault matrices, host sealing, shipping, a3b/result/
baseline denial or final repository cleanup. C4 remains incomplete.
