# Preserve source normals and camera query precision

Initiative: Q54 / P12.4 / C4 numerical input representation
Stage: bounded design amendment before implementation
Author: codex
Version: v2
Date: 2026-10-10
Status: in_review
Related Files: input-precision design-request/review v1; runtime-integration design-request/review v3; intersection-precision native-gap-review v1; input-precision diagnosis v1; scene_contract.py; native_input.py; render_parameters.py; camera_contract.py; device_scene.py; cuda_scene.py; warp_native_render.py; warp_execution.py; device.py; _cuda_direct_light_source.py
Owner Summary: Preserve source-derived float32 normals and binary64 camera/query rays in both native lanes. Replace the affected existing representations coherently for R0/R1. Keep the original CPU oracle, scene/workload and tolerances. This permits implementation only after independent design approval; it does not promise full optical acceptance from the two known counterexamples.

Base:13f2e885de3500a67e9649b374b78ea3c68d77b9, the frozen v1 proposal.
This version addresses design-review v1 F1 with an explicit asymmetric-camera
identity migration and F2 with the retained brute-force executor's normal ABI.
The complete v1 review is retained unchanged. Native evidence remains
bound to812ca29030a45d802addf10ca5811a4354ce0eb9, not this proposed amendment.

## 1. Problem, authority and fixed boundaries

Independent native-gap-review v1 establishes two failures: packed float32
vertices lose thin-triangle normal information before intersection, and
rounding the authoritative camera origin/direction to float32 changes the
surface at1920 pixel782417. Its all-triangle census and original-camera CPU
replays support targeted source preservation. They do not establish that all
geometry/shading must use float64 or that this proposal passes whole images.

This is the scoped design revision permitted by the owner's continuation
instruction when a frozen contract conflicts with real correctness. It changes
the numerical representation of the same original scene/query. Original
asset bytes/revision, registry construction, primitive filtering/order/IDs,
materials, lights, original camera formula and all frozen binary64 camera
values, pixel convention, profiles, shadows, frame order, workload rows,
physical limits, source-order/tie policy and CPU tolerances remain fixed.
No smoothing, edge tolerance, CPU fallback, per-pixel override, alternate
workload or relaxed boundary-pixel acceptance is introduced.

The old packed bytes remain authoritative for old artifacts. The new packed
schema is a versioned representation derived from the same source registry,
not a claim that old and new fixture bytes are identical. Both lanes consume
one new immutable CPU materialization and allocate independently. No lane may
derive normals from its already-rounded vertices or read the other lane's
allocations.

## 2. One additional source quantity: local geometric unit normals

`PackedOpticalSceneHost` v2 retains every existing array byte-for-byte and adds
`triangle_normal_local`: C-contiguous float32 shape(N,3), semantic
`scene.triangle.normal_local`. Row i belongs to precisely the same primitive
as row i of `triangle_vertices_local`. The existing original-float64
degeneracy filter and enumeration are unchanged.

In the existing sole `_pack_registry_arrays`, for each admitted triangle use
the original binary64 local vertices, subtract edges in binary64, cross and
normalize in binary64, then round each normal component once to float32.
This is a geometric face normal, not a supplied mesh shading normal. Empty
scenes produce shape(0,3). Reject nonfinite/zero results through the existing
materialization failure path; do not filter an additional primitive or fill
a default normal. The array is immutable and included in manifests/digests,
native uploads, exact owner custody, readback and typed argument validation.

R0 `_derive_fixed_fixture` and both R1 device scene-update kernels consume
this local normal. Compose the existing world-geometry rotation by the
existing transform path. Widen its nine scalar components and the three
stored local normal components to binary64, multiply, then normalize the
world vector in binary64 and round once to the existing float32 world-normal
output. Do not add translation or reconstruct a cross product from rounded
world vertices. Existing winding and hit-facing flip remain unchanged.
R1 is rigid motion under the existing transform contract; this does not admit
nonrigid/scaled/sheared transform matrices.

All existing vertex, edge, plane, bounds, centroid and topology arrays retain
their current dtype/layout/formulas. Their equality before/after must be
shown for the frozen CPU materializations; adding a normal cannot silently
alter BVH input geometry. The existing R0 world-edge degeneracy check remains
a validation of that geometry, independent of normal derivation. R1 refit
versus rebuild selection remains unchanged. These retained representations
still require the existing full CPU correctness tests; this proposal does
not assert universal geometric equivalence to binary64 source meshes.

## 3. Binary64 camera and internal intersection rays

The existing two owned ray arrays `origins` and `directions` become C-contiguous
binary64(N,3) in both lanes, including resident-ray compatibility callers.
There is one pair per existing buffer set, not an extra high-precision shadow
copy or CPU-generated ray upload for camera workloads. Source sensor rays
already stored in binary64 are uploaded without a float32 intermediate.
Camera workloads still execute the native raygen kernel in the renderer.

The first17 entries of the existing scalar order (fx,fy,cx,cy,origin3,rotation9,
max_distance) remain exact finite binary64 from the frozen render sequence.
The seven lighting scalars (ambient3,background3,shadow_bias) retain their
existing one-time nearest-even float32 conversion. Replace bound parameter
schema v1 with `q54.p12.4.bound-render-parameters.v2`, retaining its metadata
and replacing the old24-element `native_scalars_f32` with exactly:

- `native_camera_scalars_f64`:17 lowercase16-hex-character little-endian IEEE
  binary64 encodings in the existing first17 order;
- `native_shading_scalars_f32`:7 lowercase8-hex-character encodings in the
  existing final7 order.

The existing render_parameters module owns these orders/encoders. Host camera
objects and actual Warp packed launch/CUDA argument binding validate these
values and typed ray descriptors, not only record hashes. No NaN/inf or
caller-selected precision. Old v1 records fail closed at the new binding.

Raygen uses integer pixel coordinates with the unchanged convention:
`x=(u-cx)/fx`, `y=(v-cy)/fy`. It first computes world direction
`w[j]=(R[j,0]*x + R[j,1]*y) + R[j,2]`, then
`length=sqrt((w[0]*w[0]+w[1]*w[1])+w[2]*w[2])`, then `d[j]=w[j]/length`.
Every operation and stored origin/direction is binary64. Normalize in world
coordinates, not camera coordinates. Do not use reciprocal sqrt approximation
or a second float32 normalization. Native sqrt/division use round-to-nearest
double arithmetic with fast_math disabled. The CPU oracle stays its original
NumPy implementation; its values are not rewritten to match these kernels.

For a fixed, inspectable evaluation order, turn floating-point contraction
off for the existing Warp renderer module and CUDA direct-light recipe only:
Warp `fuse_fp=False`, CUDA `--fmad=false`; retain all other compiler settings.
The renderer's existing compatibility executors import the same kernel module
and therefore use the same setting. Warp setup validates the renderer-specific
option exception; other scene/BVH modules retain the current option contract.
This compiler change affects the whole existing renderer translation unit;
all shading/intersection regressions must be rerun, and no performance benefit
is assumed. Do not introduce a second raygen module or backend option.

Primary triangle/plane/box routines receive binary64 rays and physical max
distance without a downcast before predicates. Geometry scalars are widened
on load as in the reviewed intersection repair. Strict physical/shadow
cutoffs, determinant epsilons, source-order comparator, float32 returned
hit-distance ordering and conservative downward box-entry quantization remain
the reviewed policy. Do not narrow double ray operands at helper boundaries.
Shading continues using current float32 positions/normals/materials/lights
and existing shadow-ray construction; shadow rays are widened when calling
the common intersection helpers. Remaining shadow precision errors, if any,
are correctness failures, not grounds to silently broaden this proposal.

`origins`/`directions` validation readback now exposes the actual binary64
arrays with explicit f8 metadata. No diagnostic-only f32 copy may hide the
intersection inputs. Other output channels, sentinels and consumer semantics
remain unchanged. Memory increases by24*N bytes per owned ray buffer pair;
manifest byte sizes and existing allocation/OOM accounting must reflect it.
Both lanes pay this same additional ray storage and execute raygen within the
same existing timed stage. Source-normal derivation/upload stays in the
already-excluded preparation boundary for both; R1 normal transform remains
in the already-measured scene update.

Retained compatibility entries are `GpuBruteForceOpticalExecutor`,
`GpuDeviceSceneOpticalExecutor`, `GpuDeviceBvhOpticalExecutor` and
`GpuDeviceBvhDirectLightOpticalExecutor` in warp_execution, plus
`CudaDeviceBvhOpticalExecutor`, `CudaDeviceBvhDirectLightOpticalExecutor`
and their camera entry points in cuda_direct_light.
Their camera and resident ray paths all consume the new binary64 ray ABI.
Device-scene/BVH/direct-light paths consume the new source-normal-derived
scene outputs described above. They do not retain a float32-ray fallback.

For `GpuBruteForceOpticalExecutor`, preserve the existing sole
`build_host_optical_primitive_workload` and its primitive filtering/order.
It already derives `triangle_normals_world` from original world vertices in
binary64 before rounding the final unit normal to float32. Upload that
existing array and insert it immediately after `triangles` in the
`_brute_force_first_hit_kernel` ABI and actual launch tuple. Require float32
C-contiguous(N,3), row count matching triangles/source keys/IDs, and finite
nonzero normals before upload; retain its native owner in the returned
result resources alongside triangles. This compatibility path uses source
world normals directly, without an additional rigid transform. The kernel
loads the supplied normal and applies the existing hit-facing flip. Remove
the rounded-vertex cross/normalize used to return its normal, while retaining
the current geometry-based degeneracy check and its filtering/cutoff/order
behavior. Ray/predicate operands follow the new double-query rule.

This uses existing workload data, not a second normal materializer or a CPU
render fallback. Row correspondence comes from that sole builder's ordered
construction; tests trace original primitive indices through the actual
upload/launch and detect row swaps numerically. The legacy API does not gain
captured-session provenance/custody authority. Its existing stream/result
resource lifetime remains its contract; do not describe its returned arrays
as guarded C4 setup evidence or accept them as prepared C4 native inputs.

## 4. Versioning, ownership and replacement

Version the scene schema/materializer to `q54.optical-scene-contract.v2` /
`optical_registry_pack_v2`, fixed fixture to `q54.p12.4.fixed-render-fixture.v2`,
native-input schema to `q54.p12.4.native-input.v3`, Warp render command schema
and ABI to their v2 suffixes. CUDA scene recipe becomes
`robot_sim_cuda_scene_v2` (one added local-normal input in its triangle update);
direct-light recipe becomes `robot_sim_cuda_direct_light_v9` (double rays,
camera/query arguments and contraction setting). LBVH recipe/ABI is unchanged.
The new scene triangle-update argument is inserted immediately after
`triangle_vertices_local` in both lanes. Remaining argument order is retained.

The asymmetric resolved-camera contract becomes
`q54.p12.4.resolved-camera.v2`. Change its hashed `native_scalar_encoding`
from `binary64_to_binary32_nearest_even_v1` to
`binary64_native_no_rounding_v1`; preserve every other numeric and convention
field, including the exact original rational rotation values. Recompute its
new exact camera digest and retain the old v1 artifact as history only.
The current resolver/validator accepts only v2. The current asymmetric
sequence resolution identifier becomes `asymmetric_resolved_camera_v2`;
other camera resolution identifiers remain unchanged. Old v1 camera records
and mixed old-resolution/new-bound-record combinations fail before launch.

Regenerate derived scene/fixture/replay identities from the unchanged source
assets and documented new materializer, preserving all original artifacts.
Publish a before/after manifest diff proving unchanged old packed arrays,
primitive mappings, transform sample values, topology/geometry arrays, and
every camera/lighting/frame value. Only the new normal array, derived normals,
version/parent-identity fields, and explicitly specified camera encoding and
resolution identifiers may change in these CPU source artifacts.
Resolved camera sequences keep their schema and exact frames/bounds/lighting;
their embedded scene identity changes, and only the asymmetric sequence also
changes the resolution identifier specified above, with new derived digests. No
camera recomputation from altered rounded bounds is permitted.

Advance gate/result v13 to v14, row-semantics v8 to v9 and C3 one-shot-binding
v6 to v7 for these new representation/binding identities. The gate's resolved
render contract must replace its old single binary32-native encoding with
the same17-f64/7-f32 order and encoding owned by render_parameters. Preserve
activation=false and identical logical rows, thresholds and timing rules.
Freeze the resulting exact version/digest table
in the implementation candidate for independent review; no old record can be
relabeled as a new execution. This is migration of representation identity,
not publication of results or admission of shipping/S workloads. Existing
disabled S rows remain disabled and receive no new executable path.

Update the existing owners/consumers together: scene packing and manifests,
fixed materializer, Warp/CUDA scene upload/update, render storage/layout,
camera/query parameter binding, recorded graph/timeline descriptors, readback
metadata, compatibility allocation/validation, compiler recipes and captured
source catalog/function pins. All changed native arrays and scalar arguments
must be in the original typed custody/graph validation, and allocation return
must be adopted before another fallible operation. Existing failure routing,
quarantine, completion/lease/generation/reclaim and disposal rules apply.
No new lifecycle state machine, fallback allocator or runtime precision knob.

Remove superseded rounded-vertex normal derivations and float32 camera/ray
casts from these paths. Do not preserve a second implementation for old
result-capable callers. Historical evidence can retain its original bytes;
the current entry points reject incompatible old ABI/records. Compatibility
executors call the same current kernels. No additional paired comparator is
introduced; existing comparison tooling remains subject to the original
selected-backend cleanup gate.

## 5. Verification and continuation

The implementation must prove the representation first and full correctness
separately. Required evidence includes:

1. Original primitive346916 source-normal repro and all376139 Go2 source
   normals, thin/cancellation-prone triangles, empty/degenerate inputs,
   rotations and R1 transform samples. Confirm original primitive enumeration
   and unchanged geometry/bounds; wrong or replaced normal rows fail custody.
2. Authoritative782417 camera repro, both resolutions, first/last/wrap camera
   samples, resident rays, actual ray bytes/dtypes and original CPU ray rules.
   Wrong float32 descriptors, scalar rounding, stale records and aliases
   reject before native launch. Native camera scalar rounding cannot be
   covered by a CPU-only formula test.
   For the asymmetric camera, independently recompute the new resolved-camera
   and sequence digests, prove identical before/after numeric fields and test
   rejection of old/mixed camera, resolution-ID and bound-parameter records.
   For retained compatibility paths, trace actual upload and launch argument
   positions, reject wrong shapes/dtypes/counts, and use a thin triangle whose
   supplied source normal differs from its rounded-vertex reconstruction.
   Check the brute-force normal row's primitive order and hit-facing flip;
   its regression is required even though it is not a frozen C4 workload row.
3. Existing physical-limit, shadow endpoint, tie and conservative-BVH
   counterexamples remain required. Test the actual widened helper boundaries
   and previously repaired cases; rerun shade regressions affected by fusion.
4. Fresh fixed CUDA builds and actual Warp JIT at the frozen candidate, then
   supervised R0/R1 native execution with all affected source/admission,
   allocation/partial failure, warmup/readback, module option and typed graph
   checks. Preserve original artifacts and terminate/reap owned processes.
5. Full required CPU comparisons of rays, hit mask, range, position, normals,
   IDs, materials, RGB and intensity, profiles/shadows and sentinels under the
   unchanged parent matrix/tolerances. Eight pixels, cross-lane agreement or
   a normal census alone cannot approve optical correctness. Retain any new
   failures and diagnose them; this amendment grants no automatic further
   precision/schema/tolerance change.

Fresh independent design review precedes implementation, followed by fresh
independent code/native reviews of the changed scope. After correctness,
continue the already-required final graph/arena/barrier integration, lifecycle
matrices, C4c3/c4 and C4d/B3, systematic whole-C4 refactor and fresh whole review.
No performance/result/baseline/backend-selection/product gate opens here.
