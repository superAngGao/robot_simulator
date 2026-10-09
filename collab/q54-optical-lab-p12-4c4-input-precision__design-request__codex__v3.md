# Preserve source geometry and use the original CPU intersection policy

Initiative: Q54 / P12.4 / C4 numerical input and intersection semantics
Stage: bounded design amendment
Author: codex
Version: v3
Date: 2026-10-10
Status: in_review
Related Files: full-cpu-frame diagnosis/gap-review v1; input-precision design-request/review v2; intersection-precision native-gap-review v1; runtime-integration design-request/review v3
Owner Summary: Replace the incomplete ray/normal-only proposal with one coherent query representation: preserve original geometry, use original CPU rejection and nearest-hit rules, and round final output channels only. Remove the proposed extra local-normal array. Keep existing shading/output formats, independent native allocation, timing boundaries and lifecycle. No numerical, speed or whole-C4 approval is implied.

This proposal supersedes the implementation scope of input-precision v2 at
db062cd603464b9b3088739a2aca68acb2a44769. Its approval and unfinished isolated
implementation remain historical evidence. No v3 implementation is authorized
until fresh independent design approval. Original native evidence remains
812ca29030a45d802addf10ca5811a4354ce0eb9.

## 1. Evidence and fixed product requirements

The original full CPU frames show shared native ID, range, normal and RGB
failures at both resolutions. Source packing omits20639 CPU-admitted faces.
The native1e-8 determinant cutoff rejects a retained nearest face at1920
pixel970674, whereas the original CPU1e-12 rule accepts it. Nearby competing
surfaces expose native1e-5 versus CPU1e-9 tie rules. Rounding internal distances
before comparison loses differences larger than1e-9 even if the epsilon alone
is corrected. Float32 local/world geometry can also move CPU-admitted faces
across the1e-12 area boundary; these are not claims of exact zero-area collapse.

The CPU's world-space filter cannot be reproduced by a once-only local filter:
the Go2 source has398432 faces, CPU world-space packing admits396778, and local
versus world binary64 filtering at1e-12 already differs at12 source faces.
Retain source faces and apply the world query rule where world geometry exists.

Original assets, registry, instance/material IDs, roles, cameras, light values,
profiles, shadows, logical frames/rows, physical limits, CPU implementation and
all frozen tolerances remain unchanged. This explicitly replaces the native
filter/determinant/direction/tie policy with the existing CPU policy; it does
not relax CPU acceptance. No edge smoothing, per-pixel exception, CPU fallback,
new lifecycle machine or runtime precision selection is introduced.

## 2. One source geometry representation

`PackedOpticalSceneHost` stores the original finite binary64 values in these
existing arrays, preserving names and shapes:

- `triangle_vertices_local`:float64(N,9).
- `instance_X_body_geometry_R`:float64(I,9),
  `instance_X_body_geometry_r`:float64(I,3).
- `plane_normal_local`, `plane_point_local`:float64(P,3).

Pack every source triangle in registry-instance/source-face order, including
degenerate triangles. Remove the native local-area filter. Source order keys
remain `(instance_index << 32) | original_face_index`; original instance,
material, geometry and within-instance identifiers do not change. Packed
global ordinals are densely regenerated for the retained source rows and
are not confused with stable source keys. Empty arrays keep the exact shapes.
Reject nonfinite source values through the existing materialization path.

Remove the v2 proposed `triangle_normal_local` array and its derivation,
upload, ABI argument and manifest semantic. There is no parallel normal
source: normals derive from preserved world edges below. Existing material,
light and integer arrays retain their current types/values.

R0 fixed transforms are preserved as binary64 without an intermediate
float32 cast. The frozen static workloads have no body-transform rows; their
empty transform arrays change dtype/identity, not any numerical sample.
R1 canonical replay/physics transform inputs retain their existing source
float32 values, ownership and layouts. Load those exact values into binary64
before composition; never regenerate frozen replay samples from different
formulas or pretend to recover precision absent in a float32 producer.
The transform-replay schema/content stays unchanged. This does not admit
the still-closed real physics-to-image product path.

R0 and R1 compose body/world and geometry transforms in binary64, transform
each original local vertex into binary64 world coordinates, then subtract
world vertices to obtain binary64 edges. Do not transform previously rounded
edges or vertices. Scene update kernels use explicit ordinary double arithmetic,
without fast math or floating-point contraction, in both lanes. Fix ordered
three-term dot products as `(a0*b0 + a1*b1) + a2*b2`, then translation; use
the same composition order in the host fixed materializer. The CPU oracle
remains its existing implementation, and original CPU comparisons are required.

World query arrays `triangle_v0_world`, `triangle_e1_world`,
`triangle_e2_world`, `plane_normal_world`, `plane_point_world` become float64.
Normalize geometric normals in binary64 from those same world edges.
`triangle_normal_world` remains float32: round the unit normal once for the
existing shading/output path. For world cross-product norm<=1e-12, produce
zero triangle normal and valid geometric bounds; do not throw, remove/reorder
rows or supply a default visible normal. Plane normalization retains the
existing invalid-plane failure behavior with the CPU1e-12 threshold.

The source-preserving world geometry and CPU world-area test are authoritative
for hits; zero shading normal is not an alternative geometry-validity oracle.
Every triangle intersection helper computes the world-edge cross-product norm
and rejects norm<=1e-12 before the determinant test, including compatibility
and shadow callers. Thus retained degenerate faces cannot become hits.

## 3. Conservative existing acceleration structures

Retain float32 primitive/BVH bounds and the existing BVH builders/layouts.
Compute each world-coordinate min/max from binary64 world vertices. Convert
each lower component to float32 and, only if conversion rounded upward, take
the next representable float32 toward negative infinity. Convert upper bounds
symmetrically outward. Finite bounds must remain representable under existing
input/resource limits; no infinity padding or arbitrary geometric epsilon.
Use one CPU helper and equivalent fixed native scalar helpers for this rule.

Degenerate triangles retain their point/segment/triangle bounds. No invalid
box or additional active-mask array is required. Internal BVH bounds union
these already-conservative float32 leaf bounds. Empty scene behavior remains
the existing empty topology. Rebuild the common R0 topology from all retained
source faces, same frozen builder/leaf-size/split policy in both lanes.
R1 retains Warp refit and CUDA LBVH rebuild, with unchanged builder algorithm,
index/role mapping and timing boundaries. This is a changed source-derived
fixture identity; old/new topology equality is not claimed.

Ray/box subtraction, division, physical limits, entry/exit values and traversal
stack distances use binary64. Bounds are widened exactly on load. Remove the
old downward float32 entry-quantization repair because entries no longer narrow.
Keep conservative nearest-hit pruning: discard a node only when its entry is
greater than the current best distance and `entry-best > 1e-9`, evaluated in
binary64. Root/child tests retain the physical ray limit; no tie window extends
the physical segment. Triangle/plane acceptance still checks physical limits
before any final output rounding. No faster builder may masquerade as renderer
speedup, and R0 uses the same new immutable host fixture independently uploaded.

## 4. Original CPU numerical policy and internal distances

One CPU-safe module `optics/query_numerics.py` owns the fixed double constants
`BUILD_EPS`, `DIR_EPS` and `TIE_EPS`:
world area/determinant epsilon1e-12, direction/plane epsilon1e-12, and
nearest-hit epsilon1e-9. Native scene/render modules and CUDA source generation
consume that module; do not maintain separate mutable policy copies. Tests
compare these values/semantics with the unchanged original CPU executor.
The CPU oracle does not import rewritten native algorithms or new policy
implementation. Other shading constants remain unchanged.

Triangle/plane helpers return binary64 hit distance for candidate comparison.
`best_t`, temporary candidate distances, physical limits and pruning operands
stay binary64 for the complete primary traversal. Apply the original rule:
`t < best_t - 1e-9`, otherwise `abs(t-best_t) <= 1e-9` and smaller source key.
Never round either distance to float32 before that rule. Preserve exact source
key ordering and strict barycentric/physical acceptance; no tolerance is added
to a triangle edge or shadow endpoint.

Only after selecting the winning hit, round the existing range/position/normal
output channels to float32. Position is formed from the binary64 ray and
unrounded winning distance before this final conversion. Keep existing hit,
miss, ID/material and diagnostic output semantics and dtypes. Materials,
light shading, RGB/intensity and the existing float32 shadow-origin construction
remain as before; widen shadow rays at the common query helper boundary.
All shadow predicates use the new geometry and CPU rejection policy.
Full native CPU comparisons must still establish adequacy of that retained
shading boundary; a failure cannot silently authorize extra buffers/precision.

## 5. Camera rays and compatibility paths

Retain v2's reviewed camera design: one pair of binary64 owned origins/directions
arrays,17 exact binary64 camera/query scalars and7 float32 shading scalars,
native timed world-rotation-then-normalization, and renderer-only contraction
disabled. Retain its explicit operation order, typed ctypes/CUDA scalar binding,
asymmetric-camera v2 encoding/identifier migration, source numerical values,
and actual f8 readback. Resident source rays avoid a float32 intermediate.
There are no additional shadow copies or host-generated camera rays.

All six retained compatibility executors migrate together with the canonical
paths and call the same current kernels: GpuBruteForceOpticalExecutor,
GpuDeviceSceneOpticalExecutor, GpuDeviceBvhOpticalExecutor,
GpuDeviceBvhDirectLightOpticalExecutor, CudaDeviceBvhOpticalExecutor and
CudaDeviceBvhDirectLightOpticalExecutor. Their geometry uploads preserve
source binary64 query values; f32-only input producers are widened on load
without a claim to recover absent precision. Host triangle builders must not
drop source faces using a local or native1e-8 filter before the world query.

The brute-force kernel uses binary64 world vertices/edges for both intersection
and geometric normal derivation. It needs no v2 extra normal argument at index9.
Remove that proposed ABI expansion; preserve the existing argument order with
the changed geometry/ray dtypes and double internal distance behavior. Existing
compatibility lifetime contracts remain unchanged and confer no captured C4
storage/custody authority. No alternate old result-capable precision lane.

## 6. Version migration and integration

Use new identities distinct from both the old runtime and unimplemented v2
proposal:

- `q54.optical-scene-contract.v3` / `optical_registry_pack_v3`.
- `q54.p12.4.fixed-render-fixture.v3`.
- `q54.p12.4.native-input.v4`.
- `q54.p12.4.warp-render-command.v3` /
  `warp_bvh_direct_light_recorded_v3`.
- `robot_sim_cuda_scene_v3` and `robot_sim_cuda_direct_light_v10`.

The CUDA LBVH recipe/ABI stays unchanged because
its float32 bounds and integer topology inputs do not change dtype/layout.
Scene-update contraction is disabled in the new scene module/recipe, as well
as the renderer; builder modules retain their existing options.

Use bound-render-parameters v2 and asymmetric resolved-camera v2 exactly as
specified in the reviewed v2 camera sections. Advance gate/result to v15,
row semantics v10 and C3 one-shot binding v8 for the combined representation
and native policy change. Retain activation=false, logical rows, product
thresholds and timing boundaries. Derived scene, fixture, camera-sequence,
replay-parent and topology identities must be regenerated and listed with
exact hashes for code review. Preserve old artifacts, never relabel execution.

The manifest diff must prove original source values and stable source keys
for all398432 Go2 faces, exact retained camera/light/replay numerical values,
unchanged asset/source contracts and unchanged output/delivery formats.
It must explicitly enumerate newly retained faces, changed geometry dtypes,
outward bounds, changed topology and removed proposed local-normal array.
It must not assert unchanged old geometry arrays or primitive counts.

Update existing owners/consumers together: packing, fixed materialization,
scene uploads/updates, host/device BVH input validation, ray storage, typed
launch/graph descriptors, readback, manifests, recipes and captured source/
function pins. Account for actual new geometry/ray bytes and retained faces
in both lanes; no new fixed memory-budget threshold. Existing registration,
partial adoption, failure routing, supervised waits, quarantine, generations,
leases and disposal remain the same contract. No new control-plane layer.

## 7. Verification and continuation

Required checks, using original assets/oracle/tolerances:

1. Full source face/key enumeration, original binary64 vertex/transform values,
   thin and degenerate triangles, world/local threshold crossings, nonfinite
   rejection, empty scene, rotations and R1 updates. Verify outward bounds
   contain all binary64 vertices; degenerate faces never hit or destabilize BVH.
2. All original captured pixel repros,970674 and omitted-nearest-face failures,
   original nearest-key conflicts and a1 versus1.00000002 distance pair whose
   float32 values coincide. Exercise both visit orders and separate leaves.
   Preserve strict physical cutoff, shadow endpoint and conservative traversal
   counterexamples, updating old expected tie behavior to the original CPU
   policy explicitly rather than retaining the obsolete native1e-5 policy.
3. V2's actual native camera/ray/scalar/readback and asymmetric identity tests,
   including wrong dtype, stale/mixed ABI/identity and alias rejection. Verify
   all compatibility upload/argument paths and removal of the extra normal ABI.
4. Fresh fixed CUDA compilation, actual Warp JIT and supervised native R0/R1
   runs with correct dtype custody, source pins and scene/renderer options.
   Exercise affected partial failures and existing resource ownership checks.
5. Complete required original CPU comparisons of every output channel, profiles,
   shadows, frames, sentinels and diagnostics. Include native intensity readback;
   the old nine-channel evidence is insufficient. Cross-lane agreement and CPU
   arithmetic probes cannot substitute for this native correctness matrix.

Independent design approval precedes implementation, then fresh independent
code/native review. Continue final graph/arena/barrier integration, remaining
lifecycle matrices, C4c3/c4, C4d/B3, systematic whole-C4 refactor and whole review.
No speed/result/baseline/product/backend-selection gate opens with this proposal.

## 8. Supplemental author evidence, not implementation acceptance

The independent full-CPU gap review separately reproduces the source/predicate/
ordering findings and identifies the world/local filter distinction. Preserve
that complete report verbatim. Author-only probes below are outside tracked
source, use unchanged812ca29, and are available in local `c4_assets/` and remote
`/data2/shared/ang.gao/`. They do not modify the CPU oracle or execute a GPU.

| Script / result | SHA256 |
| --- | --- |
|`probe_native_policy_predicates_v1.py`|`6d03556211d4a2059e0d1c3e33128282da496c0c05d744873d3836a93fb0035a`|
|`c4-native-policy-predicates-v1.json`|`a2357b98e8eba5f67c713f444829f2fc22576c99aaf41f25dba8b0dd51e0a28c`|
|`probe_minimum_geometry_precision_v1.py`|`d4b119b7435e5a34e56b15fd00e48803c0e6794e5ec600ed3ee3b87a91b150bd`|
|`c4-minimum-geometry-precision-v1.json`|`0d995621d4efb54158bbfa14a774456f7951a8af3eb05f58b0fd5678abf2f5c2`|
|`probe_primary_output_shadow_rounding_v1.py`|`a185dd1ba564fa07b4c53a912dcf44fe776e0b3b604af06e4e7d6a25622a58e3`|
|`c4-primary-output-shadow-rounding-v1.json`|`0e48daa3a3ff167576e563250633344bd7a6f334da9fa1ad4f75f5ec6acadb41`|

All three processes completed normally. The first evaluates the actual Warp
helper Python body on saved rays/fixture data. The second saves original and
rounded face coordinates plus CPU rays and a concrete float32 hit-ordering
counterexample. The third exactly reproduces original CPU shading at all507
1920 and144960 Warp RGB-error pixels, then rounds the original first-hit
position/normal selectively before original CPU shading. It observes zero
new RGB tolerance failures from those roundings in that selected set. This
supports keeping the existing output boundary for now, not acceptance of all
native shadow arithmetic or untested pixels. Full native verification in
section7 remains mandatory; no additional precision expansion is automatic.
