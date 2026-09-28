# C4c1b2 storage ABI — bounded implementation checkpoint

Initiative: q54-optical-lab-p12-4c4c1b2-storage
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-09-28
Status: in_review
Related Files: optics/warp_native_render.py, tests/gpu/test_warp_render_buffers.py
Owner Summary: Caller-owned storage contract only; C4c1b2 remains incomplete.

## Scope and authority

Implements section 5 of the approved C4c1 v1 design, as retained by v2/v3.
This is a smaller review checkpoint within b2, not a new design or activation.
The b1 snapshot recovery is recorded separately; its original primary-host
Git objects are still unavailable. Base for the storage delta is `c0046c4`.

The immutable WarpDirectLightBuffers keeps all 14 primary arrays regardless
of output profile, with zero or exactly nine overlay arrays in the existing
output_contract order. CPU layout metadata does not query devices or allocate.
Construction and full validate_bound are untimed and neither allocate arrays,
initialize their values, record commands, launch kernels nor synchronize.

Validation checks exact Warp array type, device identity, dtype, shape,
independently recomputed contiguous strides, sizes, active intervals, retained
native Warp root and view ownership chain. Mutable output intervals must be
disjoint, including from additional scene/BVH inputs supplied at preflight.
Distinct slices of one retained allocation are legal. Raw nonowning pointers
are not accepted as mutable output ownership. Static interop input provenance
remains the responsibility of the future input/factory binding.

The frozen snapshot retains the original array/root objects and physical
metadata. Revalidation detects single-point object/ptr/capacity/shape/stride/
owner drift. The baseline cannot be reset through normal assignment, deletion
or a second post-init call. This is not a security sandbox for arbitrary Python
that rewrites both bindings and every internal baseline; the approved closed
worker trust boundary still applies. It is NOT prepared-input provenance,
execution authority, a lease, GPU completion or permission to release memory.

The b1 source-ownership test now checks all seven migrated kernel entries,
addressing the owner's explicitly non-blocking review suggestion.

## Verification before independent review

Fallback: 101, isolated `/data3/ang.gao/robot_sim_c4c1b2`, robot_sim environment
(Warp 1.12.0, Torch 2.9.1+cu128). The original dirty checkout was preserved.

Initial focused run: `41 passed in 3.34s` across the buffer unit/GPU tests and
seven-entry ownership checks. Ruff check passed after formatting. Real GPU
cases cover empty/single/non-square counts, diagnostics on/off, dtype/shape,
overlap/alias, disjoint slices, root retention, CPU/second GPU, noncontiguity,
raw pointers, ordinary frozen assignment/deletion, baseline re-entry,
same-object physical drift and same-shape field reordering. Validation is also
run with allocation/copy/launch/init/device-lookup/sync entrypoints forbidden.
No recorded bundle is executed; no result or benchmark file is generated.
Full clean-tree regression and independent review are recorded separately.

## Remaining work / closed gates

b2 is NOT complete. Deterministic reset, four fixed shade ABIs, camera binding,
render evidence, immutable factory and recorded command bundle remain pending.
C4c1c must supply supervised contained execution and independent CPU numerical
oracle checks. This storage-only slice does not claim those tests passed.

C4c2-C4c4, C4d/B3, product GPU preparation, host/shipping paths, a3b, result
activation and baseline outlets remain closed; no gate or worker allowlist
was changed. No performance claim is made. No push is performed.

## Key reasoning

Physical interval validation is intentionally separate from a2 input authority:
this is a caller-supplied mutable-output graph, not proof of a frozen workload.
The active byte range (shape times scalar size) determines overlap; retained
root capacity proves ownership and bounds. Rejecting every shared root would
incorrectly prohibit legitimate disjoint arena slices. A frozen dataclass
alone is insufficient because its array members remain mutable, hence the
live untimed re-read and retained snapshot.
