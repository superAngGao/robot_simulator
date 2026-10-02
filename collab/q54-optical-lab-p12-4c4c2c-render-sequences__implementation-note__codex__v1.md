# C2c3 CPU sequence and scalar-record materialization

Initiative: q54-optical-lab-p12-4c4c2c-render-sequences
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: runtime-integration design v3 section3 at674af14; asset-custody approval536b815; optical_worker/render_parameters.py
Owner Summary: Implement the approved shared CPU camera sequences and descriptive scalar records. No native ABI adoption, v13 gate migration or runtime admission is claimed by this checkpoint.

Base:b0a6f82. Source derivation is bb85f02; the following candidate also freezes
the actual four sequence artifacts and validators. This is progress toward the
whole C2c contract, not an alternative smaller terminal scope for C4.

## Actual data flow and gap scan

The public packed-scene builder delegates to one internal materializer. That
materializer alone imports/captures/packs Go2 and returns bounds from the same
actual float64 visual import. The new render-source builder receives both;
there is no second asset import/registry pack just to obtain camera bounds.
Existing packed-scene calls keep their return type, source and content identities.

Synthetic bounds reuse the existing CPU fixed-world derivation on the canonical
packed scene. Dynamic uses exact replay frame0 and its instance transform, so
the triangle is at z=.25, not z=0; subsequent z-cycle offsets never reframe the
camera. Packed geometry is float32, lifted exactly to float64 bounds. Go2 uses
the importer's actual float64 visual bounds, not rounded packed-AABB bounds.
The existing camera preset builders and asymmetric numeric-camera contract are
the sole camera math implementations. Dynamic stores120 identically positioned
camera frames; asymmetric stores one; Go2 stores120 orbit frames.

The historical AST baseline is unchanged. Its test follows the exact new public
delegation and projects only bounds-return/IO statements out of the sole internal
materializer; mathematical statements continue to hash to the original baseline.

`ResolvedRenderSequence` owns canonical immutable bytes and validates schema,
identity, dimensions, frame order, finite canonical binary64 hex and shapes.
These public records are descriptive data, NOT unforgeable execution capability.
`materialize_render_source` reproduces values; `load_frozen_render_sequence`
loads descriptive bytes only. `resolve_render_source` first performs real source
materialization and then requires exact equality to the frozen artifact before
returning its bytes with the same packed scene. Formula/host-rounding drift is
an error, never silently replaced. Missing Go2 fails; S rejects before materialization.

Artifacts are one canonical JSON record plus one LF JSONL terminator. The
sequence hash covers precisely the canonical record, not its framing LF. Explicit
gitattributes preserve LF on Windows. Extra/missing framing, noncanonical JSON,
wrong bits, source identities or record shape reject. Four actual hashes are
in render_sequence_contract.py; no invented/placeholder digest is used.

The scalar-record builder binds sample index, case-derived sensor ID, frame/time,
profile, exact shadows/pass-mode boolean, dimensions and the supplied actual
scene role table. Its24 native scalars retain the approved order and undergo one
binary64-to-binary32 nearest-even conversion, serialized as little-endian bytes.
Validation recomputes the record from source/request; copying/re-hashing a
different record does not make it validate. These records do not yet prove the
arguments reached native kernels; exact Warp descriptor/CUDA ABI adoption remains
required in the remaining integration work. No lane may substitute this CPU
record/hash for prepared storage authority or actual runtime parameter evidence.

## Validation and environment

Primary183 timed out; authorized101 fallback is reachable. The existing dirty
common worktree is untouched. Source generation used clean detached bb85f02,
Python3.12.13 robot_sim, actual frozen Go2 assets at
/data3/ang.gao/c4_asset_frozen. Both Go2 resolutions and both synthetic workloads
were materialized with forbidden benchmark/example/tool/Torch/Warp imports
blocked. The bounded probe exited0, PID1829863/timeout1829842; both absent on
round-end check, no GPU compute entries. It performs no GPU preparation.

Local development exposed an incorrect replay attribute access; this was fixed
to the existing replay array contract before candidate tests. Initial Ruff import
order/line-length findings were corrected before final freeze. No test failure
is reclassified as an environment failure. Local focused136 tests pass at the
artifact step; final local parameter/ownership/camera suite206passed and Ruff
passed. Targeted Linux tests, independent real-asset reproduction
and the exact review verdict will be recorded separately, not presumed here.

Tests cover exact camera reuse, float32 packed-world frame0 bounds, repeated
dynamic camera, cycle indexing, metadata without warmup offset, all profiles /
shadows / diagnostic passes,24 scalar order, ties-to-even and signed zero,
mutation/re-hashing, malformed sequence/bytes, missing Go2, S rejection,
artifact corruption, formula drift and fresh-process CPU import closure.

## Remaining required work and gates

No benchmark gate/schema/row or C3 binding version changes are included here.
The approved v13/v8 migration, complete before/after semantic preimages, actual
native ABI binding and final import admission still must be implemented/reviewed;
C2c as a whole is not complete. This avoids activating a partially migrated schema.
C2d/e/f, R2/S/host/depth-two, C4d/B3, product GPU preparation/submission,
a3b, result/baseline and C4 completion remain closed. Existing parent design,
timing/performance gates and cleanup obligations are unchanged. Shared collab
protocol and private agent instructions are untouched.
