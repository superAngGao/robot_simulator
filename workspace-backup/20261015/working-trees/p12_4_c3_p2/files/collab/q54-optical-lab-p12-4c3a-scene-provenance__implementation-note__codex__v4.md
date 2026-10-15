# Q54 P12.4 C3a provenance implementation note v4

Status: implementation update for final review. This note supplements v3 and
addresses the Warp R2 acceleration-class defect. No benchmark result or
baseline is produced.

## Acceleration class is now binding identity

- C3 binding schema v5 adds the mandatory `acceleration_class` field.
- Dynamic dataflow contract v2 includes `acceleration_class` in its canonical
  identity hash.
- Validation resolves the expected acceleration ID through
  `EXACT_LANE_BINDINGS[lane]["acceleration_by_class"][acceleration_class]`.
- Scene-update and acceleration operation IDs are then resolved through the
  matching `EXACT_ACCELERATION_CONTRACTS[lane][acceleration_id]` entry.
- There is no longer an R1/R2-wide `production_dynamic` assumption.
- Runner pre-enqueue validation compares the binding's acceleration class with
  the frozen row before any timed submission can occur.
- Result-schema cross-validation also binds the nested C3 evidence to the row
  class, renderer, resolved acceleration ID, output profile, and shadow mode.

The intended Warp identities are therefore distinct:

- R1 `production_dynamic` uses
  `cpu_median_leaf4_topology_warp_refit_v1` and the Warp refit operation.
- R2 `production_static` uses
  `cuda_morton_karras_lbvh_rebuild_v1` and the CUDA LBVH rebuild operation,
  while retaining the Warp scene-update operation and Warp renderer.

## Matrix and negative coverage

The test suite now constructs every frozen R0/R1/R2 mandatory row for both
Warp and CUDA. For each row it verifies the bound class, resolved acceleration
ID, scene-update operation, and acceleration operation. A dedicated Warp R2
test also proves that changing its class to `production_dynamic` while keeping
the static CUDA-LBVH identity is rejected.

## Verification

- Focused provenance/core/a3a/native tests: 85 passed.
- Full benchmark, optics, and optical GPU suite: 444 passed.
- Ruff targeted checks: passed.
- Go2 asset preflight remains unchanged and fail-closed; no missing asset was
  replaced with a synthetic product result.
