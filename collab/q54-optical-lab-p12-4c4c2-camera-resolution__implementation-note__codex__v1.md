# C4c2 camera contract implementation

Initiative: q54-optical-lab-p12-4c4c2-camera-resolution
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-09-29
Status: in_review
Related Files: design-request v1; review v1; optical_worker/camera_contract.py;
tests/unit/optics/test_resolved_camera_contract.py; cleanup_ledger.json
Owner Summary: Implement the independently approved numerical camera definition
as a CPU-only immutable record with exact-value validation. No runtime adoption.

Design base848b5a2abf24f15021b1dcab4c859beefce031ed was independently approved by
/root/c4_camera_resolution_design_review_v1. Its full report is archived verbatim
as __review__codex__v1.md, SHA-256
ed0015fad94a2309b004254b495ef3c107bdd4c1b0803da989f7699951d25e29.

## Actual changes

- One stdlib-only camera module, exact binary64 hex rotation table, descriptive
  frozen/slots dataclass and scalar accessor. No renderer, camera-spec builder,
  backend, live authority or GPU dependency is added.
- Non-rotation values are derived from the existing exact-hash source case.
  The generated canonical payload is checked against its independently computed
  SHA-25604ab7bc2571b8396f06d36b7baea66224868001894c4ce7a5ea12e4faecda1ef.
- Exact recursive type/value/shape validation rejects bool dimensions, lists
  replaced with tuples, noncanonical hex, unknown fields/cases and self-hashed
  alternatives. No hash passed by the caller can grant validity or execution.
- Construction, serialization and scalar access revalidate values. Ordinary
  mutation fails; object.__setattr__ drift is rejected on validated export.
  This is supported-path data integrity, not hostile Python memory isolation.
- The cleanup ledger records common selected-backend canonical regression
  ownership and removal if there is no selected runtime/regression consumer.

## Verification performed by author

Local Python3.14: camera tests69 passed initially; combined camera/direct-light/
scene-contract/native-contract-extraction selection122 passed in5.62s. Ruff
check/format passed and diff check clean. Exact Fraction axis-matrix composition,
all21 independently formulated rays, immutable/payload-copy tests, every-field
mutations, source drift, cold import attempts (with negative control), actual
canonical scene/content anchors, BVH and brute CPU executors, shadows on/off,
and binary32 input robustness are covered. All five case/content records still
match the unchanged gate.

The earlier design review independently ran53 CPU regressions and Windows/Linux
numerical probes. Those are design evidence only; clean Linux implementation
tests and a fresh code review are still required for this candidate.

## Boundary

The source case/scene/fixture/replay hashes, gate/runtime manifests, C0/C1
protocols, product APIs and shared collaboration rules are unchanged. No GPU
preparation, benchmark result, timing or baseline is produced. This completes
neither actual camera-to-native-ABI binding nor C4c2 runtime integration.

The next complete integration design must bind the new resolved-camera identity
AND actual binary32 camera fields to row, plan, stage and receipt, freeze the
other cases' camera builders, and satisfy existing readiness/guardian/timeline/
lease/failure requirements. C4c3/c4/d/B3, product GPU preparation, host/shipping,
a3b/result/baseline and backend selection remain closed. C4 remains incomplete.
