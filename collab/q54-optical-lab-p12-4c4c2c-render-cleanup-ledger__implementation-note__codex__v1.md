# C2c4 exact-path cleanup ledger follow-up

Initiative: q54-optical-lab-p12-4c4c2c-render-cleanup-ledger
Stage: implementation
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: render-binding-migration independent review v2 C1; runtime-integration design v3 section10; cleanup_ledger.json
Owner Summary: Close the non-blocking P3 inventory gap without changing runtime code, gates or the approved C2c4 behavior.

The ledger now lists the two production render-contract modules, four immutable sequence artifacts and their durable parameter tests under both selected-backend retain dispositions. A separate entry covers the two historical migration snapshots and their comparison/extraction test consumers. Both selected-backend branches remove comparison machinery only after retaining equivalent canonical-parameter, closed-gate and moved-body regressions; extraction tests must stop reading deleted snapshots without losing unrelated invariants. Historical collab evidence is retained. The deadline remains P12.4e; this is a future deletion obligation, not a claim that cleanup has happened.

A CPU test derives the four artifact paths from the frozen sequence catalog and verifies exact path sets, both terminal actions, existence and deadline. The existing repository-inventory validator also runs.

Local Python3.14 command:

```text
python -m pytest -q tests/unit/benchmarks/test_optical_render_migration.py tests/unit/benchmarks/test_optical_benchmark_contracts.py::test_inventory_and_cleanup_ledger_match_current_repository
```

Result:47 passed in0.86s; changed-test Ruff and diff check clean. No GPU test is required for this JSON/test/documentation-only delta. No product source manifest, gate, row/preimage artifact, collab protocol or private agent context changed. This follow-up requires a fresh independent review; C2c4 code approval remains bound to4b7bf4c. Complete C2/C4 and every existing activation exclusion remain unchanged.
