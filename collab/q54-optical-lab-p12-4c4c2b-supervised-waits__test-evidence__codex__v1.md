# C4c2b approved CPU protocol evidence

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-09-29
Status: accepted
Related Files: code-review v5; design-request v3; implementation-note v4
Owner Summary: Independent review approves exact 54f3e1a for the CPU-only
protocol slice. This file records results and scope only; it changes no gate.

## Exact approved revision

- Full HEAD: 54f3e1a3363664414e434f091e5c9ed8bc7d7c3b.
- Whole slice base: fcb9608a916d20d1c43f64a3401b3c78d823404f.
- Reviewed tree: e82d3ce8d019c9795c1e6c0110368fb6c90d32c9.
- Fresh reviewer: /root/c4_c2b_wait_code_review_v5.
- Full report archived verbatim as __code-review__codex__v5.md, SHA-256
  4a693d34751ba0e85023b29999a5f679c4f34bf795c0bf3894fdbfe7cc4f186a.
- Verdict: Approve, no substantive findings remain for this exact slice.

The interrupted v2/v4 sessions have no final verdict. Their tool failures did
not count as approval. Completed v1/v3 reports remain unchanged history.

## Author and independent executions (not pooled)

Author local C:/Python314/python.exe 3.14 core: 120 passed in 2.16s.
Author Linux robot_sim Python 3.12.13 six-entry regression: 181 passed in
16.95s, no skips. Separate clean detached worktree:
/data3/ang.gao/robot_sim_c4_c2b_code_54f3e1a.
Log /data3/ang.gao/c4-c2b-54f3e1a-cpu.log, SHA-256
66c83930fcf499b1dd9bbae5acaca6d7b7f770794bd19d5f43b84ac02ebb5fab.
The copied local log has the same hash. Ruff check/format and git diff/show
checks passed; local and remote trees were clean after tests.

Independently executed by the reviewer: local 120 passed in 2.12s; Linux
181 passed in 17.02s, no skips; original-counterexample and adjacent ordering
probes passed on both interpreters. Full commands, test limits and source
provenance are in the archived report, not inferred from author results.

Author Linux command, from that clean detached worktree:

```text
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/benchmarks/test_optical_benchmark_contracts.py::test_inventory_and_cleanup_ledger_match_current_repository
```

Primary tileops-dev 183:40031 timed out on this turn. Owner-authorized fallback
101 was used after verifying exact revision/status in its new worktree. No
existing worktree replacement, dependency installation, push or merge occurred.

## Scope

Approved: isolated stdlib-only wait client/protocol/supervisor, fixed CPU
process tests, cleanup classification and corresponding evidence at exact HEAD.
Shared collab/README.md, REVIEW_PRINCIPLES.md, AGENTS.md and CLAUDE.md unchanged.
Existing product worker/setup/lifecycle, C0 A/K, C1 S/V/E and all gates unchanged.

This does not approve C4c2 runtime integration, native Warp timeline/arena,
product GPU preparation, C4c3/c4, C4d/B3, host/shipping, a3b, result/baseline,
backend selection or defaults. C4 is not complete. The separate complete
integration design still owes numeric camera resolution, production source
closure, setup/readiness/warmup/guardian order, exact lease/event custody,
supervised wait adoption and real-device failure validation. No existing proof
for one slice substitutes for those obligations.

This archival commit is evidence-only under the shared review exception and
does not extend the review to uninspected executable or operative changes.
