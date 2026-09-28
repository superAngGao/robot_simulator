# C4c1b2 storage — approved-revision test evidence

Initiative: q54-optical-lab-p12-4c4c1b2-storage
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-09-28
Status: accepted
Related Files: storage review v1/v2, implementation note and review follow-up
Owner Summary: Record fresh clean-revision tests and the bounded independent approval.

## Revision and review chain

- Storage candidate: `fe292cf40f0eefc2a6ae999185d360e9f8a6d8c0`.
- First fresh reviewer: `/root/c4c1b2_storage_review_v1`, Request changes.
  Two P2 findings: cached launch descriptor drift and stale input-stride flags.
- Repaired and approved candidate:
  `aa97f093dd90313f5c2daee710b99e056abf05da`.
- Second fresh reviewer: `/root/c4c1b2_storage_review_v2`, Approve.
  Both exact counterexamples now fail closed. Complete reports are preserved
  in the adjacent versioned review files.
- This evidence-only commit records the unchanged v2 report and these results;
  it does not change runtime code, tests, requirements, gates or approval scope.

The additional shared-workflow consistency audit is preserved separately in
`collab/independent-review-workflow__review__codex__v2.md`. It approved the
consistency of the shared workflow, not this implementation.

## Host and environment

Primary 183 was unreachable; use of fallback 101 is owner-authorized.
Testing used a new isolated clean checkout:
`/data3/ang.gao/robot_sim_c4c1b2_review2`.
The original dirty 101 checkout was not reset or overwritten.

Python 3.12.13, Warp 1.12.0, Torch 2.9.1+cu128.
The designated robot_sim environment has NVCC 12.8.93. Warp's own initialization
banner separately reports its toolkit 12.9 and driver 13.3. These are distinct
version facts. The shell outside Conda has a different NVCC, so all Python,
pytest and lint commands use the named environment.

Real Warp devices included H200 NVL at cuda:0 and A100-SXM4-40GB at cuda:1/2.
No environment packages were installed or changed during this checkpoint.

## Fresh full regression on approved code

```sh
cd /data3/ang.gao/robot_sim_c4c1b2_review2
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim \
  python -m pytest tests/ -m 'not slow' -q
```

Observed: **1879 passed, 39 skipped, 117 deselected, 1 warning in 372.02s**.
Process exit status: 0.

The warning is the existing pytest-10 deprecation for a class-scoped fixture
defined as an instance method in `tests/unit/solvers/test_pgs_split_impulse.py`.

Log:
`/data3/ang.gao/c4c1b2-storage-v2-full-suite.log`

Log SHA-256:
`97bdb31bcaa89b338758ac4744f3952d30a33026b77d68c7de7156ccfb6514a0`

Both local and remote checkouts were clean at the approved HEAD; remote
`git show --check HEAD` and local `git diff --check` passed.

## Focused and supplemental checks

- Revised author focused run: 55 passed in 3.10s.
- Independent v2 reviewer focused run on the clean candidate:
  55 passed in 3.08s; no skips. Exact commands and extra adversarial checks are
  in the unchanged v2 report.
- Ruff check passed; format check: four files already formatted.
- A separate subprocess with `sys.modules['warp'] = None` successfully
  imported the CPU layout helper and returned all 14 zero-ray primary specs;
  runtime buffer construction raised the intended Warp-required error.
- Recomparison with the pre-extraction `2b41068` implementation confirmed
  the complete normalized Warp kernel/device-function block remains identical:
  53,156 characters. This verifies unchanged code content, not the unavailable
  original `00f1790/f771e61` commit identities.

## Exact approval boundary and next work

Only the storage contract and source-ownership test expansion are approved.
b2 remains incomplete: deterministic resets, four fixed shade ABIs, camera
binding, independent render evidence, factory and recorded bundle are pending.
C4c1c's supervised execution, fault matrix and independent CPU oracle are not
certified by this storage-only validation.

Product GPU preparation, C4c2-C4c4, C4d/B3, a3b, result/baseline outlets,
host payload and shipping remain closed. No benchmark, performance or backend
selection result was created, and no push was performed.

The recovery branch has new identities. The approved primary-host b1 objects
remain unavailable and must be compared before future history integration.
Do not treat this evidence commit as permission to merge duplicate recovery
changes or rewrite primary history.
