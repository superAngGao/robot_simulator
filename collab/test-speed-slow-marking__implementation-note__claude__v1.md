Initiative: test-speed-slow-marking
Stage: implementation-note
Author: claude
Version: v1
Date: 2026-04-20
Status: implemented
Related Files:

## Open Questions Addressed

**Q40 — Test suite execution time blocking development velocity** (OPEN_QUESTIONS.md)
- Status before: P1 open, `not slow` suite taking ~15 min, blocking commit workflow
- Status after: partially resolved — commit gate now ~3 min
- Remaining: full suite still ~21 min; Q40 can be closed once timing is validated
  in practice over a few sessions

## What Changed

Added `@pytest.mark.slow` to the 4 tests that consistently measured >= 20s in
two independent timing runs of `pytest -m "not slow"`. This drops the commit
gate (`not slow`) from ~15 minutes to ~3 minutes.

Also updated CLAUDE.md to split the workflow into two gates:
- commit gate: `pytest -m "not slow"` (~3 min)
- push gate: `pytest` full suite (~21 min, before every `git push`)

## REFLECTIONS.md / PROGRESS.md Impact

No REFLECTIONS.md entry needed — this is a test infrastructure change, not a
physics or architecture decision.

No PROGRESS.md update needed — Q40 is not a phase milestone.

## Files Touched

`tests/gpu/solvers/test_solver_backends.py`
- `TestMassSplitting::test_100_steps_no_nan` — 100-step jacobi_pgs_ms loop (~20–53s)
- `TestColoredPGS::test_100_steps_no_nan` — 100-step colored_pgs loop (~28–63s)
- `TestCrossSolverAgreement::test_single_sphere_all_solvers_agree` — runs all 4 solvers × 50 steps each (~53–82s)

`tests/gpu/collision/test_b5_d4d8_mixed_ground.py`
- `TestStep5MixedShapesGround::test_simulation_stable_100_steps` — 100-step jacobi_pgs_ms on 3-robot mixed-shape fixture (~20–63s)

`CLAUDE.md`
- Quick Reference table: updated timing estimates
- "After Every Change" section: split into commit gate + push gate

`OPEN_QUESTIONS.md`
- Q41 marked RESOLVED
- Q44 marked RESOLVED

## Tests Added / Updated

No new tests. 4 existing tests gained `@pytest.mark.slow`.

Verification:
```
pytest tests/gpu/solvers/test_solver_backends.py tests/gpu/collision/test_b5_d4d8_mixed_ground.py -m "slow" --co -q
# → 4/13 tests collected (9 deselected)
```

## Known Limitations

- Timing measurements vary run-to-run (~30% variance) due to GPU scheduling.
  The 20s threshold was chosen conservatively; a few tests near the boundary
  (e.g. `test_max_qdot_bounded` at ~10–30s) were left unmarked because they
  did not consistently exceed 20s.
- `test_max_qdot_bounded` for both `jacobi_pgs_ms` and `colored_pgs` runs 50
  steps (half of the 100-step variants). These measured 10–30s and were not
  marked slow. If future hardware is slower, they may need marking too.
- The full suite timing (~21 min) is dominated by Q28 multi-body stability
  tests and GPU ADMM warmstart tests (7–37s each). These were not marked slow
  because they are below the 20s threshold in most runs.

## Commit

(pending — to be filled after `/commit`)
