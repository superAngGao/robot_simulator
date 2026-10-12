# Physics audit evidence — 2026-10-12

Base server main: `f771e612beaeea126e26926738097512cd2d809c`.
Branch: `physics/correctness-audit-20261012`.
See the adjacent implementation note and design/code reviews for scope and provenance.

- `physics_probe_baseline.jsonl`, `physics_extra_baseline.jsonl`: measured baseline energy,
  contact, geometry, analytic motion, MuJoCo comparisons. `baseline_probe.py` preserves
  the first experiment; the current reproducible modules are under
  `tests/validation/rigid_body/audit_physics.py` and `audit_solver_contracts.py`.
- `physics_baseline_tests.log`: original environment, 356 passed / 5 missing-Pinocchio skips.
- `physics_reference_baseline.log`: after isolated dependency installation, original
  25 target tests plus 5 Pinocchio tests passed.
- `physics_inertia_before.log`: all 20 original articulated repro cases failed on base.
- `physics_gpu_baseline.log`, `physics_gpu_settling_baseline.log`: real H200 runs.
- `physics_solver_contracts.log`: diagnostic counterexamples, including unresolved
  ADMM optimum, GPU regularization and elevated terrain discrepancies. Exit zero is
  diagnostic completion, NOT a claim that these contracts pass.
- `v2-oracle-tests.log`: preserved failed intermediate run: 439 passed, 1 failed.
  The sole failure is the invalid PGS drop-to-slop expectation, superseded by the
  independently reviewed discrete-crossing reference. This is not a solver regression.
- `v2-oracle-metrics.json`: raw intermediate sweep, unchanged. Its PGS
  `equilibrium_z` and `error_m` fields use that INVALID oracle. Measured trajectories
  and energy values remain useful; do not report these reference fields as final.
  The v3 independent review supplies freshly measured PGS sweeps and corrected references.

Full energy curves (every integration state), PNG and original artifacts are also
preserved outside Git at local `../physics_correctness_evidence/physics_correctness_audit_final/`
and server `/tmp/physics_correctness_audit_final/`. Energy includes t=0 and t=2 s.
The drop recorder samples input states t=0 through T-dt while executing T/dt steps;
its last 0.1 s window follows that convention. It does not return the final step output.

Reproduce from the repository root with the isolated interpreter:

```sh
PY=/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python
OPENBLAS_NUM_THREADS=1 $PY -m pytest tests/integration/contact_pipeline/test_articulated_pgs_inertia.py tests/reference/analytical/test_solver_reference.py tests/validation/rigid_body/test_physics_metrics.py tests/validation/rigid_body/test_contact_vs_mujoco.py::TestQuadrupedDropPgsSi::test_pgs_si_vs_admm_settling_height -q
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 $PY -m tests.validation.rigid_body.audit_physics --output /tmp/physics_new_audit --sweep --gpu
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 $PY -m tests.validation.rigid_body.audit_solver_contracts
```

Warp 1.12.0 and Torch 2.9.1+cu128 are available in the original environment;
MuJoCo 3.15.0 / pin 4.1.0 were added only to the separate audit venv.
The archived CUDA rendering experiment was not resumed.
