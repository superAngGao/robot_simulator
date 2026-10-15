# ADMM regularization options

The default remains `per_row`. Both policies construct a diagonal R; they differ
in how the diagonal is computed. R changes the contact model. ADMM's `rho` changes
the iteration and should not change a fully converged solution of that model.

```python
from physics.solvers.admm_qp import ADMMQPSolver
from physics.cpu_engine import CpuEngine

solver = ADMMQPSolver(regularization="mujoco", impratio=1.0,
                     max_iter=2000, tolerance=1e-12, warmstart=False)
cpu = CpuEngine(merged, solver=solver, contact_condim=1)

from physics.gpu_engine import GpuEngine
gpu = GpuEngine(merged, solver="admm", regularization="mujoco",
                impratio=1.0, contact_condim=1)
```

`contact_condim=1` enables actual normal-only contacts. The default is 3. GPU
retains three storage slots per contact, but masks tangent equations, cached
variables and impulses in condim1. CPU `solver.last_R_diag` returns a copy of the
last completed solve's R (an explicit empty solve clears it). The engine adapter
bypasses the solver on no-contact frames, so check the current contacts before
using this last-solve diagnostic as a frame measurement.

`regularization` and `impratio` are read-only after construction. Select another
policy by constructing another solver/engine. Nondefault ADMM options with another
GPU solver are rejected. `impratio` must be finite and positive, and exactly 1
for `per_row`.

| Policy | R construction |
| --- | --- |
| `per_row` | `(1-d)/d * abs(diag(J H^-1 J.T))` at the current state |
| `mujoco` | `Rn=max(1e-15,(1-d)/d*(w_i+w_j))`; isotropic tangents `Rt=Rn/max(1e-15,impratio)` |

The GPU `per_row` path now uses raw W, correcting its previous use of a
PGS-adjusted diagonal. This can change existing GPU ADMM results. PGS assembly
and iteration are unchanged.

Reference weight `w` is the average translational COM mobility at
`tree.default_state()`, `trace(Jcom H0^-1 Jcom.T)/3`; world has weight zero.
MuJoCo's simple-slider exception uses `1/max(1e-15,mass)`. In our one-joint/body
representation this requires a leaf prismatic root (or child of a fixed root),
COM components below 1e-6, exactly one joint-axis component above 1e-14, and an
aligned principal inertial frame. An otherwise eligible slider with nondiagonal
body-frame inertia is rejected because the model does not retain MuJoCo's inertial
quaternion classification. Other sliders use general COM mobility.

CPU recomputes weights so model edits are reflected. GPU snapshots them with the
other static model data: rebuild the engine after mass/inertia/joint edits.
Neither policy derives or overwrites mass/inertia from material properties.

The `mujoco` policy normalizes finite solimp as MuJoCo: endpoints/midpoint in
`[1e-4,.9999]`, width >=0, power >=1; width <=1e-15 uses the endpoint average.
The width-average branch is canonicalized on the host before float32 conversion;
a just-above-threshold width retains the nonzero-width branch on GPU. Piecewise
interpolation keeps the complementary weight directly, returns exact endpoints
and stays within their range, including decreasing impedance curves.
Standard positive solref is required; timeconst is at least twice dt.
Unrepresentable or subnormal native reference-acceleration coefficients are
rejected. Validation and CPU assembly use the same coefficient evaluation.
In the GPU opt-in mode, public step/reset/partial-reset inputs must remain finite
after float32 conversion; invalid state, forces, model parameters or explicit
nonpositive/nonfinite dt are rejected before upload/contact/cache/publication
changes. Only `dt=None` selects the default. Device buffers are engine-owned;
this admission check does not add a host readback of private device state.

R has a conservative native domain check before assembly. Over all depths and
potential pairs, pair mobility is bounded below by the smallest body weight and
above by twice the largest; normal and tangent R must remain finite and at least
the native format's smallest normal value. Nonzero body weights must also remain
normal after conversion. This avoids dependence on CUDA subnormal flushing and
checks sums, impedance scaling and tangent division together. It can reject an
otherwise representable boundary value: R bounds reserve 16 native epsilons of
relative headroom from overflow/subnormal limits for arithmetic rounding, and an
extreme model whose particular current contacts would still be representable.
This restriction is numerical, not a material admissibility criterion. Very
small impratio is floored for R assembly; that does not guarantee force convergence
for an extremely ill-conditioned QP. `per_row` retains its prior impedance curve.

## Validated scope

Reference: MuJoCo **3.15.0**, Euler, approximate diagonals (`diagexact` disabled),
rigid contacts, isotropic condim1/3, positive solref. For body-weight comparisons,
retain bodies (`fusestatic=false`), use default simple optimization, matched qpos0,
and canonical diagonal inertias without an inertial rotation. MuJoCo is a test
dependency only, not a runtime dependency or the ADMM implementation used here.

The cross-engine fixtures exclude margin/gap, joint limits, damping, armature,
equalities, noslip and adhesion. Direct negative solref, anisotropic friction,
condim4/6, exact diagonals and other integrators have no compatibility claim.

R assembly is validated with analytic and actual MuJoCo data; normal-only forces
are checked against scalar and coupled active-set optima. Actual CPU sphere and
articulated multi-contact trajectories are compared at two timesteps. Real H200
tests exercise both policies, body-body contacts and inactive-slot handling.
See the [implementation evidence](../collab/admm-regularization__implementation-note__codex__v1.md).

The condim3 clamp/disk projector remains a known incorrect Euclidean cone
projection. Matching condim3 R does not establish frictional force accuracy.
GPU's default 30 iterations are not a convergence guarantee; the snapshot force
checks use 1500 iterations. Warmstart identity/rho handling, arbitrary moving
contacts, high-speed convergence and material calibration remain separate work.
In particular, matching another soft-contact model is not proof of material truth.
