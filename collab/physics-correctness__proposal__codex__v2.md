Initiative: physics-correctness
Stage: proposal
Author: codex
Version: v2
Date: 2026-10-12
Status: in_review
Related Files: physics/constraint_solvers.py; physics/solvers/pgs_solver.py; physics/dynamics_utils.py
Owner Summary: Audit numerical evidence and repair the CPU PGS inertia inconsistency, without claiming general contact fidelity.

## Baseline and bounded scope

Server main f771e612beaeea126e26926738097512cd2d809c is clean; GitHub main is a20d75e.
Independent branch physics/correctness-audit-20261012, remote worktree
/data2/shared/ang.gao/robot_sim_physics_correctness; local mirror robot_physics_correctness.
Use the archived collaboration protocol from ../c4_worker_completion/AGENTS.md,
collab/README.md and collab/REVIEW_PRINCIPLES.md. This proposal does not import
the rendering implementation or reopen CUDA rendering work.

## Intent / implementation / evidence / open assumptions

| Intent | Implementation | Existing evidence | Open assumption |
|---|---|---|---|
| smooth forces -> constrained dynamics -> integration | CpuEngine -> StepPipeline, ABA without contact, CRBA cache with contact; adapters can recompute ABA/CRBA | analytical and Pinocchio tests | broad engine accuracy is not established by component tests |
| articulated contact inertia | ADMM uses J H^-1 J^T; CPU PGS uses independent-body inverse mass, then adapter applies H^-1 | loose settling tests | PGS solve and applied impulse have different mobility |
| bounded integration error | semi-implicit Euler | original energy test endpoints / abs(E0), 50% | potential datum changes denominator; full trajectory not checked |
| collision fidelity | analytical dispatch plus GJK/EPA/margin; GPU hull ground manifold caps at four points | shape tests, deep-penetration workaround | shallow box-cylinder/hull misses; arbitrary hull CPU/GPU count differs |

Measured baseline: double pendulum 2 s, dt=0.4/0.2/0.1 ms full-path max energy
error 0.273532/0.142552/0.072710 J, relative to energy above hanging equilibrium
5.180004 J: 5.2805/2.7520/1.4037%. Original endpoint ratio at 0.2 ms is 26.1136%.
Four-foot drop: ADMM z=0.4197491641 m, PGS z=0.4149798895 m (difference 4.769275 mm).
The 5 mm PGS slop explains most of that offset but does not validate inertia.
Box-hull overlap 0.01/0.1/1 mm returns no contact; box-cylinder 1 mm also misses.
356 analytical/contact/collision tests pass; 5 Pinocchio and all 25 requested
MuJoCo-file tests skip in original environment because dependencies are absent.

## CPU PGS repair

Preserve the existing PGS row construction, friction box projection, restitution,
ERP/slop/CFM, warmstart and body-impulse output. At the engine adapter boundary,
compute body spatial velocity Jacobians B_i (6 x nv, body frame) by the same
forward transform recursion as body_velocities: B_i=X_up B_parent, then add S
in this body's velocity columns. Fixed joints inherit; roots begin at zero.

Supply optional body_jacobians and mass_cholesky to PGS.solve and the PGS-SI
wrapper. They must be supplied together. Assemble each constraint row using
the existing body rows: Jg_row=Ji_row B_i+Jj_row B_j (Jj already signed).
Compute W=Jg H^-1 Jg^T via the cached Cholesky factor, then apply existing row
regularization. Direct independent-body callers keep the existing API/path.
VelocityLevelAdapter supplies this context for both PGS variants; no new
solver variant, no separate iteration implementation, no GPU algorithm change.
The output is still body-frame impulses, propagated by the existing backward
pass and applied with the SAME H^-1 used to form W.

## Reproductions and acceptance

Before editing solver behavior retain a CpuEngine regression: one free body
plus a fixed child carrying a sphere, zero gravity, initial inward normal
speed, ERP=0, CFM=0. Inelastic contact must remove normal speed and provide
impulse equal to total mass times incoming speed, independent of mass split.
Add offset/rotated and revolute cases so frame or torque mistakes cannot pass;
exercise PGS and PGS-SI, independent roots/body-body contact and condim 1/3/4/6.
Use analytic rigid composite response or independently formed geometric J and
H to check impulse balance; numerical tolerance 1e-8 for well-conditioned
float64 fixtures (solver tolerance <=1e-12), not a material-accuracy claim.

Rerun representative analytical, Pinocchio, contact pipeline and collision
tests, and actual CUDA tests on available H200. Separate missing dependencies,
toolchain failures and code failures. Record solver iterations and dt effects.

Replace the energy test with a CPU-entrypoint full trajectory invariant and
step-halving study at equal duration including the endpoint. Normalize by
E(q0,v0)-E(hanging,0), invariant to potential zero; assert finite energy and
first-order convergence (halving ratios in [1.8,2.2], a 10% asymptotic-order
regression band supported by the measured 1.92/1.96 ratios). This is a numerical
regression, not a universal energy-accuracy certificate. Check one-step
acceleration against an independent reference where available. Move optional
MuJoCo skips to only comparisons that actually use it; engine import errors
must fail collection, not masquerade as a missing optional dependency.

Replace the 10 cm cross-solver height assertion by model-specific equilibrium
checks. For PGS with straight symmetric legs and normal CFM, equilibrium depth
is slop+CFM*(M*g/4)*dt**2/ERP for 0<ERP<=1; for ERP>1 it is
slop+CFM*(M*g/4)*dt/ERP (inverse-time semantics). Require zero restitution,
stationary symmetric contacts, and inactive depenetration clamp. ERP=0 has
no restoring-height oracle. Normal velocity and force balance must settle.
For ADMM, solve the scalar symmetric equilibrium using actual per-row R and
the four coupled normal rows (do NOT assume single-contact A cancels). Check
the residual of this physical balance and compare measured height to its root.
Report MuJoCo's value separately; matching it is not real-world calibration.

## Exclusions and follow-up

GJK margin/shrink correctness, GPU manifold reduction, GPU solver convergence,
diagonal friction cone anisotropy and ADMM angular condim gaps remain open.
Record minimal shallow-overlap repros; do not increase penetration to hide them.
No full RL environment, rendering work, material-calibration claim or release.
After numerical tests propose static load / dragging-incline / footpad drop /
single-leg calibration with held-out loads and speeds; real force/displacement
data are missing. Independent design approval then code review is required.

## v1 review disposition and frozen implementation details

R1: corrected the impulse/force dimensional error above. Add a resolvable CFM
fixture (M=2 kg, CFM=.001, dt=.01/.005, ERP=.8/10, slop=.005) with analytic
equilibrium initialization. Assert velocity <=1e-10 m/s and load impulse error
<=1e-10 N s; halved-dt excess depths must scale by 4 or 2, respectively. Include
four symmetric feet with total load / 4 to check coupled rows, using iterative
tolerance <=1e-12. No change to the existing ERP interpretation.

C1: retain the importable ImplicitContactStep velocity entrypoint by routing
its constraint solve through the same VelocityLevelAdapter and DynamicsCache,
with its already-computed smooth acceleration (including external forces).
Replace its duplicate velocity/mass/impulse helpers with delegates to existing
dynamics_utils helpers if compatibility is needed. Keep its acceleration path
unchanged. Add legacy-entrypoint equivalence coverage for the impact repro.

C2: body_jacobians is a finite array of shape (num_bodies,6,nv), mass_cholesky
is a finite lower-triangular (nv,nv) factor with positive diagonal. Both omitted
selects the existing direct-body API. Partial/malformed context raises ValueError
before frame/warm-cache mutation, with no fallback to independent masses.
Only PGSContactSolver/PGSSplitImpulseSolver receive these keywords in the adapter.
Require B*v versus body_velocities, sum(B.T*impulse) versus backward propagation,
and contact row/response checks from independent geometry for rotated/fixed,
revolute, multiple-root and condim 1/3/4/6 cases, as listed in review C2.

C3: energy study duration 2 s, dt=.0004/.0002/.0001 s including t=0 and t=2,
full trajectory max abs(E-E0) and endpoint error retained in joules and relative
to E0-E_hanging. First-order ratio gate [1.8,2.2] is a fixture regression only.
Height oracle for the default symmetric drop: <=1e-6 m against independent
model equilibrium, with final 0.1 s speed <=1e-5 m/s and height range <=1e-6 m;
1 micrometre resolves the baseline PGS 20 micrometre excess while staying above
double precision noise. ADMM equilibrium uses independently formed geometric
normal Jacobians, CRBA and scalar root finding, not observed final heights;
report it as model-consistency, not an independent dynamics implementation.
Use the larger-CFM fixtures above to verify tiny-CFM effects that this drop
tolerance cannot resolve. Check default and doubled solver iterations and
halved/doubled dt where used for numerical error classification.
