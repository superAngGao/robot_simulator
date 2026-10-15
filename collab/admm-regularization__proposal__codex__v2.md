Initiative: admm-regularization
Stage: proposal
Author: codex
Version: v2
Date: 2026-10-15
Status: in_review
Related Files: physics/solvers/admm_qp.py; physics/gpu_engine.py; collab/admm-bounded-verification__analysis__codex__v1.md
Owner Summary: 用户批准先实现可选 R 策略，以 MuJoCo 兼容路线通过解析及无摩擦交叉验证，再处理摩擦。先前投影优先的建议由此调整；本阶段不修 condim3 摩擦投影。

## Scope and reference

Base 119ec72e6a3e9b9ceea740e0c266bb9a1829c3e4 on physics/correctness-audit-20261012.
Supersedes v1; addresses independent design review C1/C2 (slider weights and
impratio lower bound). Preserve the v1 report and repros as evidence.
Implement CPU and real Warp GPU R selection. Keep the existing per-row strategy
as default; validation explicitly selects mujoco. This is a permanent bounded
strategy choice, not a temporary shadow solver or a new material/interface layer.
MuJoCo is test-only. No runtime dependence or copying of its computed R into our
production solver is permitted. No mass/inertia edits, friction-cone repair,
rho adaptation redesign, material accuracy or GPU performance claim.

Pin cross-engine reference to installed MuJoCo 3.15.0: Euler, diagexact disabled,
elliptic cone for condim3 assembly, no noslip, adhesion, margin/gap, armature,
equality constraints or joint limits in comparison fixtures. R compatibility is
for rigid contacts condim1/3 with isotropic sliding friction. Negative/direct
solref, anisotropic friction, condim4/6, discrete integrator and exact-diagonal
MuJoCo mode are outside this API's compatibility claim.
Source: https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_setconst.c
and https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_core_constraint.c.

## Public contract and ownership

- ADMMQPSolver gains keyword-only regularization='per_row'|'mujoco', impratio=1.
  Existing positional arguments keep their meaning. GpuEngine gains the same
  keyword-only options, used with solver='admm'; nondefault ADMM settings with
  another solver raise ValueError. regularization/impratio are read-only after
  construction, so switching models requires a new solver/engine and cannot
  reuse a previous strategy's warmstart state.
- CpuEngine gains keyword-only contact_condim=3, accepting 1/3; generated
  contacts use this value. GpuEngine gains contact_condim=3, accepting 1/3 for
  ADMM and 3 for other solvers. This narrowly enables actual frictionless engine
  validation. GPU retains its three-slot layout; condim1 means exactly zero
  tangential impulses and no tangent equations in the ADMM subproblem.
- Both strategies use raw A=J H^-1 J^T (GPU W). per_row uses (1-d)/d*abs(A_ii).
  Fix the known GPU contamination by PGS W_diag; PGS itself is unchanged.
  impratio must be finite positive, and must equal 1 for per_row (no ignored knob).
- New mujoco strategy rejects unsupported condim at solve entry, invalid body
  indices (only -1 denotes world), nonfinite data/parameters, and nonpositive
  solref components. Validate before contact-frame/cache mutation. Existing
  per_row condim behavior is not expanded or advertised as correct for 4/6.
- Expose CPU last_R_diag as a diagnostic copy for the last solve (empty on empty
  solve). GPU's existing admm_R_diag buffer remains the diagnostic source.

## R construction

At model reference q0=tree.default_state()[0], form H0 and each body's linear
COM Jacobian Jcom. Define w_body=trace(Jcom H0^-1 Jcom^T)/3; world weight=0.
This is the general-body branch and includes articulated and welded mobility.
MuJoCo's simple-slider branch is an explicit exception: for our one-joint-per-body
representation, a leaf PrismaticJoint body is simple2 if it is a root or its parent
is a fixed root, its COM components have absolute value <1e-6, its principal
inertial frame is aligned with its body frame, and exactly one joint axis component
has absolute value >1e-14. For this branch use 1/max(1e-15,body.mass), not trace/3.
Reference XML must retain bodies (fusestatic=false), use default body simple
optimization, and encode diagonal inertias with diaginertia and no inertial rotation.
Our Body stores an inertia matrix rather than a separate inertial quaternion:
for otherwise simple-eligible sliders with a non-diagonal inertia matrix, reject
the mujoco mode with NotImplementedError instead of guessing compiler frame
classification. Non-simple sliders (children, articulated parent, COM offset or
non-axis-aligned joint) use the general trace formula. Multiple slider joints on
one MuJoCo body have no equivalent in our one-joint-per-body representation and
are outside compatibility scope. Test simple and non-simple slider branches.
Compute in our dynamics code with Cholesky, no MuJoCo import. Static bodies have
zero Jacobian. Reference poses must match MuJoCo qpos0 in verification (a global
free-body translation does not affect the weights). Nonstandard reference joint
angles require matching model joint definitions; arbitrary per-reset q is not q0.

CPU recomputes these reference weights when the mujoco route is solved, avoiding
a hidden stale-cache policy for mutable CPU model inertias; existing tree FK state
must be restored to the current q before proceeding. GPU computes them once at
construction and uploads a 1D per-body float32 array, like its other static model
data. Rebuild the GPU engine after model edits. No new global state or resource
lifecycle is introduced.

For a rigid contact, w=w_i+w_j:

    Rn=max(1e-15, (1-d)/d*w)
    Rt1=Rt2=Rn/max(1e-15,impratio)    (isotropic condim3 only)

The cone's physical mu is unchanged. MuJoCo's internally scaled cone parameter
is not the physical friction coefficient passed to our projector.
For condim1 only Rn participates. GPU leaves inactive and condim1 tangent R slots
zero, uses rho on their otherwise zero matrix rows, and masks tangent rhs/output.
The current condim3 projector and ADMM iteration/rho logic stay unchanged.

For mujoco mode only, normalize finite solimp as MuJoCo: clamp endpoints and
midpoint to [1e-4,.9999], width>=0, power>=1. At width<=1e-15 use the average of
the two endpoints. Otherwise use the piecewise power curve without the old
1e-10 artificial denominator cutoff. Normalized d_width also feeds reference
acceleration; timeconst>=2*dt. Inputs remain standard positive solref. Keep
per_row's existing impedance behavior for compatibility. CPU/GPU share host
validation/constants; native numerical formulas get explicit conformance tests.

## Validation and gates

1. Analytic assembly: isolated mass, articulated chain/COM offsets, fixed child,
   two independent roots, static body. Compare reference COM mobility to MuJoCo
   body_invweight0; compare final R to actual efc_R, with matched contact ordering
   and frames. Include a 2 kg single z-slider and sliders with a child, offset
   COM and tilted axis. The simple slider at d=.9 must have Rn=.0555555555556;
   with depth=.01, solref=(.02,1), dt=.002 its scalar force is 67.658 N.
   Test impratio below/at/above 1e-15 against actual efc_R (e.g.1e-16,1e-15,2e-15)
   without treating such extreme scales as force-convergence fixtures.
   Include changing current pose, masses, depth, d, impratio, zero
   width, clipped endpoint values, and both strategies. Ensure per_row and mujoco
   differ in a lever-arm fixture. A caller-supplied R never counts as production
   assembly evidence. Changing rho must not change R.
2. CPU solves: condim1 analytic scalar optimum; independent nonnegative QP
   solution for coupled normals. Against actual MuJoCo freeze the same contact
   point/normal/depth, check J/A/R/reference rhs separately, then compare forces
   and generalized acceleration. At least an articulated multi-contact model
   exercises off-diagonal A. Disable warmstart and converge tightly to isolate R.
3. Real CPU engine: frictionless sphere drop plus articulated multi-contact
   scenario, using matched parameters. Record full position/velocity errors,
   contact impulses and penetration at dt and dt/2. Collision and integration
   differences must be diagnosed, never hidden by tuning R or loose height gates.
4. Real CUDA: both strategies in GpuEngine; R components match CPU/MuJoCo with
   float32 budgets. Verify distinct contact bodies, inactive slots, condim1 zero
   tangents and normal impulse against scalar/frozen coupled reference with a
   sufficient iteration cap. This is not a default-30-iteration accuracy claim.
5. Invalid strategy/impratio/dimension/nonfinite parameter failures; empty solve;
   new instances select independently; default per_row CPU regressions; existing
   relevant CPU/GPU contact tests and energy/settling oracles. Explicitly retain
   the known condim3 projection counterexample as unresolved evidence.

Numerical budgets: CPU assembly 1e-10 relative + 1e-12 times the fixture's inverse
mass scale; CPU forces 1e-7 relative + 1e-8 N for conditioned O(1) fixtures with
solver tolerance <=1e-11. These allow rounding, independent assembly and solve
error while excluding the measured O(1) defects. GPU assembly 3e-5 relative plus
2e-6 times inverse-mass scale (covers d endpoint cancellation in float32); impulse
5e-5 relative + 1e-7 N s for well-conditioned snapshots after convergence checks.
These are numerical regression budgets, not measured physical fidelity. Trajectory
budgets must be derived from matched equations/discretization or measured
convergence evidence before closing their gate; do not invent a universal mm gate.

## Delivery and review

Independent design review precedes implementation; independent code review binds
the final candidate and new execution evidence. Archive reports verbatim. Record
measured discrepancies and any still-open end-to-end claim. Passing R assembly
and no-friction comparisons does not approve sliding trajectories, real materials,
full MuJoCo compatibility, or overall rigid-body correctness. All temporary probe
scripts stay in collab evidence or outside production; targeted regressions live
in tests and the final model-option documentation describes the supported subset.
