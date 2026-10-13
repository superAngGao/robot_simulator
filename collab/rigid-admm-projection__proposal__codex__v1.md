Initiative: rigid-admm-projection
Stage: proposal
Author: codex
Version: v1
Date: 2026-10-13
Status: in_review
Related Files: physics/solvers/admm_qp.py; physics/backends/warp/admm_kernels.py; collab/admm-method-audit__analysis__codex__v2.md
Owner Summary: 先做扎实刚体。修复 CPU/GPU ADMM 对其既有 condim3 锥模型的投影一致性，使用解析最优解验收；不引入通用 interface，不改变 R 或材料模型。

## Scope and owner direction

The owner confirmed on 2026-10-13 that rigid-body correctness comes first.
General interaction/interface abstractions wait for soft/fluid interaction needs.
Material inputs must not implicitly overwrite model mass/inertia; margin is
distinct from an interaction law. No material/interface API is added here.

Base is 171a6fc2e71910f22bcf079681700dc3742bc968 on
physics/correctness-audit-20261012. Server main remains clean f771e612.
The previous CPU inertia repair and energy/settling oracles remain intact.
This increment addresses the confirmed non-Euclidean ADMM projection only.
GPU R sourcing, rho tuning, stopping/output policy, warmstart identity/scale,
terrain height, GJK, angular friction and real-material calibration remain open.

## Mathematical contract

Keep the declared frozen-step problem min .5 f^T Q f - c^T f, f in K,
Q=A+R, per-row R unchanged. For GPU use p=dt*f and rhs=dt*c; test its
actual assembled Q as GPU R currently differs from CPU. K for each contact
is n>=0, ||t||<=mu*n. With scalar rho*I the z update minimizes
.5 ||z-v||^2 over K. Normal-first disk clipping does not solve this problem.

For finite mu>=0 and v=(a,b), r=||b||, the Euclidean projection is:

- mu=0: (max(a,0),0,0).
- a>=0 and r<=mu*a: v unchanged.
- a+mu*r<=0: zero (polar region).
- otherwise n=(a+mu*r)/(1+mu^2), t=(mu*n/r)*b.

The last branch has r>0. There is no absolute tangent-norm cutoff: it would
break homogeneity for small forces. CPU uses float64 and Warp uses float32;
implement the same piecewise law in their native backends, with conformance
tests. Preserve GPU inactive-slot zeroing and active-slot indexing.

CPU condim1 retains scalar nonnegative projection. CPU solve and projection
reject condim other than 1 or 3 before mutating contact frames/cache: existing
ADMM angular Jacobian rows are zero and cannot support condim4/6 faithfully.
Validate finite nonnegative sliding mu for condim3 on the CPU entry path.
GPU already has fixed condim3; this increment assumes its existing finite
nonnegative scalar friction configuration. PGS angular capabilities are unchanged.

Exact projection couples normal and tangential response. Sliding can therefore
produce a different normal force from the old implementation. This is a repair
to the stated convex model, not proof that its associated friction law matches
physical Coulomb contact in every experiment. Do not retune R/solref or old
tolerances to hide this behavior change. Any trajectory regression must be
classified against the declared model before accepting the change.

## Validation and acceptance

1. Preserve a pre-fix failure using the real CPU solve for a sphere contact:
   A=diag(1,3.5,3.5), d=.9, c=(1,8,0), mu=1. The analytic optimum is
   (1.8,1.8,0), while the current implementation returns (.9,.9,0).
2. Direct projection cases cover inside, boundary, polar, negative normal with
   nonzero projected normal, mu=0, zero tangent, rotations and small magnitudes.
   Independently verify feasibility, residual in the polar cone and projection
   residual orthogonal to projected point (Moreau conditions); idempotence alone
   is insufficient. Test mixed condim1/3 and invalid condim/negative/nonfinite mu.
3. Real CPU solve across d=.5/.9/.99, rho=.1/1/10 and adaptive on/off must match
   analytic boundary optima with force tolerance 1e-8 N at solver tolerance
   1e-11 and sufficiently large iteration cap. Derive fixture optima from Q,
   not previous outputs; verify stationarity including cone normal and objective.
4. Real CUDA tests call the production projection and batched ADMM kernel,
   covering active/inactive slots and varied mu. Float32 projection tolerance
   is 2e-6 relative plus 2e-7 times input scale; kernel analytic solution error
   <=2e-5 relative plus 2e-6 impulse units for O(1) conditioned fixtures. These
   budgets are float32 roundoff/iteration regression budgets, not material accuracy.
   At least one GpuEngine sliding contact must match an independently assembled
   analytic optimum of its actual frozen-step Q/rhs, using enough iterations.
5. Run affected CPU solver/reference/contact tests, GPU ADMM tests, and existing
   CPU energy/settling oracles. Report old/new sliding forces and trajectory
   behavior; preserve failures instead of weakening expectations. Prove original
   code fails the targeted new optimum test. No mock replaces CUDA validation.

## Review and completion

Independent design review precedes implementation, followed by independent code
review of a frozen revision. Record commands, raw results and remaining limits.
Only this model-consistency correction can be marked complete; broad rigid-body
correctness, material calibration and all other known defects remain unresolved.
