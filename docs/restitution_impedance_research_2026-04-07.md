# Restitution API Survey: Engines, Papers, and Impedance

> Research conducted 2026-04-07 (session 21) for Q34 (OPEN_QUESTIONS.md). Output
> from a research agent surveying 8 physics engines + 7 academic papers, with
> a specific focus on whether anyone has bridged spring-damper contact damping
> `b` to physical mechanical impedance `Z = ρcA`.
>
> Conclusion: the math is well-known in continuum mechanics / acoustics / DEM
> literature, but no surveyed engine documents the bridge as a user-facing API
> design principle. Drake hydroelastic comes closest but leaves the dissipation
> parameter `d` phenomenological.

## Engine survey

**MuJoCo (incl. MJX, MuJoCo Warp, Newton).** Spring-damper compliance contact,
no direct `restitution` parameter. Users tune `solref = (timeconst, dampratio)`
and `solimp` instead, with the documented mapping `b = 2/(d_width·timeconst)`,
`k = d(r)/(d_width²·timeconst²·dampratio²)`. The docs explicitly note: "If the
reference acceleration is given using the positive number format and the
impedance is constant, then the penetration depth at rest is..." and that the
reparameterization is in terms of "the time constant and damping ratio of a
mass-spring-damper system" (https://mujoco.readthedocs.io/en/stable/modeling.html).
MuJoCo also offers a `direct` solref form (negative numbers) where users pass
`(stiffness, damping)` directly, and the docs say this "allows direct control
over restitution in particular." There is still no `restitution=e` setter — the
user is expected to tune `dampratio < 1` for bouncy contact. Combine rule:
priority-weighted average via `solmix`, except direct-format `solref` falls
back to element-wise minimum.

**NVIDIA Newton (Linux Foundation, GPU/Warp, successor to warp.sim).** Uses
MuJoCo Warp as primary backend (https://github.com/newton-physics/newton); also
exposes Featherstone, XPBD, VBD, SemiImplicit, and a hydroelastic contact
library. The public docs (https://newton-physics.github.io/newton/) do not
document a top-level `restitution` parameter — contact tuning inherits MuJoCo
Warp's solref/solimp under the MuJoCo solver, and per-particle `ke`/`kd` for
XPBD/SemiImplicit. No closed-form e-to-parameter translation is documented.

**Bullet/Bullet3.** Has explicit Newton-style restitution. Each `btCollisionObject`
carries a restitution coefficient; on contact creation,
`btManifoldResult::calculateCombinedRestitution`
(https://github.com/bulletphysics/bullet3/blob/master/src/BulletCollision/CollisionDispatch/btManifoldResult.cpp)
sets `cp.m_combinedRestitution = body0->getRestitution() * body1->getRestitution()`
(multiplicative combine, customizable via `gCalculateCombinedRestitutionCallback`).
The Sequential Impulse solver uses it in `restitutionCurve(rel_vel, m_combinedRestitution)`
only above a velocity threshold to suppress jitter at rest. The model is
rigid-body Newton restitution applied as a velocity bias in PGS — no
spring-damper compliance.

**PhysX 5 (Isaac Sim/Lab/Gym).** `PxMaterial::setRestitution()` plus
`setRestitutionCombineMode(PxCombineMode::Enum)` with options
`eAVERAGE | eMIN | eMULTIPLY | eMAX`
(https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/_api_build/struct_px_combine_mode.html).
When two materials collide, the higher combine-mode enum value wins. PhysX is
impulse-based (TGS/PGS variants), so restitution is applied as a velocity-level
Newton bound — no spring-damper translation.

**ODE.** `dSurfaceParameters` has `bounce` (0..1) and `bounce_vel` (minimum
incoming velocity for bounce), set per contact joint before joint creation
(https://ode.org/ode-latest-userguide.html). Implementation is Newton-restitution
velocity bias inside the LCP solver. ODE's CFM/ERP soft-contact knobs control
compliance independently and are *not* automatically derived from `bounce`.

**Drake (point contact + hydroelastic).** No `coefficient_of_restitution`.
Drake's MultibodyPlant uses a Hunt & Crossley dissipation model with parameter
`d` (s/m). The contact force is `f(x) = fₑ(x)·(1 − d·vₙ)₊`
(https://drake.mit.edu/doxygen_cxx/group__compliant__contact.html). The docs
say: "the bounce velocity after impact is bounded by 1/d, giving quick physical
intuition... typical value of 20 s/m"; `d > 500 s/m` is "unphysical." MultibodyPlant's
discrete approximations actually pin `d` so that penetration behaves as a
critically damped oscillator, giving e≈0 by default. Combined dissipation:
`d = (k₂/(k₁+k₂))·d₁ + (k₁/(k₁+k₂))·d₂`. Hydroelastic contact (Elandt et al.
2019) reuses the same Hunt-Crossley `d` but combines via hydroelastic moduli
(https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html).

**Project Chrono.** Two contact methods: NSC (PSOR/non-smooth, complementarity)
and SMC (DEM-like compliance). SMC's `ChMaterialSurfaceSMC` exposes Young's
modulus E, Poisson's ratio ν, friction `µs/µk`, and a literal `restitution`
(COR) `e`. Internally Chrono converts `e` to a damping coefficient via
Hertzian/Lankarani-Nikravesh-style closed forms
(https://api.projectchrono.org/collisions.html). NSC, being velocity-impulse-based,
uses Newton restitution directly.

## Paper survey: viscoelastic / disengagement

**Hunt & Crossley (1975)** "Coefficient of restitution interpreted as damping in
vibroimpact," J. Appl. Mech. 42(2), 440–445, doi:10.1115/1.3423596
(https://hal.science/hal-01333795/file/Hunt.pdf). Introduces
`F = kδⁿ + λδⁿδ̇` (n=3/2 for spheres). The famous closed form
`e ≈ 1 − (1/(1+α))·λ·v₀` (linear in v₀) is a small-damping perturbation
expansion. **Their derivation explicitly criticizes the Kelvin-Voigt half-period
model**: "during impact... half of a damped sine wave... is shown to be
logically untenable, for it indicates that the bodies must exert tension on
one another just before separating." So Hunt-Crossley themselves rejected the
half-period assumption and used a tension-cutoff (early-disengagement) boundary,
meaning the popular `e = exp(-π·ζ/√(1-ζ²))` formula is in fact a Kelvin-Voigt
approximation, not a Hunt-Crossley one — a common conflation in the literature.

**Lankarani & Nikravesh (1990)** "A Contact Force Model with Hysteresis Damping
for Impact Analysis of Multibody Systems," J. Mech. Design 112(3), 369–376,
doi:10.1115/1.2912617. Damping factor `μ = 3k(1−e²)/(4·v₀)`. Valid for `e > 0.7`;
degrades for high damping because the derivation assumes most kinetic energy
is recovered. Used in Chrono SMC and many DEM codes.

**Marhefka & Orin (1999)** "A Compliant Contact Model with Nonlinear Damping for
Simulation of Robotic Systems," IEEE Trans. SMC-A 29(6), 566–572,
doi:10.1109/3468.798060 (https://ieeexplore.ieee.org/document/798060/). Refines
Hunt-Crossley; gives a nonlinear-damping formula whose `(e ↔ b)` mapping is
energy-consistent for both small and large `e`, and explicitly handles the
contact disengagement boundary (`F → 0` at separation, no tension).

**Falcon, Laroche, Fauve & Coste (1998)** "Behavior of one inelastic ball
bouncing repeatedly off the ground," Eur. Phys. J. B 3, 45–57,
doi:10.1007/s100510050283 (https://link.springer.com/article/10.1007/s100510050283).
Experimental: `e` is approximately constant for moderate-to-high impact
velocity but drops to 0 as `v₀ → 0`. They show the simple half-period assumption
fails when "the duration between two successive bounces becomes of the order
of the impact duration" — the disengagement-truncated regime.

**Schwager & Pöschel (2007/2008)** "Coefficient of restitution for viscoelastic
spheres: The effect of delayed recovery," Phys. Rev. E 78, 051304,
doi:10.1103/PhysRevE.78.051304. Earlier "Coefficient of restitution of colliding
viscoelastic spheres," Phys. Rev. E 60, 4465 (1999), doi:10.1103/PhysRevE.60.4465.
Provides an analytically exact infinite-series `e(v₀)` for nonlinear (Hertzian)
viscoelastic contact and compact closed-form approximations. **The 2008 paper
addresses exactly the disengagement-boundary problem ("delayed recovery").**

**Zhang & Sharf (2019)** "Exact restitution and generalizations for the Hunt–Crossley
contact model," Mech. Mach. Theory 139, 174–194,
doi:10.1016/j.mechmachtheory.2019.04.018
(https://www.sciencedirect.com/science/article/abs/pii/S0094114X19302332).
**The first analytical closed-form solution `λ(e, v₀)` for the *full* Hunt-Crossley
equation using "an inverse restitution coefficient" series.** Documented as the
missing `(e → damping)` map valid in the high-dissipation regime where
Lankarani-Nikravesh and Hunt-Crossley's own perturbation fail.

**Stronge (2000)** *Impact Mechanics*, Cambridge UP, doi:10.1017/CBO9780511626432.
Chapters 2 and 6 derive both the linear viscoelastic (Kelvin-Voigt half-period)
and nonlinear (Hertzian + viscous) cases and explicitly discuss the energetic
vs kinematic vs Poisson definitions of `e`. Confirms the half-period formula
is only the Kelvin-Voigt fully-completed-cycle case.

## Mechanical impedance interpretation

The literature does **not** offer a single accepted bridge from spring-damper
contact damping `b` to a physical mechanical impedance `Z = ρcA`. Two adjacent
bodies of work exist but neither closes the loop:

1. **DEM literature** derives `(k, b)` from particle elastic properties (Young's
   modulus, density, radius) via Hertzian/Lankarani-Nikravesh expressions
   (e.g. https://arxiv.org/pdf/2509.07461 — review of DEM contact models).
   These give a *physical* `k` from `E` and a damping `b` calibrated to a
   measured `e`, but not a wave-impedance interpretation.

2. **Drake hydroelastic** (Elandt et al. 2019,
   https://ryanelandt.github.io/projects/pressure_field_contact/) derives the
   contact stiffness from a precomputed pressure field over the body interior;
   the dissipation parameter `d` is still phenomenological Hunt-Crossley with
   units s/m, *not* a physical impedance — combination across bodies uses
   hydroelastic moduli, but `d` itself is user-supplied per body.

**Robotics impedance control** (Hogan 1985,
https://summerschool.stiff-project.org/fileadmin/pdf/Hog1985.pdf) uses
"mechanical impedance" in the inverse sense — *prescribing* a desired
`(M, B, K)` for an end-effector — and never derives that `B` from material
wave-impedance `ρcA`. None of the surveyed engines bridges contact damping to
characteristic acoustic impedance. **No paper found explicitly equates
phenomenological contact `b` with `ρcA`.**

## Empirical / lookup approaches

DEM literature contains the closest matches: there are explicit "calibration of
damping coefficient in DEM" papers that pre-measure `(damping → e)` at fixed
step size and store it as a fit/table (e.g.
https://www.researchgate.net/publication/289463731 "Calibration of damping
coefficient in discrete element method simulation";
https://arxiv.org/pdf/2509.07461 review). For agricultural/granular DEM,
calibration is the dominant practice — Coetzee 2017 and others publish "master
calibration curves" for COR-vs-damping at fixed integration parameters. No
mainstream rigid-body engine ships such a lookup table; they all rely on
analytical closed forms (PhysX/Bullet/ODE: Newton impulse; Drake/Chrono-SMC:
Lankarani-Nikravesh; MuJoCo/Newton: user tunes solref).

## Recommendation (literature-grounded only)

1. **Most common pattern in mainstream engines for spring-damper-based contact
   (MuJoCo, MuJoCo Warp, NVIDIA Newton, Drake):** they do **not** expose `e`
   directly. They expose either a damping ratio (`solref dampratio`) or a
   Hunt-Crossley `d` (s/m), and the documentation explicitly tells users what
   `e` to expect at limiting cases (Drake: "bounce velocity bounded by 1/d";
   MuJoCo: critically damped at `dampratio=1`). Engines that *do* expose `e`
   directly (Bullet, PhysX, ODE, Chrono SMC) all use velocity-impulse Newton
   restitution or a closed-form Lankarani-Nikravesh translation — never a
   spring-damper-with-runtime-`e`-mapping.

2. **Most accurate `(parameter ↔ e)` translation in the high-dissipation regime
   documented in the literature:** Zhang & Sharf 2019 ("Exact restitution... for
   Hunt-Crossley") provides the only analytically exact closed form valid at
   high damping; Schwager & Pöschel 2008 ("delayed recovery") provides the
   exact treatment of the contact-disengagement boundary the user identified
   empirically. Marhefka & Orin 1999 is the most cited "energy-consistent at
   all `e`" formulation. Lankarani-Nikravesh is widely used but is documented
   as inaccurate for `e < 0.7`.

3. **Bridge from `b` to physical mechanical impedance:** **no engine surveyed
   makes this bridge**. DEM is the closest body of work to derive `(k, b)`
   from physical material properties, but uses Hertzian/Lankarani-Nikravesh,
   not wave impedance `ρcA`. Drake hydroelastic derives `k` from pressure
   fields but leaves `d` phenomenological. Hogan-style impedance control is
   the dual problem and does not address contact damping. If the user wants a
   physical bridge, the literature offers DEM-style `(E, ν, ρ, R) → k`, plus
   a separate `(e → b)` calibration via Zhang-Sharf or Schwager-Pöschel — but
   no single closed-form `Z → b` mapping was found.
