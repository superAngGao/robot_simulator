Initiative: physics-correctness
Stage: proposal
Author: codex
Version: v3
Date: 2026-10-12
Status: accepted
Related Files: physics-correctness__proposal__codex__v2.md; physics-correctness__design-audit__codex__v1.md
Owner Summary: Amend only the PGS drop oracle after the inertia fix shows that slop is an allowed penetration interval, not a target depth. All other v2 repair and review obligations remain.

## Superseded assumption

The v2 equation p=slop+CFM*(M*g/4)*dt^2/ERP (ERP<=1) is a valid stationary
equilibrium ABOVE the slop threshold. Its prescribed-state one/four-contact
tests remain valid, as independently verified in v2 review. It does NOT imply
that a falling body reaches that depth within one second. With near-hard
normal constraints the body can stop at first contact anywhere within slop;
CFM only produces tiny inward creep there. Do not force a corrected engine
to reproduce the old 5 mm penetration caused by incorrect inertia.

## Replacement drop oracle (v2 default 0.45 -> 0.42 m, straight legs)

Prior to contact, semi-implicit Euler gives z_n=z0-g*dt^2*n*(n+1)/2.
The first detected penetration is the smallest integer n with z_n<leg_length
(contact depth must exceed the detector's 1e-10 m numerical threshold).
Use this discrete analytic crossing, not a measured trajectory value, as the
PGS arrest-height reference. For dt=.0002, n=391, z_n=.4199280336 m.
The reference penetration .0719664 mm is well inside the 5 mm slop region;
there is no Baumgarte correction and no target at 5 mm.

After arrest, CFM predicts v_n approximately -CFM*(M*g/4)*dt, with a similarly
CFM-scaled first impact residual. Over duration T the height contribution is
bounded by dt*CFM*(M/4)*(g*n*dt+g*T), under the symmetric contact model;
it is below 5 nm here. Finite iteration residuals are separately checked with
60/120 iterations and a dt sweep .0004/.0002/.0001 s. Keep the existing 1 um
height tolerance (more than 100x the computed CFM drift, well below the old
4.95 mm error), 1 um last-0.1s range and 1e-5 m/s speed thresholds. This tests
this fixture's discrete impact response; it is NOT a universal settling law.
ADMM remains compared to its scalar coupled soft-contact equilibrium root.

Implementation: add a pure discrete free-fall crossing helper in physics_metrics;
use it only for this PGS fixture. Name output `reference_z` / `reference_kind`
instead of labelling every solver's reference `equilibrium_z`. Preserve the
failed v2-oracle run in evidence and explain why it was wrong. Do not adjust
solver parameters or relax numerical tolerances to pass the invalid oracle.

## Design audit findings are evidence, not extra implementation authorization

The owner's follow-up asks whether the overall interaction design is reasonable
and faithfully implemented. The companion design-audit records both. Reproducible
ADMM projection, GPU regularization and terrain-height discrepancies are left
explicitly unresolved; changing friction models or the GPU equations requires
a subsequent scoped design. This round's production repair stays CPU inertia.

Review disposition: design-review v3 approved the amendment with comments.
C1's 4.8 nm arithmetic transcription error above is corrected; implementation
computes the stated formula and tests its arithmetic. No tolerance changed.
