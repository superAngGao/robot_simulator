# Owner-requested systematic refactor before C4 acceptance

Initiative: Q54 / P12.4 / C4
Stage: recorded owner direction and implementation follow-up
Author: codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: C4/C4c2 approved designs; REVIEW_PRINCIPLES.md; compilation-admission design-request/design-review v3; compilation-admission implementation-note v1
Owner Summary: The owner is concerned that accumulated hotfixes may leave an unnecessarily complicated implementation and explicitly requests a systematic refactor after implementation. Include that work in the active C4 delivery, with affected regression evidence and fresh independent review before final acceptance.

## Direction and scope

The owner stated: “我比较担心的是，你可能引入了过多的hotfix，需要在实现后系统refactor一轮”.

This is authorization to perform the refactor, not a request for another permission
round. It covers the structure of the C4 implementation as a whole, including
interfaces to accepted C2d3 where the integrated changes expose duplication or
unclear responsibility. It must not be reduced to renaming the latest fixes or
adding another validation wrapper. Existing reviewed behavioral contracts, timing
budgets, ownership guarantees, schemas and phase gates remain requirements.
Any change to those requirements still needs the established bounded design review.

Complete the functional implementation and required real success/failure evidence,
then perform a concentrated structural pass before whole C4 acceptance. Continue
fixing obvious structural problems during implementation. Each earlier bounded
checkpoint still receives its required independent review; a promised later
refactor does not excuse incomplete behavior or structural findings there.

## Concrete review targets

- Give source admission, runtime custody, generated-module admission and process
  supervision explicit owners and narrow interfaces. Check whether responsibilities
  added to native_bootstrap belong there or can share existing production mechanics.
- Keep one source for recipes, phase order, role bindings, state transitions and
  validation rules. Inspect collector/parser/driver/observer interactions for rules
  that can drift independently, rather than merely comparing names or test counts.
- Consolidate partial-return retention and failure propagation. Preserve original
  failures, exact owners and deadlines across exceptions; avoid repeated failure
  notifications, restoration paths or apparent recovery after poison.
- Remove superseded branches, duplicated compatibility implementations and temporary
  wiring once their approved replacement is proven. Historical evidence remains
  unchanged; final losing-backend cleanup still belongs to its separate parent gate.
- Make tests exercise actual public or private production boundaries and meaningful
  counterexamples. Explicitly separate descriptive synthetic records, CPU behavior
  and real native/process facts; remove tests that merely repeat implementation.

Record concrete before/after findings and the resulting responsibility/call-flow
changes in the eventual review packet. Refactoring is complete only after affected
success and failure regressions pass and a fresh independent reviewer assesses both
the integrated structure and behavior. Changes to native source/runner/observer or
ABI boundaries require the applicable renewed exact-code preflight before native
execution; old approval and evidence cannot silently cover changed code.

This requirement is part of completing C4. It is not a claim that the current
compilation draft, C4, CUDA performance or final backend choice is complete.
