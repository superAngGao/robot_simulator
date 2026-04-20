# Robot Simulator — Shared Collaboration Protocol

> Status: Draft v0.2
> Audience: Project owner, Claude, Codex
> Scope: shared collaboration process for non-trivial design, implementation,
> review, and capability-gap discussions in this repo

---

## 1. Purpose

This repo is evolving toward a **multi-physics simulation + rendering +
robot training platform**, not only a rigid-body simulator.

Planned or implied major subsystems include:

- rigid body dynamics
- soft body
- fluid
- cloth
- rendering
- robot training platform

Because of that, collaboration should not rely on informal chat fragments.
Instead, Claude and Codex should communicate through **shared repo files**
with explicit naming and stage semantics, while the project owner acts as the
router and final prioritization point.

This document defines that shared workflow.

---

## 2. Core Idea

Claude and Codex should not behave like duplicate assistants.

The intended split is:

- Claude pushes domain solutions forward
- Codex pushes back on architectural, maintainability, coverage, and ecosystem gaps
- the project owner routes files between them and makes final tradeoff calls

This is a constructive adversarial optimization loop:

1. proposal
2. architecture challenge
3. implementation
4. review
5. gap scan

Each step should add a different kind of value.

---

## 3. Shared Directory

All cross-agent collaboration files should live under:

```text
collab/
```

This directory is a **neutral shared workspace**.

It is appropriate for:

- design proposals
- architecture challenges
- implementation notes
- code reviews
- post-commit gap scans
- final decisions for a specific initiative

It is **not** a replacement for:

- `OPEN_QUESTIONS.md` for long-lived unresolved gaps
- `REFLECTIONS.md` for settled design decisions
- `PROGRESS.md` for milestone progress tracking

Think of `collab/` as the working surface for a specific discussion round.
The long-term conclusions should later be condensed into the project-level docs.

Completed or inactive threads may later be moved under:

```text
collab/archive/
```

This helps keep `collab/` root focused on active initiatives.

---

## 4. Boundary Rules

Claude and Codex must have clear operational boundaries.

### 4.1 No Cross-Editing Agent Bootstrap / Operating Context

Neither agent should edit the other's bootstrap, operating-context, or
agent-specific setup files unless the project owner explicitly requests it.

This rule is about preserving clear agent boundaries, not about forbidding
discussion or critique of checked-in project documents.

In practice:

- shared process rules belong in neutral repo documents such as this one
- suggestions about another agent's checked-in project docs are allowed
- direct edits to the other agent's operating context should not happen by default

### 4.2 Shared Files Are the Communication Channel

Claude and Codex should communicate through:

- files in `collab/`
- code diffs / commits
- shared project docs when a conclusion should become durable

The project owner passes **file names**, not reformatted summaries, whenever possible.

### 4.3 Human Router Model

The project owner is the operational router.

That means:

- Claude writes a file
- the owner gives Codex the file name
- Codex reads it and writes the next file
- the owner gives that new file name back to Claude

This keeps routing explicit and auditable.

### 4.4 Owner Visibility

Agent-to-agent communication must also remain legible to the project owner.

That means collaboration files should not only contain deep technical detail;
they should also expose the key takeaways in a form that the owner can route,
evaluate, and react to quickly.

In practice:

- each stage file should include a short owner-facing summary
- when an agent creates or updates a `collab/` file, it should also present the key points on screen
- major disagreements should be stated explicitly, not buried in prose
- recommendations should be easy for the owner to approve, reject, or defer

The goal is not only traceability between agents, but also visibility for the
human decision-maker.

### 4.4.1 On-Screen Summary Requirement

Whenever Claude or Codex produces a new collaboration file, or substantially
updates an existing one, the agent should also show the owner a short on-screen
summary.

That on-screen summary should highlight:

- the main conclusion
- the most important risk or disagreement, if any
- the recommended next step

The file remains the durable record, but the screen output is the owner's
primary real-time view.

### 4.5 Git-Tracked Collaboration

`collab/` files should normally be committed into the repo.

Reasons:

- both agents can inspect the same durable history
- discussion context is not trapped in chat logs
- architectural reasoning becomes auditable
- later promotion into `REFLECTIONS.md` or `OPEN_QUESTIONS.md` is easier

If a particular thread is intentionally ephemeral, the owner can choose not to
commit it, but the default should be to keep collaboration history in git.

---

## 5. File Naming Convention

Each collaboration file should use this pattern:

```text
collab/<initiative-id>__<stage>__<author>__v<n>.md
```

Example:

```text
collab/render-scene-contract__proposal__claude__v1.md
collab/render-scene-contract__challenge__codex__v1.md
collab/render-scene-contract__decision__owner__v1.md
collab/render-scene-contract__implementation-note__claude__v1.md
collab/render-scene-contract__review__codex__v1.md
collab/render-scene-contract__gap-scan__codex__v1.md
```

### 5.1 Initiative ID Rules

`<initiative-id>` should be:

- stable across the full discussion round
- lowercase
- `kebab-case`
- specific enough to distinguish from nearby work

Good examples:

- `render-scene-contract`
- `gpu-contact-force-api`
- `soft-body-mvp`
- `multiphysics-scheduler`

Bad examples:

- `fix`
- `idea`
- `phase3`

### 5.2 Allowed Stage Values

Use one of these exact stage names:

- `proposal`
- `challenge`
- `decision`
- `implementation-note`
- `review`
- `gap-scan`

Do not invent new stage names unless the project owner explicitly wants to
expand the protocol.

### 5.3 Allowed Author Values

Use one of:

- `claude`
- `codex`
- `owner`

### 5.4 Versioning

If the same stage file is revised, increment the version:

- `v1`
- `v2`
- `v3`

Do not overwrite the semantic history by silently renaming old files.

---

## 6. Standard File Header

Every `collab/*.md` file should begin with this header:

```md
Initiative:
Stage:
Author:
Version:
Date:
Status:
Related Files:
Owner Summary:
```

Recommended `Status` values:

- `draft`
- `in_review`
- `superseded`
- `accepted`
- `implemented`

`Related Files` should list prior or downstream files in the same thread when useful.

`Owner Summary` should be short and decision-oriented. It should give the owner
the most important point of the file without requiring a full technical read.

---

## 7. Stage Responsibilities

### 7.1 `proposal` — Claude

Claude proposes a solution before code changes.

Expected contents:

- problem statement
- goal
- affected layers and files
- interface or API sketch
- implementation plan
- validation / test plan
- tradeoffs
- reference projects consulted

Claude should optimize for domain quality and technical feasibility.

The proposal should also make the recommended direction legible to the owner,
not only to Codex.

### 7.2 `challenge` — Codex

Codex responds to the proposal from the platform perspective.

Expected focus:

- maintainability
- extensibility
- dependency direction
- abstraction hygiene
- future compatibility with soft / fluid / cloth / rendering / training
- risk of local implementation choices becoming platform constraints

Required structure:

- `Keep`
- `Change`
- `Risks`
- `Future Compatibility`
- `Recommendation`

For non-trivial proposals, Codex should normally identify:

- at least 2 structural risks
- at least 1 future compatibility concern
- at least 1 alternative design point or narrowing suggestion

Codex should make the main accept / change / defer signals easy for the owner
to route.

### 7.3 `decision` — Owner

The owner records the current decision after considering the proposal and challenge.

Expected contents:

- chosen direction
- deferred ideas
- what is explicitly out of scope for this round
- acceptance conditions

This file is the handoff target for implementation.

It should be the clearest owner-facing record of what the team is actually doing.

### 7.4 `implementation-note` — Claude

After implementation, Claude records what actually changed.

Expected contents:

- files changed
- behavior added or changed
- tests added or updated
- what remains intentionally incomplete
- commit hash if available

This should describe the implemented result, not restate the original plan.

### 7.5 `review` — Codex

Codex reviews the implemented result.

Expected focus:

- maintainability of the actual patch
- architectural regressions
- hidden coupling
- test completeness beyond happy paths
- CPU/GPU or old/new interface consistency where relevant
- documentation, packaging, or user-path drift

Required structure:

- `Findings`
- `Coverage Gaps`
- `Residual Risks`

Findings should be prioritized by severity.

The top findings should be understandable to the owner without requiring a
full patch read.

### 7.6 `gap-scan` — Codex

After implementation or commit, Codex compares the landed capability against
relevant open-source systems and identifies what is still missing.

This stage is most valuable for algorithmic, architectural, subsystem, or
capability-shifting changes. It is usually unnecessary for narrow maintenance
work that does not materially change the repo's external capability profile.

Likely comparison targets:

- MuJoCo
- Drake
- Isaac Lab
- Newton / MuJoCo Warp
- SOFA
- Bullet
- Pinocchio
- hpp-fcl / coal

Required structure:

- `Compared Against`
- `What Landed`
- `Remaining Gaps`
- `Suggested OPEN_QUESTIONS entries`

Gap types should be labeled when possible:

- `[algorithm]`
- `[API]`
- `[engineering]`
- `[validation]`
- `[performance/stability]`

Important remaining gaps should later be copied into `OPEN_QUESTIONS.md`.

The gap-scan should make it clear to the owner whether the change meaningfully
advanced the repo's position relative to external projects.

---

## 8. Workflow Paths

There are two standard workflow paths: a full path and a lightweight path.

### 8.1 Full Path

Use the full path for initiatives with meaningful architectural, subsystem, or
future-platform implications.

Typical examples:

- subsystem architecture
- solver work
- contact / coupling work
- GPU backend changes
- rendering contracts
- training platform interfaces
- cross-layer refactors
- changes that may affect future soft / fluid / cloth / rendering integration

Sequence:

1. Claude writes `proposal`
2. Owner sends proposal file name to Codex
3. Codex writes `challenge`
4. Owner sends challenge file name to Claude
5. Owner writes or confirms `decision`
6. Claude implements and writes `implementation-note`
7. Owner sends implementation-note file name and commit hash to Codex
8. Codex writes `review`
9. Codex writes `gap-scan`
10. durable unresolved items are added to `OPEN_QUESTIONS.md`

### 8.2 Lightweight Path

Use the lightweight path for medium or bounded changes where a full
proposal/challenge/decision chain would be overkill.

Typical examples:

- targeted test improvements
- local refactors with clear boundaries
- narrow validation or benchmark additions
- non-architectural engineering changes
- scoped bug fixes with limited blast radius

Sequence:

1. Claude implements and writes `implementation-note`
2. Owner sends implementation-note file name and commit hash to Codex
3. Codex writes `review`
4. if the change has meaningful ecosystem or capability implications, Codex also writes `gap-scan`
5. durable unresolved items are added to `OPEN_QUESTIONS.md` when needed

### 8.3 Escalation Rule

If a supposedly lightweight change reveals architectural impact, subsystem
boundary changes, or future-platform consequences, the owner may escalate it
to the full path.

Either agent may proactively flag an escalation need to the owner if those
signals appear during proposal, implementation, or review.

### 8.4 Gap-Scan Trigger

`gap-scan` is not mandatory for every change.

It is most valuable for:

- subsystem architecture
- solver work
- contact / coupling work
- GPU backend changes
- rendering contracts
- training platform interfaces
- cross-layer refactors
- changes that materially shift the repo's capability relative to external projects

It is usually unnecessary for:

- pure test speed optimizations
- local tooling cleanup
- narrow maintenance work with no meaningful capability change
- tiny bug fixes that do not alter subsystem scope or platform surface

---

## 9. Review Standards

### 9.1 Proposal Challenge Standard

Codex should not just approve a proposal.

The challenge should answer:

- what should remain as proposed?
- what should change before implementation?
- what platform risk is being underestimated?
- what future subsystem might this accidentally block?

### 9.2 Test Audit Standard

When Codex reviews implementation, test review means more than checking whether tests exist.

Coverage should be judged across:

- nominal behavior
- boundary cases
- invalid / defensive behavior
- regression-prone paths
- backend consistency where relevant
- integration behavior where relevant

If coverage is incomplete, Codex should state the missing test matrix explicitly.

### 9.3 Gap Scan Standard

Codex should not only say "more work remains".

The gap scan should identify:

- which reference projects matter here
- what capability they provide that this repo still lacks
- whether the gap is algorithmic, architectural, engineering, validation, or performance-related

---

## 10. Templates

### 10.1 Proposal Template

```md
Initiative:
Stage: proposal
Author: claude
Version: v1
Date:
Status: draft
Related Files:
Owner Summary:

## Problem

## Goal

## Scope

## Affected Files / Layers

## Proposed Design

## Test Plan

## Tradeoffs

## References
```

### 10.2 Challenge Template

```md
Initiative:
Stage: challenge
Author: codex
Version: v1
Date:
Status: draft
Related Files:
Owner Summary:

## Keep

## Change

## Risks

## Future Compatibility

## Recommendation
```

### 10.3 Decision Template

```md
Initiative:
Stage: decision
Author: owner
Version: v1
Date:
Status: accepted
Related Files:
Owner Summary:

## Chosen Direction

## Accepted Constraints

## Deferred Items

## Out of Scope

## Acceptance Conditions
```

### 10.4 Implementation Note Template

```md
Initiative:
Stage: implementation-note
Author: claude
Version: v1
Date:
Status: implemented
Related Files:
Owner Summary:

## Open Questions Addressed

List every OPEN_QUESTIONS.md entry touched by this change:
- **QXX — <title>**: status before → status after (e.g. P1 open → partially resolved / closed)

If no open questions were affected, write "None."

## REFLECTIONS.md / PROGRESS.md Impact

State whether REFLECTIONS.md or PROGRESS.md need updating, and why or why not.

## What Changed

## Files Touched

## Tests Added / Updated

## Known Limitations

## Commit
```

### 10.5 Review Template

```md
Initiative:
Stage: review
Author: codex
Version: v1
Date:
Status: draft
Related Files:
Owner Summary:

## Findings
1. ...

## Coverage Gaps
- ...

## Residual Risks
- ...
```

### 10.6 Gap Scan Template

```md
Initiative:
Stage: gap-scan
Author: codex
Version: v1
Date:
Status: draft
Related Files:
Owner Summary:

## Compared Against

## What Landed

## Remaining Gaps
- [algorithm] ...
- [API] ...
- [engineering] ...
- [validation] ...
- [performance/stability] ...

## Suggested OPEN_QUESTIONS Entries
- ...
```

---

## 11. What Belongs In `collab/` vs Project Docs

### Keep In `collab/`

- round-specific proposals
- challenge documents
- implementation notes
- review memos
- gap scans
- owner decisions for a particular initiative

### Archive From `collab/` Root When Inactive

When an initiative is complete, abandoned, or dormant, the owner may move its
thread files into `collab/archive/` to keep the root directory readable.

Example:

```text
collab/archive/gpu-contact-force-api__proposal__claude__v1.md
```

### Promote Out Of `collab/` When Mature

- settled architectural decisions -> `REFLECTIONS.md`
- important unresolved capability gaps -> `OPEN_QUESTIONS.md`
- milestone completion / status changes -> `PROGRESS.md`

Do not treat `collab/` as the permanent home of all knowledge.
It is the working buffer, not the final archive.

---

## 12. Example Thread

For an initiative named `gpu-contact-force-api`, a normal thread could be:

```text
collab/gpu-contact-force-api__proposal__claude__v1.md
collab/gpu-contact-force-api__challenge__codex__v1.md
collab/gpu-contact-force-api__decision__owner__v1.md
collab/gpu-contact-force-api__implementation-note__claude__v1.md
collab/gpu-contact-force-api__review__codex__v1.md
collab/gpu-contact-force-api__gap-scan__codex__v1.md
```

The owner only needs to pass file names and any relevant commit hash.

---

## 13. Success Criteria

This protocol is working if:

- Claude proposals become more explicit, scoped, and reference-backed
- Codex feedback becomes more architectural and less redundant
- implementations ship with stronger and more explicit test coverage
- post-commit gap analysis continuously feeds `OPEN_QUESTIONS.md`
- the repo evolves as a platform rather than a collection of local optimizations
