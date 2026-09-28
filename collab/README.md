# Robot Simulator — Shared Collaboration Protocol

Status: owner-directed workflow, 2026-09-28

The owner sets scope and product decisions. Codex implements and coordinates
independent reviews. Every design or implementation checkpoint is reviewed in
a fresh session, fixed in response to findings, and independently re-reviewed
until approved. The owner need not manually relay each revision.

This supersedes the old fixed Claude-author/Codex-reviewer role split and
mandatory human routing. Historical proposals and approvals remain records of
their own scope. The owner may still request an external Claude review.

## Mandatory review standards

Read [REVIEW_PRINCIPLES.md](REVIEW_PRINCIPLES.md) in full. Its code and design
principles are the authoritative review checklist. No quota of findings or
number of tests substitutes for an evidence-based conclusion.

## Independent sessions

- Start a fresh reviewer for every round. For the available agent mechanism,
  use `spawn_agent(..., fork_turns="none")`. The owner authorizes these review
  agents. Give them the task packet below and repository access; do not fork
  the implementation conversation or provide hidden implementation reasoning.
- The reviewer must not have authored the candidate changes. It may use the
  same model: this provides context and role isolation, not a claim to a
  different vendor, external Claude session, or separate security boundary.
- Provide requirements, accepted designs, scope and evidence locations.
  Author notes and prior verdicts are claims to check, never approval evidence.
  On follow-up rounds provide prior findings and original repros, so each can
  be independently closed against the revised code.
- Freeze the candidate while review runs. Use an isolated checkout/worktree
  when available, particularly for tests requiring a clean tree. Record actual
  base and HEAD. Do not change the candidate underneath the reviewer.
- The reviewer inspects and tests; the author edits the implementation. The
  reviewer does not mutate the implementation or silently fix it. Scratch repros
  and generated evidence belong outside the candidate's tracked files; record
  any test side effects. The author records the returned report verbatim.
- Only requirements and evidence may be used to resolve a disagreement. Do
  not demand an approval, discard unresolved findings, or select successive
  reviewers merely to obtain a preferred verdict.
- If fresh sessions or required validation are unavailable, report the exact
  limitation. Do not present same-session self-review or static inspection as
  completed independent or mandatory runtime validation.

## Review packet

Every handoff provides:

- review kind (design or code), initiative and bounded stage;
- exact base/HEAD and candidate paths; for a design, its version and digest;
- requirements and accepted predecessor design/decision files;
- promised behavior, exclusions and gates that remain closed;
- repository location, interpreter/environment and relevant test entrypoints;
- prior findings/repros for a follow-up, plus the changes addressing them.

The reviewer independently checks the diff, working tree, call/data/state flow,
resource graph and any claimed evidence. Tests are proportional to the risk;
document-only workflow edits do not require unrelated GPU regressions.
Warp/CUDA verification normally uses `robot_sim` and `python -m pytest` from
that environment. Required real-device properties cannot pass on mocks alone.

## Review, fix and re-review loop

1. Implement or draft only the authorized scope. Freeze a concrete candidate
   as a local commit or exact content snapshot. A review commit is not release,
   push, merge, product activation, or final approval.
2. Start the independent session with the packet. Continue useful work outside
   the frozen candidate, without preempting the review conclusion.
3. Save the complete report under `collab/`, including session identity,
   reviewed revision, scope, findings, commands, results and limitations.
4. For `Request changes`, fix substantiated findings and add evidence that
   addresses the actual counterexample. If evidence contradicts a finding,
   document the reproduction and reasoning for the next independent review.
5. Freeze the revised candidate and start a new independent session. It checks
   the previous findings, changes and affected invariants for regressions.
6. Continue until `Approve` or `Approve with comments`, with no unresolved
   blocking finding or required evidence gap. Comments must be explicitly
   non-blocking and retained with their disposition or bounded follow-up.
7. Report the exact approved revision/scope, tests and remaining limitations to
   the owner. Design approval permits its implementation only; child-stage
   approval does not approve the parent or open later gates.

There is no automatic approval after a fixed number of rounds. If the fix needs
new product policy, broader authority, unavailable assets or unavailable real
hardware, keep the affected gate closed, explain the blocker, and request only
the missing input. Do not weaken the contract to force approval.

Approval binds the inspected revision or snapshot. Subsequent changes to code,
tests, design or operative workflow rules require independent review of the
delta. Recording the unchanged reviewer report, commands/results and scope in
a separate evidence-only commit does not recursively trigger another review.
Check that this commit contains only that record. It cannot change the verdict,
requirements, approved scope or gates. Reference the approved parent revision
and the separate evidence commit accurately.

## Review report

Use a versioned file with the standard header below, plus reviewer session ID,
base/HEAD or design digest, actual tree status, and one of `Approve`,
`Approve with comments`, `Request changes`, or `Blocked`.

Required content:

- Findings: severity, location, trigger, consequence, why existing checks miss
  it, proposed direction and proof needed after repair. Report no substantive
  findings explicitly if appropriate.
- Verification: inspected high-risk areas; executed commands and environment;
  observed results; reproductions; unexecuted checks and why.
- Scope: precisely what is approved and what remains closed.
- Coverage gaps, residual risks and non-blocking follow-ups.

Keep code failures, environment failures and toolchain failures distinct.
Do not reuse old outputs or caches as proof of a new execution. Normal caches
are allowed when the test contract permits them; record relevant provenance.

## Files and history

Shared decisions, requests, reports and evidence live in `collab/` and should
be Git-tracked. Never edit the other agent's private instructions, including
`CLAUDE.md`, to change shared policy. Root `AGENTS.md` points to this protocol.

Use `collab/<initiative-id>__<stage>__<author>__v<n>.md` with a stable lowercase
kebab-case initiative. Increment versions when revising a review or design;
preserve previous semantic history. Stages include `proposal`, `challenge`,
`decision`, `implementation-note`, `review`, `gap-scan` and the existing
`design-request`, `review-followup`, `test-evidence` conventions. Authors are
`codex`, `claude` or `owner`; a Codex independent reviewer uses `codex` and its
separate session ID, never impersonates Claude.

Standard header:

```text
Initiative:
Stage:
Author:
Version:
Date:
Status:
Related Files:
Owner Summary:
```

Status values: `draft`, `in_review`, `superseded`, `accepted`, `implemented`.
An implementation note states actual changes, tests, known incompleteness,
environment and candidate revision. Separate observed facts from inference.
Show the owner a concise conclusion whenever substantive collaboration files
change, and include the eventual independent verdict and scope in the handoff.

Architecture and subsystem changes use design → independent design review →
fix/re-review → implementation → independent code review → fix/re-review.
Bounded fixes within an accepted design can go directly to implementation and
independent code review. Capability-changing work also receives a gap scan;
minor maintenance need not. New architecture or policy discovered during a
bounded fix requires a scoped design decision before implementing that expansion.

Promote durable progress to `PROGRESS.md`, decisions to `REFLECTIONS.md`, and
unresolved long-term gaps to `OPEN_QUESTIONS.md` as relevant. Round reports stay
in `collab/`; archive inactive threads without deleting their evidence.
