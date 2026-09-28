# Project collaboration instructions

Read `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` before a design or
implementation task. The owner's current workflow is: Codex implements;
every review checkpoint uses a fresh independent session, followed by fixes
and independent re-review until approval for that exact scope and revision.
The owner explicitly authorizes spawning independent review agents for this
workflow. Do not inherit the implementation conversation into the reviewer.

Follow the review packet, isolation, evidence, and approval rules in
`collab/README.md`. A prior approval never opens a later phase or product gate.
Review-only evidence recording follows the bounded exception documented there.

Use `183.166.183.33:40031` (SSH alias `tileops-dev`) as the primary host.
If its SSH transport is unreachable, the owner permits fallback to
`ang.gao@192.168.80.101:22`. Check repository revision, branch, and working
tree before switching; preserve uncommitted changes and reconcile histories
before continuing implementation. See
`collab/robot-simulator-development-host-routing__decision__owner__v1.md`
when present for host-specific paths and handoff details.

Shared workflow belongs in these neutral project documents. Do not edit
`CLAUDE.md` or another agent's private operating context.
