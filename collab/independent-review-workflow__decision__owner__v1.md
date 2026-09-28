# Independent review workflow — owner decision

Initiative: independent-review-workflow
Stage: decision
Author: owner
Version: v1
Date: 2026-09-28
Status: accepted
Related Files: README.md; REVIEW_PRINCIPLES.md; ../AGENTS.md
Owner Summary: Codex implements and initiates fresh independent review sessions;
fix and independently re-review each checkpoint until it passes.

## Decision

The owner supplied eleven code-review principles and fourteen design-review
principles, recorded in `REVIEW_PRINCIPLES.md`. Each review must establish facts
from actual code/design and independently checked evidence, with scope,
ownership, failure behavior and stage gates explicit. Reviewer and author
have separate sessions. The author processes findings and repeats independent
review until there are no unresolved blockers. Operational details are in
`README.md`; the root `AGENTS.md` makes the workflow discoverable.

This changes the default human relay process. It does not authorize new product
capabilities or allow a reviewer to invent product policy or lower a frozen gate.

## Existing stage status reported by the owner

`00f1790 + f771e61` are approved only for C4c1b1 single-source Warp kernel
extraction. C4c1b2/c require separate implementation and review. Product GPU
preparation, a3b and result/baseline gates remain closed.

The owner's non-blocking follow-up is to extend `_SHARED_ENTRIES` from four to
all seven migrated compatibility kernels. Carry it into the next relevant
test change; it is not implemented by this documentation-only decision.

## Recording boundary

183 is currently unreachable. This workflow change is prepared on the isolated
branch `workflow/independent-review`, based on locally available `2b41068`.
It changes only collaboration documentation and adds the root discovery file.
It does not reconstruct or replace the later approved C4c1b1 commits. Preserve
the existing 101 worktree and reconcile the documentation commit into the
authoritative history when it is accessible.
