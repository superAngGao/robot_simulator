# Robot Simulator development host routing decision

Owner decision date: 2026-09-28

## Host order

1. Primary: `183.166.183.33`, SSH port `40031`, local SSH alias
   `tileops-dev`.
2. Fallback: `192.168.80.101`, SSH port `22`.

Use the primary host by default. If the primary host is unreachable at the SSH
transport level, work may continue on the fallback host. Do not switch merely
for convenience or because a test is slow.

## Repository and environment locations

- Primary repository: `/data2/shared/ang.gao/robot_simulator`
- Primary Conda environment: `/home/ang.gao/miniforge3/envs/robot_sim`
- Fallback repository: `/data3/ang.gao/robot_simulator`
- Fallback Conda environment: `/data3/ang.gao/anaconda3/envs/robot_sim`

## Safe handoff rules

Before doing work on either host:

1. record `git rev-parse HEAD`, `git status --short`, and the current branch;
2. compare them with the last known authoritative host state;
3. never assume two uncommitted worktrees are equivalent;
4. transfer reviewed work through commits or a verified Git bundle, not by
   silently overlaying one entire worktree on the other; and
5. do not push until the active host history and worktree have been reconciled.

When the primary becomes reachable again after fallback work, reconcile the
fallback commits into the primary repository and rerun the relevant validation
there before treating the primary as authoritative again.

## State when this decision was recorded

- Primary last known reviewed state: `f771e61`, with a clean worktree.
- Primary was unreachable when this file was created.
- Fallback was at `2b41068` and still held the earlier uncommitted C4c1b1
  mirror. That mirror must be reconciled before new implementation work begins
  on the fallback host.
