# C2c6 approved source catalog checkpoint

Initiative: q54-optical-lab-p12-4c4c2c-source-catalog
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-03
Status: accepted
Related Files: source-catalog reviews codex v1/v2; implementation-note v1;
review-followup v1; PROGRESS.md
Owner Summary: Record exact scoped approval, actual Go2 regression and process
hygiene. C4 is not complete and no product/result gate is opened.

## Revision and independent verdict

Original base: `73b4172caeb155158acdba2a3e4f3d98799fdcd6`.
Initial candidate: `f8e9c4b015fcb78e464f9a62d32e97a97e39a31c`.
Approved candidate: `3a0b5e5e5a770ef7902f314279f77743f4375d2d`.
Fresh reviewer: `/root/c4_c2c6_source_catalog_review_v2`, forked without the
implementation conversation. Verdict Approve; no unresolved substantive finding.
Full report archived verbatim, SHA-256:
`50ccbb91732559e1c84d7309a417465ca5a132f435227b308f44cfe7f9d95f35`.
The evidence-only commit containing this file is separate from that approval.

Initial review correctly rejected the mandatory Go2 tests: they omitted the
existing required asset evidence. Author initial regression319passed/2failed;
independent focused216passed/2failed. The repair invokes real production asset
preflight per case and passes its result, preserving all actual import checks.
No source guard, skip rule, catalog or product behavior was relaxed.

Independent revised regression:218passed,0skipped,74.31s, actual exit0.
Both lanes and both real Go2 cases also passed a separate actual-checkout probe;
all71 source Git blobs, disk bytes and catalog hashes matched. Full details,
commands, process identities, limitations and closure of the original repro are
in review v2. These facts are distinct from the author regression below.

## Author revised regression

Host: authorized fallback101, hostname aig-a100. Primary183:40031 transport
was retried and timed out. No alternate port was inferred or configuration
changed. Interpreter `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`,3.12.13.
Exact clean detached worktree `/data3/ang.gao/robot_sim_c2c6_v2` at approved HEAD.

```sh
cd /data3/ang.gao/robot_sim_c2c6_v2
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2 \
  /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest -q \
  tests/unit/optics/test_native_sources.py \
  tests/unit/optics/test_native_source_policy.py \
  tests/unit/benchmarks/test_closed_worker_protocol.py \
  tests/unit/benchmarks/test_optical_benchmark_core.py \
  tests/unit/benchmarks/test_optical_benchmark_contracts.py \
  tests/unit/benchmarks/test_optical_render_migration.py --tb=short
```

Observed321passed,0skipped in82.12s, actual exit0. A Python wrapper spawned and
waited for this exact pytest command inside `timeout -v -k 5s 240s`; no timeout
or signal was needed. No installation, substitute Go2, GPU import/work, benchmark
record or performance comparison occurred. Ruff/diff/show checks passed.

Process ledger: wrapper2113597; pytest2113603, parent2113597, Linux starttime
83812177; timeout parent2113576. Wrapper reported `AUTHOR_EXIT 2113603 0`.
Local SSH9192, parent27132, Windows creation2026-10-03 07:06:21, tool session
58850 completed exit0. All four identities were absent at the scoped postcheck;
GPU compute query returned no rows. Internal short-lived test children were not
individually ledgered; completed subprocess calls and scoped checks are the
available evidence. Reviewer separately checked its own children, recorded in
review v2. No broad kill, device reset, pool clearing or unrelated cleanup.

The original user's dirty `/data3/ang.gao/robot_sim_c4c1b2` was not changed:
two pre-existing modified Warp files and two pre-existing untracked tests
remain. Bundle transfer used per-invocation `git -c gc.auto=0`; no maintenance,
repack or persistent config change was attempted. Candidate worktrees stayed
clean. Current clean checks do not diagnose the old unidentified hang.

## Explicit next boundary, not additional approval

Source snapshots and actual post-import checks do not prove earlier bytecode
consumption. A separate disposable author CPU probe confirmed this on local
Python3.14 and remote3.12: a valid timestamp/size-matched cached module executes
under `-B` even when the actual source bytes contain a different marker.
The probe used only a fresh tempfile directory and retained its script outside
the repository at `c4_assets/source_consumption_probe.py`; it changed no candidate.
Source bytes `MARKER = 'FROZEN'\n` hashed to
`e321f34626931b9abe84ae1ddecfb9eed64a22f87fc49397e758ea8a54343929`,
while the existing cache produced `STALE!`. A disposable source-bytes loader
produced `FROZEN`. This diagnostic is not an implemented/approved loader policy.
Local probe2720 and remote2125707 completed exit0 and were absent afterward;
remote GPU compute query was empty. Short probe-child PID identities were not
retained. No GPU or worker lifecycle property is claimed by this observation.

Source consumption/cache handling and startup/HELLO/request wiring require
their own reviewed C2d integration; if a new loader needs a policy exception,
freeze and independently review that decision before implementing it. Current
C0 and all activation gates stay unchanged. C2d/e/f, C4 completion, B3/host,
shipping, GPU preparation, a3b, result and baseline remain closed. No changes to
collab/README, review rules or historical approval meanings are included here.
