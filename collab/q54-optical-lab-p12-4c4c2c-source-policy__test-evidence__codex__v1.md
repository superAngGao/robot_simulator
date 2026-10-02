# C2c5 author validation and scoped independent approval

Initiative: q54-optical-lab-p12-4c4c2c-source-policy
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-03
Status: accepted
Related Files: source-policy implementation notes v1/v2 and reviews v1/v2
Owner Summary: CPU primitive approved with a non-blocking mutation-test follow-up.
Complete source admission, runtime lifecycle integration and C4 remain unfinished.

Approved code: `05d9ccf370824f36cf757e5a251ad244de89b988` (includes3a365ea).
Original base: `7f32e6869e47fe31ea65a67ea317e3554144b41a`.
This commit records only reports/evidence/progress, not code, tests or policy.

## Actual author runs

Local Windows `C:/Python314/python.exe`3.14, checkout
`C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`:

```text
python -m ruff check optical_worker/native_source_policy.py tests/unit/optics/test_native_source_policy.py
python -m pytest -q tests/unit/optics/test_native_source_policy.py tests/unit/benchmarks/test_closed_worker_protocol.py
git diff --check
git show --check --oneline HEAD
```

Final result: Ruff/whitespace checks passed; **175 passed in2.37s**. No local
Torch installation was added. Earlier broader local collections were not green:
98pass/9missing-Torch failures (source/protocol/core) and62pass/25missing-Torch
failures (contracts/migration). All these collections were rerun remotely below.

Primary `tileops-dev` (183.166.183.33:40031) timed out. Authorized fallback101
connected; hostname aig-a100, H200 physicalGPU2 visible alongside two A100s.
No GPU validation or GPU submission is claimed for this CPU-only slice.
Remote exact detached clean checkout:
`/data3/ang.gao/robot_sim_c2c5_05d9ccf`.
Existing interpreter: `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`3.12.13.

The final remote command ran with GNU timeout120s, kill-after5s (neither fired):

```text
python -m pytest -q tests/unit/optics/test_native_source_policy.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_benchmark_core.py tests/unit/benchmarks/test_optical_benchmark_contracts.py tests/unit/benchmarks/test_optical_render_migration.py
```

Observed **278 passed in10.30s**, actual exit0. A stdin launch probe printed
PID2055888, parent timeout2055867 and Linux `/proc` start tick83545224, then
exec'd the test interpreter in the same PID. Subsequent exact PID inspection
returned only its header: both absent. The parent SSH tool session1024 exited0;
local Win32_Process check found no matching validation SSH/Python residual.
Remote compute-process query was empty. Candidate status remained clean.

The initial frozen3a365ea remote collection observed105pass/2fail from the
interpreter-dependent AST pin, independently reproduced by reviewv1. Existing
contracts/migration separately passed87. Those initial failures are retained
in reviewv1 and notes; they are not silently counted as successful tests.

## Independent closure and retained comment

Fresh `/root/c4_c2c5_source_policy_review_v1` requested changes for F1-F3.
Fresh `/root/c4_c2c5_source_policy_review_v2` independently approved05d9ccf
with comments, including actual POSIX byte/file validation, exact valid-source
negative probes and175 focused passes on each interpreter. Both full reports
are archived verbatim. No candidate changes occurred during either review.

C1 remains a **non-blocking test-quality follow-up**, not an unreported pass:
two existing mutation cases replace `snapshots = []` globally and accidentally
make an earlier block unparsable. Their current green result is not proof of
valid write/escape rejection. The independent reviewer scoped each mutation to
the snapshot function, asserted parseability, and observed function-pin rejection
on both interpreters. Next full source-admission candidate must scope these
mutations and assert `ast.parse` success, with a fresh review of that test delta.

## State and hygiene

Both reviewer reports record their own process IDs, parentage, reaping and GPU
queries; no owned residual, timeout or hang was observed. No broad kill, GPU
reset or package installation occurred. This does not diagnose the historical
unidentified hang. The original remote user worktree robot_sim_c4c1b2 retains
its same four pre-existing changed/untracked files; only new detached worktrees
were created through its common Git database. An initial bundle fetch naming
a nonexistent branch failed before mutation; fetching its actual HEAD succeeded.

Old C0 files/allowlist/protocol, GPU preparation and all product/result gates
remain unchanged. Literal per-lane source catalogs, actual import ledger and
bootstrap admission still need implementation and review; this CPU primitive
is not an execution capability. C2d/e/f, full C4, B3/host payload, shipping,
a3b and benchmark result/baseline generation are not approved here.

Post-record transfer observation: while fetching the evidence-only9c5e12b,
Git automatic maintenance reported `bad object worktrees/robot_sim_c2c5_05d9ccf/HEAD`
and `failed to run repack` in the common database's gc.log. The fetch/new detached
worktree still succeeded. Subsequent read-only checks found both05d9ccf and9c5e12b
as valid commit objects, the expected worktree HEAD bytes, clean status and the
correct three-commit history. A separately executed
`timeout -k 5s 30s git fsck --connectivity-only --no-dangling` exited0 with no
diagnostics. The two approved code/test files are unchanged in the evidence HEAD.
The cause of the transient automatic-maintenance failure is not established;
gc.log and all worktrees were preserved, no repack/prune/reset performed. For
the final evidence transfer only, `git -c gc.auto=0 fetch` avoids another
background maintenance attempt without changing persistent Git configuration.
