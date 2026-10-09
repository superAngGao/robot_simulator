# Private-directory creation repair before another CPU dependency run

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: independent bounded code/execution preflight
Author: codex
Version: v3
Date: 2026-10-09
Status: in_review
Related Files: setuptools-cpu-preflight review/request v2; test-evidence v1; accepted setuptools-admission design v2
Owner Summary: The previous approved single run failed before dependency import because a shared parent's setgid bit survived mkdir0700. Review the exact directory-creation fix and Linux regression before one new run. The child still requires exact0700; no dependency or later gate is weakened.

Previous approved code: `311f16f1cfec641957b1a688454bb834006ccbce`.
Evidence-only report archive: `a9dfc3f7f551eafece6f8313d1e7d15d187b7b30`.
The packet supplies the exact revised candidate HEAD. Same local repo, primary
target, interpreter and fixed dependency-only command from request/review v2;
use a fresh directory named for the new approved HEAD. Read AGENTS.md and the
shared collaboration/review principles, all prior findings and accepted design
requirements. Prior approval does not authorize a retry on changed code.

Read test-evidence v1 and original stderr/parent/request archive. The previous
command was executed exactly once and is unsuccessful: no source bootstrap or
setuptools initialization occurred. The failure is not a toolkit/package issue.

The only operative delta is `_create_run_directory` in the fixed test runner.
It requires an absolute normalized path, creates a new directory with mkdir0700
(existing path rejects), opens that directory with O_DIRECTORY|O_NOFOLLOW,
checks actual owner/type, applies fchmod0700 on the owned descriptor, verifies
current path dev/inode and exact mode, and closes the descriptor in finally.
It clears inherited setgid without touching the shared parent or any previous
run. Parent entry uses this helper; child validation remains unchanged.

The added Linux test creates a real setgid parent, first proves mkdir0700 alone
inherits2700, then calls the actual helper and requires0700, unchanged parent2700,
and rejection of reuse without altering an existing marker. Existing runner
source/denial/group tests remain required. No actual installed dependency entry
may run during this preflight; stdlib/filesystem/source-only tests are permitted.
The target parent remains2775; do not change its permission or prior artifacts.

Review the delta and affected integration without reopening already closed
findings by author assertion. The existing exact profile/catalog pins, narrow
Git subprocess allowance and CPU group ownership, actual production loaders and
RESTORED stop are unchanged. Input/device/toolchain remain explicit outer test
stand-ins. No COMPLETE, compiler, native, timing, readiness or C4 acceptance is
granted. Determine whether one new exact dependency-only run is ready.

Use a fresh independent session, freeze the candidate and write the full report
outside tracked files at
`C:/Users/AG/新建文件夹/c4_assets/setuptools_cpu_preflight_review_v3.md`, with
actual revision/tree/status, findings/reproductions, commands/results/limits,
standard verdict and exact approved or denied execution scope.
