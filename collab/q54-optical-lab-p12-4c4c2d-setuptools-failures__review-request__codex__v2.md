# Dependency failure-matrix quarantine and partial-operation repairs

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: fresh independent code and execution re-preflight
Author: codex
Version: v2
Date: 2026-10-09
Status: in_review
Related Files: setuptools-failures review/request v1; accepted setuptools-admission design v2; compilation-admission design v3
Owner Summary: Repair both reproduced driver defects without changing production code: require failed-guard quarantine, and inject partial alias faults inside the real owner's protected post-delegate checks. Review the exact repairs and same finite 33-run CPU scope before actual dependency execution.

Original candidate: `44a57da4d80c1372f12ccc2478be981e9b5afee8`.
Unchanged full independent review v1 SHA256:
`9d7892c6bce0cf9ac83dbce1482c5f125135945facd4dceff05a9616a2f02a90`.
Report-only archive commit: `82adad5668a77b800d7f08adfe3d32f3d730a0f3`.
The handoff supplies the revised candidate HEAD. Candidate files are the CPU
driver, its runner/helper unit tests and this request. All production sources,
profile/catalog/dependency bytes and accepted contracts remain unchanged.

Read AGENTS.md and the shared collaboration/review principles, original report
and its complete original counterexample/reproduction. Original probe remains
unchanged in c4_assets/setuptools_failures_preflight_probe_v1.py, SHA256
`4b165676acd12d5b484a40d40629ca3fabe0a75d69176925b0ca02875ad547a6`.
Do not execute any installed dependency during review. Preserve the original
probe as historical evidence; its assertions intentionally establish the old
bad behavior and need a separate adapted check for repaired behavior.

R1: the negative validator now requires quarantined_process_exit,
TRANSFERRED_TO_QUARANTINE, not_authorized release, full_quarantine custody,
failed guard6, six earlier completions, no terminal arm, no device work or sync.
It checks actual owner and compile-root objects remain in guardian custody.
The healthy positive branch is unchanged. New production-controller/real-peer
regression accepts the failed-guard observation and explicitly rejects the
healthy cancellation with the same validator. Parent-owned hard exit/wait and
post-run PID absence remain part of actual matrix evidence. This supersedes
v1 request's erroneous negative “host-only cancellation” expectation; failure
quarantine must not be replaced by ordinary release.

R2: partial_create now raises at the real post-delegate intermediate metadata
check, before alias_created. partial_exec raises at the second non-intermediate
metadata check while ALIAS_CREATED: after the actual delegate call, before
alias_executed. Both injections stay inside the production alias method's own
try/except, so its fail path latches the original error before shim unwind.
The driver requires the corresponding success action absent. Adapted unit
regressions use the original source/alias fixture and real owner, _AliasLoader,
module_from_spec and shim removal; they assert actual delegate calls, exact
first error, returned-source retention, sticky FAILED, restoration and no
inappropriate success event. Fixture limits remain explicit; these are not
installed-source proof.

Local Python3.14 runner/helper regression:29 passed,4 Linux-only skips11.29s.
Target source-only regression and exact released hashes are supplied separately.
The prior independent v1 source-only run33pass and profile1791 file check remain
their historical scope; the proposed installed matrix has not executed.

Requested execution is unchanged: one positive and one of each32 fixed negative
selectors listed in request v1, serial cold invocations of the same fixed parent
command. Substitute the full newly approved HEAD in a fresh campaign directory
and every case subdirectory. No old directory reuse, no changed-code retry,
and no compiler/backend/native observer/GPU/COMPLETE/readiness/performance gate
opens. Actual matrix outcomes still require inspection, raw archives, process
hygiene and subsequent checkpoint review. C4/refactor remain unfinished.

Freeze the candidate. Write the complete new independent report outside tracked
files to C:/Users/AG/新建文件夹/c4_assets/setuptools_failures_preflight_review_v2.md,
with original finding dispositions, affected invariant checks, exact revisions,
commands/results/limitations, standard verdict and exact execution scope.
