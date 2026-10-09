# Revised exact CPU dependency execution preflight

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: independent code/execution re-preflight, dependency-only
Author: codex
Version: v2
Date: 2026-10-09
Status: in_review
Related Files: setuptools-cpu-preflight review-request/review v1; setuptools-admission design-request/design-review v2; compilation-admission design-request/design-review v3
Owner Summary: Repair the two independently reproduced startup blockers using the actual Git-release profile digest and a precise allowance for source-revalidation Git commands. Account for those owned CPU descendants. Re-review the exact frozen candidate before any real dependency run; no GPU/compiler gate is opened.

## Candidate and requirements

Previous reviewed HEAD `32fba6096f6425e70a4b671d2e14800dd10c2f8f`.
Unchanged original implementation base `52fb36a8960596ddb2ac3decd8d129188e4b1eb4`.
The packet supplies the full revised HEAD containing this request, runner and
new regression tests. Repository remains
`C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`; target is
`/data2/shared/ang.gao/robot_sim_cuda_compilation_wip` on tileops-dev.
The predecessor report is preserved unchanged in review v1, SHA256
`ceaba8a08a3d8c676d890182725be6157629c004914f402b871d2080612d6774`.
Read its original R1/R2 reproductions independently; author repair statements
below are not proof. The accepted designs and complete execution/outer-stand-in/
RESTORED-stop requirements in request v1 still apply, except the process
restriction and ownership statements expressly corrected below.

## R1: exact released profile bytes

The runner now pins the target Git LF blob SHA256
`0cb6288345fc3a2644e61bb01dc6e5897ee58d52eb71050c31b2e76f8ea60753`.
The historical authoring/evidence bytes were CRLF SHA2561725bfd and are unchanged;
their parsed content matches the LF profile but their raw byte identity does not.
No runtime normalization or alternate-digest acceptance was introduced. A new
test joins the actual Git blob to the runner constant and on POSIX also requires
the real working-file bytes to match. The prior reviewer independently verified
the actual LF profile's1,791 installed files at the previous clean HEAD; repeat
the release pin/schema/file check on the revised exact target candidate.

## R2: required source revalidation and CPU process lifetime

The test-only audit closure admits exactly these two subprocess.Popen requests:

```
executable='git', argv=['git','rev-parse','HEAD'], cwd=exact_root, env=None
executable='git', argv=['git','show',exact_revision+':optical_worker/data/native_sources/v13.json'], cwd=exact_root, env=None
```

The actual target Linux audit event must have a list argv and string cwd. Any
other command, option, executable, revision, directory or environment rejects.
Backend imports and os.system/fork/posix_spawn/ctypes.dlopen still reject. The
restriction remains installed during binding and all later production source
audits. Production loaders, audits and source policy are unchanged by this fix.
This intentionally supports the target Linux ABI only; the real driver already
rejects other platforms. Windows Popen events have different argument forms and
are not added as a compatibility branch.

The v1 statement that no descendants can launch was false. Source revalidation
uses the production synchronous subprocess.run(timeout20s) Git calls. The parent
now starts the CPU child in a new session/private process group. On timeout,
failure or final cleanup it signals only that owned group and waits for its
direct child (10s). It does not kill by name, scan unrelated processes, use a GPU
reset, install subreaper/sigaction code or claim compiler-tree wait ownership.
Descendants killed after their direct parent dies belong to the OS reaper; the
CPU helper proves they cease running, not that it performs their waitpid. The
later compiler-specific sole-wait/subreaper proof is unchanged and still closed.
The wrapper avoids a second group cleanup after the timeout cleanup succeeds.

## Repair tests and permitted verification

`tests/unit/optics/test_setuptools_dependency_runner.py` adds:

- exact Git blob/runner/target-byte equality;
- finite process-command/env/cwd/revision and backend/library denials;
- actual cold source install/verify, real custody binding and repeated production
  source audits with the real revised hook installed, without package execution;
- two Linux owned-process-group cases: direct parent still running, and direct
  parent already exited leaving its stdlib-only descendant. The helper stops the
  group and waits the direct child; the assertion distinguishes a dead zombie
  awaiting the OS reaper from a still-running descendant.

Windows check passed2/skipped3 (0.57s); Linux cases remain mandatory and are
provided in the final frozen packet. An initial Windows attempt at the Linux
binding test failed because the Linux hook rejects Windows Popen event encoding;
the test now explicitly skips that unsupported platform instead of broadening
the production restriction. No actual installed dependency runner was executed.

Other permitted tests/source archives/closed gates follow request v1. A dedicated
fresh review is required before the actual command:

```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/robot_sim_cuda_compilation_wip/tests/support/setuptools_dependency_process.py parent /data2/shared/ang.gao/robot_sim_cuda_compilation_wip /data2/shared/ang.gao/setuptools-cpu-<approved-head>-v1
```

Do not run that dependency entry during this preflight. Inspect/test the fixes,
actual source/profile/driver boundary and process lifetime; no desired verdict
is implied. Whole implementation/native/compile/performance/C4 acceptance remains
unproved. Use a fresh independent session, freeze the candidate and write the
complete report outside tracked files at
`C:/Users/AG/新建文件夹/c4_assets/setuptools_cpu_preflight_review_v2.md`.
