# Approved single CPU entry: private-directory startup failure

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: actual bounded execution evidence; unsuccessful entry
Author: codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: setuptools-cpu-preflight review v2; review-request v3; setuptools-admission design v2
Owner Summary: The independently approved command executed once at311f16f but failed before loading any dependency. The shared target parent has setgid mode2775; mkdir0700 created a2700 directory, and the child's exact0700 check rejected it. The direct child exited1, was waited, and was absent afterward. This is not setuptools initialization success.

Exact reviewed/executed candidate:
`311f16f1cfec641957b1a688454bb834006ccbce`, target clean before/after.
Unchanged review v2 SHA256:
`9791a877970573ca6bbbbc671cb452b4db8959e317e52ced4469920212975c7c`.
Report archive-only local commit:
`a9dfc3f7f551eafece6f8313d1e7d15d187b7b30`.

Executed the exact approved command on tileops-dev:

```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/robot_sim_cuda_compilation_wip/tests/support/setuptools_dependency_process.py parent /data2/shared/ang.gao/robot_sim_cuda_compilation_wip /data2/shared/ang.gao/setuptools-cpu-311f16f1cfec641957b1a688454bb834006ccbce-v1
```

Parent exit1, child PID597340, child exit1, timed_out=false, waited=true. Child
stderr identifies its line69 exact mode/owner assertion. `identity.json`,
`failure.json` and `dependency.json` do not exist because this check precedes
identity writing, source bootstrap and the dependency try block. stdout is empty.
The child was checked absent in `/proc`; parent command completed and its guard
peer was closed by the existing finally. No native/GPU/compiler process ran.

Actual filesystem facts from `stat`:

```
/data2/shared/ang.gao mode=2775 uid=1006 gid=985 type=directory
/data2/shared/ang.gao/setuptools-cpu-311f16f1cfec641957b1a688454bb834006ccbce-v1 mode=2700 uid=1006 gid=985 type=directory
```

The child enforces `stat.S_IMODE(info.st_mode) == 0o700`; S_IMODE includes the
inherited setgid flag. Owner identity is correct. The creator's mkdir(mode0700)
does not clear a parent-inherited setgid bit. Keep this failed run directory and
all artifacts unchanged; do not chmod/retry it or reinterpret it as a pass.

Raw artifact archive on target and local c4_assets:
`setuptools-cpu-311f16f-entry-failure-v1.tar.gz`, SHA256
`f3ac562ac4b9848346813494038a699439a734242f583cf1181d7b4b44c23b9a`.
It contains request.json, parent.json, stdout.log and stderr.log for the original
run. The revised runner must use a new directory after a fresh exact review.

Earlier repair evidence at311f16f remains scoped CPU evidence: eight runner/
profile tests passed5.84s, released profile/blob/pin matched, all1,791 files passed
production verification; log `setuptools-311f16f-runner-cpu-v1.log` SHA256
`c0e9e2abe01db311c2b6b06a313a38eaccfa42297a9f9816df4cfd0cec3f162c`.
PID549345 and its stdlib-only descendant test PIDs550099/550102 were absent.
The prior intermediate bb29de1 probe had seven passes/one failure because its
hook expected string cwd;311f16f repaired the observed PosixPath case. None of
those checks exercised real dependency initialization or the setgid deployment
parent. Independent review v2 separately reproduced the repair and verified393
tests, one expected platform skip, and the full file profile before approving
the unsuccessful entry above.

The next bounded fix makes the newly created directory exactly0700 through its
owned open descriptor, preserving the child's strict check and refusal to reuse
a run. A real Linux setgid-parent regression is required. No import-policy or
production initialization change is implied. All later execution gates remain
closed; the one-shot preflight approval has been consumed by the recorded run.
