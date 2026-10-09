# Actual installed setuptools dependency initialization

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: actual bounded CPU execution evidence
Author: codex
Version: v2
Date: 2026-10-09
Status: draft
Related Files: setuptools-cpu-preflight review v3; setuptools-admission design v2; test-evidence v1
Owner Summary: The exact independently approved CPU run succeeded with actual installed setuptools 83.0.0 and build_ext, reaching RESTORED. This proves the positive dependency-loading path only. Real CUDA compilation, native readiness and performance remain unproved.

## Exact execution and review

Reviewed and executed code: `0177a869aec966da2e112d5f92120892aa2fb1fa`.
Fresh independent reviewer `/root/setuptools_cpu_preflight_v3` approved one
dependency-only execution, stopping at RESTORED. Unchanged report SHA256:
`2db97ef049de097a1d750244c2f5bb9be14ddf726ec22664fa4debf0c8f74c7c`.
Its archive-only local commit is `e352f35864bf53d0e2ef3ab1dab7051244d9cd3f`.

Exact command on tileops-dev, Python 3.12.13:

```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/robot_sim_cuda_compilation_wip/tests/support/setuptools_dependency_process.py parent /data2/shared/ang.gao/robot_sim_cuda_compilation_wip /data2/shared/ang.gao/setuptools-cpu-0177a869aec966da2e112d5f92120892aa2fb1fa-v1
```

Parent PID650402 completed with exit0. Child PID650414 exited0, waited=true,
timed_out=false. Both PID paths were absent in /proc at completion and again
when recording this evidence. The target checkout is still clean at the exact
executed revision. The run directory is mode0700, uid1006, gid985. Both output
logs are empty; there is no failure.json. The earlier failed311f16f run and its
inherited2700 directory remain separate historical evidence.

## Observed positive behavior

The real production source session, external finder/loaders, initialization
owner, guardian and guarded transport loaded the installed dependency sources.
The runner asserted the actual same-object distutils/setuptools._distutils
alias, setuptools and setuptools.version version83.0.0, both original shim
bindings restored, original code/closure identities, and unchanged singleton
finder. sys.path kept its original three entries and appended only the exact
setuptools/_vendor directory. Source, admission and retention audits passed;
all admitted ordinary file records completed.

dependency.json records state RESTORED, one metadata read and one resource
read with closed handle, 75 ordinary file completions, 381 external events,
36 repository trace entries and 1,165 journal events. Initialization actions:

```
begun, hooked, shim_entered, alias_spec, alias_created, alias_executed,
alias_published, shim_left, hooks_restored, resource_read, metadata_read
```

There is no completed action. Resource SHA256:
`37fedcffbf73c4eb9f058f47677cb33203a436ff9390e4d38a8e01c9dad28e0b`.
Released LF environment profile SHA256:
`0cb6288345fc3a2644e61bb01dc6e5897ee58d52eb71050c31b2e76f8ea60753`.
Source catalog SHA256:
`17387af6b8188c40707da7e1835f639e87d1464ffa664b7cdcdb301eedd9d399`.

Terminal handling used the existing host-only cancellation path, with
disposition cancelled_before_submission_exit and guardian RELEASED. No device
submission or synchronization ran. The declared descriptive flags remain
backend_imported=false, compilation_complete=false,
native_scope_observed=false and readiness_issued=false.

## Raw evidence and limitations

Archive on target /data2/shared/ang.gao and local c4_assets:
`setuptools-cpu-0177a86-restored-v1.tar.gz`, SHA256
`18ce11bb63b7f12cdde7939a743cedb820d4b104e4bc0a0f32c1148db7df0599`.
It contains the request, process identity, parent outcome, dependency observation
and both logs. Local and target archive digests match.

The runner deliberately uses outer input/device/toolchain stand-ins. This is
not a compilation receipt, actual uploaded-input-to-compiler integration,
COMPLETE, native observation, CUDA execution or performance evidence. The
single-run preflight approval has been consumed. Actual dependency failure
paths, full compilation integration, exact compiler/native preflight and the
required real cold-build/process/device matrices still precede checkpoint
acceptance. C4 and the owner's subsequent whole-C4 refactor remain unfinished.

This document and the associated PROGRESS entry record the unchanged approved
execution and its limits only; they do not change code, requirements or gates.
