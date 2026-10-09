Initiative: q54-optical-lab-p12-4c4c2d-compiler-process
Stage: implementation-note / bounded code preflight
Author: Codex
Version: v1
Date: 2026-10-09
Status: in_review
Related Files: compiler-supervision, compiler-job, compilation-admission design v3
Owner Summary: CPU process fixtures exercise the existing compiler supervisor;
  no CUDA build, readiness, timing or C4 completion is claimed.

## Scope and prerequisite

Base is ee76591e4f8705fe01eec0ca8142a5be1edd0f17. Exact frozen candidate HEAD
is provided in the independent review handoff. Accepted compilation-admission
design v3 (SHA256 028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2)
sections 6/8 require actual process ownership validation before cold CUDA builds.
The preceding installed-setuptools positive/32-failure matrix is archived in
setuptools-failures test-evidence v1; it did not execute this OS owner.

This candidate changes tests/support/cuda_compilation_process.py and adds
compiler_process_cases.py, compiler_process_campaign.py and compiler_sigaction_abi.c;
it extends tests/unit/optics/test_compiler_transport.py. Production process owner,
core, job schema, captured catalog, source policy and environment profile are
unchanged. Review must also inspect those production paths on which this runner
depends. Source/CPU tests do not prove Linux behavior.

## Actual path and cases

Each CPU fixture uses the existing cuda_compilation_process supervisor loop,
actual _CompilerJob, libc policy, pidfds, waitid/WNOWAIT, waitpid, transport,
core/outer deadlines and CompilerJobRecord parser. Its worker uses empty setup
guards0..5, compile-labelled guards6..8 and terminal9. Those empty operations
are explicitly not input admission or compilation. CPU fixture requests alone
use ACK1000ms/operation2000ms; actual CUDA compilation retains120000ms.

The fixed sequential campaign runs these 15 cases once, stopping on first
failure: normal, separate_pgid, leader_exit, leader_ack_exit, double_fork,
fork_during_cleanup, pipe_hang, missing_complete, residual_complete,
proc_failure, handle_failure, stale_identity, disappearing_identity,
signal_failure, parent_lost. No arbitrary child command or backend import is
provided. Test-only wrappers inject one declared OS error or identity mismatch
around the real adapter, then delegate to the real OS; these injected errors are
not claimed as naturally occurring kernel faults. Pending-ACK case suppresses
real output readiness at ARM6, preserving the earlier actual ACK deadline.
Ordinary child/grandchild cases use real fork, own PGID in the worker SID, actual
adoption and reaping. A sibling sentinel has its own SID and must remain alive
and identity-stable through supervisor completion; its only exit request is EOF.

Fixtures record actual kernel pidfd fdinfo, process observations, production
core/job records, sampler output, errors and stdout/stderr. Validation replays
the actual production records and checks member/reaped/adopted sets, unique
signal attempts, original deadline, immutable exited core, ECHILD, EOF, closed
pidfds and absent member PIDs. A normally worker-reaped child is kept separate
from supervisor reaping. Signal-denial fixture exits naturally after five seconds,
without a retry/second kill policy. Parent observes supervisor up to30 seconds;
if still alive it preserves the owner and writes incomplete.json. The campaign
must stop and inspect that owner, not kill it or retry. Persistent failure is not
advertised as recovered closure; existing CPU close-failure regression retains
the DRAINING owner.

These fixtures are temporary compilation validation support, not product
architecture. They follow the existing losing-lane cleanup gate. The owner's
required whole-C4 refactor remains before final C4 acceptance and must consolidate
actual duplication after functional and real verification.

## Independent ABI prerequisite and allowed preflight

Review is code preflight. First independently inspect source and execute CPU-only
tests. The accepted design explicitly requires an independent C/system-header ABI
probe before native supervisor execution. Reviewer may compile and run only the
included read-only compiler_sigaction_abi.c as that prerequisite: it reads the
inherited SIGCHLD action and GET_CHILD_SUBREAPER; it never sets either. Do not run
_CompilerJob/_SubreaperPolicy or the15-case campaign during preflight, and do not
import Torch/Warp/NumPy/SciPy/setuptools or execute CUDA compilation. After exact
preflight approval, author may run this frozen15-case CPU campaign once, using
the reviewer's retained ABI JSON. No broader native/GPU permission is inferred.

Target: tileops-dev, isolated /data2/shared/ang.gao/robot_sim_cuda_compilation_wip;
Python /home/ang.gao/miniforge3/envs/robot_sim/bin/python3.12.13. Use -I -S -B.
System ABI command (fresh reviewer-owned0700 output directory outside repo):

```
env -i PATH=/usr/bin:/bin LC_ALL=C /usr/bin/gcc -std=c11 -Wall -Wextra -Werror \
  tests/support/compiler_sigaction_abi.c -o "$probe_directory/sigaction_abi"
env -i PATH=/usr/bin:/bin LC_ALL=C "$probe_directory/sigaction_abi" > "$probe_directory/abi.json"
```

Use system gcc and headers, not Conda's sysroot. Independently hash source,
compiler, binary, included headers and libc; retain command/output. Expected
pointer8, ulong8, struct152, offsets0/8/136/144, sigset128, SIGCHLD17,
SA_NOCLDWAIT2, PR_SET/GET36/37, default_handler=true, no_cldwait=false,
initial subreaper0. Reconcile with production ctypes layout and verify no source
or host drift before allowing the supervisor. Metadata-only author reads found:

- /usr/bin/gcc -> /usr/bin/x86_64-linux-gnu-gcc-13;
  SHA256 1b99826121ae6682a634e5efe09bd3e3df58ce58e0b28f849114ab5b89139c26.
- libc /usr/lib/x86_64-linux-gnu/libc.so.6 size2125328;
  SHA256 d8db8739a1633c972cec6a4fe0566bdcec6fd088f98723492ab0361f66238f75.
- bits/sigaction.h b2f76a26bbfa3668859e8e8a93af9321e14083914df0aa0367cdffaabf4ece81;
  bits/types/__sigset_t.h 05291a7d38bb1d0da1ed01c09fe85fa2b1f4d5bc708e06bb930efec672db08d5;
  sys/prctl.h a76908ae4f9cc728b69720ede3630ed2df5462e85d03fe949b48f269ad3dade6;
  linux/prctl.h 6feff68eef81225d5885ed3dd6e0644b6ec4881591e219041fcec87a67040e42.

Post-approval invocation, fresh directory and exact reviewed target HEAD:

```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B \
  tests/support/compiler_process_campaign.py "$campaign_directory" "$reviewer_abi_json"
```

## Author verification and remaining gates

Windows Python3.14 CPU-only tests: compiler_job, compiler_transport and
compiler_process_unwind, with plugin autoload disabled and no pytest cache:
29 passed in0.43s. New counterexamples check malformed fixture rejection before
native constructor and actual transport withholding of pending ARM6 ACK through
leader exit. git diff --check passed. No C probe or real OS job has run yet.

Real process evidence and whole checkpoint review remain pending. This preflight
does not approve actual cpp_extension loading, the three CUDA cold builds, native
observer, device execution, G1..G4, C2e/f, C4, measurements or backend selection.
