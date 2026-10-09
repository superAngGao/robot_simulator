Initiative: q54-optical-lab-p12-4c4c2d-compiler-process
Stage: implementation-note / fixed deployment pidfd binding
Author: Codex
Version: v3
Date: 2026-10-09
Status: in_review
Related Files: implementation-note v1/v2; test-evidence v1; accepted compilation-admission design v3
Owner Summary: The existing exact-libc supervisor adapter now binds pidfd_open
  because the fixed Python build omits os.pidfd_open. The owner/deadline/wait
  architecture and case matrix are unchanged; fresh ABI/code review is required.

Base operative candidate0497573043c66186a912ed6998722d43677da3f7, separate v2
report archiveeee75b7. The failed actual run and native-symbol/header diagnosis
are retained in compiler-process test-evidence v1. Exact new HEAD is supplied
in the fresh independent preflight packet.

This is a bounded implementation repair within accepted design v3 section6's
dedicated Linux supervisor facility and required pidfd ownership. It changes the
platform binding for the already required operation, not ownership, deadlines,
cleanup authority or the worker's native boundary. There is no runtime optional
fallback and no second process implementation: every _CompilerJob._acquire uses
the existing pinned libc policy's open_pidfd. The same libc file/digest is checked
before loading; captured workers never import this adapter. No package change,
system-Python substitution or broad PID signalling is introduced.

Exact ABI: libc.pidfd_open, argtypes=[ctypes.c_int,ctypes.c_uint],
restype=ctypes.c_int, use_errno=True on the held CDLL. The method accepts only
an exact positive Python int below2**31; it passes flags0 unconditionally.
Negative result raises OSError from ctypes.get_errno immediately. Nonnegative
FD returns to the existing before/after PID/starttime/PPID checks; mismatch or
disappearance closes that FD and does not add an owned handle. Successful handles
retain the same lifetime. Signal delivery stays signal.pidfd_send_signal with
the original unique-attempt and positive ownership constraints. os.waitid/waitpid
and WNOWAIT remain required. No generic library/function/syscall binding is exposed.

The native ABI probe adds sys/pidfd.h and compile-time checks that pid_t is signed
int, both int/unsigned int are four bytes, and pidfd_open's prototype is exactly
int(pid_t,unsigned int). It still executes only the existing read-only SIGCHLD
and GET_CHILD_SUBREAPER queries; it never calls pidfd_open, sends a signal or sets
policy. Its source changed, so the fresh reviewer must independently inspect,
compile and run that fixed probe once with system gcc/headers before approving
any new OS campaign. Retain new source/binary/transitive-header hashes and raw ABI
JSON; output layout is unchanged. Verify current libc exports the exact symbol.
No runtime constructor/campaign/backend/GPU is permitted during that preflight.

The fixture observer now wraps job.policy.open_pidfd instead of the absent os
attribute. It still delegates to the actual operation except the one declared
EMFILE injection and records actual kernel fdinfo. All15 cases and the R1 oracle
are unchanged. Existing proof of unrelated sentinel survival, actual wait status,
adoption, original deadline, exited-core immutability and reaping remains required.

New CPU regressions exercise the actual fixed method and _acquire while removing
os.pidfd_open: flags0, ESRCH/EMFILE/EPERM propagation, rejected invalid PIDs,
unchanged identity success and changed/disappearing identity with exactly one FD
close and no retained handle. C return/FD facts are explicit CPU stand-ins, not
native evidence. Windows Python3.14 -B, plugin autoload/cache disabled, the same
four compiler suites now pass52 in1.03s. Source diff --check passes.

Requested next scope after independent approval: one fresh sequential15-case CPU
campaign at the new exact revision using the new independent ABI JSON; stop on
first failure, preserve owner and outputs, no implicit retries. The previously
failed run cannot be relabelled or resumed as this code. Production source change
is confined to compiler_supervision.py outside the captured worker catalog, so
source catalog/policy/environment digests are unchanged. Full compilation/native
observer preflight, real three-recipe builds/failures, whole checkpoint review,
later C4 milestones, whole-C4 refactor and performance gates remain outstanding.
