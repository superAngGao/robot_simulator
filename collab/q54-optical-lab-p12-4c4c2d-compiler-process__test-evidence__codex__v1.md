Initiative: q54-optical-lab-p12-4c4c2d-compiler-process
Stage: actual first CPU process campaign / stopped before worker launch
Author: Codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: implementation-note v2; independent review v2
Owner Summary: The approved first run stopped in the first supervisor constructor:
  this pinned Conda Python lacks os.pidfd_open. No worker or GPU started. The
  failed run is preserved; no retry was performed under its consumed approval.

Exact executed clean target HEAD0497573043c66186a912ed6998722d43677da3f7.
Fresh reviewer /root/compiler_process_preflight_v2 approved this one campaign;
complete report SHA256 ecb3b0c8df129cb6b89e64cbd4739ac446c95da08e9faa37c3832412b5a8d2cf
is archived unchanged in separate evidence commit eee75b7. Target stayed on0497573.

On tileops-dev at /data2/shared/ang.gao/robot_sim_cuda_compilation_wip, after
clean-HEAD verification and rechecking all original ABI/compiler/header hashes:

```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -u -I -S -B \
  tests/support/compiler_process_campaign.py \
  /data2/shared/ang.gao/compiler-process-0497573-matrix-v1 \
  /data2/shared/ang.gao/compiler_process_preflight_v1.zTQ8bY69/abi.json
```

The campaign exited1. Only normal was attempted, in0.1917875949293375seconds;
its parent and supervisor exited1. The traceback is GuardProtocolError:
`required Linux process APIs absent` at compiler_supervision.py's initial API
check. This occurs before libc policy construction, worker launch, pidfd creation
or subreaper setup. supervisor_identity.json/worker artifacts do not exist.
The next14 cases did not run. No process-tree correctness pass follows.

Actual IDs: campaign998766, case-parent998772, sentinel998777, supervisor998778.
Parent observed sentinel alive with unchanged PID/starttime/SID after supervisor
failure, then closed its private pipe; sentinel exited0 and was waited. All four
PIDs are absent in the post-run inspection. Earlier read-only inspectors999685
and1000698 also were absent. No kill, fallback, device query or GPU work occurred.

Raw archive on target /data2/shared/ang.gao and local c4_assets:
compiler-process-0497573-matrix-v1.tar.gz, SHA256
9174dfd998b88f655aadd749f424776fe531c1a6a0bc5111b7bf83140ed06419.
It includes the complete campaign directory, stdout/stderr log and inspection.
Separate compiler-process-0497573-inspection-v1.json has SHA256
d87cc21defd23fd2ed52470c48e0c89c8144897830711928b3ea99eb6c3a57af.
Inspection PID1002735 completed synchronously and is checked absent before the
next candidate synchronization. Raw per-file hashes are in this inspection.

Metadata-only diagnosis (no pidfd/native job invocation): Python3.12.13 from
Anaconda, kernel6.8.0-85-generic; os.pidfd_open=False while os.waitid/waitpid/
WNOWAIT/P_ALL and signal.pidfd_send_signal are present. Installed system header
/usr/include/x86_64-linux-gnu/sys/pidfd.h declares
`int pidfd_open(pid_t, unsigned int)` and documents fixed flags0.
Header SHA256 fcd96ba6b05d59ccf36317ccad56c586300b5b9c16c9327b7018224eb97b58ec.
Read-only readelf --dyn-syms on the already pinned libc finds
pidfd_open@@GLIBC_2.36 (and pidfd_send_signal@@GLIBC_2.36). This is symbol/header
evidence, not a successful syscall. No dependency installation or interpreter
replacement is needed or authorized by this record.

Next repair remains within the accepted dedicated-supervisor architecture:
bind the required pidfd-open operation through its existing exact-libc adapter,
with a fixed ABI, no arbitrary syscall interface and no alternate fallback path.
An updated independent code/ABI preflight precedes any new OS campaign. Actual
compiler loading, cold builds, device work, readiness, performance and full C4
remain unproved. The failed trial and earlier review reports are preserved.
