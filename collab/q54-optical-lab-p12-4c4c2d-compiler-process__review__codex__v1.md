Initiative: q54-optical-lab-p12-4c4c2d-compiler-process
Stage: independent code preflight
Author: Codex independent reviewer
Version: v1
Date: 2026-10-09
Status: in_review
Related Files: compilation-admission design-request v3; compiler-process implementation-note v1; AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md
Owner Summary: Request changes. Independent read-only system-header ABI validation and 29 CPU tests passed, but the campaign oracle accepts ineffective descendant termination followed by natural exit. The fixed 15-case OS campaign is not approved to run.

Reviewer session: /root/compiler_process_preflight_v1, fresh independent task session; not the candidate author.
Base: ee76591e4f8705fe01eec0ca8142a5be1edd0f17
HEAD: ef6c8df717f574684e88cadbeb48222e7916d3a6
Local candidate: C:\Users\AG\新建文件夹\p12_4_c4_warp_timeline_design
Target: tileops-dev (primary), /data2/shared/ang.gao/robot_sim_cuda_compilation_wip
Actual target checkout: detached HEAD at the specified commit; git status --porcelain empty before and after verification. Local HEAD matched and working tree was also clean before and after review. No candidate edits.
Requirements digest: compilation-admission design-request v3 SHA256 028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2.

Verdict: Request changes.

1. Blocking finding R1 — High: kill-required process cases can pass after ineffective signals and natural descendant exit.

Location: tests/support/compiler_process_cases.py:71, :102, :329, :345–347, :383–398; related per-case checks at :400–412.

Trigger: in leader_exit (also applicable to leader_ack_exit, double_fork, fork_during_cleanup and pipe_hang), an adapter records signal_attempted but its pidfd_send_signal operation returns without terminating the child. The child subsequently reaches the fixture's declared five-second natural exit.

Consequence: the oracle accepts the result as a validated CLOSED process fixture even though the descendant survived the intended SIGKILL. This masks the specific real OS termination behavior that the new campaign is meant to validate before cold compilation. This finding is about the test's discriminatory power; I did not observe a production kernel signal failure or execute an OS job.

Why existing checks miss it: the validator reduces child_reaped events to identities, dropping their wait_status for these cases. It checks unique attempts, original deadline, eventual absence, adoption, ECHILD/EOF and closure, but requires neither a signalled descendant wait status nor absence of the natural-exit marker. The production replay correctly permits a legal natural exit after a recorded attempt; that general replay contract cannot supply the fixture's stronger expected outcome. Five-second natural fallback is useful for finite cleanup, but its occurrence must fail a case that claims effective termination. signal_failure is the explicitly different case where the natural exit is expected.

Independent reproduction: outside-repo noop-repro.py constructs a real production GuardSupervisorCore and _CompilerObligation, runs wire guards0..5 and ARM6, records normal leader exit, child adoption, unchanged deadline expiry and one signal_attempted, then records child wait_status=0 at five seconds and valid closure. It passes these production records and explicitly synthetic member/pidfd observations to the actual validate_result. No process is forked, no native constructor is called, and no signal is sent. This models the no-op signal mutation; it is not presented as real OS evidence or a forged-evidence trust-boundary exploit.

Observed raw output:
```
COUNTEREXAMPLE ACCEPTED: leader_exit validated closed despite child natural wait_status=0 after ineffective SIGKILL attempt.
{"attempt_count": 1, "case": "leader_exit", "closed": true, "member_count": 2, "reaped_count": 2, "scope": "OS fixtures only; no compilation or readiness"}
```

Retained repro and raw inputs/results:
- Target: /data2/shared/ang.gao/compiler_process_preflight_v1.zTQ8bY69/noop-repro.py
- Target raw: same directory/repro.stdout, repro.stderr, noop-signal-counterexample/ (README, request, supervisor record, member rows, natural-exit row, accepted validated.json).
- Original local script: C:\Users\AG\新建文件夹\c4_assets\compiler_process_preflight_v1_noop_repro.py
- Full local raw copy: C:\Users\AG\新建文件夹\c4_assets\compiler_process_preflight_v1.zTQ8bY69

Repair direction: define the expected actual exit mechanism for each fixed case and bind each required killed identity to its real wait status (SIGKILL, returncode -9), unique attempt and appropriate ordering. Assert exact intended descendant roles/counts so missing work cannot silently reduce coverage. Preserve intentional natural exits separately: normal/separate_pgid, leader exit, a worker-reaped double-fork intermediate, and the denied-signal worker. A natural fallback in a kill-required case should retain evidence and fail validation. Check the injected observation/handle/identity recovery cases against their intended worker result as well; their current checks mostly establish injection and failure recording.

Proof required after repair: the preserved CPU counterexample must reject through the real validator, with a matching production-record SIGKILL positive accepted. Extend that meaningful oracle regression to each kill-required case or a common enforced exit-policy boundary. A fresh independent review must bind any subsequent execution authorization to the new exact revision. Only after that approval should the fixed campaign run once, retaining real statuses and residual inspection. No new OS execution was necessary or performed to establish R1.

2. Production flow and risk inspection.

I read the full shared protocol/principles, accepted design sections6/8 (and remaining-gate text), candidate implementation packet and complete six-file diff. I inspected the production compiler_supervision, compiler_job replay/obligation, GuardSupervisorCore and GuardSupervisorTransport; the driver dispatch, runtime-driver writer and dependency-runner directory helper were checked beyond the diff. The earlier setuptools evidence explicitly disclaims compiler OS ownership; its results were not reused as proof of this campaign.

Observed design alignment:
- The fixed worker dispatch enters the CPU fixture before captured runtime child work. Supervisor-side dynamic runtime-driver loading is stdlib-only at import; it uses the existing evidence writer. The fixture directory helper is imported without invoking setuptools. There is no arbitrary child command hook or opened product/backend gate in this delta.
- _CompilerJob retains Popen before constructor invocation, anchors worker PID/starttime/SID, acquires a pidfd before transport ARM, and is the sole waitid/WNOWAIT + waitpid consumer for its tree. Parent wait calls target the supervisor; campaign wait calls target the case parent. Descendants use separate PGIDs within the worker SID. The sibling sentinel has its own SID and is outside the supervisor's descendant ownership.
- _SubreaperPolicy uses the exact inventoried libc and read-only SIGCHLD query; the allowed C probe does not set a handler or subreaper. Native setup is confined to the future dedicated supervisor.
- JobObligation freezes the preceding effective deadline before core exit/termination; outer expiry continues after EXITED, without ticking or appending to the exited core. The pending-ACK fixture withholds real transport output after ARM6 and records the actual pending response. It does not substitute a longer post-exit timeout.
- PID/starttime/PPID checks bracket acquisition; signalling uses held pidfds only. Unique attempted identities remain sticky after signal failure. Production closure requires actual worker/adopted reaping, policy readback, ECHILD, retained identity absence, output EOF and handle closure.
- COMPLETE6/7/8 passes through registered exact owner interception before core acceptance and ACK; residual session members withhold DISARMED. Ordinary transport retains the same framing, fence and deadlines.
- Finite error wrappers delegate to real OS adapters except the declared single injection. A denied signal has a deliberate natural-exit path. An observation timeout writes incomplete.json and preserves the supervisor rather than killing its wait owner; the campaign stops on failure and records remaining known PIDs. Persistent close failure keeps DRAINING custody in the existing CPU regression.
- The raw-result validation does check immutable exited-core evidence, adopted/reaped membership, worker-reaped child separation, unique attempts, original ACK/operation deadlines and PID absence. These checks do not close R1.

No additional substantive finding was established in this bounded review. Source inspection and CPU models do not prove actual Linux subreaper, adoption, signal or reaping behavior.

3. Independent permitted verification.

Target interpreter: /home/ang.gao/miniforge3/envs/robot_sim/bin/python
Observed version: Python3.12.13, Anaconda package, GCC14.3.0.
All reviewer Python commands used -I -S -B. The test harness explicitly adds the repository and existing Python3.12 site-packages to sys.path without executing site startup/.pth files. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and -p no:cacheprovider were used. Temporary pytest output is outside the repository.

CPU checks command:
```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/compiler_process_preflight_v1.zTQ8bY69/checks.py
```
checks.py invokes pytest.main with:
```
-q -p no:cacheprovider
--basetemp=/data2/shared/ang.gao/compiler_process_preflight_v1.zTQ8bY69/pytest-tmp
tests/unit/optics/test_compiler_job.py
tests/unit/optics/test_compiler_transport.py
tests/unit/optics/test_compiler_process_unwind.py
```
Observed: 29 passed in0.15s, exit0; checks.stderr empty.

The harness rejects torch/warp/numpy/scipy/setuptools imports, replaces both native constructors with explicit failures, and rejects subprocess.Popen, os.fork, os.system and ctypes.dlopen audit events during tests. ctypes was imported first to inspect the structure layout; no _SubreaperPolicy or _CompilerJob native setup ran. The loaded-module list is retained as modules-after-tests.json. The transport fixtures instantiate inert object.__new__ owners, and unwind tests use mocked process/FD operations.

Counterexample command:
```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/compiler_process_preflight_v1.zTQ8bY69/noop-repro.py
```
Observed exit0 and the acceptance output quoted in R1. This successful counterexample is evidence of the oracle defect.

Local git diff --check ee76591e4f8705fe01eec0ca8142a5be1edd0f17 HEAD passed. Both candidate trees remained clean at exact HEAD.

4. Independently inspected and executed read-only C ABI prerequisite.

Evidence directory:
- Remote: /data2/shared/ang.gao/compiler_process_preflight_v1.zTQ8bY69
- Local retained copy: C:\Users\AG\新建文件夹\c4_assets\compiler_process_preflight_v1.zTQ8bY69

Created with mktemp -d and umask077 outside the repository. The shared parent inherited a setgid directory bit (mode02700); this was explicitly cleared with chmod g-s, leaving verified mode0700. There was never group/other access.

After inspecting every line of the included C source, I compiled only that source using the system compiler and installed system headers:
```
env -i PATH=/usr/bin:/bin LC_ALL=C /usr/bin/gcc -std=c11 -Wall -Wextra -Werror -MD -MF "$probe_directory/headers.d" tests/support/compiler_sigaction_abi.c -o "$probe_directory/sigaction_abi"
env -i PATH=/usr/bin:/bin LC_ALL=C "$probe_directory/sigaction_abi" > "$probe_directory/abi.json"
```
The added -MD/-MF only retains the actual transitive header list. Compiler stdout/stderr are empty. Probe exited0. The first shell transcript's final display-only cat command encountered a trailing Windows CR in its filename argument; compilation, execution and retained hashes had already succeeded. Subsequent reads verified all retained files; no compilation or probe execution was repeated.

Raw abi.json:
```json
{"pointer_size":8,"ulong_size":8,"size":152,"offsets":[0,8,136,144],"sigset_size":128,"sigchld":17,"no_cldwait":2,"set_subreaper":36,"get_subreaper":37,"default_handler":true,"no_cldwait_active":false,"subreaper":0}
```

Independent ctypes comparison on the target imported only the inert production structure and checked pointer8, ulong8, size152, offsets[0,8,136,144], sigset128. Result PASS; ctypes-layout.json retained. The source's 17/2/36/37 constants agree with the native C output and the production signal/flag/prctl usage. This validates inherited policy in the probe process, not future supervisor enable/readback.

SHA256 evidence:
- C source: 283e0ba7c7edb6eadfbbed4de5961d8e8b78bf26f3c7246bf1ed50798bcdcd7e
- /usr/bin/gcc: 1b99826121ae6682a634e5efe09bd3e3df58ce58e0b28f849114ab5b89139c26
- compiled sigaction_abi: 558d662991f84aa4cca2c98ff241fa67e011bdc9b4e76a68ccab24e0df240452
- /usr/lib/x86_64-linux-gnu/libc.so.6: d8db8739a1633c972cec6a4fe0566bdcec6fd088f98723492ab0361f66238f75
- raw abi.json: 037752b995cd2f1f82caa698ad655ca4b6aa8f7a5da00bd037905fd7ee19656b
- bits/sigaction.h: b2f76a26bbfa3668859e8e8a93af9321e14083914df0aa0367cdffaabf4ece81
- bits/types/__sigset_t.h: 05291a7d38bb1d0da1ed01c09fe85fa2b1f4d5bc708e06bb930efec672db08d5
- sys/prctl.h: a76908ae4f9cc728b69720ede3630ed2df5462e85d03fe949b48f269ad3dade6
- linux/prctl.h: 6feff68eef81225d5885ed3dd6e0644b6ec4881591e219041fcec87a67040e42

headers-sha256.txt contains hashes of every actual transitive included header from headers.d, not just the four key headers. hashes.txt, compiler-version.txt, source.c, binary, command description, raw JSON, stdout/stderr and reviewer scripts are retained. final-inspection.json retains source hashes for the production owner/job/transport and candidate runners.

5. Residual inspection, approval scope and gates.

Independent final-check.py inspected recorded reviewer Python PIDs965028 and966949: both absent. It also scanned /proc command arguments for exact retained checks.py, noop-repro.py and sigaction_abi paths: no remaining matches. The C probe/compiler had completed synchronously. The final inspector and read-only Git subprocesses themselves completed normally. This is an inspection of this review's own work, not host-wide process ownership proof. No process was killed or GPU queried.

This report accepts the observed ABI comparison and CPU results as evidence, but does NOT approve execution of ef6c8df's fixed15-case campaign because R1 leaves a required oracle gap. The ABI JSON is retained for a repaired candidate's fresh independent preflight; it is not itself execution authority.

Still unexecuted and closed: native _SubreaperPolicy/_CompilerJob setup; the OS campaign; actual cpp_extension loading; three CUDA cold builds; native observer/device execution; G1..G4; C2e/f; parent C4 acceptance and required whole-C4 refactor; measurements, backend selection and product activation. Torch, Warp, NumPy, SciPy and setuptools imports were neither needed nor performed. No GPU was touched.

After the repair and exact-revision independent approval, the maximum intended next scope remains one run of the reviewed sequential15-case CPU campaign on the specified target using verified ABI evidence, stopping on the first failure and inspecting the same owners. It must not be construed as approval of any CUDA compilation or later gate.

