Initiative: q54-optical-lab-p12-4c4c2d-compiler-process
Stage: independent code preflight / R1 repair re-review
Author: Codex independent reviewer
Version: v2
Date: 2026-10-09
Status: accepted
Related Files: compiler-process implementation-note v1/v2; compiler-process review v1; compilation-admission design-request v3; AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md
Owner Summary: Approve this exact code preflight. R1 is closed by a common actual-exit oracle and independently reproduced rejection of the preserved counterexample. The exact fixed sequential 15-case CPU OS campaign may now run once on the specified target. No actual OS job, compiler/backend execution, CUDA build, readiness, measurement or C4 completion is approved as already proved.

Reviewer session: /root/compiler_process_preflight_v2, fresh independent task session; not the implementation author and not the v1 reviewer session. The review packet and preserved findings were the inputs; no implementation conversation was inherited.

Original base: ee76591e4f8705fe01eec0ca8142a5be1edd0f17.
Prior candidate: ef6c8df717f574684e88cadbeb48222e7916d3a6.
Evidence-only archive: 1039f55ca2f1b2154b67a1fa5979ef93fae179bd.
Reviewed HEAD: 0497573043c66186a912ed6998722d43677da3f7.
Local candidate: C:\Users\AG\新建文件夹\p12_4_c4_warp_timeline_design, branch implementation/c4-supervised-waits.
Target candidate: tileops-dev, /data2/shared/ang.gao/robot_sim_cuda_compilation_wip, detached HEAD.
Both working trees were clean at the specified HEAD before and after verification. Twelve inspected candidate/production files have matching local and target SHA256 values. No candidate files were edited.

Verdict: Approve.

1. Findings and prior R1 disposition.

No unresolved substantive finding was established for the requested bounded code preflight. This is permission for the specified experiment, not approval of its future results.

Prior R1 (High, ineffective descendant SIGKILL followed by natural exit accepted): closed.

The new common boundary in tests/support/compiler_process_cases.py:419-466 is called by the real validate_result before validated.json publication. It checks the exact worker/child/grandchild role multiset, the presence and identity of worker-owned reaps, actual child_reaped wait statuses through the production wait_status_returncode, worker_returncode, exact signal target set, attempt-before-reap ordering, expected signal failures, natural-fallback files and expected worker-error artifact presence. The surrounding validator already requires unique identity attempts/reaps, exact member/reaped set equality, adopted-set equality and matching held pidfd observations. Production replay independently requires SIGKILL=9 and expired ownership before each unique attempt. Together these checks prevent the original legal production CLOSED record from being mistaken for a successful kill-required fixture.

The fixed expected outcomes correspond to the actual worker and supervisor paths:

| Case | Required observed outcome |
| --- | --- |
| normal | Worker exit0, compile quiescence6/7/8, no attempt or expiry. |
| separate_pgid | Worker exit0; separate-PGID child exit0 actually waited by worker, sampled by supervisor, excluded from supervisor reaps. |
| leader_exit / leader_ack_exit / pipe_hang | Worker exit0; one child SIGKILL returncode-9 and one child attempt; original operation deadline, or earlier pending-ACK deadline for leader_ack_exit. |
| double_fork | Worker exit0; intermediate child exit0 actually waited by worker; adopted grandchild returncode-9 and one attempt. |
| fork_during_cleanup | Worker exit0; supervisor-reaped intermediate exit0; grandchild created after DRAINING, adopted, returncode-9 and one attempt. |
| missing_complete / proc_failure / parent_lost | Worker returncode-9 and one attempt; proc failure is injected after expiry; parent_lost additionally requires observed liveness EOF. |
| residual_complete | Worker exit74 with error artifact after withheld COMPLETE ACK; child returncode-9 and one attempt; residual-descendant diagnostic. |
| handle_failure / stale_identity / disappearing_identity | Worker exit74 with error artifact after pre-ARM observation/acquisition failure and transport fencing; no signal attempt. |
| signal_failure | Exactly one failed worker attempt, declared natural-fallback file, actual worker exit0; no second attempt. |

The natural-fallback absence rule applies to every case except signal_failure. Exact topology prevents a missing grandchild or omitted child workload from silently shrinking the intended test. The oracle tests enter the production GuardSupervisorCore/_CompilerObligation and actual validator; process observations are explicitly synthetic. They do not execute or duplicate the native supervisor.

Independent original reproduction: I copied the preserved noop-repro.py into a new private outside-repository directory, changing only its output directory string. The original production-record construction and its original request revision were preserved. A wrapper guards forbidden operations, records its PID, executes the copy, checks the exact failure, and verifies the old evidence directory remained unchanged.

Observed output:
```
PRESERVED COUNTEREXAMPLE REJECTED at actual exit-policy boundary
Original evidence files unchanged: 36
```
The retained traceback reaches validate_result at line373, then _validate_case_exits at line451:
```
AssertionError: fixture wait status differs from required exit mechanism
```
No validated.json was produced. This was rejection by the intended exit oracle, not an import, environment, missing-file or schema failure. The four-suite CPU run also accepted the matching production-record SIGKILL positives and rejected natural-exit variants for all five required leader/descendant cases, plus natural-fallback-marker and erased-grandchild attacks.

2. Complete candidate and dependent-path inspection.

I read the collaboration protocol and all review principles, both implementation packets, authoritative prior finding/reproduction record, accepted design sections6/8 and remaining-gate text. The design SHA256 is 028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2. The archived v1 review SHA256 is 7d41ca9bae8ed77103dbd97b37191fd5f11d13000fa0064a6a3a5b985b7afeba. Archive1039f55 changes only that report.

I inspected the original candidate's full process fixture, campaign and ABI source, its cuda_compilation_process/transport-test changes, the R1 repair delta and oracle tests. I traced the dependent compiler_supervision.py, compiler_job.py, GuardSupervisorCore and GuardSupervisorTransport, including production replay, deadline transitions, launch/acquisition failures, direct-child discovery, sole-waiter reaping, pidfd signalling, COMPLETE interception, pipe draining and final closure. I also checked runtime-driver import/write behavior and the dependency-runner directory helper. Production owner/job/core/transport and captured catalog/policy/profile are unchanged by the original-base-to-HEAD candidate; the review does not rely merely on that lack of diff.

Material observations:

- Native _SubreaperPolicy construction remains confined to the dedicated supervisor; module import and ctypes structure inspection are inert. Setup checks the pinned libc, ABI, initial child/thread conditions, native SIGCHLD disposition and subreaper readback. No arbitrary child command or new captured-worker native authority is exposed by this fixture dispatch.
- Popen is retained before its constructor can partially succeed. Worker PID/starttime/SID is anchored; PPID/identity checks bracket pidfd acquisition before ARM. The production supervisor is the only waitid/WNOWAIT and waitpid consumer for the worker/adopted tree. Parent/campaign waits concern their own supervisor/case-parent children, and normal intermediate reaps belong to the worker.
- DRAINING freezes the preceding effective obligation. Leader exit does not replace an earlier ACK deadline or mutate the exited core. Pending ARM6 output is withheld through the actual transport. Child-only expiry/reaping continues through the outer obligation.
- Attempts use owned pidfds only, remain unique after errors, and have explicit real wait statuses. Missing/stale identities and acquisition errors retain custody. The signal-denial fixture has one intentional finite natural exit; it cannot masquerade as a kill-required pass after this repair.
- Compile COMPLETE6/7/8 enters exact registered-owner checks before core acceptance/DISARMED. Residual descendants cause the live core to fault and fence output. Closure requires worker/adopted reaping, policy verification, ECHILD, retained-identity absence, output EOF and pidfd closure; an empty sampler result alone is insufficient.
- Parent observations are bounded at30 seconds. Timeout writes incomplete.json and preserves the supervisor; finally closes liveness and sentinel EOF pipes without killing the supervisor. Campaign execution is sequential, records known remaining PIDs, stops on nonzero case exit or leftovers, and contains no retry or secondary kill policy. Persistent pidfd-close failure stays DRAINING in the CPU regression. No claim is made that every possible kernel/IO failure has a bounded cleanup time.
- The unrelated sentinel has its own SID, must remain alive/identity-stable through supervisor completion, and exits only on its separate EOF request. Real fork/session/adoption behavior, pidfd effects and actual future per-case outcomes remain matters for the now-permitted experiment.

3. Independently executed permitted verification.

New evidence directory: /data2/shared/ang.gao/compiler_process_preflight_v2.deDoS0SG, verified mode0700. A shared-parent inherited setgid bit was cleared before executing scripts; group/other access was never granted. Full local copy: C:\Users\AG\新建文件夹\c4_assets\compiler_process_preflight_v2.deDoS0SG.

Interpreter: /home/ang.gao/miniforge3/envs/robot_sim/bin/python, Python3.12.13, Anaconda package, GCC14.3.0 build. Reviewer Python commands used -I -S -B. The test harness explicitly inserted only the repository and existing Python3.12 site-packages path; site startup/.pth files did not run. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; pytest cache disabled; pytest temporary files outside the repository.

Commands, from the target candidate directory:
```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/compiler_process_preflight_v2.deDoS0SG/checks.py
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/compiler_process_preflight_v2.deDoS0SG/replay.py
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/compiler_process_preflight_v2.deDoS0SG/final-check.py
```
checks.py runs pytest with -q -p no:cacheprovider and --basetemp in the private directory on test_compiler_job.py, test_compiler_transport.py, test_compiler_process_unwind.py and test_compiler_process_oracle.py. Observed: 41 passed in0.26s, exit0, empty checks.stderr.

Both test and replay harnesses reject torch/warp/numpy/scipy/setuptools imports, reject subprocess.Popen/os.fork/os.system/ctypes.dlopen audit events, and replace _CompilerJob.__init__ and _SubreaperPolicy.__init__ with explicit failures. ctypes was imported before the audit guard solely for inert structure inspection. The retained loaded-module list contains none of those prohibited backend packages. Inert object.__new__ owners and mocked OS operations in tests are not native job construction.

Source checks: git diff --check from original base through HEAD passes for tests/; it also passes for the repair tests/ delta. Whole-diff --check reports only the extra final blank line in the verbatim archived v1 report at line138. That archival whitespace is non-blocking, has no runtime effect, and should not be edited in a way that violates verbatim preservation. No source whitespace failure was observed.

Final inspection found reviewer PIDs990316 and991517 absent and no /proc command arguments matching the exact new checks.py, replay.py or noop-repro-copy.py paths. The inspector and read-only Git subprocesses completed synchronously. No kill or GPU query was issued. This is review-owned PID hygiene, not host-wide OS ownership proof.

4. ABI evidence revalidation, without repeated native execution.

The prior independent system-header probe and its raw outputs remain at /data2/shared/ang.gao/compiler_process_preflight_v1.zTQ8bY69. I inspected the C source and verified every row in its hashes.txt and headers-sha256.txt against current files: 87 manifest rows total, including the source, compiler, retained binary, raw ABI, libc and every recorded transitive header. The production owner/job/core hashes also match prior final-inspection.json. The new abi-verification.json retains all verified path/hash rows and the fresh inert ctypes layout.

Unchanged raw ABI:
```
{"pointer_size":8,"ulong_size":8,"size":152,"offsets":[0,8,136,144],"sigset_size":128,"sigchld":17,"no_cldwait":2,"set_subreaper":36,"get_subreaper":37,"default_handler":true,"no_cldwait_active":false,"subreaper":0}
```
The current target ctypes _SigAction layout independently matches pointer8, ulong8, size152, offsets0/8/136/144 and sigset128. Production native query/set/readback constants match17/2/36/37. The old raw inherited-policy result describes the old probe process; the future supervisor still must perform its own native checks.

Verified SHA256 values:

- C source: 283e0ba7c7edb6eadfbbed4de5961d8e8b78bf26f3c7246bf1ed50798bcdcd7e.
- /usr/bin/gcc: 1b99826121ae6682a634e5efe09bd3e3df58ce58e0b28f849114ab5b89139c26.
- Retained sigaction_abi binary: 558d662991f84aa4cca2c98ff241fa67e011bdc9b4e76a68ccab24e0df240452.
- libc: d8db8739a1633c972cec6a4fe0566bdcec6fd088f98723492ab0361f66238f75.
- Raw abi.json: 037752b995cd2f1f82caa698ad655ca4b6aa8f7a5da00bd037905fd7ee19656b.
- bits/sigaction.h: b2f76a26bbfa3668859e8e8a93af9321e14083914df0aa0367cdffaabf4ece81.
- bits/types/__sigset_t.h: 05291a7d38bb1d0da1ed01c09fe85fa2b1f4d5bc708e06bb930efec672db08d5.
- sys/prctl.h: a76908ae4f9cc728b69720ede3630ed2df5462e85d03fe949b48f269ad3dade6.
- linux/prctl.h: 6feff68eef81225d5885ed3dd6e0644b6ec4881591e219041fcec87a67040e42.

No compiler, ABI binary, native constructor, process campaign or backend was executed in this review. Revalidation of unchanged retained evidence is explicitly distinct from a fresh ABI-probe execution.

5. Precise permissible execution and remaining gates.

This approval binds HEAD0497573043c66186a912ed6998722d43677da3f7 and permits exactly one run of its fixed sequential 15-case CPU OS campaign on tileops-dev in the specified target checkout, using a fresh private outside-repository output directory and the verified retained ABI JSON. The maximum allowed invocation is:
```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B tests/support/compiler_process_campaign.py <fresh-absolute-campaign-directory> /data2/shared/ang.gao/compiler_process_preflight_v1.zTQ8bY69/abi.json
```
Before invocation, confirm the same reviewed operative source/target environment and clean checkout. A separate evidence-only commit that verbatim archives this report may be recorded under the shared protocol exception; identify the approved code parent and evidence commit separately. Any operative code, policy, runner, test or design change requires review of that delta.

The permitted order is normal, separate_pgid, leader_exit, leader_ack_exit, double_fork, fork_during_cleanup, pipe_hang, missing_complete, residual_complete, proc_failure, handle_failure, stale_identity, disappearing_identity, signal_failure, parent_lost. Native _SubreaperPolicy/_CompilerJob execution is permitted only as part of those fixed dedicated-supervisor cases. Stop at the first failure/incomplete result, retain raw outputs and inspect the same owners and known identities. This approval provides no automatic retry, substitute case, renewed timeout, owner kill, alternate cleanup policy or permission to fabricate closure. A failed experiment requires diagnosis and the appropriate repair/re-review before another attempt.

Required post-run evidence remains real wait statuses and topology, deadlines, actual native policy/pidfd behavior, adoption/reaping, output EOF, closed handles, sentinel survival/exit and owned residual inspection. A campaign pass still requires independent review of the actual evidence before progressing to a later native/backend stage.

Still closed: actual cpp_extension loading; all three CUDA cold builds/admissions; backend imports or arbitrary native/GPU observer execution; device execution; readiness/PREPARE and downstream result/selection gates; G1..G4; C2e/f; integrated compilation checkpoint acceptance; parent C4 acceptance and required whole-C4 refactor; measurements, backend selection, losing-lane cleanup acceptance and product activation. This process-fixture preflight does not complete any of those scopes.
