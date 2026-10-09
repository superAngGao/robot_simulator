Initiative: q54-optical-lab-p12-4c4c2d-compiler-process
Stage: independent code/ABI preflight, interrupted partial inspection
Author: Codex independent reviewer /root/compiler_process_preflight_v3
Version: v3 partial
Date: 2026-10-09
Status: draft
Related Files: implementation-note v1/v2/v3; independent review v1/v2; test-evidence v1; compilation-admission design v3
Owner Summary: Review stopped at coordinator direction while the owner reassesses complexity and product priority. No approval or process-runtime proof is issued.

Session and revision

Fresh independent session /root/compiler_process_preflight_v3. Original base ee76591e4f8705fe01eec0ca8142a5be1edd0f17; prior operative candidate 0497573043c66186a912ed6998722d43677da3f7; inspected candidate 92a8ac646ac182f2dac4dc4eef8a3dd0b331f1af.
Local repository: C:\Users\AG\新建文件夹\p12_4_c4_warp_timeline_design.
Target: tileops-dev:/data2/shared/ang.gao/robot_sim_cuda_compilation_wip.
Local and target git rev-parse HEAD returned that candidate; git status --short / --porcelain returned empty output when checked. No candidate files were changed by this reviewer.

Verdict

Not issued. This is an interrupted partial review, not Approve, Request changes, completed independent validation, or authorization for another campaign. The coordinator expressly directed that no further CPU/ABI/native verification be started while product scope is reassessed. No substantive blocking finding has been established by the incomplete inspection; that statement is not assurance that none exists.

Completed inspection

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md. Initial attempted root REVIEW_PRINCIPLES.md read failed because the file is under collab; the correct file was then read in full.
Read implementation notes v1/v2/v3 and test-evidence v1; inspected prior review reports, with some combined terminal output truncated. Read the accepted design's sections 6 and 8 and remaining-gate provisions. Inspected the complete binding delta from prior operative candidate, production compiler_supervision.py and compiler_job.py, fixed cuda_compilation_process.py, compiler_process_cases.py, compiler_process_campaign.py, compiler_sigaction_abi.c, and the process unwind/oracle tests. No independent completed review of all GuardSupervisorCore/Transport internals or all original-base changes was performed before interruption.

Observed code facts, subject to incomplete verification:

- The new fixed binding is library.pidfd_open with c_int/c_uint arguments and c_int result on the existing held CDLL(use_errno=True). It passes flags zero, rejects non-exact/nonpositive/out-of-range PIDs and reads errno for negative returns. The previous os.pidfd_open capability requirement and call are removed. Signal delivery still uses signal.pidfd_send_signal.
- _acquire retains its before/after PID/starttime/SID/PPID checks and closes the newly returned descriptor if the second observation fails. The fixture observer now wraps the actual policy method, delegating except for its declared EMFILE injection.
- The new C source includes sys/pidfd.h and compile-time type/size checks. Its main executes only the prior read-only sigaction and GET_CHILD_SUBREAPER queries; the pidfd_open reference is within a type expression. No pidfd_open call, signal delivery or policy SET is present in this source.
- The stronger R1 fixture oracle is called by validate_result before publishing validated.json. It compares production child_reaped wait-status return codes with case-specific expected outcomes, role topology, signal target sets/order, worker-owned waits, signal failure records and natural-fallback/error artifacts. This source inspection was not supplemented by a fresh reproduction.
- The campaign remains the fixed sequential fifteen cases with first-failure stop and preserved owner on timeout. Production ownership uses retained Popen/handles, sole-waiter reaping, outer deadline state after worker exit, and ECHILD/EOF/handle closure conditions. These paths were read, not exercised.

Commands and evidence

All commands issued were synchronous file reads, rg searches, Git metadata/diff reads, or the single target scratch-directory creation. Representative local commands:

    Get-Content -Encoding utf8 <listed source and collaboration files>
    git status --short
    git rev-parse HEAD
    git diff --stat 0497573043c66186a912ed6998722d43677da3f7..HEAD
    git diff 0497573043c66186a912ed6998722d43677da3f7..HEAD -- optical_worker/compiler_supervision.py tests/support/compiler_process_cases.py tests/support/compiler_sigaction_abi.c tests/unit/optics/test_compiler_process_unwind.py
    rg --files collab

Single completed target command:

    ssh tileops-dev 'cd /data2/shared/ang.gao/robot_sim_cuda_compilation_wip && git rev-parse HEAD && git status --porcelain && mktemp -d /data2/shared/ang.gao/compiler_process_preflight_v3.XXXXXXXX'

It exited zero and returned HEAD 92a8ac646ac182f2dac4dc4eef8a3dd0b331f1af, clean status and scratch directory /data2/shared/ang.gao/compiler_process_preflight_v3.fMlzFq3W. The directory was left as created, with no review artifacts written inside it. Its effective permissions were not checked before interruption; do not describe it as a verified 0700 evidence directory.

Process hygiene: every issued exec/SSH command returned a completed exit result. No tool session remained running. No worker, compiler, probe, Python harness or background process was launched. No PID signal or host-wide cleanup was issued. No separate target /proc liveness inspection was needed for spawned review workloads because none were started; this is not host-wide residual-process proof.

Unexecuted work and limits

No CPU pytest run, preserved R1 replay, system GCC invocation, C ABI binary execution, current libc export/header/hash verification, local-target source hash comparison, native constructor, process campaign, backend import or GPU work occurred in this reviewer session. Earlier reports and reported test counts are historical claims, not fresh evidence from this review. The absence of os.pidfd_open in the current target interpreter and current exact libc symbol/header facts were not independently rechecked.

No new ABI JSON exists from this review. The previous failed campaign remains failed; nothing here resumes or relabels it. The candidate remains unapproved for the proposed new fifteen-case experiment. Full compilation/native observer preflight, three-recipe builds/failure verification, whole checkpoint review, later C4 work, required whole-C4 refactor and performance/product gates remain outside this partial report.