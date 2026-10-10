Initiative: Q54 / P12.4 / C4c2 Warp void-exit fault
Stage: Fresh independent bounded TEST-CODE / NATIVE-FAULT PREFLIGHT review only
Author: Codex, independent reviewer /root
Version: v1
Date: 2026-10-10, Asia/Shanghai
Status: accepted
Session: 01a12565-e595-7763-b65f-3c04530d6ad0
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; native-state-reader design-request v1 and code-preflight-review v3; terminal-target code-preflight-review v1; runtime-integration design-request v3; tests/support/native_runtime_process.py; tests/gpu/test_native_scope_runtime.py
Owner Summary: Approve exact fc23b4e38d816d5108b61363a5372710db6a2937 for execution of the one fixed Warp exit-not-restored fault under the existing pinned preparation-reader deployment contract. No substantive findings. Twenty-one focused independent CPU cases pass without skips. This is test-code/native-fault preflight permission, not a native result, whole-reader acceptance, or authority to open later gates.

Verdict: Approve.

Identity and independence

Candidate: C:/Users/AG/新建文件夹/c4_warp_exit_fault_review
HEAD: fc23b4e38d816d5108b61363a5372710db6a2937
Immediate base, independently resolved: becf2eeb3c5863404a1ba88c29137be2cb243bbf
HEAD tree: 3cb3c6cab5582ca6f12ce653cdccabd86ae5afb3
Branch: detached HEAD.
Initial and final status: clean, including all untracked and ignored files.
Immediate delta: exactly two files, 49 insertions and 3 deletions.
Approved production reference: 0be9876b9778cf4826a62f45c373a38d60ff33f1.

The complete name-only diff from that production reference contains only PROGRESS.md, the terminal-target review/implementation-note records, and these two test files. Thus production sources, catalog, native ABI, query implementation and source pins are unchanged. The immediate base-to-HEAD diff has no documentation or production change. Committed LF SHA256 values are:

- tests/support/native_runtime_process.py: 4ff2ba13a6cc5241e1fcb0c1a64c6d47ae50bd6fb3aa7df927c621da1fa9ba9f
- tests/gpu/test_native_scope_runtime.py: 31887cb5568c3f2a04a9536be7209fb7d92ac6d90fe244053600ce0030131637

This fresh session received the owner's bounded packet, not an implementation conversation. I did not author or edit the candidate and spawned no agents. AGENTS.md, the complete shared README and REVIEW_PRINCIPLES, the specified reader design/preflight, terminal-target preflight and runtime-integration design were read. Large combined reads were supplemented by focused reads; UTF-8 rereads resolved the initial Chinese display encoding problem. Prior verdicts establish predecessor scope, not evidence of this candidate's behavior.

All new scripts, test outputs, remote source excerpts and this report are confined to C:/Users/AG/新建文件夹/c4_assets/warp_exit_fault_review_scratch_v1. The directory was empty at initial listing. No candidate file, ignored bytecode, old artifact or old cache was changed. No install, signal, reset, deployment, native-library acquisition, backend import or GPU work occurred. Read-only SSH used primary tileops-dev only; no fallback or host switch occurred.

Findings

No substantive blocking or non-blocking findings in this 49-line test delta. The inspected high-risk boundaries and evidence are described below. Native execution remains required evidence for the hardware property; that is the purpose of this preflight, not a missing prerequisite to a test-code approval.

Fault model and source assessment

1. The fixed selector is admitted by RUNTIME_OBSERVATION_FAULTS. Static evaluation of the GPU parameter expression yields 17 rows, exactly one new row: warp / asymmetric_synthetic_7x3_v1 / runtime_scope_warp_exit_not_restored. It does not add this fault to the CUDA matrix. The pre-existing 16 rows retain their values. The supplied old 20-case campaign additionally includes its four foreign/unknown stream cases; it is a separate execution, not this file's new 17-row matrix.

2. At native_runtime_process.py:523-566, the original NativeStateReader.binding_returned executes first. Only after successful return is module.ScopedDevice replaced. The wrapper factory creates UnrestoredExit only while session.phase is validating_input; other phases construct the original scope. UnrestoredExit constructs the actual original ScopedDevice(target), and its __enter__ delegates once to that original object. It does not replace the entry with a declared state or nullcontext.

3. Read-only source inspection freshly obtained the installed pinned files from /home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages/warp. utils.py SHA256 is 01bd20f279f5762881c50cf8b6d766734fe21bd5f9b1ec286b34b1f3fcb27f24; context.py SHA256 is eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee. ScopedDevice lines 1243-1284 save default_device, select the original device as Python default, and delegate entry/exit to device.context_guard; exit then restores saved_device. ContextGuard lines 3153-3167 push the CUDA device context on entry and pop it on exit. These are actual installed Python source bytes, read without importing Warp. The existing ABI/driver preflight is reused only because production query code/pins are unchanged; no new native ABI claim is made from Python text alone.

4. On a body exception, the new wrapper returns the original __exit__(kind, value, traceback) result. On normal completion it deliberately does not invoke that native context pop; it assigns only original.device.runtime.default_device = original.saved_device, writes the fixed fault marker and returns None. There is no new native API call, library acquisition, getter, setter, GPU allocation, guard, synchronization, alternate cleanup or successful fact producer. This is a controlled omission of an exit operation that models a void exit failing to restore context. It is not a claim to induce or diagnose a real driver malfunction.

5. The actual readback ordering at optical_worker/native_readback.py:90 is reader.outer_scope -> real ScopedDevice -> held readback body. The report and readback owners are retained before leaving that body. NativeStateReader.outer_scope resets its nesting depth and invokes its after snapshot only following a successful body/scope return. Consequently the new void-like return reaches the existing raw after query at outer depth zero. If a body fails, the after snapshot is skipped; the original exit still runs once. The existing missing-Warp-scope test is not treated as evidence of a successful body followed by contradictory physical after state.

6. NativeStateReader._state at native_scope.py:356 queries actual current context, conditionally its device, all primary states, and the already-proved target stream cache. The target_context value was captured by production; the injector never writes it. The expected contradictory state is default0 with current context equal to that retained target context, current ordinal1, and only primary1 active. NativeStateReader.snapshot at :445 saves raw state and appends the factual native_scope journal row BEFORE checking ambient restoration. It rejects the non-null outer current context. _fail marks unknown scope and the original normal guard failed; later queries reject before getters. A successful Python default assignment cannot satisfy this physical-state predicate.

7. The wrapper instance and original scope class/method are strongly retained in probe.originals. The probe is retained in session.runtime_owners. The existing source/guardian graph already retains the runtime, target, stream, reader/native handles and returned readback owners. Omitting a pop does not introduce a separately acquired native resource needing a new cleanup authority. The error follows the existing session failure/whole-graph quarantine and hard-exit path. Test helper diagnostics remain no-ops around normal input helpers, and their later diagnostic path still delegates to the production reader.

8. At test_native_scope_runtime.py:100-117 the new assertions require the exact omission boundary, setup guard sequence 5, full_quarantine and guarded_segment_failed, exactly one readback_scope/after raw observation, Python default0, non-null current context equal to target_context, ordinal1 and only primary1 active. The post-marker lifecycle suffix may contain no normal_complete_started, normal_arm_started, terminal_arm_started, emergency_sync_started or ordinary_cleanup_started. Inherited assertions require worker return 74, unknown scope, zero emergency-sync/terminal-arm attempts, release not authorized, no sync marker, no published input validation report and no ordinary atexit marker.

9. Existing custody selection, parsed runtime record and attachment checks are unchanged. validate_runtime_custody_attachment calls the existing custody artifact validator and checks the exact lifecycle prefix. The supervisor obtains a real worker returncode and final evidence, and the parent waits for the supervisor. The external OS identity file still has to match worker, supervisor and parent PIDs. A retained valid prefix is permissible when the contradictory after row prevents a successful final runtime record; diagnostic raw/journal evidence is not promoted into a successful receipt. The new fault marker is a test cut/boundary, not completion or scope authority. These checks operate in the accepted closed trusted test-process model; they do not claim resistance to coordinated arbitrary rewriting of all evidence and internal objects.

Independent verification

Environment: Windows PowerShell, C:/Python314/python.exe, Python 3.14. CPU subprocess uses -B, PYTHONDONTWRITEBYTECODE=1, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, repository PYTHONPATH, UTF-8 output, disabled pytest cacheprovider, and unique scratch TEMP/TMP/basetemp. In-memory compile() checked syntax; compileall was deliberately not run. ctypes in the CPU model supplies scalar/pointer containers only; it acquires no native query library and performs no backend calls.

Commands from the frozen candidate (S denotes the complete scratch path above):

    git rev-parse HEAD HEAD^
    git rev-parse 'HEAD^{tree}'
    git status --porcelain=v1 --branch --untracked-files=all
    git status --porcelain=v1 --ignored --untracked-files=all
    git diff HEAD^ HEAD --stat
    git diff HEAD^ HEAD -- tests/support/native_runtime_process.py tests/gpu/test_native_scope_runtime.py
    git diff --name-status 0be9876b9778cf4826a62f45c373a38d60ff33f1 HEAD
    git diff --check HEAD^ HEAD
    C:/Python314/python.exe -B S/inspect_remote.py
    C:/Python314/python.exe -B S/run_cpu.py
    C:/Python314/python.exe -B S/static_audit.py

The scripts preserve exact expanded argv, source inspection command and outputs in remote_execution.json, remote_source.jsonl, remote_stderr.txt, cpu_execution.json, cpu_output.txt, static_audit.json and the four Ruff JSON outputs. The CPU command is:

    C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=S/pytest_cpu_v1 S/test_exit_fault_review.py tests/unit/optics/test_native_scope_observer.py::test_exit_return_with_wrong_context_preserves_row_then_poison

Observed result: 21 passed in 1.69 seconds, exit 0, no skips or failures.

- Three new integration probes execute the actual _NativeScopeProbe constructor/binding wrapper and production NativeStateReader._state/snapshot/outer_scope, using the original production lifecycle and joined CPU guard peer. ScopedDevice and ContextGuard Python definitions are compiled directly from the freshly read installed source excerpts. Only native leaves are modeled by an explicit context stack and read-only scalar-output functions. Fault mode proves one real modeled push and no pop, Python default restoration, contradictory raw and appended journal state, unknown/failed full quarantine, no normal completion/terminal arm/sync/release, strong wrapper retention, and rejection of subsequent reads. Body-error mode propagates the same exception, pops once through original exit, restores the default, and produces no fault marker or successful after row. Normal control pops/restores and records a null after context. Other phases delegate normally in all three cases.
- Seventeen assertion probes compile and execute the exact candidate's new GPU assertion block, without importing or running the GPU test. The valid modeled artifact passes. Sixteen counterexamples are rejected: null current and target, unequal current context, wrong default, wrong ordinal, extra active primary, missing/duplicate after row, wrong guard, wrong boundary, wrong custody, false guard-failed flag, and each of the five forbidden post-cut operations. These are assertion-sensitivity tests, not fabricated native acceptance records.
- One existing focused CPU regression independently verifies that a wrong context after a void-like exit is appended before failure/quarantine. The full 87-case reader suite was not rerun because this review is bounded to the small test delta.

Static results: both changed files compile in memory. git diff --check passes. Ruff was freshly run via stdin for committed base and HEAD bytes with --no-cache and the original filenames. There are 50 existing diagnostics in native_runtime_process.py and 9 in test_native_scope_runtime.py on each revision; the code/message multisets are identical (59 total). This is no new diagnostics, not a clean Ruff result. Full diagnostic JSON is retained, including changed line positions. Static matrix evaluation confirms the one new Warp row. Production equality is established by the complete Git diff, not inferred from author notes.

The author-reported 87 CPU passes, earlier 66 passes/21 skips, Ruff summary and author-checkout compileall side effect are not counted as this session's execution. No old output/cache is used as proof of a new execution. This frozen checkout remains free of ignored files. The optional ROBOT_SIM_WARP_CONTEXT_SOURCE was set by the runner but the selected tests did not need its Device extraction fixture; the independent scope models use the freshly read pinned ScopedDevice/ContextGuard definitions instead.

Tool issues, distinct from candidate failures: an initial unquoted HEAD^{tree} was misinterpreted by PowerShell, then immediately rerun with correct quoting. Initial large combined document outputs required smaller rereads. git symbolic-ref -q HEAD returns status 1 because the checkout is detached; this is expected. No test failed, no environment workaround/installation was needed, and no candidate change was made.

Process closure and preservation

Reviewer CLI PID 4292, parent 47420, creation Unix milliseconds 1791628797121; the recorded ancestor chain correlates this review's command process with that CLI. The enclosing launcher is not review-owned test work.

CPU pytest PID 27220 was waited by driver 32664 with exit 0. SSH source-reader PID 15832 was waited by driver 44532 with exit 0. At 2026-10-10T18:43:06.8094800+08:00, all four were absent; no scratch-scoped Python/SSH/SCP residual was found. CPU peer threads use fixture teardown and joined close. Static Git/Ruff commands completed synchronously. No native worker or native supervisor was started.

Remote source inspection PID 2793475 and its parent 2793473 were freshly checked absent by a bounded read-only /proc existence check, saved in remote_closure.txt; that SSH command returned synchronously. No remote candidate checkout, running campaign, GPU inventory or GPU-process list was queried. No process was signaled. These are scoped OS closure observations, not exhaustive GPU/process attestation. The final local Git status remains empty including ignored/untracked files, with unchanged HEAD/tree. No review-owned residual is unresolved.

The existing 22b1db0 reader 20-case campaign on primary183 was not inspected, interrupted, counted as passed, or adopted as this review's process tree. Its reported running state remains the owner's packet information. Likewise the previously approved 0be9876 two-lane repair probe and its waiting order were not changed or reopened.

Exact native permission and limits

This approval binds only fc23b4e38d816d5108b61363a5372710db6a2937 and the reviewed two-file test delta. It permits execution of exactly the one new Warp/asymmetric runtime_scope_warp_exit_not_restored case under the existing approved preparation-reader contract, pinned primary deployment (eight visible devices, target1, ambient0), existing original setup guard/supervisor and external OS identity recorder. It does not authorize starting another GPU round over an unresolved existing campaign. Preserve the existing old-reader completion/process-hygiene ordering and separate repair-probe sequencing.

Execution must preserve the actual marker, raw contradictory after state, lifecycle/quarantine/no-release evidence, retained prefix/attachment and observed/reaped process exit. A setup pass, CPU model, absent-scope failure, marker alone, or normal __exit__ return is not hardware proof of this fault. New native evidence still requires its own assessment; this review grants no success in advance. This reviewer initiates no native execution.

Not executed here: real GPU fault, original driver/library queries, real D2H body under this fault, hardware graph retention/destruction, actual target-context state, full reader hardware matrix or two-lane repair probe. Source inspection does not prove binary rebuild equivalence or proprietary driver behavior. The CPU model establishes the intended omission and detection/control path, not real-device behavior.

Still separate and closed: full reader hardware acceptance; final-sample physical/numerical proof; bounded preallocated control traces; complete worker fault/close coverage; full C2f/C4 and C4c3/c4/C4d/B3 acceptance; F3 predecessor compact-reply/journal-recovery approval. No performance, product activation, result/baseline, backend selection, R2/S/host delivery, a3b or shipping entry opens.

The bounded review stops here. This complete report is ready for verbatim archive under the shared evidence-only recording exception; recording it must not change its verdict, exact revision, scope or gates. No evidence commit, candidate edit, deployment or additional task is performed by this reviewer.
