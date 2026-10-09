# Warmup readback checkpoint independent code review v1

Initiative: Q54 / P12.4 / C4 runtime integration
Stage: independent code review of bounded native test-driver repair
Author: codex
Version: v1
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3 sections 5.1, 8 and 9 and review codex v3; fixed-render-warmup code-review codex v2; warmup-bootstrap-repair code-review codex v1; warmup-readback-checkpoint implementation-note codex v1
Owner Summary: Approve this exact bounded support-code repair. Refreshing the original validated runtime checkpoint after the completed RGB readback guard fixes the observed stale-cut comparison without weakening validation. Fresh CPU tests and original-artifact replay pass. The failed native case remains failed; a source-frozen positive native rerun is mandatory.

Reviewer session: `/root/warmup_readback_checkpoint_v1`, fresh independent Codex review supplied the task packet and repository, without the implementation conversation. No implementation authorship, candidate edits, or subagents.

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`
Branch: `implementation/c4-supervised-waits`
Base: `c83b35eeca9a9fec834425f066b41202c7d01059`
Reviewed HEAD: `7fe32a58b9e4bee1b6d354f42ceed942f6e9cb92`
Actual tree: clean initially and after verification; HEAD and branch unchanged.

Verdict: **Approve**, exclusively for this bounded test-driver checkpoint repair and its regression assertion. No substantive blocking or non-blocking finding remains. Native acceptance is a separate, unpassed gate.

## Scope and findings

The exact diff contains three added lines in tests/support/native_runtime_process.py (two comment lines and one checkpoint call), four added lines in tests/unit/optics/test_native_warmup_support.py, and the implementation note. No production source, source catalog/policy, supervisor, renderer, environment contract, lifecycle parser, or acceptance threshold changes occur. `git diff --check` passed.

Read AGENTS.md, the complete collaboration protocol and complete review principles, relevant accepted design sections and design review, prior scoped reviews and author note. Inspected the actual source-session warmup entry, support readback, checkpoint payload/parser, observe(), final custody persistence and existing terminal exception path. Author reports and prior verdicts were treated as claims, not new acceptance evidence.

At optical_worker/native_bootstrap.py:3431 the original warmup entry executes the ten frames, sets its warmup evidence/phase and validates a checkpoint before returning. The support helper subsequently opens a further setup segment, performs the diagnostic D2H copy under the original resource scope, adopts/retains returned storage, writes diagnostic bytes/metadata, and exits through the normal guard handshake. That later segment changes the lifecycle. The unchanged observe() at tests/support/native_runtime_process.py:1240 compares the last validated cut's lifecycle length with the live controller events and correctly rejects the old cut.

The added call at tests/support/native_runtime_process.py:1126 runs only after both nested scope and setup segment have exited successfully. It invokes the existing session observation checkpoint: original payload construction, adoption freezing, journal issuance checks and RuntimeCustodyRecord.from_payload with the original environment/trusted inputs. Only a successful parse replaces last_validated. The change does not edit an old record, synthesize completion, suppress the mismatch check, create another guard or native wait, or change terminal policy. Guard/readback/write failures cannot reach the new call. Checkpoint exceptions propagate into the existing outer worker exception path, which calls session.fail and preserves terminal custody handling.

The committed assertion compares the latest recorded checkpoint lifecycle to the actual current lifecycle for both lanes and both diagnostic modes. Its recorder deliberately replaces checkpoint parsing in native-double fixtures; it proves placement/current-cut equality, not complete live custody parsing. Original-artifact replay below separately exercises the unchanged parser, and real positive execution remains mandatory.

## Original evidence independently checked

Read the supplied original archive without modifying it. Its SHA-256 is `0b7e668a1717f52683874ea831624e2aae0b38849d4b0b27cc3d7e78c115b8b1`, matching the packet. Inspected request, identity/process records, prefix, failure, warmup completion/RGB, worker, supervisor, runtime-custody/selection/diagnostic and combined custody artifacts.

The request names source revision `0ede8d04a9ce99c9c2562cd1dfe1e692c7e5c526`, Warp, and go2_static_960x640_v1. failure.json records RuntimeError, phase fixed_render_warmups, message `normal custody checkpoint does not match current lifecycle`. The warmup frame records contain 50 warmup launches and ten terminal waits. The prefix contains 4338 lifecycle events, ending in RGB readback guard sequence24 normal_disarmed, with 25 normally completed guards. The later terminal runtime record contains4345 events and parses successfully.

The unchanged _validate_render_warmups accepts the original warmup/RGB artifacts against the actual worker trace/request. The RGB contains7372800 bytes, frame10,960x640 float32 RGB, SHA-256 `2bfa01fc5b39c61cce30c850a4d52183f3feb3a873a4ff7b1ff1e3c6931b7ba4`; this validator also checks finite values, exact byte count, bound graph/input identities, warmup order and target/stream/restoration facts in the recorded observations. The original runtime custody terminal record and combined guarded custody artifact also pass their unchanged replay validators.

The supervisor records observed worker return74, EXITED,25 normal completions, no deadline expiry and no kill request. Worker custody is full_quarantine, disposition quarantined_process_exit, zero emergency synchronization attempts. Thus the whole positive case still fails its unchanged terminal-success requirement. Passing raw diagnostic replay does not convert it into native acceptance or optical numerical correctness.

The archived process observation names parent3464184, supervisor3464352 and worker3464353 with start ticks and parentage while alive. This review made no remote query and does not independently certify their current absence; the packet's subsequent remote absence claim is not reused as a fresh process check.

## Fresh verification

Environment: Windows PowerShell, C:/Python314/python.exe, bytecode disabled with -B, pytest cache provider disabled. All commands ran from the frozen repository. No GPU launch, SSH, installation, performance run, or unrelated source/bootstrap suite.

1. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_warmup_support.py tests/unit/optics/test_cuda_artifact_runtime.py`

   **36 passed in18.75s**, exit0, tool session45916. Covers both support lanes/diagnostic modes, existing failure/hook restoration, raw artifact tampering and relevant artifact acceptance behavior.

2. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider ../c4_assets/test_checkpoint_review_v1.py`

   **8 passed in21.76s**, exit0, tool session10750. Independent scratch probes outside the candidate:
   - Four lane/diagnostic combinations remove only the new checkpoint call from an in-memory function copy and invoke the committed positive test unchanged; its new equality assertion rejects the original stale-cut behavior.
   - Both lanes inject a readback exception and confirm the exact original error, restored hook, no second checkpoint, no RGB file, and quarantine.
   - Both lanes inject an exception from the post-readback checkpoint and confirm propagation, completed RGB output, an IDLE/disarmed guard, exactly one additional normal completion since the production checkpoint, and no subsequent lifecycle mutation inside the helper.

   Probe SHA-256: `7f15b60dfdc1b70db1edf30b058460156f225bab12ef35c72432a5909075bce5`.

3. `C:/Python314/python.exe -B -c "import runpy; runpy.run_path('../c4_assets/checkpoint_raw_review_v1.py', run_name='__main__')"`

   Exit0: original warmup/RGB replay, original terminal runtime-custody parser, and original combined custody replay pass; the script explicitly requires the failed worker74/quarantine outcome. Selected original bytes were copied into an automatically removed temporary directory. For offline Linux deployment-path syntax on Windows, the scratch process binds runtime_environment.Path to PurePosixPath; it does not relax record validation, alter artifact contents, or claim a live deployment/environment audit. Script SHA-256: `86e9d0611d1d1f3ffeffbf304c85064d797b4091cc8344ebcedd2793aed162de`.

Total fresh pytest outcome: **44 passed**, no skips. Original-artifact replay is additional CPU verification, not new device execution.

Reviewer tooling issues: initial REVIEW_PRINCIPLES.md lookup at repository root failed before reading its actual collab path fully; an initial rg used nonexistent src; one stdin diagnostic's non-ASCII absolute path was replaced by PowerShell encoding and rerun using a relative path; direct external-script execution lacked repository sys.path and was rerun via runpy; Linux deployment paths initially failed the Windows Path syntax check before the explicit offline PurePosixPath adaptation. These were inspection/tooling issues, not candidate test failures. No candidate repair was made or failure hidden.

## Limits, gates and cleanup

Approval covers only refreshing the original validated observation after successful diagnostic RGB readback and its focused regression assertion. The source-frozen Warp/CUDA positive native rerun must run after this approval. This review does not establish live repaired custody parsing, successful whole-case terminal disposal, optical numerical correctness, native failure matrices, sentinel reset, performance, or complete target1/ambient0 device acceptance.

All prior unfinished/closed scopes remain so: final readiness/origin and final commands/reset, final coverage/quiescence and arena/lifecycle integration, R1 and product PREPARE/sample/lease/IPC activation, R2/S/host delivery and external holds, numerical/result/performance/baseline/backend-selection/shipping authority, whole-C4 completion, and the later systematic whole-C4 refactor and independent review. No parent gate opens.

Focused pytest PID46316, parent45044, creation timestamp1791576844014ms Unix; scratch pytest PID40748, parent41484, creation1791576942051ms Unix. Both tool sessions completed exit0. At2026-10-10T04:16:26.8577859+08:00 the scoped Win32_Process query found none of those recorded processes, no remaining direct children of the pytest PIDs, and no Python command line naming the two scratch scripts or exact focused test command. Short inspection/replay commands completed terminally without separately sampled PIDs. No reviewer-owned residual/hang was observed. No manual signal, remote/GPU query or reset occurred.

Only this report and the two scratch scripts were written under c4_assets outside the candidate. Temporary replay copies were removed by TemporaryDirectory. Candidate and original archive remained unchanged. The coordinator may archive this complete report verbatim in an evidence-only collab commit; that record cannot change the verdict, exact approved revision, scope or remaining gates.
