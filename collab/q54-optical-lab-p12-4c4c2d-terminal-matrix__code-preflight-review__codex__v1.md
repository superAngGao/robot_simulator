# Independent bounded terminal-matrix code/native preflight review

Initiative: Q54 / P12.4 / C4c2 / C2d3 terminal matrix
Stage: independent code and execution preflight review
Author: codex
Version: v1
Date: 2026-10-09
Status: accepted (bounded preflight only)
Related Files: runtime-adoption design-request codex v2 sections 7-9; runtime-integration design-request codex v3 section 8; runtime-custody implementation-plan codex v3; terminal-matrix implementation-note codex v1; runtime-faults test-evidence codex v3; native-scope-aggregate test-evidence codex v1
Owner Summary: Approve the frozen test/ledger/document delta and permit its 12 actual native terminal cases under the existing fixed supervisor. No blocking candidate finding. Independently reproduced the old singleton-stride counterexample and all 96 repaired storage mutations using retained real records. This is not acceptance of C2d3 or of native outcomes that have not run.

Reviewer session: /root/terminal_matrix_preflight_v1. Fresh independent review role; I did not author or edit the candidate. Requirements/evidence packet was checked against files, not treated as proof.

Base: 17bae2c765242d9a82e59c3b555f0bf8a0f3d713.
Reviewed HEAD: 7cde8879b307090d54350ca3511529bfd8e5d7c8.
Local tree: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design; clean at initial and final checks.
Remote CPU replay tree: tileops-dev:/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip; independently observed same HEAD and empty porcelain status.
Verdict: **Approve**, exclusively for this bounded delta and the ensuing 12 actual cases with the execution constraints below.

## Scope and independence

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md in full. Inspected the full five-file base-to-HEAD diff, the relevant accepted requirements, both evidence documents, the candidate test, original fault test and selectors, actual fixed child/supervisor, production runtime initialization/upload/readback, storage parser and guarded lifecycle/deadline/evidence chain. The root REVIEW_PRINCIPLES.md path in the packet does not exist; the authoritative document is collab/REVIEW_PRINCIPLES.md as AGENTS.md states.

The five changed files are exactly the two tests, cleanup_ledger.json, runtime-faults evidence v3 and terminal-matrix implementation note v1. git diff --check passes. The complete e585956-to-HEAD changed-path list contains only these test/ledger/document changes and prior evidence/PROGRESS records; no production source changed. In particular optical_worker has an empty diff against e585956eb29e3a33483d8cb0c7444f893b390ac4. No catalog, profile, native observer, guard protocol or product activation change is hidden in this candidate.

All reviewer scratch is in sibling c4_assets. No candidate files, private context, dependencies or remote files were changed. No reviewer GPU operation, backend/native library import, signal or host fallback was performed. The CPU replay explicitly asserts NumPy/Torch/Warp remain absent from sys.modules.

## Findings and substantive checks

No unresolved blocking finding in the frozen candidate.

1. **Both-lane expansion preserves the real path.** tests/gpu/test_native_runtime_cold_warp.py parameterization is exactly two cases x two lanes x three terminal selectors = 12. Importing PROFILES from the existing fault suite reuses its literal frozen environments and deadlines; it does not execute its tests or import a backend. The latter suite explicitly slices RUNTIME_NATIVE_FAULTS[3:], so its 85 cases do not already cover success/sync-raise/sync-hang. The new terminal cases reach the unchanged -I -S -B native_runtime_process child through guarded_custody_process. The driver calls the captured production SourceSession materialization, initialize_native_runtime, upload_native_input and validate_native_input. It does not implement a second uploader or insert a production fault hook.

2. **Counts follow actual work, not invented lane symmetry.** Production native_bootstrap.py records canonical dependency import, indexed runtime import and upload scope in both lanes. Warp additionally records indexed init, stream lookup and validation readback scope, giving runtime_import_or_init 6 versus CUDA 3. CUDA Stream(device=target) is an allocation leaf. The unchanged native_setup loops each allocate/copy one destination per manifest array; native_readback copies each array and both stages each create/record/wait one event. I inspected these call sites and read all eight actual positive artifacts. Asymmetric has 42 arrays and replay 27: Warp allocations 42/27, CUDA 43/28; H2D/D2H 42/27; event-create/record/wait each 2; runtime counts 6/3. All eight artifacts have 7 ARM and 6 normal COMPLETE. The added CUDA expectations accurately account for the first stream without weakening other counts.

3. **Original deadline and terminal ownership are retained.** Warp remains 30000 ms and CUDA 120000 ms, ACK 1000 ms. The parent communicate allowance (10*deadline +45s) covers the existing supervisor outer fail-bound (10*deadline +30s); it does not renew an operation. Finally closes the parent liveness pipe and waits deadline+15s, replacing the old Warp-only 40s parent kill. wait_supervision commits operation deadline at ARM; emergency ACK retains it, COMPLETE only accepts ARMED_NORMAL, terminal faults select the earlier existing deadline, and only the supervisor performs its owned worker kill/wait. guarded_lifecycle commits termination once, synchronizes at most once, quarantines on sync exception and never retries or sends emergency COMPLETE. The fixed raise/hang selectors enter this sole terminal callable after real preparation; they are fault injection, not claims that an actual CUDA API produced the injected exception/hang.

4. **Success, failed sync and hung sync remain distinguishable.** The test validates the actual supervisor artifact, selected durable runtime record and attachment, full lifecycle count for terminal records, current selection, input validation and no final readiness. Success requires exit 0, RELEASED and completed release. Sync raise requires exit 74, one attempted emergency ARM/sync, failed sync, full quarantine and no authorized release. Hang requires supervisor deadline expiry/kill/reap, return -9, a worker prefix, no worker terminal record and no authorized release. Seven ARM/six normal COMPLETE rejects an emergency completion or second guard. No ordinary atexit/failure fallback is accepted. The worker prefix is deliberately not promoted into a fabricated terminal observation.

5. **The singleton fix is correct and does not weaken the contract.** The actual offending CUDA replay first-active row is instance_X_body_geometry_R, shape **[1,9]**, strides **[9,1]**. The packet's [1,3] was inaccurate; the candidate documents merely say single-row and are not affected. I independently changed its strides to [1,1], recomputed the storage digest and parsed the entire real record: it is accepted. Its only changed stride belongs to a dimension of size one; this does not alter addressed elements or violate the approved nondegenerate C-contiguous progression. The other seven old [1,1] mutations reject. The new test selects an observed dimension >1 and increments its stride; for this row the mutation is [9,2] and rejects. All 96 current storage mutations passed in independent CPU replay, with all eight original roundtrips/attachments and no-live-authority tests also passing. Production parsing and live registration enforcement are unchanged. A descriptive, recomputed digest is not physical storage authority, and acceptance of another legal singleton layout is not an authority bypass.

6. **Cleanup entry is bounded and concrete.** The added entry has exactly 20 unique existing paths: custody/manifest modules, three support helpers, six unit-test modules, four manifests, two GPU-test modules and three CUDA profiles. Both lane outcomes specify collapse, selected-backend retention, removal of losing rules/profiles and paired machinery only after equivalent coverage, historical evidence retention and no parsed authority. The P12.4e deadline and open/must-complete status remain. No deletion or selection occurs. I checked path existence/uniqueness and the actual dispositions, not just a count. I did not re-run the broad benchmark pytest module, whose imports are outside this deliberately backend-free review; its author-reported one-test pass is not presented as my execution.

7. **External identity and execution runner.** The test joins external worker/supervisor/pytest PIDs and exact GPU UUID/driver rows and rejects CUDA_VISIBLE_DEVICES. These checks depend on the reviewed same-run external recorder: it reads live procfs PID/PPID/start ticks/executable/argv, checks the actual -I -S -B child and supervisor paths and per-case directory, then rechecks the worker start time. A clean unique basetemp prevents historical files from satisfying this run. I reviewed c4_assets/run_c2d3_terminal_matrix_v1.py as an execution prerequisite outside the five-file candidate. Its initial inherited 60000s outer-timeout branch called process.terminate() on pytest. I reported that this was unnecessary caller-owned terminal intervention for this scope, rather than a defect in the frozen candidate. The author removed the branch while leaving the candidate frozen. I inspected the revised script and verified SHA256 **aa2302b18d3a0b20162b9d955d2a621be991572a55480030e1a542a558018c58**; it now only communicates/observes and contains no terminate/kill call. This concern is closed. The packet prohibited reviewer signals; this observation should not be generalized into a claim that every owner-authorized signal to an owned process is forbidden.

The revised external observer intentionally has no independent terminal authority. If the host/observer is unresponsive, silence or a stalled wait is not success: preserve the failure and live identities for diagnosis. Polling is descriptive trusted-deployment evidence, not cryptographic process attestation. A short-lived missed chain must fail acceptance, not be reconstructed afterward.

## Independent evidence verification and reproduction

Local interpreter: C:/Python314/python.exe -B. Remote interpreter: /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B on primary tileops-dev. No -I for pytest imports in the reviewer CPU script; actual fixed worker's isolation flags were checked from code and retained identities.

Commands/checks included:

- git rev-parse HEAD; git status --porcelain; git diff --stat/name-only 17bae2c... HEAD; git diff --check 17bae2c... HEAD; complete changed-path comparison against e585956; remote git HEAD/status over SSH.
- AST-only terminal parameterization check: 12 cases, without executing/importing the GPU test.
- stdlib tarfile/hashlib reads of the original archives and raw logs; no extraction or rewriting of evidence.
- Local CPU script: C:/Python314/python.exe -B ../c4_assets/terminal_matrix_preflight_repro_v1.py. It verified all 85+16 external identity/artifact joins through production validate_custody_artifact, then hit a Windows-only environment-descriptor limitation: pathlib treats retained Linux deployment paths as non-absolute. This is an environment limitation, not a parser-contract failure. An earlier scratch invocation also used the wrong archive stdout member name (pytest_stdout.txt instead of stdout.txt); corrected only the scratch reader.
- Successful independent replay used the same scratch script streamed into SSH stdin, with its archive directory changed from root.parent/c4_assets to root.parent: `ssh tileops-dev 'cd /data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip && /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -'`. No remote script file or evidence was written.

Successful independent replay completed in **91.44s**, exit 0: all 101 raw identity/artifact joins; eight original record byte-exact roundtrips, durable-selection/lifecycle-count checks and attachments; eight no-live-authority checks; old [1,1] accepted only for CUDA replay; **96/96 current storage mutations passed**. These are calls of the candidate's actual mutation-test function with all 12 field/value pairs per real record, not a reimplementation of the validator. NumPy/Torch/Warp were not imported.

Retained reviewer artifacts:

- c4_assets/terminal_matrix_preflight_repro_v1.py, SHA256 2e0bfa2b9e397ce45a0dd8152acfbe76bb28524e8fc6302009b8e92e2aa9fd7c.
- c4_assets/terminal_matrix_preflight_repro_v1.log: original local attempt including its explicit platform limitation.
- c4_assets/terminal_matrix_preflight_repro_remote_v1.log, SHA256 447acaa921a9feacaed60cf6571d9018ab8d19e2fcfe062f2327d0e8c56a196b: successful independent replay and original failure excerpt.

Verified original archive hashes:

| Archive | SHA256 | Independently checked contents |
| --- | --- | --- |
| c2d3-e585956-runtime-fault-matrix-v4.tar.gz | df9f92da1bfe8dcf4095dd598381c22f4b0974f1e982b89af97ef56eed83b359 | 85 raw artifact/external identity chains, stdout 85 passed in 4488.89s; archived 172 owned PIDs with empty OS/GPU residuals |
| c2d3-e585956-actual-record-mutations-v4.tar.gz | 2b6b073073c9ab82e2af4669459c70f2eda3db3c4f45b1bba79b35ca8a7a019d | Original failing assertion and 1 failed/451 passed/36 skipped in 732.43s retained |
| c2d3-e585956-native-scope-v1.tar.gz | c84af059c9e32a8048b1b6d9bb5efa7af09e62a8a7e38d54fa3b5d0b444dc230 | 16 raw identity/artifact chains; eight actual positives replayed; actual lane work counts |
| c2d3-7cde887-actual-record-mutations-v5.tar.gz | 82f462d83c4483b48d40cf20243d9af341254132dcf02eca3900c74b78f7713f | Author stdout 96 passed/392 deselected in 49.90s, empty stderr, return 0, residual empty; all 168 recorded original-file hashes independently match original native-scope tar |

The old 85/16 runs and their residual observations remain historical e585956 execution evidence. I parsed their original records and verified their joins; I did not recreate their native executions or independently observe their past live processes. In particular the 85 matrix's cleanup-hang cases have prefixes/no authorized release, while missing-exit hangs may retain a completed worker release plus supervisor deadline-kill: these are not successful process disposal. New execution still requires its own same-run external identities and final scoped OS/GPU residual check.

## Approval boundary and remaining checks

Approve exactly clean 7cde8879b307090d54350ca3511529bfd8e5d7c8 relative to the stated base for this test/ledger/document repair. Permit the 12 actual terminal cases on the primary matching deployment, using the reviewed revised external runner, unchanged frozen profiles/30s-120s operation deadlines, original fixed native driver/supervisor, a fresh basetemp and a clean owned-process precheck. Capture stdout/stderr, requests, upstream inputs, selection/record/artifact, current external identities, failed attempts and final owned OS/GPU residual results. No skip, missing chain, outer failure, partial selection or unfinished process is a successful terminal case. Preserve any failure and return it for review.

I did not run GPU tests or load backend/native libraries. I did not rerun the entire 488-item actual-record pytest suite, the foundation/C0 regression suites or the full 85 GPU matrix; only affected CPU mutations/roundtrips/authority checks were independently executed. These limits are appropriate for permission to run the next bounded matrix and are not waived whole-stage evidence obligations.

This is **not whole-C2d3, C2d or C4 approval**. Native outcomes from the 12 cases, affected regressions, complete evidence audit and fresh independent whole-stage review remain required. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain closed. No renderer, performance result, backend selection, losing-lane deletion or product activation is authorized.
