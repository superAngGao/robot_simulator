# Warmup bootstrap ownership repair independent code review v1

Initiative: Q54 / P12.4 / C4 runtime integration
Stage: independent code review of bounded warmup bootstrap ownership repair
Author: codex
Version: v1
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3 sections 4, 5, 5.1, 8, 9 and review codex v3; fixed-render-warmup code-review codex v2; warmup-bootstrap-repair implementation-note codex v1
Owner Summary: Approve the exact bounded repair. The original source-session entry now validates its uploaded input and runtime binding inside the existing setup guard and passes the retained objects to warmup preparation, avoiding the forbidden bootstrap reimport. Fresh CPU checks pass. This does not establish native warmup acceptance; a source-frozen native rerun remains required.

Reviewer session: `/root/warmup_bootstrap_repair_v1`, a fresh independent Codex reviewer given the review packet and repository without the implementation conversation. No implementation authorship, candidate edits, or subagents.

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`

Branch: `implementation/c4-supervised-waits`

Delta base: `0dbc13b9613e6238cba58b63ba0b0b7b8e8f0845`

Reviewed HEAD: `0ede8d04a9ce99c9c2562cd1dfe1e692c7e5c526`

Actual tree status: clean at initial, intermediate, and final checks; HEAD unchanged.

Verdict: **Approve**, only for the bounded code repair and refreshed source pins/catalog at this exact revision. No substantive blocking or non-blocking finding remains. Native execution is a separate, unpassed gate.

## Inspection and findings

Read the complete project collaboration instructions and review principles, requested design sections, accepted design review, prior bounded warmup code review, and current implementation note. Inspected the complete six-file delta and affected session, import-finder, runtime-binding, helper authority, transaction, resource-scope, guard-failure, and completion paths. Author test counts and earlier approvals were treated as claims, not fresh evidence.

At optical_worker/native_bootstrap.py:3430, the only production caller of `_prepare_fixed_render_warmups` remains the existing session entry. It checks live state and the once-only graph-ready phase, enters the original normal setup guard, moves to the warmup phase, checks the exact original uploaded input identity, and invokes the existing runtime-binding validator before importing/preparing warmup. The validator requires active setup and exact session/controller/admission/module/runtime/target/stream/source bindings, stream storage identity, lane runtime configuration, and adopted owners. Its first return value is the lane runtime module, matching the prior helper behavior and existing storage/graph preparation pattern.

At optical_worker/warmup.py:114, preparation consumes that validated module/target/stream rather than reverse-importing the process bootstrap registry. There is no bootstrap module alias, replacement registry, fallback loader, source-policy exemption, public authority factory, or additional source inventory. This remains a private fixed trusted-worker call, not an arbitrary injected-runtime interface. Exact graph/readiness/adoption checks, one-shot warmup registration, session/external audits, target resource scope, retained transaction construction, and the actual leaf authority checks remain in place.

Moving the validation earlier means upload/runtime rejection occurs before warmup owner-list allocation or issuance reservation. This does not introduce a retry: it occurs inside the same active guard, whose exception path marks the segment failed and quarantines existing custody without COMPLETE or emergency synchronization. The outer session failure path retains poison. The successful path still completes the preparation guard before running the ten separately guarded transactions, and completion evidence still depends on actual terminal observation and DISARMED before advancement. No transaction, completion, readback, or final-readiness implementation changed.

The source-policy delta changes only the bootstrap whole-source pin and `_SourceSession` class pin. Independently recomputed all 86 catalog source hashes from HEAD Git blobs. The source inventory and all catalog metadata except `ast_policy_sha256` and source digests are unchanged. Exactly three digests changed: optical_worker/native_bootstrap.py, optical_worker/native_source_policy.py, and optical_worker/warmup.py. The catalog policy hash matches the policy module's derived identity. Catalog SHA-256 is `f6208cb33d29476878925e81f4f7e3c0665ca84f718aac4ff042f35b5a5fc70a`. Cold-source and negative-control tests also pass with these pins.

## Failure evidence and original counterexample

Read the preserved local tar archive without extraction or mutation:

`C:/Users/AG/新建文件夹/c4_assets/c4-render-warmup-failure-evidence-v1.tar.gz`

Independently verified SHA-256: `7cc7eb78fa664aa1439fbc61730700996433189cceaeafc8653ca8d5dd64b67c`.

The request records Warp Go2 960x640, target1, revision `3a3cd3af583cf28a9cbc3addeab360867de6770a`, and the prior catalog. failure.json reports the generic SourceBootstrapError `repository import outside CPU core or reload`; warmup_partial.json has an empty frame list. Worker evidence records failed setup guard sequence13, 13 normal completions, full quarantine, zero emergency sync and no warmup-launch count. Supervisor evidence records EXITED, observed worker return74, no deadline expiry or kill request. Parent/worker identity files and the retained live observation correlate parent3444918, supervisor3445180 and worker3445181 with start ticks. This reviewer did not independently query current remote process state or the coordinator's batch handle.

The raw exception does not identify a module. The exact bootstrap reimport is therefore a code-backed inference, independently reproduced here rather than attributed to that raw message. The scratch probe reads the two pre-repair function definitions directly from the base Git blobs and temporarily installs them into the test process. It then invokes the committed import-rejection regression unchanged. On both lanes, the call fails at old warmup.py line115, `from .native_bootstrap import _UPLOADED_INPUTS`, before any optical call, and reaches quarantine. The repaired functions pass the committed regression and complete ten transactions with the expected launch counts. The test models the cold-bootstrap import prohibition; it is not a complete native loader/GPU run.

The same scratch file restores the actual production runtime-binding validator, which the graph fixture normally doubles, and constructs its exact local binding/record shape. Separate runtime-root, target, and stream identity substitutions on both lanes are rejected by the production entry before warmup preparation or registration. Each checks zero optical calls, no completion evidence, no extra normal completion, zero emergency sync, quarantine, and unchanged ambient scope. These are CPU identity/failure checks, not real native ownership attestation.

## Fresh executed verification

Environment: Windows PowerShell, `C:/Python314/python.exe`, `-B`, pytest cache provider disabled; all commands run from the frozen candidate. No package installation, SSH, GPU job, or benchmark.

1. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_render_warmup.py tests/unit/optics/test_native_warmup_support.py`

   **64 passed in 104.32s**, exit0, tool session90720. Covers the new import rejection and missing/replaced-upload tests in both lanes, existing successful transactions and guards, authority/plan attacks, leaf/ACK failures, and driver observation/readback support. Native interfaces and Warp command construction remain explicitly doubled.

2. `C:/Python314/python.exe -B -m pytest -q -rs -p no:cacheprovider tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_bootstrap.py`

   **281 passed, 13 skipped in 180.82s**, exit0, tool session91555. Skips: two missing real-Go2 asset configurations, two POSIX release-byte checks, two actual POSIX bootstrap-release cases, and seven pinned CPython3.12 POSIX generated-module cases. These are environment/asset limitations, not native passes. Tests create current-source LF release fixtures and negative controls in pytest temporary directories outside the candidate; their generated fixture commits are not the reviewed revision.

3. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/test_warmup_bootstrap_repair_review_v1.py`

   Final probe: **8 passed in 3.69s**, exit0. Probe SHA-256: `aa871eb3b2dc13b5bdf79ea9687abbebb56d2c1f517ad7e520640fb76ec450e5`.

   The first scratch invocation produced 2 passed and 6 failed in4.26s because the reviewer incorrectly assumed the graph fixture retained the full runtime record; it deliberately supplies a one-element record and a doubled validator. All six failures were IndexError in scratch setup before invoking the candidate. The scratch probe was corrected to restore the actual validator and its record shape, then all eight passed. No candidate change or production failure was concealed.

4. Read-only Git diff/status/HEAD/branch checks, `git diff --check 0dbc13b9613e6238cba58b63ba0b0b7b8e8f0845 HEAD`, archive hashing/selected JSON reads, and independent catalog/Git-blob verification passed. An initial diagnostic wrongly asserted all non-source catalog fields were unchanged; inspection showed the expected derived `ast_policy_sha256` refresh. The corrected check verified that hash against the production policy constant and every other catalog field unchanged. An initial root REVIEW_PRINCIPLES.md lookup failed; the actual collab/REVIEW_PRINCIPLES.md was then read fully. These were reviewer diagnostic assumptions/path errors, not candidate defects.

Final successful suite/probe runs total **353 passed, 13 skipped**. The earlier scratch-setup failures are separately disclosed above. No unrelated broad timeline matrix was repeated for this small ownership/import delta.

## Scope, limitations, and remaining gates

Approval covers removing the reverse bootstrap import, retaining upload/runtime validation in the existing guarded session entry, passing the validated objects into existing warmup preparation, the focused regressions, and corresponding exact pins/catalog. It introduces no architectural or product authority change.

The repaired source-frozen Warp/CUDA native warmup campaign must still run after this approval. This review did not prove real packed-command execution, successful Go2 images, numerical correctness, target1/ambient0 native restoration, real native partial-submit/failure custody, warmup hangs/reaping, sentinel reset, or performance. The CPU import regression cannot establish that no later integration failure will occur. The existing failure archive is diagnostic evidence of the old failure, not acceptance of the repair.

R1, new final readiness/origin, final factories/commands/reset, C3 registration, final coverage/quiescence and complete lifecycle integration retain their prior unfinished/closed status. Product PREPARE/sample/lease/IPC activation, R2/S/host delivery, external holds, result/performance/baseline/backend selection/shipping authority, whole-C4 acceptance and the later systematic whole-C4 refactor/review remain closed or outstanding. No parent gate is opened by this bounded verdict.

## Process hygiene and artifacts

Focused pytest PID46728, parent43336, creation2026-10-10 04:01:42 +08:00, completed through session90720 with exit0. Source suite PID46100, parent44120, creation04:02:21, completed through session91555 with exit0. Sampled isolated bootstrap children PID39764 (creation04:04:00) and PID31348 (creation04:04:45) had parent46100. The short scratch test invocations and read-only diagnostics returned terminal tool results before separate PID sampling; they are not claimed as exhaustive child-process attestation.

At `2026-10-10T04:05:24.8546701+08:00`, the final scoped Win32_Process query found no recorded pytest/shell/child PIDs, no remaining direct children of the two recorded pytest processes, and no Python invocation naming the scratch probe. All reviewer-owned tool handles were terminal. No reviewer-owned residual or unresolved hang was observed. Existing subprocess tests own their normal timeout/reaping behavior; this reviewer sent no manual signals and did not inspect or terminate unrelated work. No remote/GPU process was started, queried, or reset.

Only this report and the independent scratch probe were written by this reviewer, both under c4_assets outside the candidate. Original evidence and candidate remained unchanged. The coordinator may archive this complete report unchanged in an evidence-only collab commit; that recording cannot alter the reviewed revision, verdict, scope, or closed gates.
