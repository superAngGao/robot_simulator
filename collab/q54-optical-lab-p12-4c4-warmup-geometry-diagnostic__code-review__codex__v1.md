# Warmup geometry readback independent code review v1

Initiative: Q54 / P12.4 / C4 render integration
Stage: independent bounded support-code review
Author: codex
Version: v1
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3 sections 5 and 9 and design review v3; fixed-render-warmup code-review codex v2; warmup-readback-checkpoint code-review codex v1; warmup-geometry-diagnostic implementation-note codex v1
Owner Summary: Approve the exact bounded geometry diagnostic support revision. Fresh verification passed 50 CPU tests, including independent partial-failure custody and replay attacks. This does not establish native geometry execution or numerical correctness. The reported P1 RGB disagreement and every wider acceptance gate remain unresolved by this review.

Reviewer session: `/root/warmup_geometry_readback_code_v2`, fresh independent Codex session supplied only the review packet and repository. No implementation authorship, implementation edits, subagents, GPU jobs or remote/native jobs.

Repository: `C:/Users/AG/新建文件夹/c4_render_execution_next`

Branch: `implementation/c4-render-execution-next`

Base: `9067e114ed609928904bf3cd95b871554bb15d26`

Base tree: `dc1d08d5c92b401b68d6397dc07037e917833128`

Reviewed HEAD: `5a5a17cdd4e75c01016e1e71738aed99d9dab8fe`

HEAD tree: `f1eeafc1b7ff8833af579ce9797ce05c7fd08f1c`

Actual status: clean on initial, intermediate and final `git status --porcelain=v1 --untracked-files=all`; HEAD and branch unchanged. The report and scratch probe are outside the candidate.

Verdict: **Approve**, only for the bounded support-code delta and regression coverage described below. No substantive blocking or non-blocking code finding was established. Native acceptance remains a separate, unexecuted gate for this candidate.

## Scope and evidence basis

Read the project instructions, complete collaboration protocol and review principles, accepted runtime-integration design sections 5/5.1/5.2 and 9/10, design-review conclusions, prior warmup and checkpoint reviews, and candidate implementation note. Inspected the full actual four-file diff and affected selector, preparation, readback, owner-retention, guarded failure, checkpoint, replay and terminal-disposal paths. Prior author/reviewer results were treated as claims rather than fresh proof.

The delta is exactly two test support drivers, one unit file and the implementation note: 189 additions and 20 deletions. There is no production source, source catalog, kernel, numerical parameter, tolerance, source-policy, supervisor deadline or product-gate change. Production/catalog equality to the base follows from the complete Git diff; this review did not rerun the broad production source-hash suite.

The task packet reports six substantial 1920 RGB discrepancies and a CPU54 diagnostic with material disagreement against CUDA at four and Warp at two selected indices. Those numerical artifacts were not independently replayed in this code review. They motivate the diagnostic; this approval cannot close that P1 blocker. The prior `c4_assets/render_warmup_native_acceptance_review_v1.md` is a separate bounded lifecycle review, not evidence of executing this geometry candidate. No result from that review is counted among the fresh checks below.

## High-risk behavior checked

1. **Selector and scope.** `tests/support/native_runtime_process.py:78` adds a single fixed selector to the existing graph/warmup case sets. `child()` still rejects selectors outside the fixed list and extra request fields. The selector uses the existing storage preparation, graph construction, ten production warmups and same native lane. It does not end in `diagnostics`, so existing preparation selects `primary_unprofiled` and ordinary storage. The graph case remains a no-injection case in `_install_runtime_fault`. No product callback or generic channel argument is exposed. `geometry=True` is supplied only by this fixed selector in the worker.

2. **Original data and ordering.** At `native_runtime_process.py:1112`, one local readback body handles RGB first, then the fixed eight-channel order. Every value comes from the existing `session.fixed_render_storage[3]`; there is no reconstruction, extra rendering, new upload, kernel launch, GPU allocation/preparation path or changed arithmetic. Both native output ABIs were inspected: Warp's `WARP_DIRECT_LIGHT_BUFFER_SPECS` and CUDA's existing buffer allocation/validation agree with the proposed shapes and scalar dtypes. Raw geometry is diagnostic data, not proof that the values are numerically correct or finite.

3. **Custody before conversion and the next copy.** `_native_resource_scope` adopts its owner list before entering the held target/stream scope. The helper appends each returned host object before individual host adoption. CUDA's CPU tensor therefore remains held even if adoption or `.numpy()` fails; its NumPy view is appended before byte serialization, and encoded bytes are also retained. Earlier results stay in that same held list across later channels. The eight additions reuse the already existing readback setup guard, after the tenth physical completion. They add copies and CPU serialization/files, not another guard or synchronization policy.

4. **Partial failure and checkpoint.** A copy, adoption, conversion or file-write exception unwinds through the original resource scope and guarded segment, quarantines the retained graph, and skips subsequent channels and the final observation checkpoint. Existing outer child failure handling calls `session.fail`, preserves terminal custody where possible, and ends by the existing hard-exit policy; no reusable-success path was added. The production warmup observer hook is restored in `finally` before readback starts. On success, the post-readback checkpoint remains outside both context managers and thus follows successful COMPLETE/DISARMED. This preserves the prior stale-cut repair. A geometry manifest is written only after all eight channel files; artifacts alone cannot prove guard success, and a later ACK/checkpoint failure still fails whole-case acceptance.

5. **Replay and original selectors.** At `cuda_artifact_runtime_acceptance.py:250`, geometry requires exactly the eight channel names, matching final RGB frame and guard, native shape/dtype, exact byte count and SHA-256. Copy declarations must equal the exact ordered list of RGB plus all eight channels. Original RGB-only selectors still require exactly one copy. The helper checks the matching completed guard; upstream `run_case()` first validates the full custody artifact and its ordered guard state machine, which enforces that DISARMED follows normal body completion. Thus the helper's `any(normal_disarmed)` is used with an already validated trace, not as standalone authentication. Final runtime-custody parsing/attachment and successful terminal disposal remain mandatory. Hashes and serialized declarations are consistency checks, not new native authority.

6. **Bounded temporary support.** The note retains the existing sacrificial-driver cleanup obligation. This is sufficient for this small diagnostic addition; it does not introduce a second renderer, materializer, readiness authority or permanent product interface. The diagnostic files and extra host storage are untimed and cannot support a performance claim.

## Fresh verification

Environment: Windows PowerShell, `C:/Python314/python.exe`, bytecode disabled with `-B`, pytest cache disabled. All tests ran from the frozen candidate. Fixtures use native doubles while exercising original source-session, lifecycle and warmup support code. No GPU, SSH, remote query, native worker/supervisor launch, installation or performance run occurred.

1. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_warmup_support.py tests/unit/optics/test_cuda_artifact_runtime.py`

   **40 passed in 26.29s**, exit 0, tool session `18170`. Includes both lanes' RGB-only and geometry support, exact bytes and tamper rejection, existing mid-copy failure, original observer-failure preservation and artifact acceptance regressions. This is a new execution, not the author's reported 10-test result.

2. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider ../c4_assets/test_warmup_geometry_review_v1.py`

   **10 passed in 19.96s**, exit 0, tool session `4250`. Independent scratch probe, SHA-256 `2476a17c16b913e2a7c6524ad8a8d404195928af1c12dbcf4296a48a1be9bbff`.

   Both lanes inject an original error at directions-host adoption, host conversion/serialization, directions-file write, and final geometry-manifest write. Probes require exact exception identity, retained current and prior returned hosts in the already adopted custody list, no later copy after the failing channel, no complete geometry manifest, only the original warmup checkpoint, restored observer hook/device/stream, quarantine, no readback DISARMED, and zero emergency-sync attempts. The manifest-write case reaches all nine copies before failing. Retention is checked before each subsequent copy as well as after failure.

   Both lanes also run the actual committed positive test through an independent validator wrapper. Missing/extra channels, changed shape/dtype/hash/frame, and reordered/duplicated/missing/cross-guard copy declarations all reject. The original positive bytes and metadata are restored between attacks. These probes call the actual implementation and replay validator; they do not duplicate the implementation algorithm.

3. Read-only Git revision, branch, tree, status and complete diff inspection; `git diff --check 9067e114ed609928904bf3cd95b871554bb15d26 5a5a17cdd4e75c01016e1e71738aed99d9dab8fe` passed.

Total: **50 passed, no skips**. No candidate failure was repaired or hidden. Test outputs are local CPU verification only.

Inspection tooling issues: the first unquoted PowerShell `HEAD^{tree}` argument was misparsed and then rerun quoted; initial Chinese text decoding was corrected by explicitly reading UTF-8 and rereading the full review principles. A few speculative file-path searches failed or used a PowerShell-incompatible wildcard path; actual resource/lifecycle and CUDA ABI files were subsequently located and inspected. Some broad output was truncated; relevant implementation/design sections were reread in bounded chunks. These are inspection/tooling limitations, not test failures.

## Limits and remaining gates

No actual Warp `.numpy()` or CUDA `.detach().cpu()` geometry transfer, pinned native source-bootstrap execution, 1920 output, real target1/ambient0 scope, readback duration/deadline, native owner lifetime or whole-case disposal was executed for this revision. The fixture checkpoint recorder proves placement and current lifecycle equality; it does not itself execute full live runtime-custody parsing. Native ACK-loss/hang and process-reaping matrices were not rerun. These properties remain for the separate source-frozen native checkpoint; mocks do not pass them.

Approval permits this bounded diagnostic support to proceed to its separately authorized native check. It does not approve numerical correctness, resolve the reported P1 discrepancy, permit tolerance/math changes, establish full CPU/all-channel/sentinel acceptance, or certify the earlier campaign. Final readiness/origin, final recording/reset, complete coverage/arena/barrier/lifecycle integration, R1, product PREPARE/sample/lease/IPC activation, R2/S/host delivery/external holds, result/performance/baseline/backend-selection/shipping and whole-C4 completion remain closed or unfinished. The later systematic whole-C4 refactor and independent review remain required.

## Process hygiene and artifact custody

Focused pytest PID `44852`, parent `25152`, creation `2026-10-10 04:53:02 +08:00`; scratch pytest PID `452`, parent `47072`, creation `2026-10-10 04:54:07 +08:00`. Both tool sessions reached terminal exit 0. At `2026-10-10T04:54:41.6137382+08:00`, scoped `Win32_Process` queries found none of these recorded processes, no remaining direct children of the pytest PIDs, and no Python process naming either exact test command/scratch file. No unresolved reviewer-owned residual or hang was observed.

Short inspection commands completed terminally without separately sampled PIDs. No manual signal, broad cleanup, GPU-process query or reset was performed. The ledger is scoped local evidence, not an exhaustive attestation about unrelated or prior native processes. No GPU-process check was needed for this CPU-only review, and no remote process-cleanliness claim is made.

Only this report and `c4_assets/test_warmup_geometry_review_v1.py` were written by the reviewer outside the candidate. Pytest's normal temporary directories reside outside the candidate; no tracked or untracked candidate side effects were observed. The coordinator may archive this report verbatim in an evidence-only commit. That archive cannot change this verdict, reviewed revision, approved scope or remaining gates.
