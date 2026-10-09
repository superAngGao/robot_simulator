# Fixed R0 render warmup independent code review v2

Initiative: Q54 / P12.4 / C4 runtime integration
Stage: independent code re-review of private R0 setup warmup and positive native-driver support
Author: codex
Version: v2
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3 sections 4, 5, 5.1, 6, 8, 9 and review codex v3; remaining-setup gap-scan codex v1 G2/G3; fixed-render-warmup code-review codex v1 and implementation-note codex v1/v2
Owner Summary: Approve the exact bounded code revision. Original F1 is closed: the actual Warp execution plan and cursor reject ordinary rebinding, and the unchanged original counterexample now stops before any launch/event or completion issuance. Positive native-driver observation/readback support passes CPU checks, including an independent host-retention failure probe. Native execution, optical correctness, performance and whole C4 remain unapproved.

Reviewer session: `/root/fixed_render_warmup_code_v2`, fresh independent reviewer supplied the task packet and repository without the implementation conversation. No implementation edits or subagents.

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`

Branch: `implementation/c4-supervised-waits`

Delta base: `04b74aca5a5f6734e1b7d84ff4fe2eb7014ef796`

Whole warmup base: `e43a86c8e63a37da7bd84a5a8f5616559c7e12a1`

Reviewed HEAD: `3a3cd3af583cf28a9cbc3addeab360867de6770a`

Actual tree status: clean at initial, intermediate and final checks; exact HEAD and branch unchanged.

Verdict: **Approve**, exclusively for the bounded code scope below. No substantive blocking or non-blocking finding remains. Missing native acceptance is an explicitly separate gate, not a passed property of these tests.

## Scope and review basis

Read AGENTS.md, the complete collaboration protocol and complete review principles, the requested design sections and accepted review, G2/G3, prior F1 and author notes. Inspected the complete eight-file repair delta, whole-warmup source changes, and the affected authority, source-session, resource-scope, guard, transaction, bundle validation, observer, readback and replay paths. Reread relevant text in bounded chunks after broad tool-output truncation. The initial attempt to read REVIEW_PRINCIPLES.md at repository root failed; the actual collab/REVIEW_PRINCIPLES.md was then read fully. No source failure or candidate change resulted.

Approval covers the private fixed R0 ten-transaction setup executor at this revision, its immutable local completion evidence, exact setup authority and source closure, plus the two fixed positive native-driver selectors and their diagnostic observation/readback/replay support. It is not native warmup acceptance. Author verification summaries and prior approvals were treated as claims; all counts below are fresh executions in this reviewer session.

## F1 — closed against the original counterexample

The original defect was validation of bundle.commands followed by execution of a separately assignable transaction._groups. Empty replacement groups therefore allowed ten FINISHED transactions, actual terminal waits and accepted evidence with zero optical launches.

At optics/warp_timeline.py:16, _WarpSetupTransaction is now a frozen, slotted dataclass with identity equality. Its actual ordered tuples, bundle, resolved parameters, runtime, target, stream, event bindings and cursor/state are initialized internally and reject ordinary assignment/deletion. Reinitialization is rejected once initialized. The groups are derived once from the validated bundle's exact descriptors and commands; neither the original frozen bundle tuple nor the new plan tuple has a supported independently mutable replacement path. Before enqueue, the existing bundle validator still checks the original recorded command identities and packed arguments. Internal progress uses object.__setattr__; this does not claim resistance to arbitrary trusted-process reflection or monkeypatching.

Independently read and SHA-256 checked the preserved original probe:

`C:/Users/AG/新建文件夹/c4_assets/warmup_review_v1/test_warp_plan_drift.py`

SHA-256: `ee93f4b16576c3b858ccd3eada07ab048a4e96e87486cfa90f323a3d6d2bccba`.

Read the supplied external wrapper and verified that it invokes that original test function unchanged, requires FrozenInstanceError, and then checks zero launch/event/wait calls and no registered completion evidence. Both diagnostic variants passed in the focused run. The new committed attack test additionally rejects empty, reversed, foreign and cross-frame plans; cursor/state, bundle, terminal and stream replacements; deletion; and reinitialization. It then executes the untouched plan and observes fifty launch calls and ten terminal waits. Copied transactions still fail exact identity registration in both lanes.

This directly closes the original ordinary-rebinding counterexample without adding another renderer or another mutable plan registry. Native packed-command execution is not inferred from the doubled bundle fixture.

## Other high-risk checks

- The original setup authority still binds the exact session/controller/guardian/graph/runtime/target/stream and active transaction/guard/resource operation. It checks the live setup phase, CONSTRUCTING guardian, known exact target and adopted owners before each leaf. Completion evidence still requires ten distinct retained transactions/completion objects, the actual terminal waits, and each original COMPLETE/DISARMED before event reuse. Provisional readiness remains warmup0. Retirement and copied-evidence rejection remain intact.
- Warp reset/phase ordering and diagnostics-off behavior are preserved. Frozen cursor updates do not add native allocation or waits to enqueue/finish. Launch, record, wait and DISARMED failures still quarantine without another normal completion, emergency synchronization or next-frame work. CUDA continues to reuse its existing render stage/transaction implementation; its ordinary transaction behavior is unchanged by the repair delta.
- tests/support/native_runtime_process.py:1049 wraps the original resource scope only for the fixed warmup frame scope, observes before/inside/after state, and restores the hook in finally. Optional failure-artifact writes cannot replace the original exception. Returned evidence is checked against actual local issuance and adopted owners before serialization. The fixed native worker remains serial; this is test-driver observation, not a product callback or new authority source.
- The additional RGB readback runs after the tenth physical completion under another original normal setup guard and the original held target/stream scope. The returned host owner enters the already adopted owner list before further conversion/adoption; CUDA's NumPy view and encoded bytes are likewise retained in that list. The RGB artifact is a diagnostic copy, not a result/sample/host-delivery authority. Normal guard counts in run_case include prepare, ten frame completions and this readback; successful terminal disposal remains checked through the existing supervisor artifact.
- The new replay check cross-checks upstream input/readiness/module/catalog/configuration and bundle identities, ordered frame/guard/terminal records, leaf kinds and returned flags, target1/ambient0 scope/restoration, and normal guard completion. It verifies final-frame dimensions, byte length/hash, finite float32 output and the completed readback-copy guard. Existing guard/custody replay remains upstream. These serialized diagnostics are consistency checks, not independently authentic local authority or numerical image correctness.
- CUDA warmup_launch's returned flag describes its declaration context immediately before the existing native entry. It is not proof that the kernel returned or completed. Acceptance still depends on the real terminal wait and original guard completion. Warp's corresponding observation surrounds the recorded launch itself. This distinction is present in the code and author note and must remain explicit in any later native report.
- Independently recomputed all 86 source hashes against HEAD Git blob bytes. The repair catalog differs from its base only in the optics/warp_timeline.py digest; lane inventories, schema and policy identity are unchanged. Catalog SHA-256 is `946ec6c70a03bce47178f9f7c40b74e8534714b54c2112ec033742a64b35e16e`. The whole warmup delta's source extension remains limited to the two warmup phases and the stated lane-specific files. No new source-policy exception or product gate is introduced.

## Executed verification

Environment: Windows PowerShell; `C:/Python314/python.exe`; `-B`; pytest cache provider disabled; commands run from the frozen candidate. No installation, SSH, native GPU job or performance run.

1. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_render_warmup.py tests/unit/optics/test_native_warmup_support.py tests/unit/optics/test_native_render_graph_support.py tests/unit/optics/test_cuda_timeline.py tests/unit/optics/test_native_render_preparation.py C:/Users/AG/新建文件夹/c4_assets/test_warmup_original_repro_rejected_v2.py`

   **152 passed in 109.49s**, exit 0, tool session 21442. This includes positive profile/shadow/diagnostic combinations, F1/copy attacks, leaf/ACK failures, observer/readback tampering, graph preparation and existing CUDA timeline regressions. Native interfaces and the Warp bundle/packed construction are explicitly doubled in warmup fixtures.

2. `C:/Python314/python.exe -B -m pytest -q -rs -p no:cacheprovider tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_scope_observer.py tests/unit/optics/test_cuda_artifact_runtime.py`

   **335 passed, 22 skipped in 187.95s**, exit 0, tool session 85704. Skips: two missing real-Go2 asset configurations; two POSIX release-byte checks; two actual POSIX bootstrap release cases; seven pinned CPython3.12 POSIX generated-module cases; nine pinned Warp-source exact-Device observer models. These skips are environment/source limitations, not native passes. The source suite creates current-source LF release/negative-control fixtures outside the candidate; their generated fixture commits are not the reviewed revision.

3. Independent scratch probe, written only outside the candidate:

   `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/test_warmup_readback_retention_review_v2.py`

   **2 passed in 11.80s**, exit 0, tool session 20623. Both lanes complete the actual production warmup driver with native doubles, then fail after the RGB host owner has returned: Warp at host dtype access, CUDA at host NumPy conversion. The probe requires the exact original error, restored hook/ambient scope, the host in both its adopted role and retained custody list, whole-graph quarantine, zero emergency retry, and no RGB data artifact. SHA-256: `d913efd1e950b753f3213544fb671cf19bbfb23b078ec1b757f608fc69d52082`.

4. Read-only Git status/HEAD/branch/diff inspection, `git diff --check e43a86c8e63a37da7bd84a5a8f5616559c7e12a1 HEAD`, and an inline Python hashlib/json/subprocess diagnostic comparing the catalog and every listed HEAD blob: all final checks passed. The catalog diagnostic asserted that its sole changed source digest was optics/warp_timeline.py. The original repro digest was checked independently; neither that repro nor the supplied wrapper was edited.

No test failure was hidden, repaired in the candidate or reported as success. Total executed pytest outcomes: **489 passed, 22 skipped** across the three runs.

## Limits and remaining gates

No real native warmup, real render output, numerical CPU-oracle comparison, sentinel reset, actual target1/ambient0 restoration, packed Warp execution, native compiled-owner reuse, native partial-submit/record/wait/ACK failure, or warmup hang/reaping acceptance was run here. Positive support is suitable for the separate source-frozen native checkpoint; structural RGB validation alone cannot pass optical correctness. The graph-only native campaign is separate and was neither inspected as evidence nor queried by this reviewer.

Measured Warp arena/timeline, R1, NEW R_final/FinalReadinessOrigin, final factories/commands and reset, C3 registration, one coverage seal/final target barrier, device-delivery and complete lifecycle integration remain closed or unfinished. Product PREPARE/sample/lease/IPC activation, R2/S/host delivery, external holds, result/performance/baseline/backend-selection/shipping authority and whole-C4 acceptance remain closed. The owner's later systematic whole-C4 refactor and fresh whole review remain required. This approval supplies neither native acceptance nor permission to weaken any of those requirements.

## Process hygiene and artifacts

Observed focused pytest PID 42560, parent 44456, creation 2026-10-10 03:46:48 +08:00; session 21442 reached exit 0. Observed source/observer pytest PID 37656, parent 8208, creation 03:47:27; session 85704 reached exit 0. Its sampled isolated bootstrap child PID 40972, parent 37656, creation 03:49:36, also disappeared in the scoped checks. No manual signal was sent. Existing test helpers owned their subprocess timeout/cleanup behavior.

The independent scratch probe returned a terminal exit-0 tool result before an OS PID was sampled; its exact command-line path was included in the final residual check. Short read-only shell/Git/Python diagnostics also completed terminally without individual PID sampling. This is a bounded process ledger, not exhaustive per-child OS attestation.

At 2026-10-10T03:50:47.8300245+08:00, the final scoped Win32_Process queries found no recorded pytest/shell/child PIDs, no remaining direct children of the two sampled pytest processes, and no Python process naming either the independent scratch probe or original-repro wrapper. All owned test handles were terminal; no reviewer-owned residual or unresolved hang was observed. No GPU worker/supervisor/remote process was launched, queried or signalled. No GPU reset or unrelated-process inspection/cleanup occurred.

Only this report and the independent readback-retention probe were written by this reviewer, both outside the candidate. The candidate remained clean at the exact reviewed HEAD. The coordinator may archive this complete report unchanged under collab in an evidence-only commit; that recording cannot alter its verdict, scope or remaining gates.
