# Fixed R0 render warmup independent code review v1

Initiative: Q54 / P12.4 / C4 runtime integration
Stage: independent code review of bounded private R0 setup warmup execution
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3 sections 4, 5, 5.1, 6, 8, 9 and review codex v3; remaining-setup gap-scan codex v1 G2/G3; provisional-render-graph code-review codex v2; provisional-graph-native-repair code-review codex v1; fixed-render-warmup implementation-note codex v1
Owner Summary: Request changes. The Warp transaction validates its recorded bundle but executes an independently mutable cached command plan. Rebinding that plan to empty groups yields ten accepted warmup completions with zero optical launches. All committed CPU regressions run in this review passed; they do not cover this drift. No native or whole-C4 acceptance is granted.

Reviewer session: /root/fixed_render_warmup_code_v1, fresh independent reviewer, no candidate implementation edits and no subagents.
Repository: C:/Users/AG/新建文件夹/c4_render_execution_next
Branch: implementation/c4-render-execution-next
Base: e43a86c8e63a37da7bd84a5a8f5616559c7e12a1
Reviewed HEAD: 04b74aca5a5f6734e1b7d84ff4fe2eb7014ef796
Actual tree status: clean at initial, intermediate and final verification; HEAD unchanged.
Verdict: **Request changes**, for the bounded code scope only.

## Scope and evidence basis

Read the collaboration protocol and complete review principles before reviewing; read the requested design sections, accepted design review, gap-scan requirements and both predecessor reviews. Inspected all ten changed files, including the implementation note, catalog v16, source-policy pins, bootstrap entry/admission, warmup authority/evidence/driver, both timeline implementations and tests. Traced the existing native resource scope, original controller/guard failure behavior, Warp bundle validation and command order, CUDA stage/template/context/finish path, source custody, and retained graph construction. Author statements and prior verdicts were treated as claims, not current proof.

This review covers only private, fixed R0 preparation and execution of ten provisional warmup transactions under the existing setup guards, with local completion evidence and the necessary literal source closure. It does not review or approve full measured Warp timeline/arena integration, R1, final R_final/FinalReadinessOrigin, final factories/commands, C3 registration, coverage seal/final target barrier, sample/lease or IPC/product activation, results, baselines or performance. Those gates remain closed. Native graph and native warmup acceptance are separate outstanding checkpoints.

## F1 — P1: the validated Warp bundle is not the command plan actually executed

Location: optics/warp_timeline.py:52, :81–83, :103; authority checks at optical_worker/warmup.py:66–96 and evidence issuance at :218–237.

Trigger: after the private driver constructs its exact registered transactions, rebind one transaction's `_groups` field. This is an ordinary assignable execution-plan field on the new class. It is neither frozen nor checked against the retained bundle. The simplest counterexample assigns `((), (), ())` to each of the ten transactions before `_run_fixed_render_warmups` begins. A stale/reordered/cross-frame cached plan is the same class of defect.

Actual behavior: `enqueue()` validates `self.bundle` and its parameters, then loops over `self._groups`. The latter can contain no commands even though the original bundle still contains all five validated commands. The loop advances `_next_phase` for each empty group. `finish()` compares that counter only to `len(self._groups)`, records the genuine terminal event and returns the transaction's completion. The driver then observes each terminal wait and exact normal guard completion, and registers a ten-frame `WarmupCompletionEvidence`. `require_observed_warmups` accepts that evidence. Thus physical completion of the terminal event is mistaken for completion of the promised render work.

Independent reproduction: outside-candidate `C:/Users/AG/新建文件夹/c4_assets/warmup_review_v1/test_warp_plan_drift.py`, SHA-256 `ee93f4b16576c3b858ccd3eada07ab048a4e96e87486cfa90f323a3d6d2bccba`. Both Warp variants (diagnostics off/on) returned accepted completion_count=10, ten waits and ten FINISHED transactions, with **zero optical launch calls**. The two CUDA fixture combinations intentionally skip because this counterexample concerns the new Warp cache. Final result: 2 passed, 2 skipped; these passes demonstrate unwanted acceptance, not correctness.

The probe uses the existing committed native-call doubles, but the production source-session method, private preparation, transaction class, authority checker, guarded driver, finish and evidence registration are unchanged. It wraps the same preparation-to-driver boundary already used by the committed terminal/copy attacks, then performs only the field assignment above. It does not replace authority validation, invent a completion, edit registry contents, change source hashes or bypass guard completion. The original bundles remain intact and still validate. With real bundles that same mismatch remains: their validator inspects bundle commands, not the separately rebound `_groups`. Actual GPU behavior is not claimed from this CPU probe.

Why current checks miss it: `_require_warmup_transaction` checks the exact transaction, bundle, runtime, target, stream, diagnostic events and terminal event, but not the cached execution plan. The full packed-bundle validation in enqueue checks a different object from the plan actually iterated. Positive tests count launches only for untampered plans; attack tests cover copied authority/transaction and terminal replacement, not this plan drift.

Requirement: design section 6 freezes all stages/outputs/events/device/stream/scalars before first enqueue; section 9.3 explicitly requires sealed-field/packed-command drift rejection. Section 5.1 requires ten actual warmup transactions before completion evidence. The shared review principles also require one source of truth and testing rebinding/drift. This finding is bounded to the newly introduced mutable execution-plan field; it is not a claim that the trusted Python process must resist arbitrary monkeypatching or reflection.

Repair direction: make the actual phase plan inseparable from the validated recorded bundle. Prefer a small immutable plan binding, or validate exact ordered command identities and phase boundaries against that bundle before the first leaf, with ordinary rebinding of static transaction fields rejected. Keep mutable execution progress separate. Do not add another renderer or broad authority framework. Checking only the number of groups/commands would still accept same-count reordering or foreign commands.

Proof needed after repair: the preserved counterexample must reject before the first launch/event and issue no completion evidence; add reordered and same-count foreign/cross-frame command-plan cases, then show positive phase/reset order and both diagnostic variants still pass. Preserve original terminal quarantine and no-retry behavior. A fresh independent re-review is required at the repaired exact revision.

## Other inspected high-risk behavior

- Source entry checks the original session, requires fixed_render_graph_ready, prepares transactions inside an existing setup segment and does not open a product dispatcher. Re-entry fails through the existing terminal session failure path.
- Authority registration ties actual session/controller/guardian/graph/runtime/target/stream and the active transaction index to the original guard ticket and native resource operation. Copies are not registered. Guardian CONSTRUCTING, exact known dirty target and live setup ticket are checked before leaves. The original controller enforces thread ownership and sticky guarded failure.
- Preparation retains transaction/completion/ticket collections and each returned transaction. It reuses registered prepared input, R_setup and compiled owners; no new final readiness or C4 lease is minted. R_setup remains warmup0. Evidence is frozen and locally registered, retaining real owner references; copied evidence fails issuance checks.
- The intended untampered Warp order uses existing recorded commands with primary resets in first-hit and shadow resets in shade. Diagnostic events exist only for the on variant. There is no copied rendering math or native allocation/wait in its enqueue/finish loop. F1 concerns the missing binding of that intended plan to execution.
- CUDA reuses the existing transaction and renderer stage. The existing stage calls diagnostic-start immediately before each raygen/first-hit/shade native entry even when diagnostics are off. The private subclass uses those boundaries to declare possible warmup submission, wraps event records, and uses the existing stage failure/finish logic. Enclosing native-resource observation keeps thrown native-entry failures sticky even though the launch declaration context itself ends before the native call.
- Every successful frame retains its actual completion before waiting, then requires the original segment's COMPLETE/DISARMED before appending observations or advancing to event reuse. Last-frame source/external audit occurs under that frame's guard. Escaping launch/record/wait/ACK failures reach the original terminal disposition and do not start a replacement emergency segment. The driver retires its live authority in finally.
- Catalog v16/schema v8 adds only optical_worker/warmup.py to both lanes and optics/warp_timeline.py to Warp. Cold/production selectors agree. All 86 source digests match exact HEAD Git blob bytes. Catalog SHA-256 is `d25e55793d1fddf2f3bf28d7cc286cde2986eb3e5129efa4a064e43bea72f5f2`. Old runtime phase admissions are unchanged across all 32 existing phase/lane combinations; the new source admissions are limited to the two warmup phases. Source-policy AST is unchanged after excluding regenerated bootstrap source/class pins; there is no policy exception expansion.

Maintainability assessment: the private setup authority and reuse of existing resource scopes/guards/CUDA renderer are justified by the accepted design. I found no other substantiated unnecessary architecture in this bounded delta. F1 is specifically an independently drifting second command-plan representation and should be repaired locally. This review does not waive the owner's required later whole-C4 refactor/review.

## Executed verification

Environment: Windows PowerShell, C:/Python314/python.exe, -B, pytest cache provider disabled; commands ran from the frozen candidate. No dependency installation, SSH or GPU work.

1. `git status --short`, `git rev-parse HEAD`, `git branch --show-current`, full ten-file diff and `git diff --check e43a86c8e63a37da7bd84a5a8f5616559c7e12a1 HEAD`: clean exact candidate; no whitespace errors. Targeted UTF-8 source/document reads followed broad output truncations. Initial exploratory filename/glob searches for nonexistent cuda_render/runtime_lifecycle paths failed and were replaced with actual cuda_direct_light.py/guarded_lifecycle.py reads; no candidate edit resulted.
2. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_render_warmup.py tests/unit/optics/test_cuda_timeline.py tests/unit/optics/test_native_render_preparation.py`: **138 passed in 77.69s**, exit 0, tool session 19221. This is a new run on the reviewed revision.
3. `C:/Python314/python.exe -B -m pytest -q -s -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/warmup_review_v1/test_warp_plan_drift.py`: final **2 passed, 2 skipped in 5.34s**, exit 0, demonstrating F1. The first probe run reached all unwanted-acceptance assertions but failed at its final diagnostic print with a reviewer-owned NameError (diagnostic was not in the test function arguments): 2 failed, 2 skipped in 6.19s. Only that scratch function argument was corrected; candidate files were untouched. The final preserved probe is the one named/digested above.
4. `C:/Python314/python.exe -B -m pytest -q -rs -p no:cacheprovider tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_bootstrap.py`: **281 passed, 13 skipped in 181.69s**, exit 0, tool session 4827. Skips: two real-Go2 assets unavailable; two POSIX release-byte checks; two actual POSIX bootstrap entry checks; seven pinned CPython 3.12 POSIX generated-module cases. None counts as native acceptance.
5. Inline `C:/Python314/python.exe -B -` read-only diagnostics used git-show bytes and hashlib/json/ast to compare lane inventories, recompute the catalog and all 86 source hashes, compare source-policy AST excluding its two pin assignments, and execute the extracted base admitted_sources method against the current method for existing runtime phases. All final assertions passed. Two initial reviewer diagnostics incorrectly assumed source_digests was a mapping and then used path instead of relative_path; they failed with AttributeError/KeyError before the hash/admission assertions. The corrected final run completed successfully and is the stated evidence.

The source suite creates current-source LF release and negative-control fixtures in pytest temporary directories outside the candidate. Their generated commit identities are fixture identities, not candidate revisions. The native-return doubles in the warmup tests do not prove actual Warp packed recording, CUDA execution or optical numerical output. No old cache/output was used as evidence of this execution.

## Limits and remaining gates

Required real-source-frozen native warmup acceptance has not been run or supplied for this candidate: actual kernel launches, terminal waits and supervisor exchange, rendered output correctness/sentinel reset, target1/ambient0 restoration, exact native compiled-owner reuse, partial-submit/record/wait/ACK failure, hangs/reaping and guardian retention remain outstanding. The separate live graph-only campaign is not warmup evidence and was neither run nor queried by this reviewer. Existing native graph acceptance likewise cannot be inferred here.

After F1 is repaired, CPU code approval would still not establish those native properties. R1, measured Warp arena/timeline, final readiness/origin and final graph, C3, one coverage seal/final barrier, device delivery and complete lifecycle composition, samples/IPC/product activation, numerical/performance/result/baseline authority and whole C4 remain unimplemented or closed as stated in the packet. No new policy or numerical threshold is proposed.

## Process hygiene and artifacts

Observed focused-test PID 32912, parent 38388, creation 2026-10-10 03:31:17 +08:00; its tool session completed exit 0. Observed source-suite PID 45696, parent 41144, creation 03:31:53; tool session completed exit 0. Sampled its isolated bootstrap child PID 45788, parent 45696, creation 03:34:29, with a current LF fixture path and the reviewed catalog digest. This was the suite's intentional response-stall child. Existing bootstrap test helpers own timeout kill/communicate handling; no manual signal was sent by this reviewer.

Both short probe runs and inline diagnostics returned terminal tool results before individual OS PIDs were sampled. Read-only shell/git commands likewise completed synchronously. This is a bounded launch ledger, not exhaustive per-child OS attestation.

At 2026-10-10T03:34:59.5994699+08:00, the final scoped Win32_Process query found no process with the two reviewer pytest PIDs, their sampled shell parents or child PID above, no remaining direct child of either pytest process, and no Python command line naming the reviewer scratch directory. All reviewer test handles were terminal. No observed reviewer-owned residual or unresolved hang remains. No remote worker/supervisor/SSH/GPU process was launched; no native job was signalled or queried, no remote cleanup or GPU reset occurred, and unrelated processes were untouched.

Only two reviewer artifacts were written: this report and `C:/Users/AG/新建文件夹/c4_assets/warmup_review_v1/test_warp_plan_drift.py`, both outside the frozen candidate. No candidate file, branch or commit was changed. The reviewer remains active only to return this report. The coordinator should record this full report unchanged under collab, preserving its verdict and scope.
