# C4 maintainability and remaining implementation gap review

Initiative: Q54 / P12.4 / whole C4 structural follow-up
Stage: independent read-only maintainability and remaining-gap review
Author: codex, independent reviewer /root/c4_maintainability_gap_v1
Version: v1
Date: 2026-10-10 (Asia/Shanghai)
Status: in_review
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request/review codex v3; remaining-setup gap-scan codex v1; systematic-refactor implementation-note codex v1; fixed-cuda-artifacts design-request codex v2; warp-module-preparation design-request codex v2; latest PROGRESS.md entries; production sources cited below
Owner Summary: There is useful implemented infrastructure to reuse, and there are concrete accumulated seams to consolidate. Do not restart compilation/admission or build a generic backend framework. Complete a coherent R0 final-graph/arena/barrier composition using the observed warmups, then R1 and integrated acceptance. Readiness alone would be another incomplete layer. Systematic whole-C4 refactoring remains required before final acceptance.

## Identity, scope and conclusion

Fresh independent session `/root/c4_maintainability_gap_v1`; no candidate authorship or implementation conversation was supplied. No further agents were spawned.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Reviewed HEAD: `9067e114ed609928904bf3cd95b871554bb15d26`.
- Reviewed tree: `dc1d08d5c92b401b68d6397dc07037e917833128`.
- Supplied last production/support revision: `7fe32a58b9e4bee1b6d354f42ceed942f6e9cb92`. Independently inspected its diff to HEAD: only PROGRESS.md and the archived readback-checkpoint review, 89 insertions/2 deletions. No source delta.
- Initial and repeated final `git status --porcelain=v1`: empty. HEAD/tree remained unchanged.
- Parent v3 design SHA256 independently checked: `a76bbe8963064876895ba56eebe72221d941af97f0c84722189b505d6536067`.

**Verdict: Request changes, limited to the remaining implementation and structural closure described here.** This is a gap/maintainability assessment, not whole-C4 acceptance, not a replacement code review of the already approved bounded slices, and not an assertion that their intentionally excluded functionality was promised by them. No new design permission is requested. The parent already authorizes the graph/readiness/barrier work and the owner explicitly authorizes the systematic refactor. Existing approvals retain their original scope.

No fresh GPU/runtime success is asserted. The coordinator's native warmup run at frozen 7fe32a5 is separate ongoing work; I did not inspect, restart, signal, or duplicate it. The current source is more complete than the old remaining-setup scan: compilation, R0 storage, provisional graph and actual ten-warmup observation code now exist. Its G1 and part of G2/G3 must not be reported as entirely unimplemented today.

## What is already present, and what actually remains

| Area | Current source facts | Remaining dependency |
| --- | --- | --- |
| Materialization | Old benchmark workload/assets and tool camera modules are explicit re-exports of production owners; render_parameters binds one frozen sequence and binary32 scalar chain. | Preserve identities and exact camera/scalar values while composing final graphs. No replacement materializer. |
| CUDA compilation | Fixed deployment artifacts enter ordinary external admission; native_setup.CudaExtensionSet and retained factory checks bind actual modules/entries/context. Live JIT/setuptools machinery has been removed; historical readers remain under historical/. | Reuse it. New compilation service, pidfd campaign, runtime dependency expansion or JIT fallback is not a prerequisite. |
| Warp compilation | warp_setup keeps the cache/module/ModuleExec set; source-session preparation validates it before and after recording and warmup. | Final factories must use the same warmed module/options/entry variant and exact owners. |
| R0 resources/graph | native_setup.py:134,242 builds views over original a2 arrays, outputs, ten resolved frame bindings and initialized slot events. | Reuse retained storage for final graph; issue new final factories/stage facts, not a second upload. Current builders are R0-specific. |
| Warmup | warmup.py:114,166 registers private setup transactions, runs ten guarded frames, waits terminal events, requires completed tickets and retires authority in finally. Warp uses actual recorded command groups; CUDA uses existing stage kernels through a setup subclass. | Required fresh native acceptance and numerical correctness remain separate. This report supplies neither. |
| Final readiness | require_observed_warmups exists at warmup.py:225; certify_native_readiness and exact readiness registry already exist. | No production FinalReadinessOrigin or final graph issuance path was found. A caller count is still not worker authority. |
| Final setup barrier | guarded_lifecycle.py:316 explicitly rejects commit_target_barrier; measured submission and clean close also reject. | Exact final graph/origin binding, immutable seal snapshot, one supervised target barrier, DISARMED-bound quiescence and later close proof. |
| Warp C2e | warp_timeline.py contains a private R0 setup transaction only. There is no optics/warp_arena.py. NativeOpticalArena already owns slot/FIFO/generation/hold/reclaim. | Measured dormant transaction/plan/lease binding and arena composition; R1 scene update/refit/render path. Do not clone slot control. |
| CUDA C2e integration | cuda_arena.py already has arena plans/active samples, exact a2/C3 binding and live lease authority. | Its public allocating constructors and direct resolve wait are not yet adopted into the guarded worker path. |
| Integrated worker | worker.py:213 still fails closed on product PREPARE; protocol names alone do not implement tokens or sample dispatch. | C2f same-driver acceptance, exact token lifecycle, guarded resolution, post-reclaim close/ACK/exit, separate activation. |

## Prioritized structural findings

### M1 — P2: source admission also owns the entire setup orchestration and stage-specific failure policy

Locations: native_bootstrap.py `_SourceSession` (2334 onward), `fail` (2448), `prepare_cuda_factories` (3212), `_fail_warp_preparation` (3253), `prepare_warp_modules` (3280), storage/graph/warmup methods (3337,3381,3431); guarded_lifecycle.py:178,194,404,449.

Concrete trigger: add final readiness/final graph, or change a setup stage's CPU entry/aggregate-issuance validation. The bootstrap class now owns captured source identity, installed-file admission, runtime owner records, compilation, render graph scheduling, phase mutations, success publication, and the special healthy-IDLE Warp disposal path. Each subsequent stage repeats original-upload checks, binding resolution, guard entry, imports, audits, retained-owner publication and checkpoint code. `fail()` unconditionally marks a guard failed, while Warp preparation has a special route expressly to preserve actual healthy-IDLE history. A new stage copied from the generic scaffold can repeat the already documented fictitious-guard ordering problem.

This is not a recommendation to make every source-admission error recoverable: integrity/admission failures retain their existing fail-closed contract. Nor does this report claim a new exploit in the approved Warp repair. The maintainability defect is that the distinction is encoded as a lane/stage exception inside a source loader, instead of at a clearly owned setup boundary.

Smallest direction: keep the loader/admission authority in the bootstrap, keep GPU construction in native_setup/warp_setup, and move the repeated optical stage sequencing into the existing setup implementation with a narrow source-session entry. Expose only the exact retained runtime/prepared binding and explicit phase completion operations that this sequence needs. Keep source-integrity failure distinct from an ordinary healthy-IDLE setup failure, with one terminal-disposition owner in the existing lifecycle. Reuse `_observe_public_operation`, `segment`, `_retain_failure`, and `terminate`; do not add a workflow engine, arbitrary stage callback registry, or second guardian. CUDA/Warp-specific compilation steps remain explicit.

Remove/consolidate: duplicated stage scaffolding and `_fail_warp_preparation`'s lane-special routing once equivalent state-based routing is installed; do not remove its semantics. Repeated failure notifications may be consolidated only after original exception, prior guard outcome, and custody facts remain observable.

Why current checks do not settle this: tests prove existing specific entries and repaired counterexamples; they do not prevent a new stage from selecting the wrong failure funnel. Required proof: original healthy-IDLE/unknown-entry and aggregate-issuance repros; active/pending/failed guard failures with no new ARM/COMPLETE/sync; ordinary source-integrity violation remaining terminal; original exception preserved if audit/retention/checkpoint itself fails; late publication unable to advance phase. Include real scope restoration for changed native entry boundaries after renewed preflight.

### M2 — P2: event creation/initialization has two implementations with different operation order

Locations: native_setup.py:199 `_prepare_render_events`; cuda_arena.py:1754 `_prepare_events`, called by builders at 1254/1377; native_setup.py:550 `_native_resource_scope`.

The guarded R0 path adopts each event before the next fallible leaf, initializes diagnostic pairs, then records/waits the terminal event last. The arena helper creates/records its completion first, creates/records diagnostics afterwards, and waits that earlier completion. The latter does not establish that the later diagnostic initialization records have completed. Whatever the legacy helper's bounded contract, these are already independently drifting implementations of slot setup, and directly calling the arena constructor cannot deliver the new guarded ownership/barrier contract.

Smallest direction: one event-construction/initialization body parameterized by the already frozen R0/R1 phase tuple, using the existing narrow resource-scope mechanism. Legacy callers may wrap that core with ordinary waiting; captured callers use actual guarded context and return/adopt each owner. Keep the final terminal initialization after all diagnostic initialization. Do not add an event factory/plugin interface or reuse an old aggregate constructor under one outer guard while ignoring partial returns.

Remove/consolidate: the duplicated CUDA event allocation/record loop after callers share the core; keep compatibility entry names where required. Consolidate the R0 three-phase and R1 five-phase definitions rather than adding a third copy in warp_arena.

Proof: fault each create, record, returned-owner adoption and wait boundary; verify all returned events retained before a next leaf, no subsequent work after failure, zero diagnostic events when off, exact 3/5 pairs when on, distinct terminal/diagnostic identities, and terminal initialization ordered last. Prove real target1/ambient0 restoration and hang disposition. Existing warmup tests cover the new guarded R0 path, not equivalence of the two construction bodies.

### M3 — P2: CUDA setup submission latching is hidden inside a diagnostic hook

Locations: cuda_timeline.py:1025 `_CudaSetupTransaction`, especially record_diagnostic_start at 1030; cuda_direct_light.py:493,500,542; cuda_scene.py:516; cuda_lbvh.py:529.

`record_diagnostic_start` declares `warmup_cuda_launch` even when diagnostics are disabled. Its `_warmup_leaf` context exits with an empty body before the stage actually invokes the kernel. Correctness currently relies on every native stage calling this nominal diagnostic API immediately before its leaf, also when diagnostics are off; the enclosing setup guard catches later exceptions. That reliance is visible in code and comments, not a newly reproduced execution failure.

Trigger/consequence: a future legitimate cleanup that skips diagnostic hooks for an off pass, or inserts another leaf between the hook and its kernel, silently changes safety latching. R1 brings more native stages to this convention. Current tests verify existing off/on launch behavior, but the API still conceals the required contract.

Smallest direction: make the existing CUDA transaction/stage boundary explicitly mark native leaf entry, independently of diagnostic recording, and share it between setup and measured authority modes. Keep one kernel invocation and one transaction state machine. A small fixed internal method/context is sufficient; do not introduce arbitrary callables from the worker/IPC. The enclosing native-resource scope and guard remain authoritative.

Remove/consolidate: the empty safety context embedded in the diagnostic method once the explicit per-leaf boundary is wired to each existing stage. Keep diagnostic methods strictly diagnostic. Retain the setup subclass only as the exact setup-authority adapter over the common transaction, not as a copied timeline.

Proof: diagnostics off still latches before every possible launch; submit-then-raise remains post-submit; early pre-leaf rejection performs no launch/event; caught inner failure cannot finish normally; changed stage ordering poisons; setup authorization never accepts a final lease and final authorization never accepts warmup authority. Validate R0 three-phase and later R1 five-phase traces against actual kernels.

### M4 — P2: final graph work must converge setup/arena assembly rather than append another graph builder

Locations: native_setup.py:242 `_prepare_fixed_render_graph`; warp_timeline.py:17 `_WarpSetupTransaction`; cuda_arena.py:1144/1318 dynamic/fixed builders and 1428 `_seal_cuda_arena`; provenance.py:1164 `build_c3_one_shot_native_execution_binding`; native_arena.py:695 `NativeOpticalArena`.

Current setup R0 binds scalar/camera values, records Warp commands or CUDA stages, and holds outputs/events. CUDA arena builders independently allocate outputs/events and bind render stages. The approved next step needs new final factories and facts, but not duplicate storage, upload logic, camera math or another long-lived setup-only renderer implementation. Warp's small existing setup transaction is a useful execution body, not full C2e and not grounds to clone NativeOpticalArena.

Smallest direction: extract/reuse the exact frame binding and preallocated-slot assembly beneath current builders. R_setup and R_final intentionally use different factory/stage identities over the same permitted physical storage. Final sealing invokes existing provenance/dataflow validation and arena plan registration. Generalize the Warp recorded-phase executor only enough for its fixed R0/R1 stages and two distinct authority modes, preserving current immutable command groups. The same launch/reset ordering must serve both; no parallel kernels. Add the missing Warp arena adapter over NativeOpticalArena, not a new FIFO/generation/hold state machine. Extend CUDA builders to accept the exact privately prepared/adopted resources through their internal production core; do not accept arbitrary public caller-labeled aggregates.

Proof: identical physical prepared-array/output owners where reuse is allowed; new final factory/stage/readiness identities; unchanged R_setup; no extra upload/allocation during acquire/enqueue/finish; final sample starts index zero after warmup 0..9; actual scalar/descriptor mutations rejected; wrong frame/slot/generation and copied/cancelled/reclaimed authority reject; reset sentinels and exact event counts. This must include 1/120/600-frame acquire operation counts, not just a fixed ten-frame setup test.

### M5 — P2: wait and resolution commit remain coupled in the CUDA adapter

Location: cuda_arena.py:722 `CudaActiveArenaSample.resolve` directly synchronizes then advances the lease. Warmup's wait loop independently uses guard completion before crediting a frame (warmup.py:200 onward).

This is a known integration gap, not a defect in the legacy in-process contract. Copying resolve into a worker-only adapter would create a second state transition implementation. Merely wrapping the existing resolve in a guard advances the arena before DISARMED succeeds, violating the parent.

Required consolidation: keep one arena resolution-commit core. The worker's shared Warp/CUDA resolver observes the exact terminal event under the existing guard and supplies an internally registered observed-wait ticket only after successful DISARMED; the same core validates exact active owner/lease/event/generation. A legacy synchronous wrapper can continue to wait and call that core under its own established authority. No boolean `already_waited`, caller receipt, public callback or new supervisor.

Proof: event wait success plus lost DISARMED never resolves/reclaims; foreign event or lease, stale generation, copied ticket, duplicate resolve and parsed record cannot commit; failed wait quarantines without another emergency sync; completed resolution alone does not drop the result hold. This is a dependency of C2f, not something final-readiness issuance can solve.

### M6 — P3: replace opaque setup bookkeeping and scattered literals as part of the structural pass

Locations: warmup.py `_SETUP_WARMUPS` records `[authority, transactions, active, completions, tickets]`, read through numeric indices across preparation/validation/run; native_setup.py hardcoded ten-frame/R0 phase assembly; native_bootstrap.py:2250 schema/summary literals versus runtime_custody.py:24/1091; runtime_phases.py already centralizes INPUT_PHASES.

Use one small private named warmup record with explicit active-frame/ticket/operation and terminal retirement fields. Share existing literal contract constants where execution and replay must agree; extend an existing CPU-safe contract owner rather than introducing a generic schema framework. Avoid independent scalar-offset tables when the canonical NATIVE_SCALAR_ORDER already supplies names. Preserve exact frozen bytes and historical schemas.

Strong registries/retained lists are not automatically leaks: one worker is terminal, and dirty/quarantined owners deliberately remain rooted. Do not replace them with weak references or clear them merely to reduce retained memory. The three warmup registries need not all disappear: session issuance, transaction lookup and completed evidence are different roles. Consolidate only demonstrably redundant storage/positional coupling.

Proof: original copied-authority/transaction/evidence attacks, one-shot failed preparation, permanent retirement after success/failure, byte-identical historical records, and source-closure import tests. This is nonblocking relative to the currently bounded warmup run, but a useful owner-requested cleanup before whole-C4 acceptance.

## Smallest coherent next implementation packet

Do not stop at a readiness-only helper or add another microcheckpoint whose output no production graph consumes. The next useful bounded unit is **supervised R0 final graph admission**, still through the existing fixed drivers with product dispatch closed:

1. Finish/review the already running native R0 warmup evidence at its exact source revision. Keep any native failures and image correctness limitations explicit; no inferred acceptance from one returned image or CPU tests.
2. In one setup composition, consume the exact locally registered ten-warmup evidence, recheck the same compiled/ABI set, invoke the existing certifier for NEW R_final, and register FinalReadinessOrigin. Hold both readiness objects and all provisional owners. No mutable upgrade or third readiness state.
3. Use the same storage and common frame-construction core to make NEW final factories/stage facts for every required final frame/slot. Build actual C3 dataflow/camera/storage bindings, minimal R0 Warp/CUDA arena plans, dormant transactions and fixed-driver admission outputs before the barrier. This includes the required C2e graph dependencies; do not claim full C2d was completed before them. Final authority must be mechanically tied to the exact graph, not to a fake placeholder arena or test loop.
4. Freeze complete guardian coverage once, retaining an immutable SEALED snapshot/digest. All barrier owners already exist. Perform one exact-target sync under a normal setup guard; only its successful DISARMED may commit final quiescence. A failed barrier preserves the sealed graph and takes the original failure path, with no reseal/retry/emergency sync. No GPU-capable preparation after the barrier.
5. Review/verify the complete R0 composition and its counterexamples as a unit. This can honestly be a bounded R0 checkpoint. It does not establish full C2d/C2e/C2f acceptance, R1, product activation or final clean-close authority.

Then extend the same graph/timeline machinery for **R1**, using existing WarpSceneUpdateCommandFactory, WarpBvhRefitCommandFactory, allocate_warp_world_scene_buffers and immutable leaf4/sort topology; CUDA uses its scene/LBVH rebuild factories and existing world/LBVH/workspace allocators. Each slot owns mutable world/bounds/render outputs. Extend guarded incremental custody to the existing scene/refit recording and allocation boundaries rather than copying their implementations. R1 has replay selection plus five ordered phases; compile coverage alone does not prove replay/refit/render composition.

The C2f packet must then join the common worker token lifecycle, guarded terminal resolver, release/reclaim, exact PostReclaimIdleEvidence and ACK-before-cleanup/observed-exit contract, covering six R0 and two R1 rows in both lanes and the parent's correctness/failure matrix. R0 final admission cannot bypass those prerequisites. Keep input runtime-custody v1/v2/v3 descriptive and its `final_execution_readiness_issued=false` meaning unchanged; represent downstream final setup with an explicit versioned record, not a silently repurposed input summary.

## Necessary constraints versus avoidable layers

Necessary: source-held-byte consumption and exact native module/entry custody; explicit target/stream scope; registration before the next fallible leaf; R_setup versus R_final and origin; distinct actual warmup transactions/completions; one guard per normal segment and original failed deadline; immutable coverage before barrier; actual completion versus release/reclaim/exit; exact live identity registries; backend-specific refit versus rebuild; historical compatibility contracts.

Avoidable: source loader owning every render orchestration step; duplicated slot-event loops; safety latching hidden under diagnostics; another final-only camera/storage/renderer builder; another slot state machine; another wait/commit implementation; another readiness schema or a standalone final-readiness subsystem; recreating deleted compiler/setuptools machinery; public generic backend callbacks; permanent comparison infrastructure.

Already successful cleanup must be preserved. Workload/assets/camera compatibility modules re-export production definitions rather than copying algorithms. Fixed-artifact replacement moved obsolete compiler vocabulary to historical readers and removed live JIT. The common `_native_resource_scope`, `_output_array_allocation`, `_warp_command_recording` are useful convergence points. Do not delete either optical lane or historical evidence during this refactor; losing-lane and transitional-harness removal belongs to the later P12.4e decision after equivalent retained coverage.

One nonblocking watch item: guarded_lifecycle.py:99 `_record` reconstructs/replays the whole prefix on every append, and guarded_evidence.py already has a single `_Replay.apply` transition implementation. Static complexity grows quadratically with the number of trace events. No runtime cost was measured here; current work is untimed setup. Do not turn this observation into an optimization project or a second transition implementation. In particular, do not carry whole-prefix replay into the required O(1) measured acquire path. Any eventual incremental application must preserve atomic rejection (apply currently mutates state before all checks finish), offline replay equivalence, original failure handling and immutable historical evidence.

## Verification performed and limits

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely, parent runtime-integration design/review v3, remaining-setup scan v1, owner systematic-refactor direction, fixed-artifact v2 and Warp module-preparation v2 designs, and current PROGRESS tail. Large document outputs were revisited in targeted chunks after truncation. Prior verdicts were treated as bounded historical records, not fresh runtime proof.

Inspected actual call/state/resource paths in native_bootstrap, native_setup, warmup, warp_setup, guarded_lifecycle/guarded_evidence, native_contract/native_input, runtime_phases/runtime_custody, worker, render_parameters/provenance, Warp timeline/render factories, CUDA timeline/arena/stage call sites, NativeOpticalArena and existing Warp scene/refit allocation/factory entrypoints. Read targeted existing warmup and lifecycle regression bodies, including exact transaction counts, immutable execution plan, copied authority, failed leaf and DISARMED-loss checks. This is targeted broad-boundary review, not a line-by-line audit of every C4 source file.

Commands: PowerShell UTF-8 Get-Content/Select-Object, Get-ChildItem for file sizes, Get-FileHash; rg/rg --files; git status/rev-parse/branch/diff --stat. All tool executions returned synchronously without a running session. The initial guessed root REVIEW_PRINCIPLES.md path failed and was corrected to collab/REVIEW_PRINCIPLES.md. Some Windows wildcard rg operands/guessed filenames failed and were corrected to literal paths/discovered files. These are inspection-command errors, not code/test failures.

No tests, Python/native imports, compiler/build, SSH/remote queries, CUDA/GPU query, deployment, process signals, cache cleanup or broad regression suite was executed. No pass count is claimed. No scientific/numerical, device-scope restoration, OS watchdog/exit, resource-lifetime or performance proof was generated. Those remain required at the applicable changed-code checkpoints. No candidate edits or commits were made; only this report was written outside it.

Process hygiene is correspondingly bounded: no reviewer worker/supervisor/compiler/background process was launched; short-lived PowerShell/Git/rg utilities completed synchronously. Their individual PIDs were not sampled, so this is not an exhaustive OS attestation. The coordinator's live frozen campaign was not touched. No owned unresolved tool session or hang was observed.

All later gates remain CLOSED: whole C4 and full C2d/C2e/C2f acceptance, product PREPARE/native sample dispatch, activation, R2/S/host payload, C4c3/c4, C4d/B3, external host holds/depth two, result/baseline/a3b, backend selection and shipping. The report may be archived unchanged as review evidence; doing so opens none of these gates.
