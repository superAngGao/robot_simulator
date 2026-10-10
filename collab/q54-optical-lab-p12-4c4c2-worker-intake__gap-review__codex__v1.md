```text
Initiative: Q54 / P12.4 / C4
Stage: Fresh independent bounded worker-intake GAP REVIEW
Author: Codex, independent reviewer /root
Version: v1
Date: 2026-10-10, Asia/Shanghai
Status: in_review
Session: 01a124d3-a32b-7d43-b6bb-1262e8c90d69
Owner Summary: CUDA worker intake depends on observations supplied only by
the test driver. Production integration needs a bounded design/policy
amendment. This report approves neither the unfinished worker nor C2f/C4.
```

**Verdict: Request changes — bounded intake gap.** The failure is reproducible from the actual custody rules. The smallest honest repair requires a **bounded architecture/source-policy amendment**, followed by implementation and fresh independent review. It cannot be completed honestly by weakening replay, fabricating observations, or merely installing the test observer in the worker test.

**Identity, isolation and scope.** I read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md`. This session received the review packet, not an implementation conversation. No additional agents were spawned.

| Item | Verified result |
|---|---|
| Local checkout | `C:\Users\AG\新建文件夹\c4_worker_intake_gap_review` |
| Local and remote HEAD | `26542543d067d20c64670d375065535420ede203` |
| Branch state | Detached HEAD on both |
| Initial and final working trees | Clean, including untracked-file checks |
| Remote candidate | `tileops-dev:/data2/shared/ang.gao/robot_sim_worker_2654254` |
| R1 baseline/merge-base | `fb49e7eba847d2168a0de81d791357c64c3f9e1c` |
| R1 native revision | `13d17dd293f9c79028260b856e99e39d5f344da5` |
| Baseline equivalence | Those two R1 revisions differ only in `PROGRESS.md` and implementation-note v9 |
| Current WIP delta from R1 candidate | 27 files, 2,215 insertions, 177 deletions |
| Whitespace checks | Passed |

I used only the frozen candidate. No implementation edits, GPU workloads, dependency changes, resets or signals were performed. Scratch and copied evidence are outside the checkout.

The inspected requirements include runtime-integration v3, runtime-adoption v2, source-custody v1, runtime-custody implementation-plan v3, native-scope-aggregate v2, native-scope-cache v1 and fixed-CUDA-artifacts v2. Their approval records were checked for scope; none was treated as evidence that this worker integration already works.

**Observed runtime results.** I decoded the preserved framed journal and replayed the available records using the frozen candidate’s production parsers.

| Evidence | Independently observed result |
|---|---|
| Warp request trace | HELLO, OPEN, PREPARE, two EXECUTE/RESOLVE/RELEASE sequences, then CLOSE; all responses successful |
| Warp terminal lifecycle | Both samples `RECLAIMED`; historical submission latch retained; close ACK committed; guardian released; `clean_reclaimed_exit` |
| Warp supervisor | Worker `1984161`, supervisor `1984157`; observed exit 0; 27 normal completions; no kill/deadline expiry |
| CUDA request trace | HELLO and OPEN successful; PREPARE receives no response |
| CUDA supervisor | Worker `1989673`, supervisor `1989666`; observed exit 74; response-channel EOF; no kill/deadline expiry |
| CUDA first journal fault | `ValueError: CUDA artifact create return was not retained inside its native pair` |
| CUDA second journal fault | `ClosedWorkerProtocolError: canonical JSON nesting exceeds the protocol limit` |
| CUDA journal terminal record | Valid guarded lifecycle: full quarantine, release not authorized, zero emergency synchronization attempts |

CUDA’s terminal trace contains all six artifact create/exec leaf descriptors. It records extension-set adoption at event 2480, normal body return at 2481, COMPLETE initiation at 2482 and guard 3 DISARMED at 2483. Checkpoint failure then produces `guard_failed` at 2484 and terminal quarantine through event 2490.

Thus this was **not an observed failure of the first native create call to return**. Initialization reached substantially further before aggregate replay rejected its missing observations. Nor does the exception establish that the module physically escaped retention.

**F1 — P1: normal CUDA intake requires native pairs that production does not produce.**

Locations:

- `optical_worker/worker.py:367`, `_prepare_native`
- `optical_worker/native_bootstrap.py:2981`, `_create_cuda_artifact`
- `optical_worker/native_setup.py:86`, `_cuda_artifact_create_exact`
- `optical_worker/native_bootstrap.py:668`, `_retain_created_module`
- `optical_worker/runtime_custody.py:881`, `returned_owner`
- `optical_worker/runtime_custody.py:950`, `cuda_extensions_bound`
- `tests/support/native_runtime_process.py:469`, `_NativeScopeObserver`
- `tests/support/native_runtime_process.py:1541`, observer installation

The actual production sequence is:

1. The external loader starts its exact create frame.
2. The source session enters the artifact’s target leaf.
3. `_cuda_artifact_create_exact` enters the Torch target-device and stream contexts.
4. The actual delegate creates the module.
5. `_retain_created_module` stores the module in the external registry and journals its returned identity **before context unwind**.
6. Loading, entry adoption and set binding proceed.
7. `_runtime_phase_completed` invokes custody replay.

The physical retention ordering in steps 4–5 is present. What is absent is the required `native_scope` before/after pair surrounding it.

For v3 CUDA artifacts, replay requires the create return to belong to the active `cuda_artifact_create` pair. Successful extension binding additionally requires all six matched create/exec pairs, joined to their exact file frames, leaves, owners and guard.

The only inspected producer of these pairs is `_NativeScopeObserver`. It wraps the actual helpers, queries native state, appends the pairs and checks the observed state. The `runtime_worker_success` branch does **not** satisfy its installation condition.

Therefore normal CUDA intake currently has a real dependency on **test-only injected observations**. The production loop has no corresponding producer. This is a deterministic integration gap, not evidence of an unavailable CUDA dependency or a numerical-kernel failure.

Existing tests miss it because the earlier preparation drivers install the observer, while CPU codec fixtures explicitly supply modeled native pairs. Both can validate their intended component behavior without exercising observation-free production intake.

Removing the first returned-owner requirement would not solve the contract: file completion and extension-set binding also require the pairs. Removing those requirements would weaken the accepted ownership/scope joins.

**Why earlier native preparation passed.** I independently replayed:

`/data2/shared/ang.gao/c4-dynamic-matrix-13d17dd293/cuda-r1_dynamic_full_shadow-runtime_final_graph_success`

Its record, durable selection and custody attachment validate. Its request identifies native revision `13d17dd293f9c79028260b856e99e39d5f344da5`. It contains **201 native-scope rows**, including six create-edge rows and six exec-edge rows—three before/after pairs of each kind.

The earlier final-graph cases belong to the observer-enabled driver groups. That driver delegates to production preparation while supplying native observations through wrappers.

As a narrow CPU counterexample, I removed only the native-scope rows from an in-memory copy of that valid record, adjusted journal indices and first-reference positions, and replayed it through the current production parser. It failed with the **exact current CUDA intake error**. No preserved record was modified.

This explains the earlier success without invalidating its bounded R1 approval: that approval covered observed preparation, warmups and dormant final composition. It expressly excluded final worker execution and complete C2f acceptance.

**F2 — P1 acceptance gap: Warp’s successful worker smoke lacks the required physical-scope evidence.**

The current Warp record has 1,383 journal entries and **zero `native_scope` entries**. Its lifecycle and attachment replay successfully. Unlike CUDA artifact binding, ordinary Warp replay does not require the missing observation coverage to exist.

This explains the lane difference: Warp can complete the framed lifecycle without the observer; CUDA’s mandatory artifact joins reject it.

The Warp result is useful evidence of two sample lifecycles and clean close. It does not independently establish native context restoration, primary-context history, target-current-stream behavior or complete same-worker scope coverage. Those facts cannot be inferred from graph hashes, target descriptors, successful exit or the custody schema version.

The retained `assert_native_scope_success` checker also describes an earlier sacrificial terminal path requiring one emergency synchronization. It cannot simply be reused unchanged as acceptance for normal integrated post-reclaim close, whose successful path correctly uses zero emergency synchronizations.

**Design/policy verdict and minimum concrete amendment.**

The fixed-artifact design already requires the six native pairs and retained create return inside each pair. That requirement should remain.

However, the accepted native-scope design explicitly says:

> “The new observer is confined to the fixed sacrificial test driver.”

It also keeps production collection CPU-only, without invoking getters. Runtime-custody’s plan forbids extra device queries in its collector. Runtime-integration section 4 and the current source policy restrict `ctypes` to the pinned read-only packed-command snapshot.

The test observer uses capabilities outside those limits: `ctypes.CDLL` with `RTLD_NOLOAD`, fixed runtime/driver queries, Torch native getters, Warp core getters, library-map inspection and test monkeypatch installation. The Warp cache supplement explicitly limits its proof to the fixed sacrificial worker and disclaims a general production error policy.

Consequently, the artifact-pair requirement is already authorized, but **the production mechanism needed to satisfy it is not fully specified or admitted by the existing combined contracts**. This requires a narrow amendment, not a wholesale C4 redesign.

The minimum amendment should freeze these decisions:

1. **One production observation component.** Give the captured source session a fixed internal native-state reader used at the actual production boundaries. Keep `runtime_custody.py` and `_RuntimeObservation` as CPU-only consumers of recorded facts. No caller-supplied observer, callback, module selector or test import.

2. **Exact admitted operations.** Specify and narrowly pin the required library identities, symbols, ABI declarations, query order and source-policy exceptions. Preserve the audited CUDA logical/raw distinction and Warp cache/error-buffer prerequisites. No setter, cache-initialization fallback, extra allocation, synchronization or dependency change.

3. **Exact custody and call ordering.** Retain reader/library/function owners and raw diagnostics in the existing session graph. Preserve file-frame start → leaf entry → native before observation → original scoped delegate once → immediate returned-module retention → original scope exit → native after observation → frame completion. Check contradictory physical state before allowing the next normal leaf.

4. **Existing failure semantics.** Getter errors must not become successful nulls or fabricated snapshots. Retain returned owners and the original diagnostic prefix before fallible processing. Contradictions and uncertain scope remain sticky; failed guarded work must not COMPLETE, retry synchronization or ordinarily release resources.

5. **Both-lane acceptance coverage.** Define scope coverage for the integrated setup/sample/resolve/close paths, including normal close without emergency sync. Continue deriving expected occurrences from actual lifecycle operations. Do not reinterpret old records or let an early valid prefix satisfy terminal coverage.

6. **Consolidation and timing.** Extract reusable query/state logic once; remove the test-owned duplicate producer after equivalent coverage exists. Keep fault injection in tests. Reuse the existing loaders, helper bodies, guardian, replay and arenas. Freeze observation placement relative to future R0/R1 timing boundaries for both lanes; this repair must not silently alter the fair comparison.

Merely adding `runtime_worker_success` to the observer-enabled test list could exercise an instrumented loop, but would leave production dependent on test mutation. Copying the entire observer into bootstrap would introduce the duplication and unreviewed native policy that the user explicitly wants removed.

**Required evidence after that amendment and implementation.**

- Fresh independent design review, followed by exact-code/native-call preflight.
- CPU proof of unchanged v1/v2 historical meanings and v3 file/pair/owner joins; missing, crossed, stale or forged observations must still reject.
- Retention tests for create-return, context-exit, getter, append and adoption failures, with original exceptions and no premature owner destruction.
- Fresh **same production framed-loop** executions for both lanes, with the production observation mechanism active and no test-installed successful-fact producer.
- Real target1/ambient0 restoration, foreign-touch and unknown-scope cases; preserved contradictory observations and correct quarantine behavior.
- Full parent C2f matrix: six R0 and two R1 rows in both lanes, profiles/shadows/diagnostics, CPU-oracle correctness of actual final samples, generation/sentinel behavior, guarded resolution, reclaim, cancellation, close/ACK, setup/post-submit faults, disconnects and hangs.
- Exact final evidence attachment, supervisor-observed exit and scoped process closure. No skips or CPU models may replace mandatory real-device properties.

This is one bounded integration amendment serving completion of C4, not permission to declare an infrastructure subset complete.

**F3 — P2: the terminal response exceeds the canonical nesting limit.**

`worker.py:211` embeds termination evidence beneath the response payload. The preserved CUDA evidence has maximum depth 7; its journal envelope reaches 8, which passes. The response envelope reaches **9**, exceeding `protocol.py`’s limit of **8**.

I reconstructed that response from the preserved terminal evidence and reproduced the exact exception using the production encoder on CPU.

This is a separate bounded encoding defect. A repair must preserve terminal evidence and reviewed protocol bounds, with a regression using real guarded terminal payloads. It does not address the missing native observations.

The failed driver’s top-level `FileNotFoundError` for `worker_arena.json` is also secondary. CUDA never reached arena preparation. The primary intake failure is preserved in `worker_journal.bin`.

**Evidence coverage limits.** CUDA’s durable runtime-custody record is only the initial setup-definitions cut: two journal entries and 29 lifecycle events. Its sidecar’s `current` designation describes that persisted cut; it does not establish coverage of the later 2,491-event terminal trace.

The combined custody artifact contains a matching worker prefix. The fuller guarded terminal lifecycle is in the journal and validates separately. I did not infer the missing runtime-observation tail from either record.

**Executed verification and preservation.**

| Check | Result |
|---|---|
| Current Warp and CUDA records: production parser, selection and attachment | Both valid for their actual coverage |
| Earlier R1 CUDA record: same checks | Valid |
| CUDA journal terminal lifecycle replay | Valid quarantine record |
| In-memory removal of historical native observations | Exact intake failure reproduced |
| Actual terminal-response encoding | Exact nesting failure reproduced |
| Observer and CUDA-artifact replay CPU suites | 36 passed; nine initially skipped for unavailable pinned Warp source text |
| Those nine CPU models after copying the pinned source text into scratch | 9 passed; no native Warp execution |
| Remote replay imports | No Torch, Warp or NumPy modules imported |
| Raw evidence preservation | 64 files, 4,690,753 bytes; every copied file matches remote size and SHA-256 |

Remote replay used `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B`; local CPU tests used Python 3.14 with bytecode and pytest cache writing disabled.

Scratch, scripts, logs, inventory and exact raw copies remain at:

`C:\Users\AG\新建文件夹\c4_assets\worker_intake_gap_scratch_v1`

The original remote evidence remains untouched. Initial scratch inspection errors involving list/null handling were corrected; successful replay results are retained in `replay_cpu_v2.log`. Local checkout files use CRLF; committed LF design bytes were checked separately rather than treating newline-converted hashes as execution facts.

Reviewer CLI PID was `1012`, parent `45080`, started at `2026-10-10 16:00:11`. Final scoped checks found the recorded campaign parent/workers/supervisors and identified remote CPU replay processes absent. No scratch-scoped Python/SSH/SCP residual was found locally. GPU-process inventory contained no recorded campaign worker PID; unrelated processes were left untouched. These are scoped checks, not exhaustive process attestation.

**Remaining scope.** No fresh GPU execution, numerical recomputation of the new final samples, full C2f fault campaign or whole-WIP code review occurred here. Historical runtime evidence was inspected and replayed, not presented as a new execution.

The R1 approval retains its exact bounded meaning. CUDA production intake remains blocked; Warp has a useful successful smoke result with an unresolved scope-evidence gap. Complete C2f and C4 acceptance, activation, results/baselines, speed comparison, backend selection, host/R2/S delivery and all later product gates remain closed.