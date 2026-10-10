Initiative: Q54 / P12.4 / C4c2 terminal-target binding
Stage: Fresh independent bounded CODE/PREFLIGHT review only
Author: Codex, independent reviewer /root
Version: v1
Date: 2026-10-10, Asia/Shanghai
Status: accepted
Session: 01a12555-9ed3-79c3-9b48-a811e5677ae7
Related Files: runtime-integration design/review v3; worker-intake gap-review v1; native-state-reader design v1 and preflight v3; native-worker-source-audit code-preflight-review v1; guarded_lifecycle.py; native_sources/v18.json; test_integrated_worker_lifecycle.py
Owner Summary: Approve exact 0be9876b9778cf4826a62f45c373a38d60ff33f1 to resume bounded native validation. The CUDA terminal-target correction fits the accepted design and preserves native resource identity and guarded resolution ordering. Original rejection reproduced; 39 focused independent CPU cases pass. This is not hardware, whole-integration, C2f/C4 or product acceptance.

**Verdict: Approve — bounded CODE/PREFLIGHT scope.**

There is no substantive blocking or non-blocking finding in this delta. Exact candidate **0be9876b9778cf4826a62f45c373a38d60ff33f1 may resume bounded native setup/framed-worker validation** under the existing approved preparation-reader design and pinned deployment. This permission does not assert that this candidate has run natively or passed any hardware acceptance checkpoint. Stop this review after recording its evidence; no native execution is initiated here.

**Identity, independence and preservation**

| Item | Independently verified result |
| --- | --- |
| Candidate | `C:/Users/AG/新建文件夹/c4_terminal_target_review_v1` |
| HEAD | `0be9876b9778cf4826a62f45c373a38d60ff33f1` |
| Source base and merge-base | `22b1db0ee619f05e30f03bd5e339f03e883e97a5` |
| Immediate parent | `d5274e819a9352f78a8c8d45f6abced90a38ed3e` |
| Branch | Detached HEAD |
| Initial/final tree | Clean, including untracked and ignored-file checks |
| Repair commit delta | Three files, 79 insertions, two deletions |
| Full source-base-to-HEAD delta | Six files, 385 insertions, two deletions |
| Additional intervening files | PROGRESS.md, source-audit implementation note v1, archived source-audit preflight v1 |
| Catalog v18, committed LF SHA256 | `4a400516facc458763158738153e78fda98d6ce2fc2ffe1b4f20a110103bd532` |
| Unchanged AST policy SHA256 | `a219791cda442b3f79da5b72becd6d2bfe85f17452b3e096377036107c80da8b` |
| Whitespace | `git diff --check 22b1db0 HEAD` passes |

This fresh session received the bounded review packet without an implementation conversation. I did not author the candidate, spawn agents, edit candidate/implementation files, install dependencies, signal processes or change private agent context. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely, followed by the specified design/review/gap/preflight documents. Truncated combined document reads were supplemented with focused reads. The reader design-review v1 was also consulted for its phase-boundary comment. Earlier verdicts establish their scopes, not proof of this candidate.

All new scripts, tests, logs, temporary files and this report are confined to `C:/Users/AG/新建文件夹/c4_assets/terminal_target_review_scratch_v1`. The candidate, preserved raw evidence and archive remain unchanged. No new native worker, GPU work, GPU query, backend execution, native query-library acquisition or installation occurred. CPU fixtures use their existing NumPy dependency and modeled runtime leaves. Read-only SSH used primary `tileops-dev` only; no fallback or host switch occurred.

**Design classification and findings**

This is a bounded implementation correction within runtime-integration v3, not a new policy or architecture decision. Section 6 binds a strongly held context to an indexed CUDA device and exact native stream; section 7 requires the exact pending sample and terminal event, a supervised wait, DISARMED before lease resolution, and no caller-supplied completion authority. It does not require two separately constructed immutable CUDA device values to be the same Python object.

The unchanged `CudaOpticalExecutionContext.__init__` at optics/cuda_timeline.py:382 constructs `torch.device(device)`, validates indexed CUDA device/stream compatibility and retains that resolved value plus the original stream and handle. Treating that value as the original source-binding object was an implementation mismatch. The correction does not change context construction, renderer behavior, native ABI, source admission policy, IPC operations, physical-state query placement, guardian coverage, release policy or product gates.

The new acceptance predicate is not unrestricted equality-based admission. It requires the transaction's context to be the exact factory context, the supplied target to be that context's device object, its stream to be the exact supplied stream, and its device value to equal the original graph target. The supplied target is then replaced locally with the graph's original retained target before ordinal/device-scope checks. Active owner, transaction, terminal event and graph stream identity checks remain before this predicate. Warp retains original target object identity.

An independently tested subtlety is intentional: replacing the **device value on the original context** with another equal modeled device value can still resolve. Replacing only the wait argument with a copy rejects. An equal indexed device value is not a native stream/event/resource replacement; the original target still supplies scope identity, and native resource objects remain exact. This distinction is consistent with the template's existing value comparison for device and identity/handle comparisons for native resources. It does not weaken a required native resource identity.

The supported trust boundary remains the closed trusted worker. Parent IPC cannot provide Python objects, contexts, equality callbacks, raw handles or rewritten registries. No claim is made that coordinated arbitrary in-process reflection, malicious equality implementations or simultaneous rewriting of sealed signatures and registries is sandboxed. The review tested ordinary substitutions and field drift against the real supported resolver, rather than treating that out-of-contract power as a new requirement.

**Inspected call, data, state and ownership paths**

- The original source/runtime target and stream flow into the registered frozen `_FinalRenderGraph`. Final preparation retains context, factory, modules, arrays, events, arena, plans, active samples and transactions before sealing and the final barrier. `_require_final_graph`, `_CAPTURED_ARENAS`, `_FINAL_RENDER_GRAPHS`, final admission and source/controller checks remain unchanged. This repair creates no owner, container, guard, wait or release graph.
- Worker EXECUTE acquires the next immutable plan, registers the exact active sample, latches possible submission before leaves, finishes the transaction and exports a pending token only after terminal recording. `_sample_for_token` and `_resolve_native` require that pending token's original active generation. No token or response shape changed.
- `CudaActiveSample.resolve` at optics/cuda_arena.py:734 validates the configuration, executable template, registered live lease authority, completion object/event/device/stream handle before invoking the controller. It performs the binding validation again after the controller returns. Template/stage signatures retain the context identity, stream identity/handle, stage identities and terminal event identity. The production renderer checks its exact stage context and sealed binding signature.
- `observe_terminal_completion` at guarded_lifecycle.py:346 calls `_require_worker`, checks admitted graph and exact pending identities, applies the repaired target predicate, validates target-only known scope and event/stream devices, and requires adopted owner/transaction/event/stream. The original target remains part of the already sealed graph custody; the new device value does not acquire ownership authority.
- All these checks precede entry to the normal resolve segment and its event wait. A successful wait alone does not resolve a lease. ARM/ARMED precedes the wait; COMPLETE/DISARMED and completed-ticket verification precede `_worker_resolution`; the arena revalidates, then commits the lease. The worker records RESOLVED_HELD and consumes the exact resolution ticket before exporting the resolved token. RELEASE alone drops the device hold and reclaims. A prior ticket cannot resolve a later generation.
- Wait or DISARMED failure follows existing failed-normal-segment quarantine, with no emergency retry or resolution/reclaim. A binding failure before any resolve segment leaves the guard idle; the enclosing worker fault path applies existing conservative terminal disposition. This explains the preserved old campaign's one emergency synchronization after its pre-guard identity rejection. Rejection itself is not presented as a successful normal wait or reclaim.
- Post-wait binding drift is still rejected by the arena before lease commit, even if the normal wait ticket exists. This was tested explicitly; the sample remains unresolved/poisoned. The ticket is an observation, not a bypass of the second validation.
- The request loop still executes handler -> source audit -> encode -> complete response write -> `commit_written_response`. Exceptions bypass success response construction/write and enter terminal handling. Normal CLOSE still needs actual post-reclaim idle proof and retains the graph until the written close ACK is committed. Existing two-lane tests verify no release before that commit, preserved historical submission latch and no extra sample work in close. The compact fault reply/journal recovery implementation itself remains outside this approval (predecessor F3).

**Original regression and independent CPU verification**

The committed fixture executes the actual CUDA context constructor under modeled device-copy behavior **before provisional/final sealing**, then places its resulting device value into the retained fixture context. It deliberately leaves native leaves doubled. It is stronger than the earlier constructor-bypassing same-object fixture, but is not a real Torch execution or full native constructor acceptance test.

For the original counterexample, the scratch test reads the original `observe_terminal_completion` method directly from Git at 22b1db0, compiles that method into the production module's globals and temporarily installs it only in the CPU test process. The candidate on disk is never changed. The same constructor-based fixture and real dispatcher/arena path then reproduce **`WorkerSetupProtocolError: terminal wait is not the worker's exact pending sample`** at RESOLVE, with no native-wait leaf, no resolution ticket, zero lease resolutions/reclaims, poisoned active owner and still-idle guard. The expected-rejection test passes.

| Fresh independent run | Observed result |
| --- | --- |
| Focused committed integrated-worker selection | **27 passed, 30 deselected**, 57.57 seconds; exit 0 |
| Original-method regression reproduction | **1 passed**, 4.16 seconds; exit 0 |
| New independent boundary/order probes | **11 passed**, 22.76 seconds; exit 0 |
| Source catalog audit | **91/91** committed LF bytes and actual AST validations pass |
| Ruff on both changed Python files | All checks passed, `--no-cache` |
| Raw inventory/archive preservation, repeated at closure | All sizes/hashes/member bytes match |

Total fresh pytest evidence: **39 passed, no skips**. No competing full-suite rerun was started. The author's earlier full-suite construction failure and subsequent six-case summary, the ongoing related regression, and the separate old-reader matrix were not counted as independent results or assumed complete.

The 27-case selection covers the new distinct-device two-sample success, all six committed wait substitutions (device argument, stream, event, transaction, exact-type replacement owner, transaction context), both-lane two-sample reclaim/ACK close, cancellation while pending, copied idle-proof rejection, consumed/out-of-order tokens, old completion replay rejection, normal wait/DISARMED failures, pre-enqueue failure and submit-then-raise behavior.

The 11 scratch probes add six mutations at the actual RESOLVE entry: factory context, context stream, context stream handle, foreign device, event handle and transaction context. Each rejects before waiting and leaves resolution/reclaim counts zero. The success probe records wait -> DISARMED -> lease commit -> returned response, verifies the exact ticket and retained target custody, and records zero emergency synchronization. Three failure probes use the distinct-device fixture for failed wait, lost DISARMED and stream replacement during a completed wait. The final probe verifies the equal device-value distinction described above over two samples. No test assertion failed and no candidate repair was made to obtain these results.

Exact runners and expanded argv/cwd/PIDs/times/exit codes are in `run_checks.py` and `focused_execution.json`, `original_execution.json`, `probes_execution.json`. Tests use `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider`, explicit repository PYTHONPATH, disabled plugin autoload, `PYTHONDONTWRITEBYTECODE=1`, and unique TEMP/TMP/basetemp under scratch. Tests use production CPU control flow and existing runtime doubles; they do not establish real physical context restoration, GPU arithmetic or resource destruction safety.

Commands executed from the frozen candidate, with `<scratch>` denoting the exact scratch directory above:

```text
C:/Python314/python.exe -B <scratch>/audit.py
C:/Python314/python.exe -B <scratch>/run_checks.py focused
C:/Python314/python.exe -B <scratch>/run_checks.py original
C:/Python314/python.exe -B <scratch>/run_checks.py probes
C:/Python314/python.exe -B <scratch>/inspect_evidence.py
C:/Python314/python.exe -B <scratch>/replay_evidence.py
C:/Python314/python.exe -B <scratch>/remote_check.py
C:/Python314/python.exe -B -m ruff check --no-cache optical_worker/guarded_lifecycle.py tests/unit/optics/test_integrated_worker_lifecycle.py
C:/Python314/python.exe -B <scratch>/audit.py
C:/Python314/python.exe -B <scratch>/final_checks.py
git diff --check 22b1db0 HEAD
```

The LF source audit independently matches every source pin against Git bytes and the worktree after LF normalization, validates every source with the actual AST policy, checks unchanged lane inventories/policy, and confirms the **only** changed source digest is guarded_lifecycle.py. Native reader, bootstrap, constructor, renderer, ABI/query implementations and runtime contract are unchanged from source base. The catalog hash is the value recorded above. The extra intervening documentation records prior evidence and preserves the closed gates; it is not a second implementation delta.

**Preserved native evidence: independently checked, not a candidate execution**

All 80 inventory rows match local size/SHA256, totaling **18,971,513 bytes**. The archive SHA256 matches **53629420b09408b5c48933f7850667f67b5af9244475e8814f435285fb3fd703**. All 81 regular archive members, including its inventory, match their local counterparts. The audit was repeated at closure with unchanged results; no extraction or raw-file mutation occurred.

`replay_evidence.py` decodes each framed request/response and the actual CUDA journal with the production codec, validates each worker terminal lifecycle with `GuardedLifecycleEvidence.from_payload`, and validates the supervisor/worker custody artifact with the production validator. It saves bounded extracted results in `replay_evidence.json`; it imports no Torch, Warp, NumPy or ctypes. The source audit similarly imports none of those modules.

| Old 22b1db0 campaign | Independently observed |
| --- | --- |
| Warp | HELLO/OPEN/PREPARE, two EXECUTE/RESOLVE/RELEASE sequences and CLOSE all successful; two terminal records/resolutions/reclaims; guardian RELEASED, release completed, clean_reclaimed_exit, zero emergency sync |
| Warp supervisor | Worker 2670775, supervisor 2670774; observed return 0; 27 normal completions; no kill or deadline expiry |
| CUDA | PREPARE and first EXECUTE successful; RESOLVE sequence 4 returns the exact terminal-pending-sample error; one terminal record, zero resolutions/reclaims, no normal CLOSE |
| CUDA terminal lifecycle | aborted_synced_exit; one successful emergency sync; release completed; emergency guard remains ARMED; 22 normal completions |
| CUDA supervisor | Worker 2678363, supervisor 2678361; observed return 64; 23 arms/22 normal completions; no kill or deadline expiry |
| CUDA raw journal | Original WorkerSetupProtocolError fault followed by full transport termination evidence matching the error reply's evidence digest |

Read-only primary SSH confirms the original remote checkout remains clean/detached at 22b1db0. Its cuda_timeline.py hash is `8d24e7286377b381fb562103c62ae07ac8ac3a010441483488eee9e08b4419bd`, matching this candidate's unchanged LF source. Reading the original constructor verifies the same device-value construction path without importing the backend. Remote Python uses `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B` and read-only Git/filesystem operations.

This round replays terminal lifecycle/framing/custody evidence; it does not repeat the full setup native-scope/input/environment reconstruction from predecessor reviews. No final physical-scope or numerical conclusion is inferred from valid terminal evidence. No new native execution exists at candidate 0be9876 in this review.

**Process closure and coverage limits**

Reviewer CLI is PID **42004**, parent **37896**, creation Unix milliseconds **1791627730433**, correlated with the current session through the final ancestor chain. The enclosing launcher remains outside review-owned test work. Focused pytest PID **40340**/driver **40852**, original-repro pytest **46504**/driver **43956**, and boundary pytest **39376**/driver **7916** were waited/reaped with exit 0. The CPU guard peers are fixture-owned threads with joined teardown, not native worker processes.

Initial SSH PID **36332**/driver **17808** completed with exit 0. Remote inspection PID **2741832**/parent **2741830** are absent at final read-only closure. The six supplied old-campaign PIDs **2670393, 2670408, 2670775, 2670774, 2678363, 2678361** were also freshly observed absent, both at inspection and final closure. The final remote closure probe **2745247**/parent **2745239** returned synchronously through SSH PID **45396**, exit 0.

At **2026-10-10T18:27:20+08:00**, the scoped local process scan found no recorded test/driver/SSH residual. Its sole scratch-scoped Python was the currently executing final-check process **35808**, which then returned successfully; all tool execution sessions completed. `final_checks.json` records parentage/start identities, exact final HEAD/detached state and empty status including ignored files. No process was signaled. No GPU-process query was performed. These are scoped OS checks, not exhaustive process attestation.

The old 22b1db0 native-reader 20-case real-input/Go2/foreign-scope matrix at `/data2/shared/ang.gao/c4-reader-22b1db0-native-scope-v1` was not inspected, interrupted, adopted as evidence or treated as complete. Its supplied wrapper/driver/pytest identities were not used as this review's processes. Related author regression results remain outside the fresh evidence above.

Inspection limitations: oversized combined reads required focused follow-ups, and an early line-based inspection of a one-line JSON inventory/catalog produced more text than intended; subsequent evidence inspection used parsed bounded summaries. This caused no mutation or failed verification. There were no test failures, unavailable-test skips, installation workarounds or candidate modifications in this review.

**Exact permission and still-closed gates**

Approval binds **only 0be9876b9778cf4826a62f45c373a38d60ff33f1 and this bounded terminal-target/source-pin/test delta**, building on the unchanged predecessor scopes. It permits resuming bounded native validation on the pinned deployment: same-production framed-worker setup and sample lifecycle in both lanes, the motivating CUDA resolve/normal-close path, affected binding/wait/guard/fault behavior, retained custody, response/ACK discipline and observed/reaped process closure. Actual native results and affected failures must still be preserved and independently reviewed. No success is granted in advance; changed code, pins or newly discovered policy follows the existing review workflow.

Still required: full final-sample physical and numerical evidence, preallocated fixed-work traces, complete worker fault/close coverage, real workload/profile/diagnostic/generation/sentinel matrices, reader real-input/Go2/foreign-scope evidence, and C2f/C4 acceptance. The predecessor F3 compact-reply/journal-recovery change is **outside this approval**. Inspection of its unchanged fault-path use does not approve that separate scope.

**Hardware/whole-reader acceptance, full C2f/C4, C4c3/c4/C4d/B3, performance, product activation, results/baselines, backend comparison/selection, R2/S/host delivery, a3b and shipping remain closed.** No speed, final numerical, backend or product claim is made.

The bounded review stops here. This complete unchanged report is ready for verbatim archive. The owner may record it separately under the shared review-only evidence exception; such recording does not alter the approved revision, scope, verdict or gates. No candidate edit, evidence commit, deployment or new native run was performed by this reviewer.
