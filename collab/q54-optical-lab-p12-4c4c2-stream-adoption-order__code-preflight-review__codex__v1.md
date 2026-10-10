Initiative: Q54 / P12.4 / C4c2 stream-owner adoption ordering
Stage: Fresh independent bounded CODE/NATIVE-PREFLIGHT review only
Author: Codex, independent reviewer /root
Version: v1
Date: 2026-10-10, Asia/Shanghai
Status: accepted
Related Files: native-state-reader design-request v1 and code-preflight-review v3; runtime-integration design-request v3; terminal-target code-preflight-review v1; exact four-file candidate delta
Owner Summary: Approve exact cf7f5a6d9ff9b3971144410a7c3ad156a806646c for bounded native validation of the four foreign/unknown stream faults and normal scope/input/framed probes. Independent original-method reproductions confirm the ordering defect and its repair. This grants no native outcome, whole-reader, parent-stage or product acceptance.

**Verdict: Approve — bounded CODE/NATIVE-PREFLIGHT scope.**

No substantive blocking or non-blocking finding was identified in this exact delta. The existing adoption authority now receives the actual returned stream before reader rejection can close normal work. Known foreign ordinals are recorded while retaining their owners; unknown failures still quarantine. Ownership evidence remains separate from successful physical observations. No native execution was initiated by this review.

**Reviewer identity and preservation**

Session ID: **01a12574-b4a1-7d92-b575-8308db14ce63**. Reviewer CLI PID 37468, parent 47368, created 2026-10-10 18:56:07 local time; the initial ancestor chain is preserved in `reviewer_ancestry.json`. This session received the bounded packet, did not author the candidate, did not inherit or seek an implementation conversation, and spawned no agents. Author claims and earlier approvals were treated as evidence to inspect and limits to preserve, not as approval of this revision.

AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md were read fully. The first PowerShell read displayed Chinese text incorrectly; REVIEW_PRINCIPLES was reread fully as UTF-8. The specified reader design/preflight, runtime-integration design and terminal-target preflight were read; oversized combined output was followed by focused reads. No private operating context was read or changed.

| Item | Verified value |
| --- | --- |
| Candidate | `C:/Users/AG/新建文件夹/c4_stream_adoption_order_review` |
| Base and merge-base | `91e11eab275f6434754b9449200bf5cb7bf45655` |
| Base tree | `3de28aa4af02e9d24e8a87ed7d1e79fe7d829673` |
| HEAD | `cf7f5a6d9ff9b3971144410a7c3ad156a806646c` |
| HEAD tree | `2ab19d4d0e9a221516584fa302e87d57bdcd7888` |
| Branch | Detached HEAD |
| Initial/final status | No tracked, untracked or ignored entries; only the detached-branch header |
| Exact delta | Four files; 60 insertions, 11 deletions |
| Whitespace check | `git diff --check <base> HEAD` passed |
| Interpreter | `C:/Python314/python.exe`, Python 3.14 |
| Sole scratch root | `C:/Users/AG/新建文件夹/c4_assets/stream_adoption_order_review_scratch_v1` |

The four files are `optical_worker/native_bootstrap.py`, `optical_worker/native_source_policy.py`, `optical_worker/data/native_sources/v18.json`, and `tests/unit/optics/test_guarded_runtime_binding.py`. No candidate file was edited. All reviewer scripts, tests, logs, temporary files and this report are in the designated scratch directory. No installation, reset, signal, old-artifact/cache deletion, compileall, deployment or host switch occurred. Read-only SSH used primary `tileops-dev` only.

**Code, state and resource conclusions**

1. At native_bootstrap.py:2866–2877 and :2898–2904, the original stream helper still runs once under the original setup guard and declared indexed leaf. Its returned object immediately enters `session.runtime_owners`, then the original `returned_owner` journal, before any returned-device metadata read. The two existing `_hold_runtime_owner` calls now precede `reader.stream_returned`. No new return owner, registry, controller, guardian, cleanup path or physical query was introduced.

2. `_hold_runtime_owner` at :2796 retains the same object in the already-adopted session graph before registration can fail. The original `NativeGuardedSetupLifecycle.adopt_returned_device_owner` delegates under the original live setup ticket to lifecycle.py:638. That implementation adopts the exact owner, appends its adoption, adds its actual ordinal, then rejects compromised scope. Thus device0 returned against target1 is retained with scope0 and expands the dirty set to `[0,1]` before rejection. The caller's existing failure path also marks unknown initialization scope. The reader is never entered after that rejection, so no cache/error-buffer/current-state/stream query follows it. A foreign adoption is not a successful observation of the target stream.

3. A target1 stream is guardian-adopted before a subsequent reader/cache/query failure. Such a failure still poisons the original guard and transfers the intact available graph to quarantine. It cannot mint a runtime binding, complete the failed normal guard, arm an emergency retry, synchronize or authorize ordinary release. Repeated initializer and raw reader-state calls reject before modeled getters. Known target custody does not certify physical restoration.

4. Metadata failure, return-journal failure, or adoption failure can prevent an individual guardian record. The immediate source-session graph still retains the returned object. These paths truthfully retain target-only known scope plus unknown scope, then quarantine; the repair does not guess a foreign ordinal that was never successfully obtained. Pre-return unknown failure similarly retains enclosing owners and the original controller, without inventing a returned stream or device0 record. This is consistent with the accepted distinction between known returned scope and unknown failure.

5. Reader construction, its seven-slot source/controller/module/runtime/target/stream/binding record, query ticket checks, raw query order, cache prerequisite and snapshot validation are unchanged. Static AST comparison shows the entire bootstrap except `initialize_native_runtime` is unchanged. The reader source remains SHA256 `3a67a897834c7c118d9cac850ea1f3f68a8f0686daf318b0a0715d0ddd3b40f7`. No native ABI, loading flag, getter, context selector, synchronization, guard handshake or production policy extension is part of this delta.

6. The pinned Warp Python source hashes to `eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee`. Inspection confirms Stream stores `self.device`, Device stores `self.ordinal`, and `get_stream(device)` returns `get_device(device).stream`; that stream property may initialize context/streams through `context`. Reading the existing returned owner/ordinal for adoption does not call that context property. The unchanged reader still demands raw target `_stream` identity, positive raw context/handle and exact empty error buffer before permitting the cached native stream getter. This is source/model evidence, not hardware proof. The existing CUDA metadata access is moved intact; no new metadata API is introduced.

7. Custody journal adjacency remains valid. Adoption adds lifecycle events, not an intervening `_RuntimeObservation` journal row. Consequently the successful `runtime_stream` after-row still immediately follows the retained-return journal row, but observes a later lifecycle cut containing adoption. The same entered-leaf index, before/after pairing, controller/owner identities and subsequent binding point remain required. A foreign rejection truthfully leaves an unmatched before and no binding. Runtime custody, lifecycle replay and source ownership still have their original separate responsibilities.

The committed binding fixture's Warp option doubles the runtime and omits native state facts; its four new tests alone do not establish cache/query correctness or hardware behavior. This limitation is explicit in the fixture. The independent probes below restore the actual reader, cache prerequisite, `_state` and snapshot bodies while replacing only query acquisition/getter leaves and runtime objects with CPU models.

**Independent verification and reproductions**

All successful pytest runs use `python -B -m pytest -q -p no:cacheprovider`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, explicit repository PYTHONPATH, and unique scratch basetemp directories. TEMP/TMP point to scratch. The optional Warp model source is supplied through `ROBOT_SIM_WARP_CONTEXT_SOURCE`. No real Torch/Warp backend import or CUDA/Warp shared-library acquisition/query occurred. Existing observer tests use standard-library ctypes CPU value containers; the new probes use Python `Cell` objects. Neither executes a native getter.

| Fresh successful run | Result |
| --- | --- |
| Binding, reader observer and reader integration suites | 125 passed, 12.03s |
| Independent actual-initializer ordering probes, base and candidate | 32 passed, 4.40s |
| Custody and CUDA-artifact replay suites | 35 passed, 5.11s |
| Independent normal stream-pair/adoption replay and mutations | 6 passed, 3.16s |
| Independent pre-return unknown failure, both lanes/base and candidate | 4 passed, 1.73s |
| Focused catalog/cold-import/reader source-policy tests | 9 passed, 166 deselected, 1.20s |

Total successful final runs: **211 passed, no skips**. These are CPU/model/parser checks. They do not establish real device execution or native destruction safety.

`test_independent_order.py` obtains the original initializer directly from `git show <base>:optical_worker/native_bootstrap.py`, extracts that method by AST, compiles it into the actual production module globals and temporarily installs it only in the test process. Candidate bytes remain untouched. The same probes exercise both old and candidate methods on CUDA and Warp.

The eight situations per lane/revision are foreign owner0, Warp cache mismatch or CUDA reader-identity rejection, actual `_state` getter failure, returned-device metadata exception, adoption exception, returned-owner journal exception, after-observation journal exception, and normal initialization. They inspect actual stream identity in the original session graph and guardian, lifecycle replay, binding absence, dirty/unknown scope, normal completion/sync/release counters, reader invocation and getter order. Metadata access itself asserts the stream was already retained and journaled.

The old method reproduces foreign `[1]` and absent individual guardian adoption when the reader fails. In the repaired method the foreign case has `[0,1]`, individual adoption, and zero reader/post-return getter entry; target reader failures retain individual guardian ownership with `[1]`. Both revisions safely quarantine their failing executions: the original defect is missing adoption/scope information, not failed quarantine or premature release. Both normal lanes preserve successful observation pairing and bind the actual stream.

`test_pair_replay.py` independently attaches explicitly synthetic runtime before/after/binding observations to the complete CPU lifecycle/file fixture, puts the returned journal row at the pre-adoption lifecycle cut and the after-row at the later adoption cut, and runs the public `RuntimeCustodyRecord.from_payload` parser. It accepts that order. Removing before, substituting the returned token, changing guard or leaf, and removing adoption all reject. These synthetic facts test the parser only and are never represented as collected native observations.

`test_unknown.py` separately checks both lanes with the original and candidate methods when the helper raises without returning an owner. The exact exception is preserved; no second helper call, completion, sync or release is permitted. Unknown remains true without inventing a known foreign ordinal.

The author's preserved original-repro log reports four meaningful expected-repair assertion failures (foreign `[1]` instead of `[0,1]`, and target guardian ownership false at reader failure); the regression log reports 125 passes. Those log contents were inspected but are not counted as independent executions. The reported earlier author draft import AttributeError was not independently reconstructed and supplies no evidence for this verdict.

**Source release and lint**

`audit.py` independently reads all 91 release entries from Git, checks their SHA256 against v18, checks the working copy after LF normalization, and invokes the actual AST source validator on each committed LF byte string. All pass. The lane inventories and catalog schema remain unchanged; only bootstrap and native_source_policy source digests change. The policy AST is identical after masking the two refreshed bootstrap pin assignments. Every bootstrap class pin independently matches the exact physical LF source lines.

Catalog SHA256: **0433890aa9f5720001e91426c8bd37bb9061370338f56861a22f852dce5297cc**.

Bootstrap SHA256: `7822eba259d80919cb5791e884d1084b4c099ba3fa5bf78fe3c6e38cc962546c`.

Policy descriptor SHA256: `673e0269b7588198eaa912ff2f93e2675fcf7dd8eb317677058a3e60b1765ff7`.

The descriptor digest changes because it binds the refreshed pins; the AST admission rules do not expand. Independent rehashed-comment mutations of bootstrap and reader are rejected. The prospective release is the committed LF bytes: normalization in this audit compares checkout content; it does not make Windows CRLF bytes a valid production release. The production capture path still requires exact deployed bytes.

Ruff was run with `--no-cache`, JSON output and `--stdin-filename` on exact base and HEAD bytes. Bootstrap reports 76/76 diagnostics, source policy 0/0, and binding fixture 2/1. Comparing code/message multiplicities finds no new diagnostic. This is unchanged existing lint debt, not a clean full-lint claim.

**Original native evidence, independently inspected and replayed**

The local archive SHA256 is exactly `f57b304eaee6af6f73a04a3038bd9e832754112cfc758eccae884da50f9b7df7`. All 33 inventory entries match size/hash against the 34 regular archive members including the inventory itself. Archive bytes remain unchanged at closure. No archive extraction into the candidate or modification occurred.

Read-only primary SSH inspected `/data2/shared/ang.gao/c4-reader-22b1db0-foreign-scope-v2`. Its 26 retained run files match the archived inventory individually. The associated repository is detached at `22b1db0ee619f05e30f03bd5e339f03e883e97a5`, with no tracked/untracked change but pre-existing ignored pytest/bytecode files. Its initial and final ignored-file status is identical; none was deleted. This old checkout is not the candidate.

The remote interpreter ran `-I -S -B`, imported CPU parser modules only, and recorded no torch, warp, numpy or ctypes module. The inspected reader and four parser/environment/supervision source hashes exactly match candidate Git bytes; old bootstrap matches the supplied base. Thus the replay uses unchanged production parser implementation on its actual POSIX path semantics. It is not a candidate native run.

Observed facts:

- Request: Warp, target1, asymmetric canonical input, original revision22b1db0, pinned environment `183_warp_go2_author_v4.json`, environment SHA256 `02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb`.
- `foreign_work.json`: actual returned stream device0 against configured target1, foreign context present; `fault_entered.json`: actual return handed to production, retained owner, original setup guard sequence3.
- Failure stack: `initialize_native_runtime -> stream_returned -> _warp_cache_precondition -> _require`, with `ValueError: native scope prerequisite or state mismatch` before adoption.
- Worker lifecycle: dirty `[1]`, unknown true, guard3 FAILED, full quarantine, guardian transferred, release not authorized, zero emergency attempts. Three prior normal completions belong to earlier successful phases; the failing guard does not complete.
- Supervisor: EXITED, observed return74, no kill request/deadline expiry. Full custody artifact validation passes; runtime custody selection and attachment validate as `worker_prefix_matched`. The retained record has one native before-row, no binding, and no input/final readiness claim.
- The preserved pytest body reaches its original `0 in possible_dirty_device_ordinals` assertion and fails. Collection lists four intended cases; `-x` stops after the first actual Warp foreign-return case. This is one observed worker, not four completed fault cases.
- The retained actual pytest parent command installs the profile hook on each collected `item.module`, checks the function globals are that module, and sets the current profiles. It changes harness profile selection, not the test bodies/assertions. The earlier obsolete-profile issue is not mistaken for the observed adoption-order failure.

The retained closure record has no remaining owned OS/GPU processes. Fresh read-only OS checks also find all five old campaign identities absent: wrapper2821013, driver2821020, pytest2821025, supervisor2821086 and worker2821087. No new GPU-process query was made.

The separate fc23b4e Warp exit-not-restored and earlier0be9876 two-sample campaigns were not independently replayed in this round. Their supplied outcomes remain bounded predecessor evidence, not proof or acceptance of this repair.

**Commands and retained artifacts**

Exact expanded pytest argv, cwd, environment, PIDs, timestamps and exit codes are preserved in the corresponding `*_execution.json` files, with stdout/stderr in the matching `.txt` files. The principal commands from the candidate cwd were:

```text
git status --porcelain=v2 --branch --untracked-files=all --ignored
git rev-parse HEAD 'HEAD^{tree}' '<base>^{tree}'
git merge-base <base> HEAD
git diff <base> HEAD -- <four candidate files>
git diff --check <base> HEAD
C:/Python314/python.exe -B <scratch>/run_checks.py regression
C:/Python314/python.exe -B <scratch>/audit.py
C:/Python314/python.exe -B <scratch>/run_checks.py independent
C:/Python314/python.exe -B <scratch>/run_checks.py independent_v2
C:/Python314/python.exe -B <scratch>/run_checks.py replay
C:/Python314/python.exe -B <scratch>/inspect_evidence.py
C:/Python314/python.exe -B <scratch>/run_remote.py
C:/Python314/python.exe -B <scratch>/run_checks.py pair
C:/Python314/python.exe -B <scratch>/run_checks.py unknown
C:/Python314/python.exe -B <scratch>/static_checks.py
C:/Python314/python.exe -B <scratch>/run_checks.py policy
C:/Python314/python.exe -B <scratch>/final_checks.py
```

Here `<base>` and `<scratch>` mean the exact values recorded above. `source_audit.json`, `static_checks.json`, `archive_inventory.json`, `probe_*.json`, `remote_stdout.jsonl`, `final_checks.json`, `final_remote_stdout.json`, `initial_status.txt`, `final_status.txt`, `reviewer_ancestry.json` and `local_process_closure.json` preserve the evidence. Ruff argv is literal in `static_checks.py`; remote code and its execution ledger are also retained.

Review-harness issues, distinct from candidate failures: an unquoted PowerShell `HEAD^{tree}` initially parsed incorrectly and was rerun quoted; the first audit draft used `path` instead of the catalog's `relative_path`, then failed closed on its empty changed-set assertion before making a verification claim; a PowerShell scratch-write attempt failed parameter binding and was replaced with an ordinary scratch patch. The first independent test run had 30 passes and two Warp normal-path failures because the reviewer model returned a fresh device object instead of the actual target. That scratch fixture was corrected, and the final 32-case run passed. The first log and execution record remain separately retained. No candidate repair, skipped check or weakened assertion was used to obtain passing evidence.

**Process closure and limitations**

Successful pytest child/driver pairs were regression40260/41624, independent39312/39812, replay30712/4824, pair29896/45504, unknown47828/37892 and policy43004/10308. The initial draft child42524/driver43704 exited1 and was reaped. All successful drivers waited/reaped their children with exit0. The policy suite's short isolated import subprocesses return synchronously; their individual PIDs were not separately logged. Guard peers are fixture-owned threads with joined teardown, and their enclosing test processes exited.

Read-only SSH46136/driver17444 returned0. Remote inspection2854689/parent2854688 were freshly absent at final closure. Final SSH8176/driver45724 returned0; final remote probe2860468/parent2860467 completed through that synchronous SSH call. No process was signaled.

At **2026-10-10T19:03:50+08:00**, a scoped Win32_Process scan found no recorded prior test/driver/SSH PID alive. The reviewer descendant tree contained only the ongoing review tool host/console infrastructure and the current closure PowerShell34088, which then returned. No owned test/SSH residual or native worker was present. These are scoped OS observations, not exhaustive process or GPU attestation. The enclosing launcher/reviewer infrastructure is not claimed to be terminated.

Final preservation at 19:03:36 rechecks all 91 source digests, exact HEAD/tree, clean candidate status including ignored files, raw archive/inventory, and unchanged primary evidence/status. No ignored file appeared in the candidate.

Remaining limitations: no candidate hardware execution, CUDA/Warp query-library acquisition, physical context restoration test, native destructor/release proof, complete whole-reader assessment, final sample numerical/physical proof or full worker fault/close matrix was performed. The full historical source-policy and native evidence suites were not rerun. Real-library ABI/source preflight is reused only for unchanged reader/query placement prerequisites; no new upstream binary/source rebuild attestation is claimed. CPU source, model and parser success cannot supply those missing native outcomes.

**Exact native permission and closed gates**

This approval binds only **cf7f5a6d9ff9b3971144410a7c3ad156a806646c**, its exact four-file delta (including fixture and refreshed pins), and the bounded ordering repair within the existing accepted designs.

It permits proceeding on the pinned deployment, with target1/ambient0 and the established visible-device/environment/source controls, to the four existing fault cases: `runtime_foreign_stream_return` and `runtime_unknown_stream_raise`, each on Warp and CUDA. It also permits the affected existing normal scope/input probes and same-production framed-worker probes on both lanes, including ordinary sample resolve/reclaim/ACK/close under the predecessor terminal-target scope. This review does not launch them, change their assertions, select a backend, or pre-grant any result. Preserve actual returned-owner/ordinal evidence, original guard/custody, physical observations where eligible, terminal disposition and observed/reaped owned-process closure; independently review the actual outcomes. A foreign failure must not be forced to emit a successful after-row or perform a post-compromise query for test convenience.

Whole-reader acceptance, C2f/C4 and C4c3/c4/C4d/B3, final-sample physical/numerical proof, fixed-work/preallocation refactoring, remaining worker fault/close coverage and F3 acceptance remain separate and unapproved here. Performance, product activation, later gates, backend comparison/selection, results/baselines, R2/S/host delivery, a3b and shipping remain closed.

This complete report is ready for verbatim archive under the shared evidence-only recording exception. Archiving cannot change its verdict, approved revision or permission scope. The bounded review ends here; no implementation or additional task follows.
