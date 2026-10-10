Initiative: Q54 / P12.4 / C4c2
Stage: Fresh independent exact-code/native-call PREFLIGHT, review only
Author: Codex, independent reviewer /root
Version: v3
Date: 2026-10-10, Asia/Shanghai
Status: accepted
Session: 01a12529-1730-7533-a977-8fccfc754ade
Related Files: native-state-reader design-request/design-review v1; code-preflight-review v1/v2; worker-intake gap-review v1; native-scope-aggregate v2; native-scope-cache v1; runtime-custody implementation-plan v3; runtime-integration v3; fixed-CUDA-artifacts v2; source-custody v1; runtime-adoption v2; binding-fixture review v1
Owner Summary: Approve with comments for exact HEAD df1a46ae5039fe8a25718c7ec60f52fa7c60c549 to proceed to bounded native setup/framed-worker validation. The two v2 blocking counterexamples are closed, all original identity probes pass, and the shared query algorithm/ABI is preserved. This is preflight permission, not hardware or parent acceptance. A non-blocking test-runner bytecode side effect is recorded below.

**Verdict: Approve with comments.** This exact code may proceed to **bounded native setup/framed-worker validation** under the approved preparation-reader design and pinned deployment. There are no unresolved blocking implementation findings or preflight evidence gaps in this scope. No native/GPU execution was performed here. Hardware acceptance remains a subsequent checkpoint, requiring actual evidence and fresh independent review.

**Identity, independence and preservation**

| Item | Independently verified value |
| --- | --- |
| Candidate | `C:\Users\AG\新建文件夹\c4_native_reader_code_review_v3` |
| HEAD | `df1a46ae5039fe8a25718c7ec60f52fa7c60c549` |
| Prior reviewed base and merge-base | `5d804f73b5ae00e89fc18455080b1e339d20dade` |
| Original reader base and merge-base | `868225db0ac5794bd7089cd73357b5c920d0c0e9` |
| Original v1 candidate | `4dcf73d78ef36dbfd40ac3025a228cf9be1b21cd` |
| Branch | Detached HEAD |
| Initial/final porcelain including untracked files | Empty |
| Final ignored-file status | `!! optical_worker/__pycache__/`; six generated `.pyc` files, detailed below |
| Delta from prior base | 11 files, 645 insertions, 89 deletions |
| Whitespace | `git diff --check 868225d HEAD` passed |
| Approved design, committed LF SHA256 | `0b6dab55c496e730219dfbb253b7d9530dff40faa6c90676c71a2b90e0f41ca5` |
| v1 review, committed LF SHA256 | `604e8711e6d71395f4f272c368bc8c46e6319b2bf400d4af9c8f8f9fd29f259e` |
| v2 review, committed LF SHA256 | `9dea7939c12c86e1f42375172476408eee00db854ba0dcae87e3fdd82c4f78ac` |
| v2 reviewer session | `01a1251b-f2f2-77f1-9b5c-2c815bedf4db` |
| Current reader, committed LF SHA256 | `3a67a897834c7c118d9cac850ea1f3f68a8f0686daf318b0a0715d0ddd3b40f7` |
| Bootstrap, committed LF SHA256 | `5a8f54ce74f9f991f11ad554cf0a7fa1a5f24793c48be4738a2444aa49fff277` |
| Catalog v18, committed LF SHA256 | `644f4077e0e2420c0b742b4e0068a669b0ce35f939e6659c5fc9eeeb06d897f0` |
| Current AST policy digest | `c74bb4de60886d26f153fbf5d607b6cdf7cbbe4f4e8571b8e04375e70d38653f` |
| Integrated runtime contract, committed LF SHA256 | `cff830f90fca7c966916ceb43b7ad0dfebd621011df66b9aa66d313653c7ebe2` |

This fresh reviewer received the review packet, not an implementation conversation, and did not author the candidate. No agents were spawned. AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md were read fully in UTF-8, followed by the approved reader design, C1, complete v1/v2 reviews, predecessor requirements and binding-fixture review. Oversized truncated reads were followed by focused reads. Author summaries and prior verdicts were treated as claims and scope records, not proof of this revision.

All reviewer-authored scripts, probes, logs and this report are under `C:\Users\AG\新建文件夹\c4_assets\native_reader_code_scratch_v3`. No tracked candidate file or old evidence file was edited. All **48 pre-existing files** inventoried under original v1/v2 scratch were hashed before execution and remained byte-identical at closure. Original v2 probes were executed directly, unchanged; new probes reside only in v3 scratch. There were no installs, resets, signals, native-library acquisitions, backend execution, GPU queries or host switches. The optional host-routing decision file is absent. Read-only SSH used the primary `tileops-dev` host only.

The binding-fixture source at HEAD is byte-identical to the separately approved file at `eab674c6f9c7eb9161a4f731db51e69352561634`. Its supplied report hash `2ff1bb4b6554126165bc06bfe4a93a7a7b82e001a8fb529a1b556e33e58d8c26` is the **raw scratch report** hash (14,284 bytes). The committed report is 14,283 LF bytes, hash `3304d16becc76e655a6f1a91ced1e123869a46bcc8a3ce8e115e31e46c7b5b38`. Replacing CRLF with LF in that raw original produces the committed bytes exactly. This is a newline normalization difference, not changed verdict/content. Its test-only approval is not reused as production/native permission. The entire affected binding test file ran independently in the current combined suite.

**Findings and disposition of prior findings**

No new substantive production finding was identified. The following conclusions close the exact prior counterexamples at this revision, without claiming arbitrary hostile-Python resistance.

**R1 — closed.** The original post-release and explicit emergency-reader-failure cases still reject before getters. `_ticket` retains its existing setup/owner-thread admission and sticky failed/compromised/terminal-failed checks. Emergency admission remains limited to the original started target-sync interval with retained custody, unauthorized release and PARTIAL_FROZEN guardian. It cannot inherit the entire emergency ticket lifetime through release/cleanup/terminal intent. The v3 diff does not change this predicate. Original probes and committed active-sync, post-sync, failed-sync and cleanup tests pass. No new ARM, wait, deadline, retry or synchronization authority was added.

**R2 original identity/custody cases — closed.** The original source-link loss, substituted reader, copied runtime binding, caught reader rejection, restored record and registered upload context cases remain rejected. `_bound_source_session` uses existing guardian-held source custody, `require_reader` checks exact reader/binding custody and the issued seven-slot record, and mandatory input entry rejects phase drift. These remain actual identity checks, not acceptance of matching fields/hashes. The original 13 v1-derived repair assertions pass unchanged in the preserved v2 repro file.

**R2-followup — closed at `optical_worker/native_setup.py:1412`.** Registration is popped before validation. If the record names this exact live registered context, its original controller is taken from `record[1]` **before** checking mutable lifecycle/runtime/target/stream/source fields. All field checks, original adoption verification and source-link check execute within that controller's existing `_observe_public_operation`. A substituted `context.lifecycle` is not used as failure authority. The same controller is returned on success; no new registry or guardian was introduced. An absent/nonmatching registration has no newly inferred authority. C0 remains under its separate null-context branch.

The original two-lane source-field counterexamples now pass their existing expected-repair assertions. New independent probes additionally replace each of the five fields with a foreign sentinel, including a lifecycle sentinel that would fail if invoked. For both lanes, the actual upload entry rejects, poisons the **original** controller, consumes the registration, and performs zero modeled scope entry/allocation/copy/query. Restoring the original field does not revive the consumed context; constructing a replacement context on the failed controller rejects. Final summaries have zero normal completions and emergency attempts, with ordinary release not authorized. These are 10 new independently executed field-drift cases; the committed comparable cases also pass.

**R3 — closed at `optical_worker/native_scope.py:356`.** `_state` keeps its original `_ticket()` call, then wraps the **same** raw-query body in `try/except BaseException`, calls existing `_fail(error)`, and re-raises. Independent AST comparison proves the entire old body after its ticket equals the new try-body, including getter order and returned dictionaries. All other NativeStateReader method ASTs are unchanged from 5d. `_initialize_queries`, `_warp_cache_precondition`, `_getter`, `_result` and `snapshot` also remain AST-identical to v1. There is one query/state implementation; snapshot and the diagnostic facade now reach the same query failure boundary.

The two original raw-CUresult cases pass unchanged. Additional independent tests use the actual `_NativeScopeProbe._state` facade, actual reader, Peer and lifecycle with modeled native/Python getter leaves, on both CUDA and Warp branches in normal setup and the original active terminal sync. They catch return-code 999 and Python getter exceptions, then restore the getter and attempt another facade read. Each records failure, rejects continuation before **any** instrumented getter, emits no complete state/journal row, and forbids ordinary release. Normal failures add no COMPLETE or emergency attempt; terminal failures use only the already-entered original sync and add no further attempt. Earlier successful preparation is counted separately. Assertions are outside the terminal callback so the lifecycle cannot swallow a failed assertion and create a false pass.

Those probes retain the actual raised error object. Python getter failures retain all ten earlier getter rows plus the exact exception row; raw first-getter failures retain the 999 result with value None. Successful-null distinction remains covered by the existing result test. New before/after helper probes exercise `_install_shared_diagnostics` in both lanes: success calls the helper once and adds zero input receipts; before failure calls it zero times; after failure retains its returned owner before failing; subsequent calls reach neither helper nor getters. Committed exact pinned-Device cache models cover the Warp prerequisite independently. The supplementary generic both-lane facade models explicitly assume an established cache and are not claimed to establish it.

Two further CPU checks verify that an injected controller diagnostic-recording exception does not replace the original getter exception and that both errors are retained. They deliberately end the surrounding guard with an explicit failure; they are evidence of exception retention only, not proof of recovery from an arbitrarily replaced/broken lifecycle recorder. Ordinary sticky behavior is proved by the unmodified-lifecycle cases above.

**C1 — satisfied for this preflight.** Source-bound upload/readback entry requires the reader at the input phase and rejects omission/substitution before native work. Shared later BVH/resource helpers keep their established phase distinction and produce no input-stage receipts. The actual later diagnostic facade delegates to the sole raw reader. C0 receives no captured-source reader. Existing zero-size array/readback occurrence tests, mandatory phase/source tests, upload-core normalization/C0 checks, and new actual facade/helper cases pass. The shared CPU acceptance checker derives eligible input operations from actual completed input guards, not from all later lifecycle helpers. Sacrificial one-sync and integrated zero-emergency normal-close requirements remain distinct.

**C2 — P3, non-blocking test-harness comment: isolated subprocesses write ignored bytecode.**

Location: `tests/unit/optics/test_native_source_policy.py:290` and `:325`. Both spawn Python with `-I -S -c` but omit `-B`. `-I` ignores inherited Python environment settings, so the outer `-B` and `PYTHONDONTWRITEBYTECODE=1` do not prevent these children from writing bytecode. Six ignored files appeared with timestamps matching the source-policy run: `__init__`, lifecycle, manifest, native_scope, native_source_policy and protocol `.cpython-314.pyc`, all under `optical_worker/__pycache__`.

Consequence: the CPU review has filesystem side effects outside its requested scratch despite a clean porcelain status. This did not change source, pins or production behavior, and it does not invalidate these CPU assertions or the pinned-source preflight. Initial ignored-file status was not separately recorded; creation timestamps and the inspected subprocess commands support attribution to this run. Existing assertions test cold imports/policy, not cache placement.

Direction: in a separate test-only change, add `-B` to those fixed subprocess argv lists (or explicitly redirect bytecode within the child before imports). Proof: run both tests with empty cache state and confirm the candidate has no new ignored files. This is a non-blocking hygiene follow-up, not a condition on the native reader implementation.

Disposition in this review: original cache bytes and hash/creation inventory were copied to `generated_pyc/` in v3 scratch. An attempted PowerShell preservation/cleanup command was rejected before execution by automatic approval review with only **“blocked by policy”**; it was not bypassed. The six ignored files remain in the candidate. No tracked edits, deletion or forced cleanup occurred. Final porcelain is clean, but final ignored status is explicitly **not** empty. This report does not claim otherwise.

**Native-call, placement and resource assessment**

The source/ABI conclusion reuses predecessor evidence only where exact identities match. Read-only SSH freshly rehashed all **16** installed files against the preserved v1 identities: Warp context/utils/native core/error/ContextGuard sources, Torch CUDA/stream/version/lazy-seed sources, CUDA headers and the two query-library files. Every hash matches. The inspected remote evidence checkout remains clean/detached at `26542543d067d20c64670d375065535420ede203`; this candidate was not deployed there. Remote Python ran `-I -S -B`, read file bytes and Git state only, and imported no backend or native library. Exact rows and command are retained in `remote_hashes.jsonl`, `remote_inspection.py`, `check_remote.py` and empty `remote_stderr.txt`.

I re-read retained pinned source excerpts for Warp Device.context/stream/_init_streams/set_stream, getter declarations, native cache lookup/setter, persistent error buffer, cache erasure and ScopedDevice/ContextGuard void exit; CUDA headers and Torch Python stream/callback paths. The unchanged C10/native Torch analysis from v1/v2 is reused under the matched installed-version identity and unchanged query AST; this review does not claim a new upstream source download or binary/source rebuild proof.

- Library path/regular-file/bytes/mapped-path checks precede acquisition. Acquisition remains exactly RTLD_NOLOAD | RTLD_LOCAL; handles/functions are retained before metadata or calls. No fallback library, alternate ABI, dynamic caller-selected symbol, setter, primary-context retain/release, memory query, allocation or synchronization was added to the reader.
- CUDA retains `cudaGetDevice(int*) -> int`, separate count/primary cold queries, exact three queued callbacks and empty seed tracker before Stream. Cold logical0 is derived only after raw0/uninitialized/all-inactive observations. Warming logical and explicit-target stream getters remain after the retained real stream return. Deferred Triton nonexecution remains checked at binding. This permits the pinned logical0/raw1 postbinding model, not an invented raw0 restoration requirement.
- Warp enumerates actual CUdevice handles and maps them back to ordinals. CUresult is checked before consuming outputs. Successful null context is distinct from failed query; current-device query remains conditional. `wp_get_error_string` ABI remains `[] -> c_char_p`, and stream getter `[c_void_p] -> c_void_p` is validated, not rewritten.
- Warp stream observation is conditionally readonly. The actual original path must start with raw target context/stream absent and exact empty error bytes; after original get_stream returns, production retains/journals the stream before checking raw cache identity, positive context/stream and exact empty bytes again. Only then can the native getter run, with its first handle matching the held return. The inspected cache-miss branch can select context and allocate; it is not admitted as an observer fallback. Cache destruction is outside the retained normal setup interval. R1's interval checks stop queries after release/terminal failure.
- SourceSession retains the reader and its held/raw containers before query acquisition; the existing guardian adopts that reader. Stream/module/array/event returns enter an already-held owner graph before after-query, metadata, append, adoption or scope exit can fail. No second ownership/release graph was introduced.
- Direct call inspection confirms initialization leaf -> reader precondition/before -> original stream -> retained/journaled return -> cache/after -> binding/adoption/journal -> binding point. CUDA artifact frame start precedes leaf/before; the original create delegate executes once and retains/journals its module inside target/stream scopes; original scope exit precedes after and file completion. Exec follows its corresponding same-frame pair. Upload/readback bracket actual empty/copy/event-create/record/wait calls; Warp outer before precedes original ScopedDevice entry and outer after follows successful exit. Failed bodies skip successful after rows and unwind original scopes once. A void exit is not accepted as physical restoration without its actual subsequent observation.
- Complete contradictory snapshots are retained/appended before expected-state rejection. A getter failure preserves raw prefix and emits no complete snapshot. The original eligible input-stage terminal sync alone receives production terminal pairs; later raw diagnostics do not expand the input terminal grammar or ordinary close.
- `_RuntimeObservation` AST and `runtime_custody.py` bytes are unchanged from the original reader base. Existing v1/v2 meanings, v3 six-pair CUDA joins and source/environment/owner/guard checks are not weakened. The test facade contains no second ABI/query/state producer; later helper diagnostics call the production `_state`.

All **91** catalog entries independently match committed LF source bytes and pass the actual source AST validator; their current worktree LF-normalized bytes also match HEAD. Both lane inventories match, policy/catalog pins match, and only native_scope/native_setup/native_source_policy source digests changed from 5d. The reader and bootstrap full-source pins are exact; the frozen integrated contract still names v18 and retains its digest above. The source-policy suite rejects changed setter symbols, loading modes, paths/count rules and transferred exceptions despite recomputed caller hashes. The independent catalog audit imports no torch, warp, numpy or ctypes. The packed-render exception is unchanged.

**Independent verification and reproducible commands**

Environment: Windows PowerShell, `C:/Python314/python.exe` (3.14). Pytest runners set repository PYTHONPATH, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, UTF-8 output, TEMP/TMP and unique basetemp under v3 scratch. Pytest cacheprovider is disabled. `ROBOT_SIM_WARP_CONTEXT_SOURCE=C:\Users\AG\新建文件夹\c4_assets\worker_intake_gap_scratch_v1\pinned_warp_context.py`; exact-Device models independently assert its pinned SHA256. The isolated subprocess bytecode exception is disclosed above.

Commands executed from the frozen candidate, where `<v3>` denotes the full scratch directory:

```text
C:/Python314/python.exe -B <v3>/run_checks.py inventory
C:/Python314/python.exe -B <v3>/run_checks.py regressions
C:/Python314/python.exe -B <v3>/run_checks.py originals
C:/Python314/python.exe -B <v3>/run_checks.py boundaries
C:/Python314/python.exe -B <v3>/audit.py
C:/Python314/python.exe -B <v3>/check_remote.py
C:/Python314/python.exe -B <v3>/static_checks.py
C:/Python314/python.exe -B <v3>/final_checks.py
```

The runner scripts and `*_execution.json` preserve exact expanded argv/cwd/PIDs/exit codes, not pseudocommands. Each pytest command is `python -B -m pytest -q -p no:cacheprovider --basetemp=<v3>/pytest_<group>` followed by the files below.

| Independent run | Files/evidence | Observed result |
| --- | --- | --- |
| regressions | test_native_scope_observer.py, test_native_scope_integration.py, test_native_upload_core.py, test_guarded_native_readback.py, test_guarded_lifecycle.py, test_native_source_policy.py, test_guarded_runtime_binding.py, test_cuda_artifact_replay.py | **502 passed in 31.57s**, exit 0, no skips |
| originals | Unchanged v2 scratch test_independent_identity.py and test_ast_normalization.py | **21 passed in 3.46s**, exit 0: 17 identity/failure cases including all four old expected-repair failures, plus four AST mutation checks |
| boundaries | New v3 test_review_boundaries.py | **26 passed in 3.26s**, exit 0: eight actual facade error cases, six later-helper cases, ten field-drift cases and two exception-retention cases |
| source audit | audit.py; identity_hashes.json | **91 LF source/AST entries**, both lanes and policy digest pass; no backend/FFI imports |
| static audit | static_checks.py; static_checks.json | Original raw query body and ticket preserved; unchanged method/ABI/cache ASTs; exact fixture/report identities; only intended catalog digests changed; CPU collector/replay unchanged |
| dependency identity | check_remote.py; remote_hashes.jsonl | **16/16 installed hashes match**, primary SSH exit 0 |
| final preservation | final_checks.py; final_audit.json | Exact HEAD/detached/porcelain, 91 current source-byte matches, all 48 original evidence files unchanged; remote inspection PIDs absent |

Total freshly executed pytest cases: **549 passed**, no skips, no candidate test failure. The author's 203/37/4/17 summaries and concurrently running source suite are not claimed as independent results. No old outputs/caches were used as proof of a new test execution. Ordinary imported CPU modules and test doubles do not constitute native acceptance.

Inspection/tool issues, distinct from candidate failures: large combined reads needed focused rereads; one guessed support filename did not exist and was replaced with the actual `native_scope_acceptance.py`; one PowerShell-to-Python Unicode literal was misencoded and corrected using cwd-relative paths; initial static audit treated the fixture's raw SHA as LF and correctly failed until raw-versus-LF identity was checked explicitly. No content mismatch remained. The cache cleanup policy rejection is recorded separately above; it was not bypassed or portrayed as a successful cleanup.

**Owned process closure and limits**

Reviewer CLI PID **37324**, parent **47388**, creation Unix milliseconds **1791624812103**, is recorded in `final_audit.json`. The enclosing launcher is not a child spawned by this review. Pytest PIDs **46672**, **7124** and **4708** were waited/reaped by their runners with exit 0; every tool session returned. A final Win32_Process traversal rooted at the reviewer, plus scratch-scoped Python/SSH/SCP command-line checks, found no owned residual and none of those test PIDs alive. Evidence: `process_closure.json`, observed at `2026-10-10T17:40:54.9010338+08:00`.

Read-only remote inspection PID **2521064** and parent **2521062** were absent at closure. The final synchronous closure probe PID **2538056** completed with SSH exit 0. No process was signaled. These are scoped OS observations, not exhaustive process/GPU attestation. No GPU inventory was queried because the packet prohibits GPU/native queries.

Unexecuted: new native reader/library acquisition, either real framed-worker lane, CUDA six-pair runtime success, Warp physical restoration/error-buffer/void-exit cases, real Go2 final numerical samples, full C2f workload/failure/ACK/disconnect/hang matrix, and formal performance. The full historical source-custody/native integration suites were not rerun; the exact eight CPU regression files and scratch probes above bound fresh runtime evidence. Pinned-source analysis does not establish proprietary driver internals, rebuild equivalence, actual device state or real native destructor/release safety.

**Exact permission and gates**

Approval binds only **df1a46ae5039fe8a25718c7ec60f52fa7c60c549** and its approved preparation-reader design/pins. It permits moving to bounded native setup/framed-worker validation on the pinned deployment, including same-production-loop setup coverage, all six true CUDA create/exec pairs, complete Warp setup observations, affected before/after query/cache/scope failures, retained ownership, and observed/reaped process closure. It does not permit treating CPU mocks or older campaign data as those results. Changed code/pins or newly discovered native policy require the existing review workflow; this report grants no native acceptance in advance.

Predecessor F1's producer gap is structurally repaired, but actual CUDA intake still needs that fresh native proof. F2 final-sample physical/numerical evidence remains open. F3 predecessor compact-reply/journal-recovery code is outside this approval. C1's mandatory-input/later-helper/C0 distinction is retained.

**Full C2f/C4, final-sample physical/numerical proof, F3 predecessor compact-reply approval, C4c3/c4/C4d/B3, product activation, formal performance, result/baseline, backend selection, R2/S/host delivery, a3b and shipping remain closed. This bounded preflight is not parent acceptance.** No further work or native execution is initiated by this reviewer. The complete report is ready for verbatim archive.
