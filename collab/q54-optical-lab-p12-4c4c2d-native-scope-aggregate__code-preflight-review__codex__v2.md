# Independent native-scope code/native preflight re-review

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: independent code/native preflight re-review
Author: Codex, independent reviewer /root/native_scope_code_preflight_v2
Version: v2
Date: 2026-10-08 (Asia/Shanghai)
Status: in_review — Request changes
Related Files: native-scope-aggregate design-request v2, code-preflight-review v1, implementation-note v2; native-scope-cache design-request/design-review v1; tests/support/native_runtime_process.py; tests/unit/optics/test_native_scope_observer.py
Owner Summary: The original cache-safety and ordered C10-diagnostic findings are repaired. One narrower required diagnostic is still missing: the raw Warp initialization prerequisites are asserted without preserving the facts that caused rejection. Two independent CPU cases demonstrate the missing data through the real failure writer and lifecycle. No new native case ran or is approved by this report.

## Verdict and revision

**Request changes.** F1 and F2 from v1 are closed for this revision's bounded code/source proof. New F3 below remains a required-contract gap. It concerns failure evidence, not a demonstrated unsafe cache getter or release. The requested permission for the next 16 actual native-scope cases remains closed until repair and fresh independent review.

- Reviewer session: `/root/native_scope_code_preflight_v2`, independently tasked, no implementation authorship, inherited implementation reasoning, subdelegation or candidate edits.
- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Base: `b0f22fb4f6eda395f72ae34f37044ed918254184`.
- Frozen inspected HEAD: `4da4ed271f20f847bea45ee005516c9d7cb21197`.
- Initial and final inspected tree clean; `git diff --check b0f22fb HEAD` passed. Production and tests/data have no base-to-HEAD changes. The code delta is limited to the fixed test observer and CPU tests; other changes are collaboration/progress records.
- Independently matched governing aggregate design SHA256 `9e678c7aaffa1147b49a6f1f3b946619f0c316e3bc68883ff0fa9a4d1ed9a0ad`, supplement SHA256 `393a32720ce10ddc525f132c38e5109a45b537a8a5df4ffa08524013e078a68c`, and complete v1 report SHA256 `b18527c80bc1daa951879de99ed39aeed4c8942685635e5e39e48a7daf3adeca`.

Read AGENTS.md, collab/README.md and REVIEW_PRINCIPLES.md completely, rereading the principles as UTF-8 after an initially garbled default decode. Read the governing design, supplement, prior full report, original repros, supplement review, implementation packet, complete code delta and relevant unchanged observer/bootstrap/readback/collector/parser/acceptance paths. Prior approvals and author counts were not treated as approval evidence.

## F3 — P2: raw Warp prerequisite facts disappear on rejection

Location: `tests/support/native_runtime_process.py:530-551`, particularly the raw-field assertions at 536 and 539-542. The supplement's exact-ordering step 1 requires checking absent raw `_context` and `_stream`, reading raw vars without properties, and explicitly says **“Preserve this raw diagnostic.”**

Trigger: the target already has a non-None raw `_context` or `_stream` before original get_stream. `_warp_cache_precondition(before=True)` reads those fields and raises a bare AssertionError before appending any diagnostic for them. `_fail` preserves prior query rows and exception type/message, but neither the field values, their presence/null status, nor which conjunct failed. Even successful raw prerequisites have no corresponding raw record. The analogous post-return identity/handle checks also have no own raw record before their assertions.

Independent reproduction: `c4_assets/test_native_scope_raw_precondition_v2.py` reuses the real CPU observer/lifecycle fixture, sets either `_context=9000` or `_stream=9000`, enters the actual modeled live segment, calls the implemented precondition, and reads the file written by the implemented `_fail`. Both cases yield:

```json
{"schema_version":"q54.c2d3.native-scope-diagnostic.v0","validating":false,"raw":[],"exception_type":"AssertionError","message":""}
```

Both retain UNKNOWN and `release_status=not_authorized`, make no error-buffer query and never establish the cache flag. These assertions passed in **2 CPU cases**. In a complete native initialization the raw list would additionally contain earlier driver-enumeration results; those do not contain or recover the missing Python prerequisite facts. This report does not claim a native run normally starts with the injected condition.

Consequence: the required diagnostic cannot distinguish a preexisting context from a preexisting stream, preserve the observed value, or show the raw prerequisite facts used by the new proof. The failure remains conservative, but the accepted supplement's explicit evidence requirement is unmet. This is similar in category to the prior lost-prefix diagnostic finding, with a narrower trigger.

Why existing tests miss it: they verify native-error-buffer output, cache gating, owner retention, and UNKNOWN; none checks the diagnostic of absent/non-absent raw fields. The stale-error tests start with valid absent fields and therefore reach the already-recorded native error getter.

Required repair: preserve a bounded primitive diagnostic from the exact raw reads before assertions can discard their evidence, identifying the pre/post boundary and the actual presence/null/value/identity facts needed by those checks. Keep returned owners retained before post-return work. Do not call properties, `repr`, native getters, setters, or other fallible resource discovery merely to render a diagnostic. Do not fabricate an aggregate snapshot for a rejected precondition. Add tests that distinguish both original counterexamples in the persisted diagnostic and retain existing query ordering, flag gating, failure and release behavior. This can remain a test-observer-only repair under the approved supplement; no new query/ABI or production policy is requested.

## Closure of prior F1: failure-sensitive native-cache proof

The actual original hooks establish the intended order: production enters `indexed_runtime_stream`; the observer resolves retained queries, checks the precondition and records the pristine snapshot; original get_stream executes once; production retains and journals its returned owner; the original injected after-return failure runs first; only then the postcondition executes. Its positive context, positive returned handle, exact target-stream identity and exact-empty native error result establish the local flag. Every non-pristine Warp stream getter goes through the sole guarded `_state` call site. The first completed native stream snapshot must equal the held returned handle before baseline assignment; zero/mismatching responses are appended truthfully and rejected.

The retained function is taken from the actual held runtime core, held before use, and its existing `[]`/`ctypes.c_char_p` declaration is checked without mutation. Bytes are preserved as hex; None, text, stale and nonempty outputs reject. The two reads do not replace direct post-scope driver observations. No error clear, retry, setter, new stream or observer cache initialization was added.

I independently rechecked the pinned source chain as text. Device.context -> _init_streams -> set_stream publishes Python `_stream` after the void native setter, so the old unsafe inference still cannot be used. Native get_context_info with a positive context either finds/inserts its cache entry or reaches the failing check_cu branch. check_cu/check_cuda failures write a nonempty `Warp CUDA error` prefix. The static error getter reads without clearing; successful original stream/create/register/restore calls do not overwrite that buffer with empty data. Therefore the exact-empty postcondition excludes the modeled NULL/cache-miss failure before the observer getter. ContextGuard and the malloc/free workaround errors also make the postcondition fail.

All g_contexts references in the pinned warp.cu were inspected: declaration, lookup, insertion, and the erase inside wp_cuda_context_destroy. The fixed initialization/helper/scope/terminal path does not call context destroy, primary release or shutdown during observation. Strongly held runtime/target/stream/observer owners survive through the original terminal disposition. This remains a proof about the exact fixed single worker and admitted native dependency, not arbitrary concurrent external mutation or source/binary rebuild equivalence.

The author-added exact-Device CPU model uses AST methods from the hash-matched full pinned context.py, with modeled native behavior and no Warp import. I executed its positive and void-setter-error paths and its retained-return hook tests. The failed original setter now rejects before its modeled getter, keeps the original owner, marks UNKNOWN, and permits no sync/completion/release. The positive path calls the modeled native stream getter once. The unchanged v1 repro was independently adapted in memory only to supply the new local flag and assert the new contract: the exact old absent-cache state now stops before the modeled cache-miss getter. The original repro file was not edited. These are CPU/source proofs, not GPU observations.

## Closure of prior F2: ordered C10 diagnostics

Count, every primary ordinal, logical device, and explicit-target current stream now use `_getter`. Each result is recorded immediately with the exact fixed getter name and primitive arguments. On exception the exception is strongly held before formatting/recording and re-raising; no retry or later getter executes. Both Warp getters use the same bounded wrapper. Existing direct CUresult checks retain nonzero-error versus successful-null handling.

I executed all ten actual-observer modeled failure positions and the count exception test. They preserve the preceding returns and exact failing call, emit no complete row, and keep UNKNOWN/no release. An independent in-memory adaptation of the unchanged v1 F2 repro now observed cudaGetDevice followed by primary ordinals 0=false, 1=true, and ordinal 2's named exception. Original call order `[0,1,2]` remained unchanged.

## Affected invariants inspected

No production or schema delta was introduced. The collector remains CPU-only; raw diagnostics do not extend issued aggregate cuts. Record-before-sidecar writing and current/prior_validated_cut/unavailable selection remain intact. Parser lifecycle operation references stay separate from journal before references, bounded by the lifecycle cut and tied to exact guard/descriptor/phase. Native pairs stay nested and unique. Runtime-stream after requires the immediate actual returned_owner, binding point joins the actual adoption, and terminal rows remain limited to the original live emergency-sync interval. Contradictory completed states remain appendable evidence, then fail the fixed live checker.

Readback retains its one Warp ScopedDevice around the original factored body after identity and host-retention checks, returning the report after scope exit. Helper and scope wrappers retain original functions and returned owners, delegate once, and skip post-failure observations. The direct wrong-context-after-exit CPU test still appends the contradiction before sticky failure. The observer changes do not change release authority, operation counts, deadlines or watchdogs. Acceptance still checks actual occurrence coverage, full terminal trace and external worker identity; these CPU checks do not supply that external native evidence.

## Commands and observed results

Local environment: PowerShell, `C:/Python314/python.exe`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONDONTWRITEBYTECODE=1`, `-B`, `-p no:cacheprovider`. Exact-Device source: `ROBOT_SIM_WARP_CONTEXT_SOURCE=C:/Users/AG/新建文件夹/c4_assets/native_scope_pinned_warp_context.py`.

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_scope_observer.py tests/unit/optics/test_guarded_native_readback.py tests/unit/optics/test_runtime_custody.py
78 passed in 18.55s

C:/Python314/python.exe -I -S -B ../c4_assets/native_scope_preflight_independent_v2.py
exit 0; original F1 getter blocked; original F2 ordered prefix retained; raw prerequisite omission printed

C:/Python314/python.exe -B -m pytest -q -s -p no:cacheprovider ../c4_assets/test_native_scope_raw_precondition_v2.py
2 passed in 0.95s; both demonstrate F3 through the real failure writer/lifecycle
```

Fresh remote read-only verification piped a stdlib pathlib/hashlib/json/sys script to `ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -`. It hashed installed source files and returned error.cpp/error.h text. Python reported 3.12.13 and `backends_loaded=[]`. Fresh hashes matched:

| File | SHA256 |
| --- | --- |
| _src/context.py | eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee |
| native/warp.cu | 7c7e6e7a02db7b285ca1800d0ec52e28d2cf8a704435d925c70a8c5ccd8f176b |
| native/warp.cpp | b9feea467eb11e0d9016ca33fceb7b1a4358fc21c65513a692378155f1d05fc7 |
| native/cuda_util.cpp | 21d9dfe57b861ada6b98d0fa2ccfb3b8ed0b14972ed840eb04d0ad7e5f70a89f |
| native/cuda_util.h | 59207856d17907bca935612ccb3ce7cde0952d8491011e1d45e8bb2d5b9ed1b8 |
| native/error.cpp | 877b9f1e889d3e5bf8950c3a3deaf28f44bc4a9a76e780dcc19606d80f6229f2 |
| native/error.h | 4ebf4a1dcd7ed121df72f9390888c5f9eb30547f0e42fde5e04ace50de7face0 |

The retained JSONL fulltext source hashes were independently recomputed before source-path reasoning. Original repro SHA256 values remained `c39a41003cc5be93da391d0c8ae326d92eae96af16e1dd2bfa584b9ff7584f34` and `c7c03ba9c55006a430461b3ddec4877e6562dd363266c704b12afdb1542d3d24`.

The coordinator supplied exact-HEAD remote CPU archive `c4_assets/c2d3-4da4ed2-cpu-regression-v2.tar.gz`. I independently hashed its 12217 bytes to `9c10b2f587ca13b70fec321272768331c08f81182455e3ae6bb7b5617f02f03f` and read its stdout (`110 passed in 9.39s`) and empty stderr. This is coordinator-produced CPU evidence including the 32 POSIX codec cases, not my independent execution or native proof. I did not rerun the full 501-test baseline; its distinct old HEAD was not relabeled as current evidence.

After report drafting, the coordinator additionally reported a current foundation CPU run with 303 passed and 5 failed: five older C0 closed-worker HELLO cases reportedly reject `optical_worker/lane_contract.py` outside their static source allowlist. I have not independently executed or inspected that failing run, so neither its root cause nor a future repair is approved here. The coordinator proposes separate bounded repair/review. These reported failures must remain visible as outstanding regression evidence and cannot be represented as a green full-C2d3 result or waived by any future narrow observer approval.

Inspection command limitations: an initial oversized combined dump was truncated and replaced by focused reads; one stdlib local read of the saved error.cpp used the platform GBK default and failed decoding, followed by the successful fresh UTF-8 remote source read. These were inspection/tooling errors, not candidate failures. Pytest may create normal temporary files; bytecode/cacheprovider writes were disabled. Durable reviewer additions are only the two scratch files named above and this report in c4_assets. No candidate or remote files were edited by the reviewer.

## Remaining scope and gates

Fix F3, freeze the revision and request a fresh independent re-review including these retained counterexamples. The coordinator acknowledged the omission and said it would repair only after this frozen review ends; that intention is not a reviewed repair.

No backend import, native library call/load, GPU query, native worker, remote mutation or process signal was performed in this review. Real error-buffer emptiness after quiet Warp initialization remains unobserved here. The 16 actual cases, affected full fault/source/persistence regressions, external process/GPU residual checks and whole-C2d3 independent review remain outstanding. No C2d3/C2d/C4 acceptance, preparation readiness, final graph seal, warmup, measured optical work, result, performance or backend decision is issued. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host, R2 and shipping stay CLOSED. C4 stays active and unaccepted.
