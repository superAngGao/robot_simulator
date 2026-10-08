# Independent native-scope preflight and bounded C0 compatibility re-review

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: independent code/native preflight re-review and bounded C0 compatibility review
Author: Codex, fresh independent reviewer /root/native_scope_code_preflight_v3
Version: v3
Date: 2026-10-08 (Asia/Shanghai)
Status: accepted for the exact bounded scope below
Related Files: native-scope-aggregate design-request v2, code-preflight-review v2, implementation-note v3; native-scope-cache design-request/design-review v1; tests/support/native_runtime_process.py; tools/optical_closed_worker.py; closed_worker_runtime_contract.json
Owner Summary: Approve the bounded preflight. F3 is closed by persisted raw prerequisite facts before rejection; F1/F2 remain closed under the fixed-worker source/model proof. The exact two-owner C0 allowlist repair preserves the existing policy and passes the supplied exact-HEAD CPU regression evidence. This permits the next 16 actual native-scope cases; it does not accept those cases, whole C2d3, or any downstream product gate.

## Verdict, identity and candidate

**Approve.** No substantive blocking or non-blocking finding was established in this bounded re-review. Approval binds only the inspected revision and the scope stated below.

- Reviewer session: `/root/native_scope_code_preflight_v3`, fresh independent task with no inherited implementation conversation, no candidate authorship, no subdelegation and no implementation edits.
- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Base: `4da4ed271f20f847bea45ee005516c9d7cb21197`.
- Frozen inspected HEAD: `e585956eb29e3a33483d8cb0c7444f893b390ac4`.
- Initial and final local tree were clean, including untracked files. `git diff --check 4da4ed2 HEAD` passed.
- Governing aggregate design SHA256: `9e678c7aaffa1147b49a6f1f3b946619f0c316e3bc68883ff0fa9a4d1ed9a0ad`.
- Supplement SHA256: `393a32720ce10ddc525f132c38e5109a45b537a8a5df4ffa08524013e078a68c`.
- Complete archived v2 report SHA256: `7be029dba351e713c46d5fe904983b0f709235f25f29d55fde76133b463f36db`.

Read AGENTS.md, collab/README.md and REVIEW_PRINCIPLES.md completely. Reread the principles as UTF-8 after the initial default decoding was garbled. Inspected the governing design, supplement and its review, prior complete v2 findings and original repros, v3 packet, actual diff, observer and relevant unchanged production/bootstrap/readback/parser/worker paths. Author notes and previous verdicts were treated as claims to check, not independent approval evidence.

The eight-file base-to-HEAD delta contains the raw observer diagnostic and CPU tests; exactly two added C0 source paths in both literals; one catalog digest repin; and collaboration records. It adds no new native query, callback surface, production GPU activation, schema, watchdog or admission mechanism.

## F3 closure: original failures now preserve their facts

At `tests/support/native_runtime_process.py:530` the actual raw dictionary reads now construct and append a diagnostic before their assertions. The record distinguishes pre/post boundary, context presence/type/value, stream presence/nullness and prior local cache flag. Post-return adds exact stream-object identity and returned raw handle presence/type/value. Nonprimitive context/handle values are represented by type and no invented pointer. The path reads raw dictionaries, not the `_context`, `_stream` or `cuda_stream` properties, and does not invoke repr, allocate object tokens, discover resources or query the native error buffer merely to explain a rejected prerequisite.

I ran the original unmodified `test_native_scope_raw_precondition_v2.py`. Both tests now fail specifically at their historical `diagnostic['raw'] == []` assertion: the context case contains `context=9000, stream_is_none=true`, while the stream case contains `context=null, stream_is_none=false`. These two expected failures are evidence that the old bug assertion no longer holds; they are not counted as passing candidate tests.

I then executed independent repaired-contract assertions against the same real fixture, failure writer and lifecycle in `c4_assets/test_native_scope_independent_v3.py`. The original two cases require the exact complete persisted diagnostic. Six additional cases cover missing post-return context/stream/handle, bool context/handle, and an unexpected handle object whose repr and handle property must not execute. All **8 passed**. They also require no error query, no established cache flag, no complete aggregate row, UNKNOWN, no normal completion, no emergency sync and release not authorized. The returned object remains held in the post-return cases.

The candidate's eight diagnostic tests independently passed in my broader local execution. Their cases additionally distinguish missing versus present preconditions and exercise hostile target properties. Production already holds the returned stream before the observed-return hook; the new fallible diagnostic work did not move ahead of that retention. The existing injected after-return failure still runs before the postcondition. F3 is therefore closed for the required raw evidence, not merely by changing an assertion message.

## F1/F2 retention and affected native invariants

F1 remains closed under the exact admitted runtime and fixed single-worker assumptions. The initialization hook enters the original indexed ticket after quiet init, captures queries, checks absent raw target context/stream and exact empty error bytes, then takes the pristine snapshot. Original get_stream delegates once. Production appends the actual returned stream to `runtime_owners` and journals returned_owner before the hook's injected fault and postcondition. Only exact returned identity, positive exact-int raw context/handle and exact empty second error result establish the local flag. Every non-pristine Warp native-stream getter still requires that flag. The first native stream result must equal the actual returned handle before baseline assignment; zero/mismatch produces a factual completed row and sticky rejection.

I reran `native_scope_preflight_independent_v2.py`, which adapts the original v1 counterexamples in memory without modifying their files. The original modeled absent-cache state was blocked before the potentially initializing stream getter; the modeled cache stayed absent. The exact pinned Device AST model's success, void-setter error, stale/NULL/nonbytes error, unset flag, first-zero/mismatch and actual retained-return hook tests also passed. These execute CPU models, not Warp.

I independently recomputed retained source-text hashes, reread the relevant call chain, and freshly hashed installed sources using a read-only stdlib SSH command. Device.context -> _init_streams -> set_stream publishes Python metadata after the void setter, so metadata alone remains insufficient. The positive-context native get_context_info path either finds/inserts its entry or reaches the failed check_cu branch. check_cu/check_cuda failures write nonempty error text, including ContextGuard and allocation-workaround failures. The error getter returns the static buffer without clearing it; successful original calls do not write empty data. Both exact-empty checks therefore reject the prior NULL-cache counterexample before the observer getter.

All `g_contexts` references in the pinned full warp.cu are declaration, lookup, insertion and erase in `wp_cuda_context_destroy`. The fixed initialization, helper, scope and terminal observation path does not invoke context destruction, primary release or shutdown. Runtime/target/stream and original functions remain in the held graph during observations. This is not a proof about arbitrary concurrent callers or external cache mutation.

The actual error function is taken from the held runtime core, retained before use, and checked against its existing `argtypes=[]`, `restype=c_char_p` declaration without changing it. Its bytes are retained as hex. Driver query ABI remains c_int CUresult/CUdevice, c_void_p CUcontext and c_uint flags with the declared pointer arguments. Native errors remain distinct from successful null results. The stream getter's pinned declaration remains `[c_void_p] -> c_void_p`. No retry, clear, setter, extra stream or observer cache initialization was added.

F2 remains closed. The unchanged independent original-C10 adaptation observed cudaGetDevice, primary ordinal 0=false, 1=true, and the named ordinal-2 exception in order, with calls `[0,1,2]`. Candidate tests exercised all ten primary/logical/current-stream failure positions and count failure. The wrapper immediately preserves successful results and holds an exception before fallible diagnostic formatting, performs no retry, and emits no complete snapshot on failure.

Readback still enters its one explicit Warp validation leaf and original ScopedDevice after identity/retention checks, holds the original body results, and returns its report after exit. The pinned ScopedDevice restores its saved default metadata and delegates ContextGuard exit; void native pop alone is not treated as successful restoration. The contradictory post-exit context CPU test still appends the actual contradiction before UNKNOWN/no release. Wrappers retain original functions and returned owners, delegate once, and skip successful-after observations on sticky failure.

The unchanged parser keeps lifecycle operation references separate from journal before references, bounds them by the cut, ties them to the exact guard/phase/descriptor, enforces most-recent helper leaves, and requires unique nested pairs. Runtime-stream after requires the immediate actual returned_owner; binding points join the actual adoption; terminal rows remain in the original live emergency-sync interval. Raw diagnostics neither extend immutable aggregate cuts nor grant release authority. Record-before-sidecar writing remains intact. The supplied full CPU run additionally covers the POSIX historical-record codec mutations and source-policy/catalog regressions.

## Exact C0 compatibility repair

The static bootstrap list and matching runtime contract add exactly `optical_worker/lane_contract.py` and `optical_worker/native_manifest.py`. The actual import chain is worker -> native_setup -> native_contract -> native_manifest -> lane_contract. Both extracted owners are CPU-only. I imported the actual worker in a fresh `-I -S -B` stdlib process and enumerated its actual repository modules: package init, lane_contract, lifecycle, manifest, native_contract, native_manifest, native_setup, protocol and worker. Both added files appeared; all were in the exact list; Warp, Torch, NumPy and ctypes were absent. Both owners and all listed files passed the actual AST validator.

This extends the existing explicit list for the already-extracted owners; it does not replace it with discovery, prefixes or exemptions. Bootstrap still compares the whole ordered path tuple before hashing/importing. Runtime validation still requires exact contract equality, source hashes and AST checks, and the actual import ledger still rejects any loaded repository path outside the list. The two missing-owner negative cases passed independently. Product PREPARE_FROZEN_ROW still routes to `_reject_gpu_preparation`, with the false activation flag; raw launch remains outside the protocol. No production initialization path was connected to C0.

Catalog v12 SHA256 independently matched `8c8554e30a09319c8bc41d3eccefa0ef301e827341be6b626f91c7ba1687e2dc`. A structural comparison found exactly one changed scalar: the tools/optical_closed_worker.py entry from `efbd167c54a596efeb3f3cc45c18514b05a7df5637fe6281217019d8df73be66` to `6114b52172e437c8f8a4035b1f3ee0e46814e133b49b12ce26bb7ecfba740894`. No lane, policy, source-set or other digest changed. The new digest matches the Git blob and the Linux execution manifests. The local Windows file has 152 CRLF line endings and raw SHA `2e8a78a47b0662e891b8ac142775e6c0b8bd0d83dd0bc65cdfd9b552088005fe`; removing CR bytes yields the exact Git blob. This checkout encoding difference is recorded, not confused with an unreviewed code change or native deployment proof.

I independently hashed and read the old foundation archive `c2d3-4da4ed2-foundation-cpu-v1.tar.gz`, matching `b87d22bb0caf9fd1eefe8172c079645af3bf7e38cb538bb93490938ec8d78ae5`. The five raw termination journals really name lane_contract outside the source allowlist. The old 303-pass/5-fail result remains historical failure evidence.

The coordinator then supplied the new `c2d3-e585956-foundation-cpu-v2.tar.gz`: 16,327,350 bytes, independently matched SHA256 `7396ebb1ae39d692a3d58648ce4dae29085fe4b50dac23f23d47073121d7ba9d`. Its raw stdout is **637 passed in 125.94s**, stderr empty, summary returncode 0, and no skips. I inspected the runner, which asserts clean exact e585956 before execution and a clean tree afterward. The actual five C0 manifests independently embed the full e585956 revision, false product GPU flag and the exact three relevant source hashes. Both HELLO supervisors report worker exit 0 with no restart/watchdog; their termination journals are empty. Product PREPARE's tested error path exits 0; raw launch records request_protocol_failure; control EOF records channel_eof and the expected nonzero worker termination. The tests assert the added paths in the HELLO import ledger, disabled preparation, zero submission behavior and raw-request rejection. This is supplied exact-HEAD execution evidence inspected by me, not my own independent rerun of those Linux workers.

Its residual file reports 65 observed owned process identities, remaining_os=[] and remaining_gpu=[]. The process-identity file and runner explicitly limit this to live descendants observed by procfs polling; very short-lived children may be missed. I did not independently perform GPU queries or reproduce the process polling and do not turn this into complete attestation of every spawn.

## Commands and results

Local environment: PowerShell, C:/Python314/python.exe, bytecode disabled with `-B`, pytest plugin autoload disabled and cacheprovider disabled. For the pinned Device models, `ROBOT_SIM_WARP_CONTEXT_SOURCE=C:/Users/AG/新建文件夹/c4_assets/native_scope_pinned_warp_context.py`.

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_scope_observer.py tests/unit/optics/test_guarded_native_readback.py tests/unit/optics/test_runtime_custody.py tests/unit/benchmarks/test_optical_closed_worker.py -k 'not clean_spawn and not product_gpu and not object_bearing and not control_eof'
88 passed, 5 skipped, 3 deselected in 6.18s

C:/Python314/python.exe -I -S -B ../c4_assets/native_scope_preflight_independent_v2.py
exit 0; original modeled cache getter blocked, original C10 prefix retained, new raw prerequisite row present

C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider ../c4_assets/test_native_scope_raw_precondition_v2.py
2 expected historical-bug assertion failures: raw is now nonempty

C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider ../c4_assets/test_native_scope_independent_v3.py
8 passed in 1.05s

C:/Python314/python.exe -I -S -B ../c4_assets/native_scope_static_review_v3.py
exit 0; hashes, one-field catalog delta, actual CPU closure, AST policy, old raw journals and pinned source inspection
```

The first local command contains the 86 observer/readback/parser passes plus two C0 missing-owner negatives; Windows POSIX skips and the explicit filter are not represented as a full C0 execution. The separate supplied 637-pass Linux run executes all 14 suites listed in its summary: C0 protocol/processes, guarded wait core/processes/lifecycle, native/CUDA arena contracts, cold custody failure doubles, observer/readback/custody/native codec and source/source-policy suites.

Fresh read-only SSH used a pathlib/hashlib/json/sys script piped to `ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -`. It returned full error.cpp/error.h text and the following source hashes, with Python 3.12.13 and backends_loaded=[]:

| File | SHA256 |
| --- | --- |
| _src/context.py | eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee |
| native/warp.cu | 7c7e6e7a02db7b285ca1800d0ec52e28d2cf8a704435d925c70a8c5ccd8f176b |
| native/warp.cpp | b9feea467eb11e0d9016ca33fceb7b1a4358fc21c65513a692378155f1d05fc7 |
| native/cuda_util.cpp | 21d9dfe57b861ada6b98d0fa2ccfb3b8ed0b14972ed840eb04d0ad7e5f70a89f |
| native/cuda_util.h | 59207856d17907bca935612ccb3ce7cde0952d8491011e1d45e8bb2d5b9ed1b8 |
| native/error.cpp | 877b9f1e889d3e5bf8950c3a3deaf28f44bc4a9a76e780dcc19606d80f6229f2 |
| native/error.h | 4ebf4a1dcd7ed121df72f9390888c5f9eb30547f0e42fde5e04ace50de7face0 |

Retained utils.py source text used for ScopedDevice was separately hash-checked at `01bd20f279f5762881c50cf8b6d766734fe21bd5f9b1ec286b34b1f3fcb27f24`. The original v1 repro files remain unchanged with SHA256 `c39a41003cc5be93da391d0c8ae326d92eae96af16e1dd2bfa584b9ff7584f34` and `c7c03ba9c55006a430461b3ddec4877e6562dd363266c704b12afdb1542d3d24`.

Some initial combined reads and one rg against JSONL exceeded output limits; focused UTF-8/source-AST reads replaced those outputs. No candidate error is inferred from that tooling truncation. Pytest created normal external temporary artifacts; cacheprovider and bytecode writes were disabled. Durable reviewer additions are the two new scratch scripts and this report under c4_assets, outside the frozen candidate. No implementation, remote worktree, package, private instruction or original repro was edited.

## Approval scope, limits and required next work

This approves the e585956 bounded code/native preflight and exact C0 compatibility repair and permits proceeding with the next **16 actual native-scope cases** under the already-approved design. It does not certify any actual native outcome. No reviewer backend import, native library load/call, GPU query, native observer worker, remote mutation or process signal occurred. Source/models cannot establish actual hardware error-buffer emptiness, real native restoration, binary/source rebuild correspondence or proprietary driver internals; the admitted dependency boundary remains necessary.

All 16 actual cases, affected full native/fault/source/persistence regressions, same-run external process/GPU residual checks and fresh independent whole-C2d3 review remain required. The supplied CPU run does not waive those requirements. No C2d3/C2d/C4 acceptance, final graph seal, preparation readiness, warmup, measured optical work, performance result or backend decision is issued. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host, R2 and shipping remain CLOSED. C4 stays active and unaccepted.

The unchanged full report may be archived through the collaboration protocol's evidence-only exception, accurately binding this approved code revision and the separate archive commit. Any operative code, test, design or workflow change requires review of that delta.
