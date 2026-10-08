# Independent native-scope implementation and native preflight review

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: independent code/native preflight review
Author: Codex, fresh independent reviewer /root/native_scope_code_preflight_v1
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: in_review — Request changes
Related Files: native-scope-aggregate design-request/design-review v2 and implementation-note v1; optical_worker/runtime_custody.py; native_bootstrap.py; native_readback.py; tests/support/native_runtime_process.py; tests/support/native_scope_acceptance.py; tests/gpu/test_native_scope_runtime.py
Owner Summary: The v1/v2 codec, Warp readback scope, occurrence joins and fixed observer are substantially implemented, and 360 independent local CPU tests passed with 13 explicit skips. Native preflight is not approved: a normal Python stream return does not prove that Warp's native context-info cache exists after the original void setter's error path. The proposed observer getter can therefore execute the forbidden cache-miss context/allocation path. CUDA getter failures also omit successful preceding C10 query returns from the required ordered diagnostic. No new native observation case executed in this review.

## Verdict and exact boundary

**Request changes.** This verdict addresses the frozen bounded implementation/preflight, whose requested approval would permit the specified next 16 actual observation cases. It does not reject every existing C2d3 property, claim those 16 cases failed on hardware, or replace the later whole-C2d3 review. The new native observation cases remain **UNEXECUTED and gated**. F1 must be resolved before approving the Warp getter for execution; F2 requires the bounded diagnostic repair and CPU proof below.

- Independent reviewer session: `/root/native_scope_code_preflight_v1`. No implementation authorship, inherited implementation conversation, subdelegation or candidate edits.
- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Base: `c243875b87873d2e57a252e3113492feaa774af3`.
- Frozen HEAD: `b0f22fb4f6eda395f72ae34f37044ed918254184`.
- Initial and checked local tree clean; `git diff --check c243875 HEAD` passed.
- Governing v2 design SHA256 independently matched `9e678c7aaffa1147b49a6f1f3b946619f0c316e3bc68883ff0fa9a4d1ed9a0ad`; unchanged independent design-review SHA256 matched `563d8661a8b1463cef09921d07b58244c4b63f5d00d43421475798d21b5c3818`.
- Production v12 catalog raw SHA256 independently matched `e24ec12fd99d5b5bc4d6dabd826d4cc9b3d30d992b94241d7ab0abf1de5509f6`.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely before substantive review. The initial PowerShell Chinese decoding was garbled; an explicit UTF-8 reread corrected it. Read the governing v2 design, current design review, prior design-review v1 and gap report, implementation packet, actual diff and relevant unchanged call paths. Focused rereads replaced oversized/truncated output. Prior approvals and author test counts were treated as scope/evidence claims, not code proof.

## F1 — P1: successful original Python stream return does not establish the required native cache invariant

Candidate locations: `tests/support/native_runtime_process.py:391-394` (post-return identity checks), `:499` (selecting `wp_cuda_context_get_stream`), `:575` (calling it). Governing design: lines 240-243 explicitly require proving the cache already exists before approving that getter for execution, with a fresh reviewed getter change if it cannot be proved. This is the code-preflight condition left open by design approval.

Pinned installed source establishes the following control flow:

1. `warp/_src/context.py:6245-6257` returns `get_device(device).stream`. `Device.stream` at 3661-3668 accesses `self.context` and returns `_stream`.
2. First `Device.context` at 3638-3653 retains the primary context, calls `_init_streams`, and restores the saved context. `_init_streams` at 3607-3613 invokes `self.set_stream(Stream(self))`.
3. `Device.set_stream` at 3696-3697 calls `wp_cuda_context_set_stream(...)` and **unconditionally assigns `_stream`** after that void call. Its ctypes restype is `None` at 4844-4845.
4. `warp/native/warp.cu:2285-2303` obtains `get_context_info`; when that returns NULL the setter simply returns, without reporting failure to Python or writing a cache entry. The helper at 366-399 returns NULL if its `cuCtxGetDevice_f` call fails. There is therefore an explicit native failure branch on which `_context` and `_stream` exist and the original Python `get_stream` returns, but `g_contexts` has no entry.
5. Other original-path operations do not close this gap: primary-context acquisition at 1925-1937 returns the retained context, while stream creation/registration at 2549-2558 and 2586-2596 creates/registers a stream/event in `g_streams`; it does not populate `g_contexts`. The sole `g_contexts.insert` in the inspected source is inside the successful branch of `get_context_info`.
6. The observer's subsequent getter at 2276-2282 again calls `get_context_info`. On that absent-cache branch, lines 379 and 385-389 enter a native context guard and, on a mempool-supported device, issue the one-byte `cudaMallocAsync`/`cudaFreeAsync` workaround. It is not a passive cached observation.

Trigger: the original native setter's cache initialization reports a driver error through `check_cu` and returns void, while Python retains its target/stream metadata and returns normally. This does not require substituting the positive uploader or misusing either explicit negative selector. A source-level failure path is sufficient to invalidate the promised unconditional cache prerequisite; this review does not claim to have produced the driver error on real hardware.

Consequence: the observer can perform new context-selection/allocation work in its alleged getter, after its current-context/primary-state samples were already taken. The exact `_stream`/`_context` identity checks do not prevent it. A failed cache insertion followed by a successful getter-side insertion can also yield the default cached stream value 0, which the current baseline comparison does not inherently reject merely because the held stream is nonzero. Later stderr/test failure cannot retroactively make the already-issued getter passive. The concern is the forbidden call path and missing proof, not a claim that every successful normal run would take that path.

Independent reproduction: `c4_assets/native_scope_cache_preflight_repro_v1.py` extracts the **exact pinned Python AST methods** `context`, `stream`, `_init_streams` and `set_stream`, executes them with CPU objects, models the native setter's documented NULL-return branch, and invokes the candidate's real `_NativeScopeObserver._state`. The original return passes the candidate's `_stream`/`_context` conditions with an empty modeled cache. The observer then reaches the modeled cache-miss context/allocation branch and records `target_current_stream=0`, while `held_stream.handle=123`. The script and log label this **model_only**, and no Warp/Torch/NumPy import, native library or GPU call occurred. The C++ branch is established by the independently hash-checked pinned source, not by pretending that the model executed C++.

Why current tests miss it: readback tests model scope entry/exit, and observer CPU tests model direct driver query errors and wrong post-exit context. None models an original Warp void setter that returns to Python without initializing `g_contexts`. The current same-object post-return assertions test Python object identity, not native cache insertion. Prior design-review success-path source analysis established feasibility, not exception safety on this explicit branch.

Required direction: do not use the possibly initializing Warp getter merely to test whether its cache exists, and do not warm/fix the cache from the observer. Establish a source-supported, failure-sensitive prerequisite that genuinely excludes the miss before calling the getter; if the pinned interface cannot supply it, follow design v2's stated route of proposing a changed observation/getter contract for fresh review. This report does not prescribe or authorize a new getter, native shim, setter, warmup, or ABI.

Proof needed after repair: retain this exact void-setter/absent-cache counterexample as a CPU negative. Demonstrate that it causes sticky failure before any observer operation capable of initializing cache/selecting context/allocating; demonstrate the normal original-return case still delegates once and retains the returned owner before fallible work. Reinspect the exact native ABI/source path and revise any affected design/pins through the required independent checkpoint before real observation runs.

## F2 — P2: CUDA C10 getter failure loses the ordered partial native diagnostic

Locations: `tests/support/native_runtime_process.py:472` (`_cuda_getDeviceCount`), `:545` (primary-context comprehension), `:553` (`self.logical()`), `:556` (`self.stream_query(...)`); compare the explicit `_result` recording at 506-510. Governing design lines 304-306 require an ordered raw diagnostic of successful getter returns and the exact error, held by the session, without a fabricated complete row.

Trigger: a CUDA C10 primary-context query throws after earlier ordinal queries succeeded, or a later logical/current-stream query throws after those queries. Those C10 calls are invoked directly; only the raw cudart call uses `_result`. `_state` never reaches its completed state dictionary, so the successful C10 values are not appended to `self.raw`. The top-level failure diagnostic has the final exception type/message but not the complete ordered successful calls or necessarily the exact getter/ordinal that failed.

Independent CPU reproduction: `c4_assets/native_scope_cuda_raw_repro_v1.py` invokes the candidate's actual `_state(False)` with CPU callables. The modeled primary getter succeeds for ordinals 0 and 1 and raises for ordinal 2. Observed calls are `[0,1,2]`; the retained raw list contains only `{"getter":"cudaGetDevice","result":0,"value":1}`. Neither successful primary result nor a named failed primary call is present. No real backend was imported; a temporary `torch.cuda` namespace stub supplies only the modeled initialized flag and is removed afterward.

Consequence: the failed execution is still conservative (the existing snapshot exception path poisons and does not emit a complete native_scope row), but the required partial diagnostic cannot reconstruct which native observations actually succeeded before failure. This is a diagnostic-contract gap, not evidence of an authorized release after failure.

Why current checks miss it: the CPU error/null test exercises `_result` directly; the fixed real-test getter-error selector raises before the allocation snapshot getter. Neither exercises an actual C10-call exception after earlier successful getters. Historical ambient success traces do not prove this new failure behavior.

Required direction: record each fixed C10 getter's name, arguments/ordinal and returned primitive immediately, and record the failing getter and exception before propagating. Preserve actual invocation order, original ticket, no retry, no additional getter, and no complete row after partial failure. Keep this confined to the fixed test observer; do not add a production callback or change the CPU parser's authority.

Proof needed: CPU tests using the real observer with a failure at a later primary ordinal and at the logical/current-stream stages must retain all earlier results in order and identify the failed call. Assert no later getter, no fabricated row, no second synchronization or ordinary release. Preserve the existing successful-null versus nonzero-CUresult distinction.

## Inspected behavior and properties that were not found defective

The codec explicitly accepts v1 and v2, rejects native_scope under v1, and leaves top-level/component/selection/input structures unchanged. V2 is not treated as coverage proof. `_RuntimeReplay` keeps lifecycle references separate from journal pairing indices; bounds operations against the lifecycle cut; ties them to original guards, descriptor kind/runtime identity and phase; rejects stale leaves, crossing/reused pairs, prior-return pristine observations and stream-after rows without their immediate real returned_owner predecessor. Scope-after rows intentionally retain the opening scope occurrence while admitting nested helper leaves. Runtime binding cross-checks earlier module/runtime/target/stream tokens and stream identity; binding point uses the actual adoption and immediately preceding journal entry. Normal completion cannot leave an unfinished pair. The terminal exception is specific to the original live emergency sync interval with completed validation and a binding; native observations remain descriptive.

The added tests explicitly insert **synthetic** native rows into actual old provenance. They include the real journal1343/lifecycle438/future-leaf447 counterexample, stale and same-kind occurrences, identity substitutions, crossed/duplicate/missing pairs, lawful adoption/nesting, pristine return order and emergency-domain attacks. Reading these and passing local generic parser tests does not claim that the new 32-case POSIX archive suite was independently rerun here or that any inserted row was a native observation.

Warp readback factors the existing single body under one target ScopedDevice after identity/storage checks and adoption of the retained list. CUDA uses the same body without that outer scope. The report is adopted/held before scope exit and returned only afterward; the new leaf accounts for one runtime_import_or_init operation without a new ARM or deadline. The pinned ScopedDevice/ContextGuard source saves/restores metadata and delegates CUDA push/pop, and the observer checks actual post-exit context rather than treating Python return as restoration proof. The dedicated CPU wrong-context test preserves the contradictory row before sticky UNKNOWN and prevents normal continuation/release.

The fixed observer is installed only for the new explicit selectors. It holds itself in session.runtime_owners, retains original functions and exact returned owners, delegates the original runtime leaf/retention/return/binding hooks, and observes inside the existing ticket. Helper returns are retained before their after snapshot. Scope wrappers call original entry/exit once on normal paths, unwind on body exceptions without later observation, and leave emergency synchronization to its original scope. The terminal wrapper observes around the one original callback before release; the existing controller catches a callback/after-observer exception as failed sync and quarantines without retry.

The library path/hash/maps/RTLD_NOLOAD|RTLD_LOCAL checks have no fallback load. CDLL/function objects are held before subsequent fallible setup. The direct Warp driver ABI uses c_int CUresult/CUdevice, c_void_p context and c_uint flags; device handles come from cuDeviceGet and are mapped back to ordinals. A nonzero return is rejected before consuming output; current-device query is conditional on a successful nonnull current-context result. The F1 concern is the later Warp-managed stream getter, not these typed direct-driver getters.

The acceptance helper derives expected before/after occurrences from actual lifecycle leaves and descriptor identities, demands exact set equality and uniqueness, matches the full terminal worker trace, and checks same-run external PID/ancestry/start ticks plus lane-specific ambient/context/primary/current-stream relationships. The opt-in test file has eight normal cases (two lanes, four workloads) and eight observer/counterexample cases. CUDA stream-only copy and Warp omitted-readback-scope are intentional negative controls; they were not misclassified here as failed positive delegation. The external live identity recorder and OS/GPU residual verification remain required when running the eventual matrix. No CPU check substitutes for them.

The CPU production observation collector still only captures bounded facts, serializes immutable issued rows, freezes adoption mappings, and selects current/prior_validated_cut/unavailable without new native queries. The existing ordered record-before-sidecar persistence and full worker-prefix attachment are retained. A partial/missing observation remains incomplete evidence; a current version or matching count alone does not establish native success.

## Independent commands and results

Environment: local PowerShell, `C:/Python314/python.exe`; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONDONTWRITEBYTECODE=1`, `python -B -m pytest -q -p no:cacheprovider`. Tests used their existing CPU models/fixtures, not a new backend/native worker.

1. `tests/unit/optics/test_native_scope_observer.py tests/unit/optics/test_guarded_native_readback.py tests/unit/optics/test_runtime_custody.py`: **56 passed in 17.02s**.
2. `tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_manifest.py`: **304 passed, 13 skipped in 152.32s**. POSIX/release/symlink/asset-dependent skips are not passes or native proof.
3. Both scratch reproduction scripts above ran successfully and wrote their explicit CPU-model outputs to adjacent `.log` files. Their findings are described in F1/F2. No candidate file was modified.
4. Git status/branch/HEAD/base diff and diff-check; document/catalog SHA256 checks; UTF-8 source reads and rg searches. Production source-policy/catalog tests passed their newly pinned closure.
5. Fresh primary-host read-only command: a stdlib script piped into `ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -` hashed installed Warp source files. Python reported 3.12.13 and `backends_loaded=[]`. No CUDA/Warp native-library load, backend import, GPU query, native worker, remote process signal or remote file mutation was performed by the reviewer.

Fresh installed hashes matched the retained source strings used for the control-flow review:

| Source | SHA256 |
| --- | --- |
| warp/_src/context.py | eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee |
| warp/_src/utils.py | 01bd20f279f5762881c50cf8b6d766734fe21bd5f9b1ec286b34b1f3fcb27f24 |
| warp/native/warp.cu | 7c7e6e7a02db7b285ca1800d0ec52e28d2cf8a704435d925c70a8c5ccd8f176b |
| warp/native/warp.cpp | b9feea467eb11e0d9016ca33fceb7b1a4358fc21c65513a692378155f1d05fc7 |
| warp/native/cuda_util.h | 59207856d17907bca935612ccb3ce7cde0952d8491011e1d45e8bb2d5b9ed1b8 |
| warp/native/cuda_util.cpp | 21d9dfe57b861ada6b98d0fa2ccfb3b8ed0b14972ed840eb04d0ad7e5f70a89f |
| warp/native/warp.h | 0bb164157668ad83ce5fe78d3a583c68258ee333ce37d97e746b5d85a2e94511 |

The two headers are inspected installed source, not an assertion that the existing runtime inventory pins them. No source/binary rebuild equivalence or driver-internal correctness is claimed. The pre-existing pinned-dependency trust boundary remains explicit.

The coordinator reported advancing the separate runtime worktree to the frozen implementation and running CPU regressions. A reviewer read-only `git rev-parse HEAD` independently confirmed `/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip` at b0f22fb; the coordinator reported that tree clean and the original binding worktree unchanged at a3d750b. This review did not change either tree. Coordinator CPU results, old 389/13 and codec 32-pass counts, and historical 41-case/85-fault successes are not represented as this reviewer's executions or new native-scope acceptance.

Inspection command limitations: one initial combined source dump exceeded output limits, then focused JSON/AST readers were used. One scratch reader initially failed on empty/BOM JSON lines; another hit a trailing metadata row without `source`/`path`; corrected readers explicitly filtered those records. One initial document command used `__review__` instead of `__design-review__`; the actual design-review was subsequently read. These are reviewer command errors, not candidate failures. All Python/backend source was read as text; none was imported for source inspection.

Scratch artifacts outside the candidate:

| Artifact | SHA256 |
| --- | --- |
| native_scope_cache_preflight_repro_v1.py | c39a41003cc5be93da391d0c8ae326d92eae96af16e1dd2bfa584b9ff7584f34 |
| native_scope_cache_preflight_repro_v1.log | 135324b7f47b1029d917c47f120e46fac355f66a0eb63a2dc957afe5b338228d |
| native_scope_cuda_raw_repro_v1.py | c7c03ba9c55006a430461b3ddec4877e6562dd363266c704b12afdb1542d3d24 |
| native_scope_cuda_raw_repro_v1.log | 3c8461365fb9274b4f86def8cfc0cfb20b22fbf8269ecf7746efae7001de5983 |

The local tests may create ordinary pytest temporary directories under the user temp area; cacheprovider and Python bytecode writing were disabled for reviewer test commands. Scratch scripts/logs and this report are the review's durable filesystem additions, all in `c4_assets`. No tracked candidate edits, fixes, backend/native worker launches or remote mutations occurred.

## Remaining evidence and gates

Fix the substantiated findings, freeze the revised candidate, and request a fresh independent review. If F1 requires changing the frozen getter/observation contract, obtain the expressly required design review first. Do not normalize the failure or silently broaden the observer to make the current getter usable.

The 16 actual new native cases, actual nonempty Warp D2H restoration, new observer failure/persistence behavior, full required affected fault/ACK/EOF/hang/cleanup/exit and C0/source regressions, external process/GPU residual checks, and fresh whole-C2d3 code review remain outstanding. The 32-case POSIX codec suite was inspected but not independently executed in this session; its historical-provenance rows are synthetic by construction. Real foreign-stream/UNKNOWN cases remain part of the larger required matrix, not evidence supplied by these 16 cases alone.

No full C2d3/C2d/C4 approval, preparation/execution readiness, final graph seal, warmup, measured optical work, result, performance or backend decision is issued. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host, R2 and shipping remain CLOSED. C4 remains unaccepted. The complete unchanged report may be archived through the collaboration protocol's evidence-only exception; doing so does not open a gate or approve a repaired revision.

Final integrity check after report creation: local HEAD remained b0f22fb4f6eda395f72ae34f37044ed918254184, `git status --porcelain` was empty, and `git diff --check c243875 HEAD` passed. Both reviewer pytest processes completed. The coordinator acknowledged F1 and described possible error-buffer research for a separate future design supplement; that proposal is outside this inspected frozen candidate, was not reviewed here, and does not close F1.
