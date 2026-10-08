# Independent Warp stream-cache precondition design review

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: bounded independent design-review supplement
Author: Codex, fresh independent reviewer /root/native_scope_cache_design_v1
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: accepted for the exact design scope below
Related Files: native-scope-cache design-request v1; native-scope-aggregate design-request/design-review v2; native-scope-aggregate code-preflight-review v1; tests/support/native_runtime_process.py
Owner Summary: Approve the frozen supplement for bounded implementation. The two exact-empty error-buffer checks supply a source-supported way to reject the prior native cache-miss failure before the observer getter. The ordered C10 diagnostic repair is sufficiently specified. This is not approval of the unchanged implementation, native preflight, or any real native case.

## Verdict and inspected candidate

**Approve**, design only. No substantive blocking or non-blocking design finding was established. Implementation must still demonstrate the specified ordering, failure behavior and actual getter precondition in fresh independent code/native preflight before the 16 actual native-scope cases may run.

- Independent reviewer: `/root/native_scope_cache_design_v1`, started with `fork_turns="none"` and no implementation conversation. No candidate authorship, subdelegation or implementation edits.
- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Actual frozen HEAD: `083688ee95bdef9c7386a41f9828b0bff7cb5c16`.
- Review base: `b0f22fb4f6eda395f72ae34f37044ed918254184`.
- Candidate: `collab/q54-optical-lab-p12-4c4c2d-native-scope-cache__design-request__codex__v1.md`.
- Independently verified candidate SHA256: `393a32720ce10ddc525f132c38e5109a45b537a8a5df4ffa08524013e078a68c`.
- Governing aggregate design v2 SHA256: `9e678c7aaffa1147b49a6f1f3b946619f0c316e3bc68883ff0fa9a4d1ed9a0ad`.
- Prior complete code/preflight report SHA256: `b18527c80bc1daa951879de99ed39aeed4c8942685635e5e39e48a7daf3adeca`.
- Initial and final inspected tree clean. Base-to-HEAD changes only the archived 131-line prior report and this 134-line design supplement. `git diff --check b0f22fb4f6eda395f72ae34f37044ed918254184 HEAD` passed.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely. The initial default-encoding principles output was garbled, so I reread it explicitly as UTF-8. Read the complete supplement, governing design and design review, prior full code/preflight findings, original repro scripts, relevant observer/production call paths, and pinned native source. Author descriptions and prior verdicts were treated as claims to verify.

## Source feasibility and the prior cache finding

The existing implementation still has F1. I reran the original CPU-only `native_scope_cache_preflight_repro_v1.py` against this frozen tree. Exact pinned Python Device methods returned the same stream and a positive context while the modeled native cache remained absent. The actual current observer then reached the modeled cache-miss branch and reported native stream 0 against held stream 123. This confirms the old counterexample remains live in the unchanged implementation; design approval does not label it repaired.

The supplement adds evidence that was absent from that implementation. Its proof is a conjunction of the fixed original call chain, initially absent raw Python context/stream, two actual native error-buffer results, exact retained returned owner and positive handles. It does not infer cache existence from a Python return or a caller-supplied flag alone.

I independently verified the relevant chain:

1. Pinned `context.py:6245-6257` returns `get_device(device).stream`. `Device.stream:3661-3668` accesses `self.context`. Initially absent `_context` on the selected original primary Device takes `Device.context:3638-3653` through primary-context acquisition, `_init_streams`, and original restoration. `_init_streams:3607-3613` calls `set_stream(Stream(self))` before constructing the null-stream wrapper. `set_stream:3696-3697` calls the native setter and unconditionally publishes `_stream`. The supplement correctly preserves the prior finding that this publication is insufficient by itself.
2. In `warp.cu:2285-2304`, the native setter calls `get_context_info` with the positive original target context. `get_context_info:366-399` has a cached return branch. Its nonnull-context cache-miss branch enters ContextGuard, calls `check_cu(cuCtxGetDevice_f(...))`, and either inserts ContextInfo or returns NULL. The NULL branch under review necessarily goes through the failed `check_cu` call. The null-input/current-context fallback is excluded by the required positive original context.
3. `cuda_util.cpp:326-343` returns directly on CUDA_SUCCESS; on error it calls `wp::set_error_string` with a format beginning `Warp CUDA error`, with or without an error-description string. Both branches produce nonempty output. `check_cuda_result:315-324` similarly writes nonempty output on the malloc/free workaround's error path. `cuda_util.h:253-270` checks its context query, selection and restoration with the same check_cu machinery. An earlier context-guard or later restoration error therefore also prevents the exact-empty postcondition.
4. Freshly read installed `error.cpp:23-61` defines a static 4096-byte buffer initialized to the empty string. `get_error_string` only returns its address. `set_error_string` writes the formatted error; output enablement controls printing, not the buffer write. `append_error_string` preserves the existing prefix. `warp.cpp:152` is a direct forwarding getter. No successful original-path call clears the buffer. The original stream constructor/registration, native setter, null-stream registration and final context restoration contain no empty-string writer. In particular, null-stream registration returns immediately for the null handle. Later nonempty errors may overwrite earlier errors, but cannot turn a detected failure into exact empty output in this fixed chain.
5. Consequently, after an initially empty read, a normal original return followed by exact empty output excludes the documented void-setter/NULL-cache branch before the observer can call the possibly initializing getter. If the original call raises or never returns, the postcondition is never established. C++ allocation failure/process termination is not a successful observer path. This reasoning relies on the existing admitted native dependency, not on a claim to have executed C++ under fault injection.

The new query is feasible without a new loader or native ABI mutation. `context.py:4050-4051` already declares `wp_get_error_string.argtypes=[]` and `restype=ctypes.c_char_p`. The design requires retaining that actual function and checking, rather than changing, the declaration. The source getter performs no CUDA call, context selection, GPU allocation, error clearing or synchronization. ctypes produces a Python bytes copy for this declaration; saving its hex preserves the returned byte content without lossy decoding. NULL and unexpected return values must fail, even though the inspected native implementation normally returns its static buffer.

The two reads are deliberately conservative initialization preconditions. A stale error rejects the run before stream creation rather than being cleared or tolerated. A new error rejects after the real returned owner is retained. They are not a replacement for direct post-scope current-context observations. The design expressly preserves a completed contradictory after row when a void scope exit returns without restoring native state.

## Cache lifetime, ownership and failure ordering

The cache lifetime portion is also source-supported in the bounded worker. All occurrences of `g_contexts` in the pinned complete `warp.cu` are its declaration at 240, lookup at 374-376, insertion at 394, and erase at 2200. The erase is inside `wp_cuda_context_destroy:2182-2205`. This is not part of the selected original initialization, allocation/copy/event helpers, upload/readback ScopedDevice scopes or terminal target synchronization. Searches of the relevant worker sources found no context-destroy, primary-release or shutdown call in that path. The admitted target/runtime/stream and observer remain in the held graph; observations end before authorized terminal release. The native stream destructor destroys/unregisters stream state, not a g_contexts entry, and retained owners also keep it from running in the observation window.

This does not establish safety for arbitrary external callers, concurrent Warp applications or arbitrary native-memory mutation. The design explicitly binds the proof to the fixed sacrificial worker and exact admitted runtime/call chain. Its observer-local flag is justified only by that execution; it is not general cache authority. No independent mechanism may destroy/reset the context or mutate the cache in that window.

The current production hooks provide the necessary insertion points. `native_bootstrap.py:2701-2715` performs quiet original initialization, holds the runtime and target, enters `indexed_runtime_stream`, calls original get_stream once, appends the returned stream to `runtime_owners`, and records returned_owner before the observer's post-return work. The design keeps the existing injected after-return failure before its new checks. The `_runtime_leaf` hook permits the first read and pristine raw checks inside the original ticket without triggering Device properties. All poststream native stream queries are explicitly gated by the locally established prerequisite.

The first positive native current-stream value must equal the actual held original stream handle. A zero/NULL or mismatching first response cannot establish a successful baseline. Later comparisons remain against that first established value. This closes the additional zero-baseline acceptance avenue described in prior F1 while retaining native evidence rather than replacing it with metadata.

On precondition/getter failure, the supplement requires retained raw results/exceptions, sticky UNKNOWN, no later native observations and no normal completion, synchronization or release. Current initialization propagates through `native_bootstrap.py:2719-2724`'s unknown-scope path, and `guarded_lifecycle.py:426-447` retains the original synchronization/quarantine boundary. Implementation must attach every new failure to those existing rules. It may not recover by retrying the native setter/getter, clearing the buffer, creating a stream, changing context or performing observer cache initialization.

## Ordered diagnostics and prior F2

I reran `native_scope_cuda_raw_repro_v1.py` against the unchanged actual observer. The CPU model invoked primary ordinals `[0,1,2]`, threw at ordinal 2, and retained only the raw cudaGetDevice result. Thus prior F2 is also still an implementation defect, not something this design review claims to have repaired.

The supplement specifies an implementable correction: record every fixed Python/C10 getter's exact name, primitive arguments and primitive success or exception in invocation order, including count, each primary ordinal, logical device and explicit-target stream. It includes both Warp getters in that bounded wrapper; retains exceptions before fallible diagnostic processing; forbids retries and fabricated complete snapshots; and retains the existing CUresult success/error/null distinction. This addresses the concrete lost-prefix case, including failures in initialization's count getter rather than only `_state`.

Raw diagnostics remain descriptive and held by the session. They do not add fields or authority to the aggregate schema, extend immutable selected cuts, or replace complete rows. Tests must use the implemented observer to fail each getter position, verify earlier successes and the exact failed call survive, and verify no later call, duplicate invocation or complete snapshot is produced.

## Commands, results and limitations

All local inspection used PowerShell in the named repository. I executed Get-Content with UTF-8, rg searches, Git status/revision/branch/diff checks, Get-FileHash, and stdlib-only Python readers with `C:/Python314/python.exe -I -S -B -`. Focused readers parsed the retained JSONL and printed selected source lines without importing those modules. Recomputed source-string hashes matched the candidate pins and fresh installation hashes.

Fresh remote read-only verification used a pathlib/hashlib/json/sys script piped to:

```text
ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -
```

It read and hashed files below `/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages/warp`, returning full error.cpp and error.h text. It reported Python 3.12.13 and `backends_loaded=[]`. The independently observed hashes were:

| File | SHA256 |
| --- | --- |
| _src/context.py | eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee |
| native/warp.cu | 7c7e6e7a02db7b285ca1800d0ec52e28d2cf8a704435d925c70a8c5ccd8f176b |
| native/warp.cpp | b9feea467eb11e0d9016ca33fceb7b1a4358fc21c65513a692378155f1d05fc7 |
| native/cuda_util.cpp | 21d9dfe57b861ada6b98d0fa2ccfb3b8ed0b14972ed840eb04d0ad7e5f70a89f |
| native/cuda_util.h | 59207856d17907bca935612ccb3ce7cde0952d8491011e1d45e8bb2d5b9ed1b8 |
| native/error.cpp | 877b9f1e889d3e5bf8950c3a3deaf28f44bc4a9a76e780dcc19606d80f6229f2 |
| native/error.h | 4ebf4a1dcd7ed121df72f9390888c5f9eb30547f0e42fde5e04ace50de7face0 |

Original CPU repro commands, each completed with exit 0 and the failure observations described above:

```text
C:/Python314/python.exe -I -S -B ../c4_assets/native_scope_cache_preflight_repro_v1.py
C:/Python314/python.exe -I -S -B ../c4_assets/native_scope_cuda_raw_repro_v1.py
```

These execute CPU models and the current observer only. The first extracts the exact pinned Device AST methods; the second temporarily installs a CPU namespace stub under torch.cuda and removes it. Neither imports a real backend or invokes a native library. The prior report's descriptions of adjacent logs were treated as historical context, not reused as proof of these executions; the fresh outputs are in this session's command record.

Inspection-command limitations: an initial combined document/source dump exceeded output limits; focused reads replaced it. One rg command named nonexistent `optical_worker/guardian.py`; the guardian was subsequently located at `optical_worker/lifecycle.py`. Some absence searches returned expected no-match exit 1. These were review command issues, not candidate failures.

No backend import, native library call/load, CUDA/GPU query, native worker, remote mutation, process signal, package change or implementation edit occurred. I did not operate on either remote worktree or treat the coordinator's concurrent CPU results as independent evidence. No new pytest suite was run for this document-only supplement. The revised implementation does not exist in this frozen candidate, so its positive/negative CPU tests and native tests cannot be claimed here. Source hashes do not prove a source/binary rebuild correspondence or proprietary driver internals; the existing admitted-dependency trust boundary remains necessary. In particular, this review has not observed that the actual quiet initialization's error buffer is empty on hardware.

## Required next proof and unchanged scope

Design approval permits only this bounded implementation and diagnostic repair. Freeze the revised implementation and obtain a fresh independent code/native preflight. It must recheck the actual already-loaded function identity and ABI; exact-empty/NULL handling; raw fields without property effects; single original delegation; owner retention and injected-failure order; local cache flag coverage; first-stream equality; source/lifetime assumptions; diagnostic exceptions; and sticky failure behavior. Replay both original counterexamples against the repaired implementation and require the specified actual-observer CPU positives and negatives. Models are not GPU evidence.

After that preflight approves, all 16 actual native-scope cases, affected full fault/persistence/source regressions, same-run external process/GPU residual checks and the fresh whole-C2d3 independent review required by design v2 remain mandatory. The aggregate schema, native library rows, profiles, operation counts, watchdogs, immutable cuts, acceptance coverage, upload/readback and original lifecycle authority remain unchanged.

No C2d3/C2d/C4 acceptance, final graph seal, product preparation/execution readiness, measured optical work, performance result or backend decision is issued. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host, R2 and shipping remain CLOSED. C4 remains active and unaccepted. The complete unchanged report may be archived through the collaboration protocol's evidence-only exception without changing this verdict, approving code or opening those gates.
