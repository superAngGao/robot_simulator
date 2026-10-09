# Independent guarded output allocation code review v1

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: independent code review, bounded slot allocation prerequisite
Author: codex, independent reviewer /root/slot_allocation_code_v1
Version: v1
Date: 2026-10-10 (Asia/Shanghai)
Status: in_review
Related Files: runtime-integration design-request codex v3 sections 4, 5, 8; remaining-setup gap-scan codex v1 G2; warp-slot-construction implementation-note codex v1; candidate paths listed below
Owner Summary: Request changes. Ordinary exception propagation retains partial arrays correctly, but a caught native allocation failure can be followed by additional allocations and a successful normal COMPLETE. The new guarded allocation boundary must make its own failed native leaves sticky before returning control to a caller. Existing 316 regressions pass; independent CPU counterexamples fail. No native or later-stage acceptance is granted.

## Identity and verdict

Fresh independent reviewer session `/root/slot_allocation_code_v1`, given the requirements packet and repository without implementation conversation. I authored no candidate code and delegated no work. I read root AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md in full, the relevant accepted design sections and gap G2, and the implementation note as claims to verify.

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
Base: `fa17f34d3302ed203a32d2fc71dec4ecf96cef66`.
Reviewed HEAD: `1b45c342676aeea199fc4dcb9811649706e2060c`.
HEAD tree: `aceb61a9235cd5e26df5a8d6cf1fbd827e855064`.
Branch: `implementation/c4-supervised-waits`.
Initial and final `git status --porcelain=v1` were empty; HEAD remained frozen.

**Verdict: Request changes**, limited to this private output-allocation prerequisite. F1 below is blocking. C1 is explicitly non-blocking for the present fixed allocator callers, but the broader original-guard binding claim is not proved by this implementation.

Actual delta: ten files, 383 insertions and 62 deletions: native_setup.py, native_source_policy.py, data/native_sources/v15.json, cuda_direct_light.py, cuda_lbvh.py, cuda_scene.py, device_scene.py, warp_native_render.py, test_native_output_allocation.py, and the implementation note. No dispatcher, session entrypoint, renderer execution, graph or readiness implementation is included.

## F1 — P1: caught native allocation failure can be turned into successful setup

Location: `optical_worker/native_setup.py:349–364`, especially the direct `runtime.empty` call and metadata inspection at lines 357–360; scope unwinding at 374–377. This affects the new guarded paths of the existing world/render/LBVH allocators.

Trigger: inside a real `NativeGuardedSetupLifecycle.segment()`, call the existing world allocator with a valid issued context; let its second native empty raise; catch that exception inside the segment; create another valid single-use context and invoke the same existing allocator again. This needs no context forgery, mutation of lifecycle state, native bridge or arbitrary callback.

Observed independently in BOTH lanes: the first builder made two empty calls, including the failed call; the second builder made another eight calls. The segment then completed successfully. Actual replay summary before fixture teardown was `guard_state=IDLE`, `controller_state=RUNNING`, `guarded_segment_failed=False`, `normal_complete_count=1`, `custody=none`, with ten allocation calls and no emergency sync. A simpler repro catches the exception within the private allocation context itself and admits a third empty call.

Cause: lifecycle-owned API failures use `_observe_public_operation` and become sticky. Exceptions from `runtime.empty` or direct returned-device metadata inspection do not. The helper assumes every exception will escape the outermost segment, so catching it before that point loses the failed-leaf fact. Retaining returned owners does not repair this lifecycle error.

Consequence: a failed guarded native allocation can be followed by further work and COMPLETE/DISARMED. That contradicts parent sections 4 and 8.1/8.2: any failing leaf within the armed segment is terminal, cannot manufacture IDLE, and must retain/quarantine under its existing deadline. If allocation submitted work before raising, successful normal completion falsely erases that failure history.

Why existing tests miss it: the failure tests let the exception escape the entire `controller.segment()`; the segment's outer exception handler then correctly quarantines. They never catch an allocator failure inside that segment or attempt a subsequent supported builder call.

Direction: observe and irreversibly fail the existing lifecycle around the actual fallible native/metadata boundary, preserving the original exception and held owners. Also account for allocation scope entry/exit failures. Use the existing guard/failure machinery, without adding an allocator state machine or a new emergency guard. A context-manager-only catch around `yield` is insufficient for a failure caught inside the yielded scope; the native `empty` boundary itself must participate.

Required repair proof: both lanes must reject the subsequent call before another native allocation, produce no successful normal completion and no replacement ARM/sync, and retain all actual returned arrays. Cover caught allocation failure, caught metadata failure, and scope enter/exit failure, as well as the existing escaping-exception cases. Re-run the actual world-builder repro below unchanged in its trigger; adapt its assertions if repair correctly raises earlier.

## C1 — non-blocking: lexical allocation scope is not exact guard identity

Location: `native_setup.py:347–354`. `active` binds the returned allocator to its context-manager lifetime; `_require_setup()` only checks whichever setup ticket is currently live. Holding the allocation context with `ExitStack`, completing segment 1, then invoking its allocator inside segment 2 succeeds in both lanes. Two allocations are observed under two successful guards. The implementation does not capture or compare the original guard ticket.

This disproves a literal assertion that the helper is bound to the original guard. Current production allocator functions keep their helper context lexically nested within one synchronous call, so I found no existing fixed caller that performs this rollover. I therefore do not make it a second blocking product-path defect. Before a later graph builder retains or shares the helper, either bind the exact existing guard identity and reject rollover or constrain/document the claim to the present lexical caller discipline. The scratch repro records this independently; it is not evidence of product activation or a hostile-Python security boundary.

## Inspected behavior and verification

I traced `_NativeUploadContext` creation/registration/consumption, new helper scope/closure, guard segment exception and completion handling, returned-owner adoption, guardian retention and termination, and each changed allocator. The custody list is adopted before entering native scopes; each successful `runtime.empty` return is appended before device inspection or adoption. Escaping allocation/metadata/adoption failures preserve partial resources in the existing quarantine path. Single-use context copying/reuse and runtime/target/explicit-stream substitutions reject in the supplied tests.

Static comparison confirms the existing CUDA render allocation order and conditional diagnostics, the eight world output fields/shapes, and the LBVH buffer sizes/order are preserved. Workspace alignment still uses one owner allocation and its aligned view; retaining the allocation owner preserves the backing storage even if later view/validation construction fails. The Warp render allocator consumes the existing exact layout and diagnostic-name tuple instead of duplicating its ABI. No new kernel, cross-runtime array conversion, slot state machine or public arbitrary callback was introduced. Exact snapshot/function pin remains unchanged; the whole renderer source pin and catalog change are the bounded admission updates exercised by source-policy regression tests.

Executed locally in PowerShell with `C:/Python314/python.exe`, CPU/runtime doubles:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_output_allocation.py tests/unit/optics/test_warp_render_buffer_contract.py tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_cuda_scene_native.py tests/unit/optics/test_cuda_lbvh_native.py tests/unit/optics/test_cuda_artifact_runtime.py tests/unit/optics/test_native_source_policy.py
316 passed in 8.34s
```

`git diff --check fa17f34d3302ed203a32d2fc71dec4ecf96cef66 HEAD` completed with no output.

Scratch repro: `C:/Users/AG/新建文件夹/c4_assets/slot_allocation_repro_v1.py`.
SHA-256: `313b694576a1f2906a1829825a574b03a123e2b12f88f4f1004d80b329f40fb9`.
It imports the candidate's CPU fixture and calls actual lifecycle/helper/world allocator implementations; it does not modify the candidate. Set `PYTHONPATH` to the repository, then:

```text
C:/Python314/python.exe -B -m pytest -q -s -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/slot_allocation_repro_v1.py
```

Execution history: the initial four cases (caught native failure and guard rollover, each lane) failed their safety assertions in 1.41s. After appending the actual-world-builder cases, a second run with `-k existing_world` reported 2 failed, 4 deselected in 1.33s. These are observed candidate behavior failures, not environment/toolchain failures. The final combined six-case command above is the reproducible entrypoint; I did not claim a separate combined execution.

Read-only command corrections: the initially guessed root REVIEW_PRINCIPLES.md path did not exist; the required collab file was read fully. Large combined output was truncated, followed by targeted reads of requirements and code. An unquoted PowerShell `HEAD^{tree}` argument was misparsed; quoted re-execution produced the tree identity above. No result depends on these failed exploratory commands.

## Limits, remaining gates and owned process hygiene

No real Warp/CUDA runtime allocation, GPU probe, compiler, renderer launch, native source-session acceptance or remote command was executed. CPU tests do not establish target1/ambient0 restoration, native exception safety, device-memory lifetime, native module admission or numerical correctness. Existing CUDA allocator preambles still perform availability/device-count checks before the shared helper; native integration must account for their actual execution scope as part of the fixed guarded builder. This review does not certify those real-device properties from doubles.

No candidate implementation or tracked evidence file was edited; only the scratch repro and this report were written outside the repository. Pytest used `-B` and disabled its cache provider. All reviewer commands completed synchronously. Fixture Peer threads joined without teardown errors. A final scoped Win32_Process check found no remaining Python process whose command line matched this review's scratch repro or regression entrypoint. No background helper, supervisor, compiler or worker was launched by this review, and no process was signalled. This is a bounded owned-process/hang check, not exhaustive OS attestation. The separately live remote admission batch and its parent/tool processes were neither inspected nor touched.

The requested code-only prerequisite is not approved until F1 is repaired and independently re-reviewed at a new frozen revision. Even a future approval of that repair will cover only these allocation mechanics and CPU evidence. Separate compilation/native acceptance, full one-slot graph/transactions, private warmup execution, final readiness/origin, recording/seal/barrier, C2e/f integration/activation, complete C4/refactor, C4c3/c4, C4d/B3, product PREPARE/native samples, R2/S/host payload, a3b, performance/results/baselines, shipping and backend selection all remain closed. No earlier approval or this report opens those gates.
