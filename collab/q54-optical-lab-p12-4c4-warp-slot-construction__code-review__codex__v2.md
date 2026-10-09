# Independent guarded output allocation code review v2

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: independent code re-review, bounded guarded output-allocation prerequisite
Author: codex, independent reviewer /root/slot_allocation_code_v2
Version: v2
Date: 2026-10-10 (Asia/Shanghai)
Status: accepted
Related Files: collab/q54-optical-lab-p12-4c4c2-runtime-integration__design-request__codex__v3.md sections 4, 5, 8; collab/q54-optical-lab-p12-4c4c2d-remaining-setup__gap-scan__codex__v1.md G2; warp-slot-construction code-review v1 and implementation-note v1/v2; candidate paths below
Owner Summary: Approve the private allocation prerequisite at fdb86a03bc771e9422d7b5b04e0179b2a792684f. F1 and C1 are closed by inspected code and independently replayed original triggers. Fresh regressions: 338 passed; unchanged original trigger bodies with repaired rejection/terminal assertions: 6 passed. This is CPU evidence only; native acceptance, graph integration and whole C4 remain unapproved.

## Identity and scope

Fresh independent reviewer session `/root/slot_allocation_code_v2`, provided the review packet and repository without the implementation conversation. I authored no implementation, delegated no work, and treated author notes and previous verdicts as claims to verify. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md, the required design sections and G2, both implementation notes and the complete previous report.

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
Branch: `implementation/c4-supervised-waits`.
Whole allocation base: `fa17f34d3302ed203a32d2fc71dec4ecf96cef66`.
Repair base / prior rejected candidate: `1b45c342676aeea199fc4dcb9811649706e2060c`.
Reviewed HEAD: `fdb86a03bc771e9422d7b5b04e0179b2a792684f`.
HEAD tree: `c7aafd1acbec80b96d640a9983bbe5091954ada6`.
Initial and final `git status --porcelain=v1` were empty. HEAD remained frozen at the revision above.

**Verdict: Approve**, exclusively for the private guarded output-allocation mechanics and the bounded repair, with the evidence limits below. No substantive unresolved finding was identified in that scope. This does not approve G2 as a whole.

The repair changes native_setup.py, its focused tests and the source catalog, plus the archived v1 report and v2 note. The complete allocation delta additionally routes existing CUDA world/render/LBVH and Warp world allocators through the helper, adds the Warp render allocator using the established layout, and updates the renderer source pin. No lifecycle state machine, supervisor protocol, worker dispatcher, rendering graph or product entrypoint is changed.

## Prior finding dispositions

**F1 (P1, blocking in v1): closed.** At `optical_worker/native_setup.py:356–369`, the existing lifecycle `_observe_public_operation()` now surrounds each actual native empty call, owner retention, metadata inspection and adoption. A thrown native or metadata exception records the guard failure before a caller can catch it. The observer at lines 372–383 also surrounds native scope construction, entry, body and exit. The helper reuses the existing lifecycle failure/retention mechanism; it introduces no parallel failure state.

I traced that observer through `_fail_guard`, `_check_running`, segment completion and termination. The original error is retained, subsequent guarded work is rejected, the segment checks sticky failure before normal completion, and the existing termination path transfers custody without an emergency synchronization. Nested observation of the same failure does not clear it or create a replacement guard.

I executed the original v1 scratch file unchanged. All six cases stopped earlier than the old assertions: same-scope retries raised `WorkerSetupProtocolError` before native empty; actual Warp/CUDA world-builder retries raised that error while creating the second context; rollover raised `ValueError` before its second allocation. The old script therefore reported six test failures due to its pre-repair expectations, not six persisting counterexamples.

An independent outside-repository wrapper then invoked those exact original test bodies unchanged, adapting only the expected rejection and post-failure assertions. Both actual world-builder triggers made exactly two native-empty attempts including the failed attempt, normal COMPLETE count stayed zero, only the original ARM occurred, emergency-sync count stayed zero, and custody was `TRANSFERRED_TO_QUARANTINE`. Every returned array remained in the pre-adopted list and the original exception remained retained. The same checks passed for the same-scope trigger. All six adapted cases passed.

The candidate's newly added tests also passed for caught allocation, metadata and adoption errors; actual world-builder device/stream entry and exit errors; caught retries followed by attempted normal completion; original exception identity; partial owner custody; and scope restoration in the CPU double. These tests assert consequences and terminal replay, not just exception types.

**C1 (non-blocking in v1): closed.** `native_setup.py:338` captures the exact original ticket, and lines 358–360 compare it before admitting each native allocation. The unchanged ExitStack rollover trigger now fails in guard 2 before its empty call in both lanes. Observed totals are one allocation, one earlier successful normal completion, two ARM attempts and zero emergency syncs, followed by quarantine. Thus the second guard cannot reuse the first guard's allocator. This retains the original lexical lifetime check too.

## Inspected call, data and custody flow

I inspected both the repair diff and the entire allocation diff, followed context creation/identity registration/consumption and guarded ownership checks, and traced the common helper into the actual changed allocators. The custody list is adopted before native device/stream scope entry. Each returned array is appended before fallible metadata inspection or lifecycle adoption, so failed adoption still leaves a retained owner. Runtime, target and stream identities remain tied to the consumed context and previously adopted roots.

CUDA world shapes/field order, render allocation order and optional diagnostics, and LBVH output sizes and workspace alignment remain in their established builders. The workspace allocation owner continues to hold storage backing its aligned view. Warp render allocation consumes the existing layout and diagnostic-name tuple. There is one runtime allocation path per builder, no array conversion, alternative renderer, arbitrary public callback or new allocator authority registry. Ordinary callers use the optional context's absent path without guarded lifecycle changes.

The complete source-policy change updates only the renderer whole-source pin; the snapshot-function pin and AST rules are unchanged. Parsed catalog comparison showed unchanged lane lists and schema, with digest changes confined to the seven changed production source files. The policy regressions passed against frozen HEAD. Catalog coverage is not execution permission, and no new native source-session graph is claimed.

## Fresh verification

Environment: local Windows PowerShell, `C:/Python314/python.exe`, CPU/runtime doubles. All tests were newly executed with `-B` and pytest cache provider disabled.

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_output_allocation.py tests/unit/optics/test_warp_render_buffer_contract.py tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_cuda_scene_native.py tests/unit/optics/test_cuda_lbvh_native.py tests/unit/optics/test_cuda_artifact_runtime.py tests/unit/optics/test_native_source_policy.py
338 passed in 9.15s
```

`git diff --check fa17f34d3302ed203a32d2fc71dec4ecf96cef66 HEAD` passed with no output.

For the scratch commands, `PYTHONPATH` was set to the repository only in the command shell:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/slot_allocation_repro_v1.py
6 failed in 1.55s (expected earlier safety rejections; detailed disposition above)

C:/Python314/python.exe -B -m pytest -q -s -p no:cacheprovider C:/Users/AG/新建文件夹/c4_assets/slot_allocation_repro_v2_review.py
6 passed in 1.36s
```

Original v1 repro preserved unchanged, verified SHA-256 `313b694576a1f2906a1829825a574b03a123e2b12f88f4f1004d80b329f40fb9`.
Independent v2 wrapper SHA-256: `2ba53f445a23b5de9fabc2da2c152aa2c5e527d60098e7778cd6a0d63b79152f`.

No environment/toolchain failure affected these results. Some combined exploratory output was truncated; targeted reads recovered the relevant code/requirements, and parsed comparison avoided relying on the single-line catalog diff. The initial Windows text read rendered Chinese incorrectly; REVIEW_PRINCIPLES.md was reread fully as UTF-8.

## Limits, follow-up boundaries and process hygiene

No real Warp/CUDA allocation, GPU probe, compilation, renderer launch, native acceptance or remote command was executed. CPU doubles prove the exercised Python control flow and retention references; they do not establish actual target1/ambient0 restoration, native exception safety, device-memory lifetime, numerical output, module admission or complete production resource coverage. Those remain mandatory later evidence, not waived checks.

Existing CUDA allocator preambles still perform actual availability/device-count calls before the shared helper. They were not altered by this repair. The future fixed native graph must cover those operations and their real scope, as well as aggregate/factory/event/command ownership, before claiming native integration. This is a bounded integration obligation rather than a reason to add another framework in this prerequisite.

I wrote only this report and the independent repro wrapper outside the repository. No candidate implementation or tracked evidence was edited. The regression process returned a tool session after printing its result; I polled it to observed exit code 0. All remaining commands completed. Fixture peer teardown reported no errors. A final scoped Win32_Process check found no Python process matching this review's repros or regression entrypoint. No background worker, compiler or supervisor was launched or signalled. This is bounded owned-process hygiene, not exhaustive OS attestation. The separately live remote admission batch at `4fe6b3a` and its parent/tool processes were not touched.

Approval does not open compilation/native acceptance, complete one-slot graph/transactions, warmup execution, final readiness/origin, recording/seal/barrier, C2e/f integration or activation, whole C4/refactor, C4c3/c4, C4d/B3, product PREPARE/native samples, R2/S/host payload, a3b, performance/results/baselines, shipping or backend selection. These gates remain closed under their own requirements and independent reviews.
