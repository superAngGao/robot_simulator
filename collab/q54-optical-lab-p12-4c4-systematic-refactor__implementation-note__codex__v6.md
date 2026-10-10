Initiative: Q54 / P12.4 / C4
Stage: M1 optical preparation orchestration consolidation
Author: Codex, implementation session
Version: v6
Date: 2026-10-10
Status: in_review
Related Files: systematic-refactor gap-review v1 M1; code-review v4; runtime-integration design-request/review v3
Owner Summary: Six optical preparation entrypoints now delegate their orchestration to the existing setup module. Shared input verification, publication and failure routing replace repeated stage code and the Warp-only idle failure path. CPU validation has passed; native evidence and fresh independent approval remain required.

The base is evidence-only commit 6db106caaeeac66d3bfc939b2bee44c15ce104d9,
following bounded R0 approval at640cf9f24490e9bf7d259c37ec9adfb0412ac1ba.
The approved candidate remains unchanged in c4_final_graph. This work is
isolated in c4_setup_orchestration, branch implementation/c4-setup-orchestration.

## Changes within the approved M1 direction

Source admission, captured modules, the exact upload registry and runtime
binding validation remain owned by native_bootstrap. Its six original entry
methods delegate to explicit functions in native_setup: CUDA factories, Warp
modules, fixed storage, provisional graph, warmups and final graph. The loader
exposes its retained input record and original condition/error check; it does
not delegate source or native identity admission.

The existing setup module owns the stage sequence. It shares validated upload
lookup, paired source/environment audits and checkpoint-before-ready publication.
Phase drift, poisoning or a failed checkpoint cannot publish a ready stage.
Backend-specific compilation and the original normal guard segmentation remain
explicit. There is no callback registry, workflow engine, extra guardian or
second slot control implementation.

One operation boundary retains the original exception and routes failure by
actual guard/scope state. Source-integrity rejection still fails closed. A
failed/active/pending segment retains the original failure and deadline; it
cannot acquire a fresh normal guard or synchronize again. Healthy IDLE with
proven target-only outstanding work uses the existing emergency synchronization;
unknown custody quarantines. CPU-only or current final-quiescent state reuses
the existing cancellation classifier and needs no runtime binding or GPU wait.
The Warp-only _fail_warp_preparation method is removed. CUDA and Warp select
their concrete target synchronization inside the same failure handler.
If secondary intake fails, the controller retains the original and secondary
exceptions and quarantines; the original stage exception is re-raised.

Input upload/materialization and source-integrity failure policy outside these
six entries are unchanged. R1 construction, product request dispatch, measured
submission and clean CLOSE are not opened by this consolidation.

## CPU verification so far

Windows C:/Python314/python.exe; native Warp/trimesh unavailable.

- Initial relocation plus common failure handling: 228 passed in206.45s across
  Warp module preparation, final graph, provisional graph and fixed storage.
- Consolidated input/audit/publication and failure regressions: 235 passed
  in218.34s across those suites plus optical orchestration tests.
- Source/catalog policy, actual bootstrap, guarded lifecycle and runtime custody:
  413 passed, 13 platform/asset skips in189.01s. POSIX source-release and real
  Go2 checks still require their native environment.
- A new CPU-only out-of-order-stage counterexample first failed in both lanes:
  missing runtime was incorrectly treated as unknown scope. Reusing the existing
  may_cancel_without_sync predicate fixes it without another classifier. After
  this last change, orchestration and original Warp module preparation suites:
  36 passed in7.38s. Earlier GPU-state paths of the predicate are unchanged.
- New regression file passes Ruff; git diff --check passes. Whole legacy setup
  and fixture modules retain import-order/long-line lint findings; no claim of
  a globally clean lint run is made. Imported dependency order is preserved.
- All90 prospective committed LF bodies pass the existing AST policy after
  bootstrap body/class pins and the catalog were regenerated. Current catalog
  SHA256 d2c1d277c263a70ed02422a60b40e3c0453086385b6158a3b8f185a510cd35fd.

The original healthy-IDLE, unknown/incomplete custody, aggregate issuance,
active-guard and ABI/module ownership tests are retained. New tests exercise
both-lane out-of-order calls with and without prior GPU work, late checkpoint
throw/poison/phase drift, source-integrity rejection and a secondary retention
failure preserving the original exception. CPU doubles do not establish native
device/context behavior.

## Remaining proof and continuation

The fixed native driver adds one out-of-order optical-stage case immediately
after actual native input validation. It observes the production-selected
emergency synchronization under its original live guard, including native
target state and ambient restoration, and checks original guard counts, exact
error, single sync, release and observed exit. It does not replace a renderer,
uploader, synchronization function or lifecycle with a test implementation.
Native execution of that case and full R0 preparation at the exact frozen
candidate remains pending, followed by a fresh independent review of M1.

This note does not claim M1 approval or full C4. Continue the approved R1
composition, C2f token/resolve/reclaim/close, numeric/native failure matrices,
C4c3/c4 and C4d/B3 and final systematic whole-C4 review. Performance, backend
selection, results/baselines/a3b and shipping remain closed.
