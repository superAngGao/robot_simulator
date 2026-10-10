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

The first actual624a71f CUDA idle-entry run reached the expected production
error, synchronized target1 once, restored its prior scope, released and exited74.
The driver then failed its acceptance assertion: it incorrectly counted a
seventh CUDA factory phase although this injection precedes factory preparation.
The six original input phases completed. The assertion now uses the existing
RUNTIME_PHASES inventory, common to both lanes. This is a driver-count repair;
production source bytes and the catalog are unchanged. Preserve the original
failed driver run under c4-final-graph-624a71f11f-cuda-runtime_optical_idle_entry
and obtain new evidence at the corrected revision. Do not relabel its original
driver exit as a pass or count this as a hardware/production failure.

This note does not claim M1 approval or full C4. Continue the approved R1
composition, C2f token/resolve/reclaim/close, numeric/native failure matrices,
C4c3/c4 and C4d/B3 and final systematic whole-C4 review. Performance, backend
selection, results/baselines/a3b and shipping remain closed.

The initial pending-evidence statements above describe the frozen handoff.
The following completed evidence and independent verdict supersede them.

## Native verification at the frozen M1 candidate

Candidate c3472f22f1d52bfb6c853bf95cef4040cf666316 is clean; its production
optics/optical_worker bytes equal624a71f11f2c4e0766fa6b3f03be3c90460e0850.
The count-only driver repair does not change production code or the catalog.

On primary tileops-dev, exact Python3.12.13 robot_sim/Warp1.12.0/Torch2.9.1+cu128,
H200 target1 and frozen Go2 assets, all five sequential acceptance drivers
returned0. Raw worker exit74 is expected for injected failures, not a successful
worker render completion.

| Native case | Lane | Normal completed guards | Emergency syncs | Worker exit / disposition |
| --- | --- | --- | --- | --- |
| post-input healthy-IDLE out-of-order optical entry | CUDA | 6 | 1 | 74 / aborted_synced_exit |
| post-input healthy-IDLE out-of-order optical entry | Warp | 6 | 1 | 74 / aborted_synced_exit |
| complete R0 final preparation | Warp | 27 | 0 | 0 / cancelled_before_submission_exit |
| complete R0 final preparation | CUDA | 23 | 0 | 0 / cancelled_before_submission_exit |
| original Warp module aggregate issuance failure | Warp | 10 | 1 | 74 / aborted_synced_exit |

All five release_status values are completed; supervisors observed EXITED,
without deadline expiry or requested kill. Idle-entry observers verify actual
target1 synchronization and prior-scope restoration under the existing emergency
guard. CUDA prior state is logical0/raw1; do not describe it as raw0. Final
preparation includes actual ten warmups, final graph/barrier and acquire/cancel;
it does not execute a final measured sample. The initial624 failed driver
remains a failed assertion with no result.json, separately preserved.

Linux source/catalog-policy/bootstrap suites at624:293 passed,1 Windows-only
skip in131.97s, including real Go2 and POSIX source checks unavailable locally.
The wrapper returned0; the production bytes match c347 as described above.

Archive c4_assets/c4-setup-orchestration-native-evidence-v1.tar.gz and the same
basename under /data2/shared/ang.gao:16432007 bytes,191 unique members,SHA256
 e41a65b368720b1f083c47896a3a51b08a5a28b51ffe4809984cc3da88fee90e.
It preserves all six raw runs, driver scripts, matrix log, CPU command/results/log,
environment contract and process closure. The closure joins each recorded
parent/supervisor/worker PID and verifies20 owned PIDs absent, including matrix
and CPU wrapper. This is scoped process closure, not a claim about unrelated
host processes. Original author sessions23548 and86204 are terminal0.

These facts do not establish R1, C2f, final-sample numerics/performance, product
activation or whole C4. The fixed native drivers remain the existing acceptance
surface; no shipping/selection gate is opened.

## Independent verdict and continuation

Fresh reviewer01a12422-6645-7760-929a-0aacefd5bce4 returned Approve with comments
for exactc3472f22f1d52bfb6c853bf95cef4040cf666316, base6db106c. Full report is
archived byte-for-byte as systematic-refactor code-review v5, SHA256
2039240945f6992716091b1c67651ef0b5c6e987288299260b8c9d275b58a082.
M1 is closed within that bounded scope. The fresh reviewer inspected the whole
delta, actual custody/phase flows, all90 committed source bodies and replayed
all six native records using production validators on primary Linux.

Main independent CPU selection:618 passed,13 skipped,4 failed in481.49s.
The four failures are the pre-existing graph-observer fixture missing fixed_row;
the reviewer independently reproduced all four on base. Independent additional
probes plus provisional preparation:102 passed in18.61s. The twelve added
independent cases cover pending/active guard and secondary disposal/final-idle
exceptions. These facts are not an all-green full-suite claim.

N1 remains a nonblocking fixture follow-up in the approved revision. A working
repair initializing fixed_row=None exists in separate R1 branch c4_dynamic_graph,
with all four existing mutation/observer cases passing; that later delta remains
unreviewed. The R1 working commit590e333 also begins scene/refit recording reuse;
it is incomplete, not R1 acceptance. Continue its full composition and the
remaining C4 obligations. No new gate is opened by this evidence-only record.

Reviewer tool session51894 returned0 and its report file was verified. Native
and author CPU sessions are terminal. The independent report documents its
completed child-test/SSH sessions and scoped process closure. This archival
commit contains only the unchanged report and factual evidence/progress updates.
