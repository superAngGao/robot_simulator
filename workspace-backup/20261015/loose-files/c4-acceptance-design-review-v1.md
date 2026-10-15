# C4c1c acceptance closure — independent design review

Initiative: q54-optical-lab-p12-4c4c1c-acceptance-closure
Stage: review
Author: codex (independent reviewer)
Version: v1
Date: 2026-09-28
Status: accepted
Related Files: acceptance-closure design v1; C4c1 ABI designs v1/v2/v3; GPU-validation independent review v4; tests/support/sacrificial_lifecycle.py
Owner Summary: Approve the bounded design supplement. Its fixed two-pass schedule, actual recorded-leaf audit and target-only context matrix are implementable without weakening the accepted lifecycle. This is design approval only, not closure of the three outstanding implementation findings or approval of C4c1c execution evidence.

## Identity and reviewed snapshot

- Fresh independent session: `/root/c4_acceptance_design_review_v1`; no implementation conversation inherited, no candidate authorship, no delegated agents.
- Candidate: `C:/Users/AG/新建文件夹/p12_4_c4_sacrificial_gpu`.
- Base: `55365c528cbdb5208804be9e46220e4bdf6a7335`.
- HEAD: `c747fb6f175632de4cae323541e736088aca4d03`.
- Design: `collab/q54-optical-lab-p12-4c4c1c-acceptance-closure__design-request__codex__v1.md`.
- Independently computed SHA-256: `9dec80201a66c96b47e8a03b8ccab8e74e968efb526371477c782f000c2c6f59`.
- Actual local tree status was clean before and after review. Base-to-HEAD is exactly two added Markdown files, 294 lines: the supplement and the retained v4 report. No executable or operative workflow changes are present.
- Read-only remote source inspection used the authorized 101 fallback, `/data3/ang.gao/robot_sim_c4_gpu_verify_55365c5`; its HEAD is exactly the code base above and its working tree was clean before and after inspection. This remote checkout does not contain the new supplement and was used only to examine the unchanged code/dependency basis.

Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` completely; reread the Chinese principles with explicit UTF-8 after initial PowerShell decoding. Examined original ABI v1, corrections v2, lifecycle v3 and the complete independent v4 findings. Prior author/reviewer test totals are historical context, not new evidence in this report.

## Verdict: Approve

No substantive blocking design finding. The supplement makes the missing acceptance work implementable inside the existing one-shot authority; it does not substitute diagnostic flags for runtime evidence or authorize a product execution surface.

The approval binds only the design digest and reviewed HEAD above. F3/F4/F5 in the prior code review remain open until the revised implementation and fresh required runtime evidence receive independent code approval.

## Requirement-by-requirement assessment

### 1. Same recorded bundle under two sentinel patterns

Design section 2, lines 38–86, preserves exact bundle/command identities and original storage bindings, and adds separate first-pass snapshot destinations and both host observation graphs before setup quiescence. The camera case includes ray outputs; resident ray inputs are correctly not corrupted as output sentinels. Zero-ray coverage still exercises the two scalar-reset commands.

The ordered sequence is sufficient: render A → target-stream D2D snapshot → target-stream output refill B → render B → one supervised terminal validation drain/readback → inspect both. Same-stream ordering protects snapshot contents without an intermediate wait, event, host observation or release. Delaying first-pass CPU inspection until the common final drain is a valid lifecycle-preserving clarification of v1 section 14.4, not removal of first-pass verification. Both snapshots must still be checked against independent CPU truth (and the retained counter invariants) and against each other.

The present guardian has a single `_command_count`, `_attempted`, `_returned` and `_launch_active` protocol. It currently rejects repeated positions; the supplement explicitly requires extending that authority to a frozen render/auxiliary schedule rather than bypassing it in the child. This is a feasible local change. It does not call for resealing, clearing the latch, resetting counters, reopening a released graph, or granting a generation/lease.

Pre-call accounting applies to every auxiliary copy/fill and every second-pass launch. An exception cannot erase the possibility of submission even when the operation never returns. Active/missing/out-of-order leaves prevent progression and V. Full-graph retention and exact-target E/quarantine continue to apply. Once V is entered, no second E sync is allowed. These are consistent with v3 sections 2, 4, 5 and 6.

### 2. Actual runtime contamination instrumentation

Design section 3, lines 90–114, audits the executed `Launch.launch` interval rather than recording or preflight. Actual wrapped entry observations and exact native launch count/identity are required; a caller-filled flag is insufficient. Fixed negative operations must enter the monitored API, be rejected before its original body, produce no native render launch for that rejected entry, and retain post-latch failure disposition.

The existing inventory is available in `benchmarks/optical/instrumentation.py:49`; it already covers allocation, host conversion/materialization, synchronization, interop and extension-load families. The supplement explicitly adds missing test-only copy, Warp array materialization, module, event and ordinary-launch coverage. It does not turn the benchmark registry or its evidence into product activation authority.

Installed Warp's actual recorded leaf is narrow and hookable: `warp/_src/context.py:7310` dispatches the frozen command through `runtime.core.wp_cuda_launch_kernel`, with its bound device context, kernel handle, parameter address and explicit stream handle. It does not call ordinary `wp.launch`. Thus the allowed operation can be distinguished from a newly constructed launch and observed below the top-level wrapper.

Restoration in `finally`, callable-identity verification and exclusion of legitimate setup/auxiliary/drain operations from the audit interval are explicit. There is no arbitrary callback/module/native-handle IPC extension. The trusted fixed-process boundary and the express exclusion of arbitrary Python/C++ heap interception are consistent with the predecessor designs, not an additional security claim.

### 3. Context matrix and absence of hidden foreign setup

Design section 4, lines 118–151, satisfies the retained v1 section 14.3 requirement. That requirement is ambient device/current-stream restoration with single-device and target-1/ambient-0 execution; it does not specifically require creating a nondefault current stream on an otherwise untouched foreign device. The prior v4 repair direction also expressly requires an ambient fixture compatible with target-only setup and calls for a reviewed clarification if foreign setup would be needed.

The supplement supplies that clarification without extending dirty-device scope:

- target 0 / ambient 0: a real nondefault current-stream fixture, distinct from the execution stream, is strongly retained and compared;
- target 1 / ambient 0: a real nondefault target-1 current-stream fixture is retained and compared, while ambient-0's actual preexisting state is preserved exactly;
- an untouched ambient device is observed as actual UNINITIALIZED metadata, not replaced by a dummy stream or treated as an unchecked value;
- success and injected-exception paths compare ambient device, target-current stream, ambient-current state, and explicit execution stream before/after each recorded leaf;
- a deliberate state drift must be detected, and mismatch is quarantine/hard exit rather than exact-target cleanup pretending to cover unknown scope.

This does **not** prove restoration of a nondefault foreign GPU-0 stream; the design accurately disclaims that case. That limitation is not a waiver of the original requirement: the two mandatory rows preserve their actual ambient state, and the target-0 row independently exercises a nondefault ambient stream. Creating a foreign fixture merely to satisfy a stronger label would violate target-only setup and is not authorized here.

Independent installed-source checks support feasibility:

- `Device.__init__` initializes `_stream = None` and `null_stream = None`.
- `Device.stream` accesses `self.context`; `Device.context` can acquire a primary context and call `_init_streams()`; `get_stream()` returns this property. Therefore querying it on untouched foreign hardware is genuinely side-effectful.
- Reading an already-created device's `_stream` and an existing stream's stored handle is CPU metadata and does not invoke that property.
- `Device.set_stream(..., sync=False)` is available. `ScopedStream` supports explicit `sync_enter=False, sync_exit=False`; its default `sync_enter=True` must not be used for the newly specified no-wait auxiliary/context work.
- `ScopedDevice` saves/restores the runtime default-device object and native context rather than constructing a foreign stream on exit.

The implementation must use these real runtime facts, not manufacture UNINITIALIZED evidence. Setup/observer owner adoption and source admission remain required before sealing. A malformed metadata observer or a lazy stream getter would fail this approved design; neither is licensed by it.

## Inspected integration and lifecycle facts

- `tests/support/sacrificial_warp_child.py`: current single sentinel, one recorded pass, device-only assertion, final retained readbacks and supervised drain; confirms the prior three missing assertions have not silently been implemented by this documentation commit.
- `tests/support/sacrificial_warp_case.py`: guardian-scoped target setup, register-before-next-leaf ownership, explicit retained input readback, failure hard exit and E/quarantine. The new fixtures can use these target-scoped adoption points.
- `tests/support/sacrificial_lifecycle.py`: one-way seal, pre-call attempt versus return counters, active-leaf exclusion, supervised S/V/E, whole-graph quarantine, and release authority based on quiescence rather than successful-return count.
- `tools/optical_sacrificial_warp_supervisor.py` and `tests/support/sacrificial_supervisor.py`: fixed canonical request, enumerated faults, source ledger, one child, deadline-before-ACK, parent-loss kill and no restart. New fixed cases require explicit schema/enum updates in this test surface, not a generic executable request.
- `tests/unit/optics/test_sacrificial_lifecycle.py`, recording/buffer GPU tests and runtime-hook inventory: existing regression surfaces can be extended without duplicating production rendering mathematics.
- Installed Warp contiguous copy path (`context.py:8708` onward) uses explicit-stream same-device `wp_memcpy_d2d` without staging allocation; `array.fill_` (`types.py:3745`) uses device memtile for contiguous storage and can run under the explicitly non-synchronizing target-stream scope. Zero-sized copy/fill returns are compatible with still-accounted fixed auxiliary positions.

## Verification performed and limits

Executed read-only commands included:

```text
git status --short / git status --porcelain=v1
git rev-parse HEAD
git diff --stat 55365c528cbdb5208804be9e46220e4bdf6a7335 HEAD
git diff --check 55365c528cbdb5208804be9e46220e4bdf6a7335 HEAD
Get-FileHash <design-v1> -Algorithm SHA256
Get-Content -Encoding UTF8 <requirements/design/code files>
rg / rg --files <local code and test surfaces>
ssh -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 <read-only git/grep/sed commands>
```

Remote installed sources were read under `/data3/ang.gao/anaconda3/envs/robot_sim/lib/python3.12/site-packages/warp/_src/` (`context.py`, `types.py`, `utils.py`). No runtime probe, recorded launch, pytest suite, GPU allocation, synchronization, package installation or benchmark was executed in this design review. This is source-backed feasibility analysis, not fresh proof of real-device behavior. The packet identifies this environment as Warp 1.12 / robot_sim; the implementation review must verify and record its actual runtime environment anew.

`git diff --check` passed. Local and remote candidate trees remained clean. Initial combined reads that exceeded output limits were narrowed; initial mistaken/glob paths were replaced with actual `rg --files` paths. The optional host-routing decision file named by AGENTS was absent in this checkout; AGENTS and the review packet explicitly permit the 101 fallback. These were inspection/tooling details, not candidate defects. No candidate file was edited; only this outside-candidate report was created. No files were deleted.

## Required next evidence and residual limits

The supplement's acceptance section is substantive: CPU schedule/order/reopen/active-leaf/retention tests, auxiliary and second-pass failure accounting, all four real shade variants using the same bundle twice, full camera-output sentinels, zero-ray resets, actual audit negative controls, both real device/stream rows, and preserved setup/readback/partial-submit/transport/watchdog faults are still required. Existing historical 98/181/165 or aggregate 346 passes cannot replace any missing assertion. No skip is real-device proof.

Implementation review should specifically check explicit `sync=False` use, hook restoration before E/V or auxiliary work, snapshot disjointness and retention, unchanged counters/latch across the pass boundary, and runtime observations on both success and failure. These are implementation checks derived from the approved text, not unresolved design blockers or authority to relax the requirements.

This test infrastructure remains selection/validation-only, follows the existing losing-lane cleanup ledger, and creates no permanent product execution protocol. No hostile-process/native-memory security guarantee, performance result, foreign-nondefault-stream coverage or product lifecycle proof is granted.

## Approved boundary

Approve implementation of this exact bounded C4c1c acceptance-gap repair. Do not treat it as approval of C4c1c code, C4c1 as a whole, or C4. A fresh independent code review of the revised frozen candidate and runtime evidence is required before C4c2 may start under its own applicable authorization.

C4c2–C4c4, C4d/B3, product GPU preparation, a3b, result/baseline access, host payload/shipping and backend selection remain closed. C0 A/K protocol, product source allowlist and activation authority are unchanged. Earlier narrow approvals retain only their original scopes.
