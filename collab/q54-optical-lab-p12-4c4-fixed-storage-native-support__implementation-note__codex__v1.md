# Native fixed-storage validation support

Initiative: Q54 / P12.4 / C4 fixed render storage
Stage: bounded test support
Author: Codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: fixed-render-storage code-review codex v1; runtime-integration
  design-request codex v3 sections 5, 8, 9; remaining-setup gap-scan v1 G2
Owner Summary: Exercise the approved production storage entry with the existing
  sacrificial worker, native scope queries and guard supervisor.

Base 219d7d4 (evidence-only Warp compilation acceptance). Production storage
remains the approved12c1619 implementation. No production, policy, catalog,
controller or supervisor change is included. This support serves the existing
G2 requirement for actual target1/ambient0 slot allocation and partial custody.

Five fixed selectors call SourceSession.prepare_fixed_render_storage after the
original lane-specific compile/factory preparation: diagnostics off/on positives,
second allocation raise, second returned-array pre-adoption raise, second
allocation hang. Warp retains the existing compiled-executable reuse check;
CUDA retains the existing exact factory-module checks. Both use the original
ACK1000ms / operation120000ms supervisor via the existing run_case entrypoint.

The observer reads the already-established native queries under the live guard
at each output allocation. It records target context/device/stream and only the
target primary context active. Successful preparation must restore ambient0
and the previous stream. Output IDs are checked against the actual production
pre-adopted array list before each adoption; the observer stores primitive
metadata, not extra array owners. Input views must borrow exact prepared arrays.
No diagnostic work occurs between runtime.empty return and production retention.
CUDA allocation order differs from tensor_items order, so ID sets are compared;
neither allocation order nor renderer ABI order is changed.

After a failed allocation/adoption, production has already failed the ticket:
only CPU root inspection is allowed. No native restoration query runs there and
no failure-path native restoration claim is made. The primary error survives a
failed diagnostic write. The runner checks exact original guard counts, original
error, retained partial arrays, full quarantine, no next leaf/ARM/COMPLETE/sync,
and actual exit. Hang acceptance requires deadline expiry/kill/exit-9 and honest
partial evidence, without a fabricated terminal worker record. Normal storage
success still ends by the existing one-sync disposal; it issues no readiness,
command, event, warmup, rendered image, timing, token or product capability.

The existing fixed-storage CPU setup was extracted into one fixture, preserving
its original tests and adding actual-session/allocator support checks in both
lanes, with primitive native-state doubles. They exercise both diagnostics modes,
allocation/adoption failures, original-error preservation on diagnostic I/O
failure, and reject altered retention/stream/device/restoration observations.
Initial checks caught a bound-method identity assertion and CUDA allocation-order
assumption in support tests; corrected to exact function/self and actual ID sets.
An initial failure observation attempted a now-failed ticket; removed all native
post-failure queries, preserving only CPU custody observations.

Native allocation execution and independent support review are still required.
This is a test-support checkpoint, not whole G2, rendering or C4 acceptance.
C4 implementation, full validation, systematic refactor and fresh whole review
remain required; all later product/performance/results/baseline gates stay closed.

Author verification: Python3.14 -B -m pytest -q -p no:cacheprovider, selected
prepared_fixed_render_inputs, native_output_allocation, native_scope_observer,
warp_failure_diagnostics, cuda_artifact_runtime and runtime_native_scope_codec:
150 passed,41 skipped in29.34s. Host-specific native-scope/codec cases skip on
Windows; these skips are not native acceptance. Tool session58975 exited0.
