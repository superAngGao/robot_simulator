# Both-lane terminal validation and actual-record fixture repair

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: implementation-note and bounded preflight request
Author: codex
Version: v1
Date: 2026-10-09
Status: in_review
Related Files: runtime-adoption design v2 sections7-9; runtime-integration
design v3 section8; runtime-custody implementation-plan v3; runtime-faults
test-evidence v3; tests/gpu/test_native_runtime_cold_warp.py
Owner Summary: Close a specific mandatory coverage gap by extending the existing
terminal matrix to CUDA, fix an invalid actual-record stride mutation, and add
the missing exact aggregate/profile paths to the existing cleanup ledger.
No production source, profile, guard protocol, native observer or gate changes.

Base17bae2c765242d9a82e59c3b555f0bf8a0f3d713; exact candidate HEAD in review
packet. Running85 native faults finished before this candidate is deployed.
Their production e585956 remains byte-identical in this test-only delta.

The existing85-case matrix excludes runtime_success, runtime_terminal_sync_raise
and runtime_terminal_sync_hang. The old cold-Warp suite covers these three for
asymmetric/replay only in Warp. Extend that same test to both lanes, yielding12
cases. Reuse the existing literal PROFILES from the fault suite, unchanged
30s Warp/120s CUDA operation deadlines, fixed native driver/supervisor and
production uploader/readback/terminal implementation. No second preparation
path, fault selector or native callback is introduced. Opt-in now uses the
explicit ROBOT_SIM_TEST_C2D3_TERMINAL=1 variable because the scope is both lanes;
the historical filename/test function name remain to preserve their history.

The existing CPU canonical builder still supplies per-array expected content/
layout/topology. CUDA's indexed first Stream counts as one allocation leaf;
its runtime_import_or_init count is3 versus Warp6, because Warp also counts init,
stream lookup and explicit readback scope there. The current production call
sites and archived actual positive artifacts substantiate these differences.
All other per-array/event counts and original release/failure/reap assertions
remain. Full terminal counts bind durable selection; killed workers remain
prefixes. Original emergency ARM count7/normal COMPLETE count6 still excludes
any emergency COMPLETE. The test requires same-run external live process
identities and exact target/foreign UUID/driver rows. Raw supervisor output and
artifact are saved before later assertions, preserving failed attempts.
Its finally closes the liveness pipe and waits for the unchanged per-lane
supervisor deadline; it no longer kills the supervisor after a Warp-only40s.

Separately, the actual-record test's [1,1] stride mutation was legal for the
CUDA replay first active array's singleton leading axis. See the original
451pass/1fail archive. Choose an actual nondegenerate axis, increment its stride
and recompute the same storage digest, preserving rejection. No parser/schema
change. Run all96 storage-descriptor mutations on the8 actual records after
deployment, rather than count an inapplicable singleton edit as invalid layout.

The existing cleanup ledger lacked exact aggregate/schema/test and CUDA-profile
paths introduced during C2d3. Add those20 existing paths with both terminal
collapse dispositions under the already-approved selected-backend retention and
P12.4e removal rules. This neither performs deletion nor opens a product gate.
The existing inventory/cleanup validator passes1test0.95s; terminal collection
reports12 cases. Collection is not native execution. Actual terminal validation
awaits independent bounded preflight approval and a clean owned-process check.

Review must inspect the test/protocol/driver chain, both-lane count/deadline
semantics, the original stride counterexample and schema rule, exact cleanup
paths/dispositions, and preservation of production bytes. This requests approval
for the bounded test/ledger repair and ensuing12 actual terminal cases only.
It is not whole-C2d3 review. Complete native results, affected regressions and
independent whole-stage review remain required. C4 stays active and all later
gates stay closed as recorded in runtime-faults test-evidence v3.
