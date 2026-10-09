# Guarded output allocation failure repair

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: implementation, bounded parent section 5 / gap G2 repair
Author: Codex
Version: v2
Date: 2026-10-10
Status: in_review
Related Files: runtime-integration design-request codex v3 sections 4, 5, 8;
  warp-slot-construction implementation-note v1 and code-review v1;
  optical_worker/native_setup.py; test_native_output_allocation.py
Owner Summary: Caught native allocation, metadata and scope failures now use
  the existing irreversible lifecycle failure handling. No new state machine.

Repair base: 1b45c342676aeea199fc4dcb9811649706e2060c. Original allocation base:
fa17f34d3302ed203a32d2fc71dec4ecf96cef66. The full independent v1 report is
archived unchanged; SHA256 35627cec8af751dcc1b775f582f0699bd4aa40b88a2172f67f931693458b9cb3.
Its original repro remains in ../c4_assets/slot_allocation_repro_v1.py,
SHA256 313b694576a1f2906a1829825a574b03a123e2b12f88f4f1004d80b329f40fb9.

F1: _output_array_allocation observes the actual empty/metadata/adoption
operation with the existing _observe_public_operation. A caught exception
therefore remains a failed guard before the next attempted leaf or normal
completion. The same observer surrounds native scope construction, entry and
exit. Partial arrays remain in the pre-adopted list; original exceptions are
retained by the existing lifecycle. C1: the closure also captures and compares
the exact existing guard ticket before each allocation, rejecting rollover.
Ordinary unguarded builders use nullcontext and keep their previous behavior.

Tests preserve the real lifecycle and actual world-builder triggers from the
review and catch failures inside the guard and inside the allocation scope.
Both lanes cover allocation/metadata/adoption plus device/stream entry/exit,
reject retry and normal completion, retain original exceptions/returned owners,
restore CPU-double scopes, and forbid additional ARM or emergency sync. The
two-guard repro rejects before the second allocation. These are CPU doubles,
not native device-scope or memory-lifetime acceptance.

Author verification: Python 3.14, -B, pytest -q -p no:cacheprovider against
test_native_output_allocation, test_warp_render_buffer_contract,
test_warp_render_command_contract, test_cuda_scene_native, test_cuda_lbvh_native,
test_cuda_artifact_runtime, test_native_source_policy: 338 passed in 9.26s.
An initial test collection used a wrong lifecycle import; corrected to the
existing lifecycle module. A subsequent run had 46 pass / 8 assertion failures:
the new retention assertion checked direct guardian membership for arrays whose
adoption intentionally failed. It now checks the actual pre-adopted owner's
list, as the original tests do. Production retention was not changed for this.
All 84 prospective LF source files pass the unchanged source policy; refreshed
catalog SHA256 54c1269919963f94924c0bca2639188a0cccaf7b11fb4e47b9394ee7db1a141a.

No native allocation driver, graph builder, renderer execution, performance
measurement or gate activation is included. Existing CUDA preamble scope must
be accounted for by the subsequent fixed native integration. Remote admission
batch remains independently frozen at 4fe6b3a; it has not run these changes.
Whole C4, rendering integration, systematic refactor and later product gates
remain incomplete. This repair requires fresh independent code review.
