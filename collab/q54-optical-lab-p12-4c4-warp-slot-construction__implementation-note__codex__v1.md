# Guarded output allocation through the existing builders

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: implementation, bounded parent section5 / gap G2
Author: Codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: collab/q54-optical-lab-p12-4c4c2-runtime-integration__design-request__codex__v3.md;
  collab/q54-optical-lab-p12-4c4c2d-remaining-setup__gap-scan__codex__v1.md;
  optical_worker/native_setup.py; existing Warp/CUDA world/render/LBVH allocators
Owner Summary: Retain every returned output array before the next fallible
  operation while preserving the existing buffer layouts and native allocators.
  This implements the already identified partial-construction custody boundary
  needed by the real one-slot graph; it creates no alternate renderer, slot state
  machine or public callback. Graph construction and native acceptance remain.

Base fa17f34d3302ed203a32d2fc71dec4ecf96cef66. Parent v3 section5 and the independent
remaining-setup gap scan G2 require allocation owners to be adopted before the
next fallible/device leaf. Existing multi-array builders returned their aggregate
only after all allocations. Wrapping only that final return cannot satisfy the
partial-failure requirement.

The shared private _output_array_allocation context manager consumes the existing
single-use native setup context (historically named _NativeUploadContext). No
new authority registry or guardian is added. Its optional guarded path requires
the exact NativeGuardedSetupLifecycle, an active setup guard, matching runtime,
lane, target and stream, and existing runtime-owner adoption. It retains a list
before entering native scopes; each native empty result enters that list before
device metadata inspection/adoption. Native device/stream scopes restore on exit;
exceptions propagate to the original guard. The yielded internal allocator is
invalid after that exact scope ends. This is not a user-supplied callback.

Existing Warp/CUDA world allocators, CUDA render allocator and CUDA LBVH output/
workspace allocators use this one allocation primitive. Their private _context
argument is optional; ordinary callers retain the same layouts and call order.
Both paths still call their own runtime.empty directly, with no array conversion
or cross-backend ownership bridge. The CUDA render path requires preexisting ray
arrays to be adopted when used with guarded custody. Workspace alignment/view
logic remains in its original allocator and its allocation owner is retained.

Warp's existing layout function supplies the new matching render-buffer allocator,
constructing the existing WarpDirectLightBuffers and diagnostic tuple. It does
not invent a second list of fields, new kernels or alternate output representation.
No prepared graph, command/event, readiness, sample token or result is issued here.
The next fixed source-session graph builder must supply its validated binding and
retain aggregate/factory/command/event owners; this delta does not authorize raw
arbitrary allocation requests through the worker dispatcher.

Tests exercise the actual common allocator with the real lifecycle and peer, for
both lanes: successful retention, second-allocation failure, returned metadata
failure, returned-owner adoption failure, context/target/stream/runtime substitution,
scope restoration and use after scope exit. They also inject failure at each of
eight actual existing world allocations.32 passed in1.58s. Runtime objects are
CPU doubles: no native scope/restoration, numerical output or memory claim.

An initial combined regression run reported314 passed and2 failed in9.06s. Both
failures were source-policy release tests that deliberately load the committed
HEAD blob while the working renderer already had its new source pin. Freeze the
candidate, then rerun against that same commit; do not weaken those assertions.
The actual read-only _render_launch_snapshot AST and its function pin are
unchanged. Only the complete renderer source pin and existing source catalog are
refreshed; all84 prospective LF sources pass the unchanged AST rules. Current
catalog SHA2562c41ba180d4cb6722e52d8e29495bb3f20d9476b841ac749f7f453052e28c616.

Remote module-admission batch remains frozen at independently approved4fe6b3a,
parent3095592/tool96843; it has not received or executed these changes. Separate
compilation/native acceptance remains open. This checkpoint needs fresh code
review and subsequent real partial-allocation/scope validation before use in the
supervised graph. Full C4 still requires slot/transaction integration, warmups,
new final readiness, recording/seal/barrier, rendering and lifecycle matrices,
C4c3/c4,C4d/B3, systematic refactor and fresh whole review. Product PREPARE,
performance/results/baselines and backend selection stay closed.
