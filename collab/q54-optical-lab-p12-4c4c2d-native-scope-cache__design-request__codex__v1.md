# Bounded Warp stream-cache precondition

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: design-request supplement
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: native-scope-aggregate design-request/review v2;
native-scope-aggregate code-preflight-review v1;
tests/support/native_runtime_process.py
Owner Summary: Repair the failed preflight assumption without permitting an
observer cache miss, allocation, context selection or dependency change.
This supplements the approved v2 design; all its other obligations remain.

## Evidence and bounded decision

The code preflight of b0f22fb found that Python Device.set_stream calls a void
native setter and then publishes _stream even when get_context_info returns
NULL. The observer must not infer a populated native cache from Python return.
The review's exact pinned-Device CPU counterexample remains authoritative
failure evidence for that implementation; no new native attempt has executed.

Keep the existing target-context stream getter ONLY after a new, independently
reviewed precondition proves the original stream path populated the cache.
Do not add a new stream, invoke the setter again, initialize the cache from the
observer, modify Warp, or substitute metadata for actual current native stream.

The additional query is the already-loaded Warp core's wp_get_error_string(),
whose pinned context.py declaration is argtypes=[] and restype=c_char_p.
It returns a static CPU buffer via warp.cpp:152 and error.cpp:27. Reading it
does not call a CUDA API, change a context, clear errors, allocate GPU memory,
or synchronize. Retain the actual function before calling it. Require its
existing ctypes declaration exactly; do not alter the shared declaration.
Preserve raw bytes as hex in the ordered diagnostic, including nonempty output.
NULL or any value other than exact b'' is a failed precondition, never success.

Source anchors on the pinned installation:

- context.py SHA eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee:
  Device.context 3638-3653, _init_streams 3607-3613, set_stream 3676-3699,
  stream 3661-3670, Stream.__init__ 3304-3353, declaration4050-4051.
- warp.cu SHA 7c7e6e7a02db7b285ca1800d0ec52e28d2cf8a704435d925c70a8c5ccd8f176b:
  get_context_info366-399, native setter2285-2304, getter2276-2283.
- cuda_util.cpp SHA 21d9dfe57b861ada6b98d0fa2ccfb3b8ed0b14972ed840eb04d0ad7e5f70a89f:
  check_cu_result326-343 and check_cuda_result315-324 write nonempty errors.
- warp.cpp SHA b9feea467eb11e0d9016ca33fceb7b1a4358fc21c65513a692378155f1d05fc7.
- error.cpp SHA 877b9f1e889d3e5bf8950c3a3deaf28f44bc4a9a76e780dcc19606d80f6229f2;
  error.h SHA 4ebf4a1dcd7ed121df72f9390888c5f9eb30547f0e42fde5e04ace50de7face0.
  Static4096-byte buffer starts empty; get does not clear it. Error setters in
  this original call chain only write nonempty messages; no success clears it.
  These are source-preflight anchors, not invented admission of new modules.
  The runtime core library remains the actual admitted, held, hashed owner.

## Exact ordering and proof obligation

1. Inside the original indexed_runtime_stream ticket, after quiet runtime init
   and before original get_stream: require target's raw _context is None and
   raw _stream is None, and require exact empty native error buffer. Read raw
   vars only; do not trigger either property. Preserve this raw diagnostic.
   The existing pristine context/primary snapshot remains required.
2. Delegate original get_stream once. Production holds and journals the real
   returned owner first. Preserve existing after-return injected failure order.
3. Check the exact returned owner is raw target._stream, raw target._context is
   positive, and returned stream handle is positive. Require the same native
   error query to return exact empty again BEFORE any native stream getter.
   Any stale or new nonempty error fails closed; never clear, compare only a
   suffix, whitelist an error, retry, or recover by taking the getter anyway.
4. Given initially absent context/stream, the pinned Device.context path must
   run _init_streams -> set_stream -> native context setter. Its cache miss
   either inserts ContextInfo after successful cuCtxGetDevice or reports a
   nonempty check_cu error before returning NULL. The latter is now rejected
   before the observer getter. Errors in ContextGuard and malloc/free workaround
   also persist and reject. Native-return absence alone still proves nothing.
5. Only after all preceding facts set an observer-local cache-established flag.
   Every non-pristine Warp _state requires that flag BEFORE its native stream
   getter. This flag is a derived local precondition from this exact execution,
   not a caller-supplied token or aggregate authority. The fixed original path
   has no context destroy/cache erase before terminal release; retained graph
   and original guard prevent release during observations. Source preflight
   must inspect that lifetime and cache erase sites, not merely Python metadata.
6. Retain native current-stream queries at all v2 sites and require the first
   poststream native handle equals the held returned stream handle. A NULL/0
   response is contradictory after the original nonnull stream and cannot set
   a successful baseline. Later queries must match that established baseline.

The two error-buffer reads are initialization preconditions. They do not
replace the direct post-scope context observations or suppress a contradictory
completed row after a void scope exit. No additional error-buffer polling at
helper/scope edges is proposed. Original sticky failures still prevent later
native observations and ordinary release, including precondition failure.

This proof is for the single fixed sacrificial worker, exact admitted runtime
and pinned call chain. It does not claim global error-buffer concurrency safety
for an arbitrary application, nor introduce a production error policy. Review
must reject this supplement if the original chain can clear/overwrite the
error with empty data or erase a proved cache before a subsequent query.

## Ordered diagnostics correction

The same implementation revision fixes code-preflight F2: every Python/C10
getter invocation records exact getter name, primitive arguments, successful
primitive return or exception type/message in execution order. This includes
count, each primary ordinal, logical device and explicit-target current stream;
the new Warp error getter and existing Warp native stream getter use the same
bounded wrapper. Hold exceptions before any fallible diagnostic processing.
Do not retry a getter, fabricate output after an error, or drop earlier success.
Raw diagnostics remain non-authoritative; complete aggregate rows are appended
only after the entire snapshot exists. CUresult queries retain return-code
checking and successful-null distinction under the existing v2 design.

## Required verification and remaining gates

Before implementation: fresh independent design review of this supplement and
its exact source feasibility. After implementation: fresh independent code/
native preflight, including prior F1/F2 reproductions and affected invariants.
No new native/GPU observer run occurs until that preflight approves.

CPU checks must drive the actual observer through the original pinned Device
methods with modeled native success, NULL-with-error and stale-error paths.
The old failing-setter model must fail before any cache getter; neither a
metadata-only match nor an unset cache flag may reach it. Require original
returned owner retained before precondition failure, sticky UNKNOWN and no
normal completion/sync/release. A positive CPU model must establish the
precondition and preserve native getter evidence; label it a model, not GPU
proof. Exercise a mismatching/zero first stream response. Ordered CUDA raw
checks must fail each getter position and preserve all earlier successes plus
the exact failing call without a second call or complete snapshot.

After preflight, resume all16 actual native-scope cases, affected full faults/
persistence/source regressions and whole C2d3 independent review required by
v2. The aggregate schema, library rows, profiles, operation counts, watchdogs,
immutable cuts, acceptance coverage, upload/readback and all downstream gates
are unchanged. C4 continues; this repair is not C2d3 or C4 acceptance.
