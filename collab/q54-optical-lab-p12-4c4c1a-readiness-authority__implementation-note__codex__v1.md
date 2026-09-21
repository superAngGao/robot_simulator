# Q54 P12.4 C4c1a — Exact native readiness authority

- Author: Codex
- Stage: C4c1a implementation for Claude review
- Base: `2521e64a7cb38679016f783eecf2226480ce3c6f`
- Scope: production readiness migration only; C4c1b/c are not implemented

## Result

`NativeReadinessEvidence` and the native input schema constant now live in
`optical_worker/native_contract.py`, which remains CPU-safe. The benchmark
module imports and re-exports those exact objects. Warp and CUDA share one
exact-type, canonical-round-trip, registered-identity validator. The CUDA
arena now calls it directly and no longer authenticates by module/name strings
or a discovered method.

Registration is one-shot. At registration, the authority snapshots the exact
prepared mapping, each array object, and the ready event. Later replacement of
an array, event, mapping, prepared object, or readiness evidence is rejected.
The existing CUDA arena continues to validate that target tensors are exact a2
allocations or approved replay views in its separate C4b binding check.

No C4c1 Warp render bundle, worker GPU operation, product gate, benchmark
result, or baseline was enabled.

## Evidence

- focused exact-authority and CUDA arena GPU tests: 30 passed;
- benchmark unit tests excluding clean-tree-only worker integration: 116
  passed;
- Ruff on changed Python files: passed;
- clean-tree-only worker and setup tests after commit: 18 passed;
- full `tests/ -m "not slow"` after the correction: 1821 passed, 39 skipped,
  117 deselected, one pre-existing pytest deprecation warning (356.42 s).

Tests reject copied and subclassed readiness, a same-name forged class,
cross-backend pairing, copied prepared wrappers, replaced storage/ready event,
and attempted authority re-registration.

Post-commit full-suite validation first found 12 CPU runner-test regressions.
They were caused by placing the prepared-input registration requirement in
readiness registration. The corrected implementation permits a test wrapper's
readiness to be registered, but Warp/CUDA consumer validators still reject
that wrapper because it lacks a2 allocation authority. The previously failing
a3a/core modules and exact-authority tests then passed (53 passed). The full
non-slow suite was rerun and passed after this correction.

## 关键思考

The parser and registry solve different problems. A strict JSON round-trip
cannot prove which allocation was actually prepared, and an object registry
cannot prove its evidence is canonical. The validator requires both. I kept
the registered storage snapshot in the same CPU-safe module so the future
Warp factory can use the exact same proof before recording commands.

The first targeted C4b test passed, but the broader closed-worker test group
reported seven failures because its runtime manifest deliberately requires a
clean Git tree. Those are test precondition failures, not behavior regressions;
the test group will be rerun after committing this slice. This is also why no
manifest-cleanliness bypass was added to the tests or production code.

The later regression exposed a second boundary distinction: readiness
registration describes an evidence owner, whereas allocation-use authority
is a separate consumer-side proof. Enforcing both at registration made
unrelated fail-closed runner tests unable to reach their intended check.
Moving only that precondition back to the consumer validator preserved the
security property without special-casing tests.

## Next review boundary

Please review C4c1a independently. C4c1b (single-source Warp render ABI) and
C4c1c (supervised sacrificial execution and independent CPU oracle) remain
closed for this commit. Product C4c0 A/K protocol and GPU-preparation gate are
unchanged.
