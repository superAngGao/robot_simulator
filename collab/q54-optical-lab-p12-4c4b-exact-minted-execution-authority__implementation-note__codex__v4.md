# Q54 Optical Lab P12.4 C4b Exact-Minted Execution Authority

- Initiative: Q54 optical computation workflow / P12.4 backend selection
- Stage: corrective implementation note
- Author: Codex
- Version: v4
- Date: 2026-09-09
- Status: ready for short re-review
- Corrects commit: `c5438d0d825337e54e9752732164a03dc8554abb`

## Outcome

The remaining forged-authority P1 is closed. The timeline no longer accepts an
authority based on `__module__`, `__name__`, or the presence of lifecycle
methods. Those checks and the caller-supplied lifecycle controller have been
deleted.

A controlled transaction now accepts only an exact
`_CudaArenaLiveExecutionAuthority` instance created by its own timeline binding
path after a real C4 acquire. The authority must also be present in two
identity registries:

- transaction → exact authority;
- exact authority → exact transaction and `C4ActiveOutputLease`.

Both relationships are checked at authorization entry and again inside every
controlled stage or finish operation.

## Exact mint and lease binding

`_bind_arena_execution_lease()` does not accept an authority object. It accepts
the static construction capability, the real arena-issued
`C4ActiveOutputLease`, the active sample owner, and the frozen plan identities.
The timeline constructs and registers the sealed authority itself.

Minting requires:

- `type(lease) is C4ActiveOutputLease`;
- lease state is exactly `LEASED`;
- active owner contains the exact transaction and is still `DORMANT`;
- lease slot, terminal event, transform frame, C3 binding, ordered-stage, and
  dataflow identities match the sealed plan.

The exact authority then retains the prior lifecycle coupling:

```text
authorize enqueue -> C4 begin_enqueue -> exact stages
authorize finish  -> terminal event    -> C4 record_completion
```

Therefore a valid authority cannot finish a transaction while leaving the C4
slot `FREE` or its terminal-event record count at zero.

## Forgery resistance

The following objects are rejected before a CUDA stage can enqueue:

- an ordinary class whose module and class name impersonate the internal
  authority and whose callbacks are no-ops;
- an exact-type object created with `object.__new__()` but absent from the mint
  registry;
- an authority registered for another transaction;
- an unregistered replacement installed into an acquired active sample;
- a copied or stale authority whose transaction/lease registry identity no
  longer matches.

The authority remains invalid before acquire and after cancel, reclaim, or
poison.

If an authority replacement is detected before the first enqueue, no device
work has started: the C4 lease is cancelled, its generation is burned, and the
slot returns to `FREE`. If failure occurs after enqueue starts, the existing
poison-and-retain rule applies. This avoids both unsafe reuse and a stranded
pre-submit `LEASED` slot.

## Tests

Real-GPU negative tests reproduce the reviewer attack using a same-named class
with no-op lifecycle methods. They also exercise an unregistered exact-type
instance and replacement of a legitimate live authority.

Verification on `192.168.80.101` in `robot_sim`:

- CUDA arena GPU tests: 20 passed;
- C4b/C2/C4a focused set: 77 passed;
- combined C2/C3/C4 set: 126 passed;
- broad `tests/unit/optics` plus `tests/unit/benchmarks`: 446 passed;
- repository-wide pytest: 1820 passed, 115 skipped, with one pre-existing
  pytest fixture deprecation warning;
- Ruff lint and `git diff --check`: passed.

## Scope remains closed

The a2 provenance and O(1) acquire corrections remain unchanged. This patch
does not modify gate activation, benchmark results, arena layout, output
payloads, delivery, C4d, or B3. a3b, baseline generation, host-payload arenas,
S, C4d, and B3 remain closed.

## Review request

Please rerun the forged same-name authority reproduction and verify:

1. authority truth is exact type plus mint-registry identity, never labels or
   callable-method discovery;
2. only `_bind_arena_execution_lease()` can associate a controlled transaction
   with the exact live C4 lease;
3. copied, replaced, and unregistered authorities fail before CUDA enqueue;
4. pre-enqueue rejection frees the slot, while post-enqueue failure poisons and
   retains it;
5. legitimate direct live-authority use still commits the matching C4
   lifecycle transitions.
