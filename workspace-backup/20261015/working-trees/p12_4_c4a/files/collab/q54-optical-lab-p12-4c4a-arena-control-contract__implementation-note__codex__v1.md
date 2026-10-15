# Q54 Optical Lab P12.4c4a Arena Control Contract (Codex v1)

Author: Codex

Date: 2026-09-07

Status: implementation submitted for Claude review; C4b/C4c/C4d, B3,
P12.4a3b result activation, benchmark samples, and baseline access remain
closed

Base: `d703bcdef06f7825e64edc303d74cb29f5d0910d`

Approved designs:

- `q54-optical-lab-p12-4c4-native-output-arena-leases__design-request__codex__v1.md`
- `q54-optical-lab-p12-4c4-native-output-arena-leases__design-request__codex__v2.md`

## Delivered C4a scope

### Production control plane

`optics/native_arena.py` is CPU-safe and does not import the benchmark package,
Torch, Warp, or a CUDA extension. It adds:

- immutable `NativeOpticalArenaSpec` with exact row, input, lane, device,
  stream, renderer, acceleration, dimensions, profile, payload policy, and
  ring identity;
- complete session/slot `NativeOpticalArenaResourceManifest` records with
  exact resource, owner, event, storage-manifest, and byte identities;
- distinct non-live `C4PreparedSamplePlanEvidence` and arena-issued live
  `C4ActiveLeaseBindingEvidence`;
- `C4PreparedSamplePlan` and `C4ActiveOutputLease` live objects whose
  constructors require the module-private arena authority;
- a sealed `NativeOpticalArena` with strict FIFO/round-robin selection,
  monotonically increasing generations, single-generation plan activation,
  typed non-mutating backpressure, explicit pre-enqueue cancellation, and
  explicit reclaim;
- arena-produced lifecycle transitions and canonical
  `C4ArenaLifecycleEvidence`.

Prepared plans and active leases both report `result_capable=False`.
Canonicalized evidence by itself is never accepted as a live plan or lease.
The arena additionally retains the exact registered plan object, precreated
lease object, and both evidence snapshots; object/evidence substitution fails
before slot mutation or first enqueue.

Arena configuration, plan, lease, signature baseline, and seal fields cannot
be reopened or deleted through ordinary assignment. The arena recomputes its
spec/resource/owner-token-to-object binding signature before prepare/acquire.

### Normal state and evidence

C4a implements and tests:

```text
FREE
  -> LEASED
  -> ENQUEUING
  -> COMPLETION_RECORDED
  -> PAYLOAD_READY
  -> RECLAIMABLE
  -> FREE
```

It also supports only the approved pre-enqueue exceptional transition:

```text
LEASED -> FREE  # generation remains burned
```

After first enqueue, an illegal transition poisons the arena and changes all
active storage to `RETAINED_POISONED`; no completion or reclaim evidence is
fabricated. C4a does not yet perform emergency device synchronization or
resource disposal. Those C4d operations remain closed.

The lifecycle validator already freezes the approved v2 safety predicate:
`resource_release_authorized` cannot be true unless device synchronization
succeeded, device quiescence is proven, the external hold count is zero, and
the exact shutdown disposition authorizes release. This prevents C4a evidence
from pre-authorizing the later C4d path.

The schema also freezes all v2 shutdown fields before C4d implementation:
hold counts at poison and after device sync, discharge and emergency-sync
counts, poisoned-slot and retained-pinned-storage counts, shutdown disposition,
and its independent transition-trace digest. C4a emits only truthful zero/null
values for operations it does not yet implement.

### Frozen-row adapter

`benchmarks/optical/arena.py` derives the production spec only from a validated
frozen gate row plus the exact prepared input/readiness pair. It mechanically
checks case, content, input contract, scene, replay/fixture, lane, device,
manifest, readiness, renderer, acceleration class/identity, image dimensions,
payload policy, and ring depth.

R0/R1 are forced to `native_only/native_only`, depth one. R2 keeps its frozen
payload/policy/depth. Host-payload specs may be represented for preflight, but
constructing a host-payload arena raises
`NativeOpticalArenaNotActivatedError` until B3 and joint activation review.

### Submission/lifecycle split

Gate/schema versions advance before any result exists:

```text
gate                    q54.p12.4.gate.v11
result                  q54.p12.4.result.v11
row semantics           q54.p12.4.row-semantics.v6
native timing           q54.p12.4.native-timing.v4
operation trace         q54.p12.4.operation-trace.v3
lane eligibility        q54.p12.4.lane-eligibility.v3
```

Guarded submission operations now include measured `arena_acquire` where the
boundary requires it and the one `terminal_completion` record. `reclaim` is no
longer a required submission-trace operation. It remains a mandatory,
source-audited lifecycle entry point and must be proven by arena transition
evidence during resolve/drain. A submission trace containing reclaim is
rejected.

The cleanup ledger now accounts for the production arena, benchmark adapter,
and tests under either terminal backend branch.

## Deliberately not claimed by C4a

- No CUDA or Warp slot allocation exists yet.
- No C2/C3 transaction template is bound to an arena slot yet. The C3/stage
  digests accepted by the control plane become exact runtime-object bindings
  only in C4b/C4c; until then every live lease remains non-result-capable.
- No host payload, pinned staging view, external payload hold, B3 stage, or
  ordered D2H implementation exists yet.
- No emergency device synchronization, external-hold discharge, quarantine
  transfer, guardian handoff, or finalizer cleanup is implemented yet; these
  are C4d/B3 lifecycle work under the approved v2 rules.
- No runner path accepts C4 evidence. The existing C3 one-shot rejection and
  versioned native-result activation gate remain in force.

## Verification

Executed in the server `robot_sim` environment:

```text
ruff check (all changed Python files)                         passed
new C4a unit tests                                            17 passed
all tests/unit/optics + tests/unit/benchmarks                416 passed
git diff --check                                              passed
```

No GPU test is required for this CPU-only control/evidence slice. C4b starts
the CUDA resource/transaction GPU coverage.

## Review request

Please approve or request changes on C4a only:

1. production arena/spec/resource ownership separation;
2. live authority versus serialized evidence separation;
3. FIFO slot/generation/backpressure/cancel/reclaim state rules;
4. arena-produced lifecycle evidence and submission/reclaim trace split;
5. frozen-row adapter provenance checks;
6. host payload and result activation remaining fail closed;
7. C4b/C4c/C4d and B3 responsibilities remaining explicitly deferred.
