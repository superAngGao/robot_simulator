# Q54 Optical Lab P12.4c0b CUDA Publication Binding

Author: Codex

Date: 2026-08-13

Status: P1/P2 shutdown-safety revision implemented for Claude re-review

Approved C0a design:
`a1e603bebdd939696d2ea4f174fc1cf82dcd8e35`

## Outcome

P12.4c0b implements the approved physics-publication lifecycle contract. A
lossless borrowed device consumer can bind one sealed, externally owned CUDA
stream/resource guardian to `GpuEngine`, borrow physics-published slots in
strictly increasing order, enqueue the publish-event wait on that stream, and
record a physics-owned done event after its last physics-buffer reader.

This slice does not add a CUDA optical product lane. It does not construct a
Torch optical scene, CUDA LBVH, output arena, delivery/readback ring, backend
selector, benchmark lane, sample, or baseline. C2-C4/B3 must still prove the
complete optical storage provenance before the binding may be used by a
shipping CUDA lane.

## Public ownership boundary

`CudaConsumerResourceGuardian.seal(...)` is the only consumer-side lifetime
input. Before bind it freezes:

- an explicit CUDA device ordinal;
- the external CUDA stream owner and its native handle;
- a non-empty identity-unique tuple containing every storage/session/module
  owner that the future reader command graph may access or destroy;
- an opaque stream token and resource-manifest digest.

The guardian is single use. Publication validates the stream owner, handle and
device on bind and before every borrow/completion/normal close. A cooperative
resource owner can call `assert_resource_disposal_allowed(...)`; bound or
quarantined storage cannot be individually destroyed. C0b supplies this safety
mechanism, while C2-C4/B3 remains responsible for sealing the actual complete
CUDA optical graph.

`physics.cuda_publication` deliberately imports neither Torch nor any optical
module. `GpuEngine` supplies a narrow Warp-on-CUDA runtime which:

- wraps the external native CUDA stream under `wp.ScopedDevice`;
- waits the physics publish event on the consumer stream;
- allocates/records publication-owned non-timing done events;
- waits those events on the physics producer stream before slot reuse;
- exposes target-device synchronization only to poisoned shutdown.

Raw CUDA handles are not emitted in evidence.

## Ordered lifecycle

The binding and lease state machines enforce:

- one binding per consumer and one binding per guardian;
- enabled, lossless, borrowed, device consumers only;
- strictly increasing unique borrows and FIFO completion;
- rejection of duplicate, fabricated/non-borrowed and out-of-order completion;
- rejection of legacy Warp stream borrow/complete, consumer replacement,
  unregister and reset while the binding is active;
- reset rejection before the engine changes run identity, arrays or counters;
- explicit completion on normal context exit;
- safe real-event completion followed by a programming error when explicit
  completion was forgotten;
- safe real-event completion while preserving an ordinary body exception;
- poison if exception unwinding cannot preserve FIFO.

Each successful completion records the done event on the stable external
stream, appends a retained completion record, then performs the exact ring
commit. Ephemeral zero-copy view handles remain owned by the lease until both
event recording and the exact ring commit succeed. A later event covers prior
work on that stream, but event references are released only after the producer
reuse wait has been enqueued or normal final drain has been established.

Normal close rejects active leases, enqueues one producer-stream wait for the
newest retained event, and only then returns publication's guardian hold to the
consumer session. It performs no host/device synchronization. This is a
physics-lifetime handoff, not a claim that consumer buffers have physically
completed their separate stream-aware cleanup.

The new binding accepts only a fresh device consumer whose legacy
`latest_seen_frame_id`, `device_completed_frame_id`, and `device_done_event`
remain at their initial values. An incomplete or completed legacy borrow cannot
move to a different external stream under the same consumer identity. A legacy
consumer with an incomplete borrow also cannot unregister. Once its real done
event exists, unregister first enqueues that event on the physics producer
stream and only then removes logical reclaim accounting; callers may afterward
register a new consumer ID for the CUDA binding.

## Poison and cross-owner teardown

CUDA wait, stream identity, event allocation/record, ring commit, producer
reuse wait, or final-drain wait failures poison the binding and every active or
still-retained slot. The failed lease does not receive false logical
completion. Future borrow, completion, publication, reset, and normal close
fail closed.

`abort_poisoned()` is idempotent after its first terminal result and has exactly
two dispositions:

1. `emergency_sync_succeeded`: one target-device synchronization proves all
   outstanding accesses finished; only afterward are lease/event/wrapper holds
   released and ordinary guardian cleanup authorized.
2. `quarantined_until_process_exit`: if synchronization fails, an append-only
   process registry atomically takes strong ownership of the engine, ring,
   slots, events, binding, active leases and the complete consumer guardian.
   The device becomes unavailable to new engines in that process. Logical
   unregister and `GpuEngine.close()` do not release covered objects, and
   guardian close remains forbidden.

`GpuEngine.__del__` never synchronizes. An unsafe unclosed binding is transferred
directly to the same process-terminal quarantine. Explicit `GpuEngine.close()`
is idempotent and is the normal lifecycle endpoint. If normal binding close
itself poisons while enqueueing final drain, the same engine-close transaction
catches that poison and immediately runs `abort_poisoned()`; callers never need
a second close call to reach a legal resource disposition.

## Evidence boundary

`CudaPublicationLeaseEvidence` is produced only by binding transitions and
records publication facts needed by the later S lane:

- physics run/frame/slot/step/device and ordered borrow identity;
- binding, guardian and opaque stream/resource/event identity;
- publish event presence and consumer-stream wait enqueue;
- last physics-buffer reader and exact consumer completion;
- logical reclaim, done-event retention and producer reuse-wait state;
- normal final drain, poison category, emergency-sync counts/failure category;
- guardian state, teardown disposition, quarantine acceptance and
  process-terminal status.

It does not claim Torch zero-copy transform construction, scene/LBVH ownership,
optical output completion, payload drain, or final shipping reclaim. Those
remain later lane-generated receipt responsibilities.

## Repository cleanup

The `physics_cuda_publication_binding` P12.4e ledger entry now covers the new
module and its unit/GPU lifecycle tests. If CUDA is selected, the binding may
remain only after the C2-C4/B3 guardian storage-provenance review. If Warp is
selected, the CUDA-only binding API, module, exports and dedicated lifecycle
tests are removed together. No permanent dual-backend dependency was created.

## Verification

```text
conda run -n robot_sim ruff check \
  physics/__init__.py physics/gpu_engine.py physics/publish.py \
  physics/cuda_publication.py \
  tests/unit/physics/test_cuda_publication.py \
  tests/gpu/test_cuda_publication_binding.py

conda run -n robot_sim python -m pytest -q tests/unit
conda run -n robot_sim python -m pytest -q tests/gpu
conda run -n robot_sim python -m pytest -q \
  tests/unit/benchmarks/test_optical_benchmark_a3a.py \
  tests/unit/benchmarks/test_optical_benchmark_contracts.py \
  tests/unit/benchmarks/test_optical_benchmark_core.py
```

Observed before this implementation commit:

- focused publication state-machine, ring and GPU API tests: 113 passed;
- a3a/benchmark contract isolation tests: 58 passed;
- full unit suite: 1091 passed, 20 skipped;
- full GPU suite: 316 passed;
- Ruff and `git diff --check`: passed.

No benchmark sample or baseline artifact was generated or inspected.

## Review request

Please review whether:

1. borrow/completion/reuse/final-drain ordering implements the approved C0a
   state machine without host synchronization;
2. guardian identity and strong ownership close the normal cross-owner
   lifetime boundary without overclaiming current optical arena coverage;
3. poison retains active/retained slots and cannot advance false completion;
4. emergency-sync success and process-terminal quarantine are the only unsafe
   teardown outcomes and ordinary cleanup cannot bypass quarantine;
5. evidence is transition-produced, handle-safe and narrow enough for C0;
6. reset/rebind/legacy/unregister/finalizer behavior fails closed;
7. the cleanup ledger guarantees a clean CUDA-selected or Warp-selected repo;
8. no C2-C4/B3, a3b, shipping lane, sample, or baseline work leaked into C0b.

The requested P1/P2 re-review should additionally confirm that legacy device
state cannot cross into a new stream binding, legacy unregister retains its
producer dependency, and a final-drain failure reaches emergency sync or
quarantine exactly once in the original `GpuEngine.close()` call.
