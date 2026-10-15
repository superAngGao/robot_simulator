# Q54 Optical Lab P12.4 C4c0 Closed Worker and Setup Lifecycle

- Author: Codex
- Stage: C4c0 implementation note
- Design authority: approved C4c design v7 at `79fa00c`
- Scope: C4c0 only
- Result/baseline authority: none
- Review status: revised to close the import-ledger and acknowledged
  abort-watchdog P1s

## Outcome

C4c0 now has a CPU-safe, spawn-only closed-worker control plane and the
backend-neutral setup resource state machine required before either optical
candidate can prepare GPU state inside that worker. Product
`PREPARE_FROZEN_ROW` remains deliberately disabled. No C4c1-C4c4 render path,
C4d/B3 host delivery, a3b result record, or baseline outlet was opened.

The implementation separates three facts that earlier in-process code could
conflate:

1. setup GPU work may have started;
2. the exact target device has a current setup-quiescence proof; and
3. a measured sample may have submitted work.

Only the first two exist in the C4c0 GPU fault harness. The product worker
cannot reach sample acquisition or submission yet.

## Closed process and protocol

`tools/optical_closed_worker.py` is the stdlib-only `python -I -S` bootstrap.
Before adding repository or site-package paths it checks the exact frozen C4c0
source allowlist and each source digest. Full validation then checks:

- interpreter and Python version;
- frozen environment identity and site-package paths;
- clean Git revision;
- gate digest and canonical workload-set digest;
- the reviewed AST policy and every allowlisted source;
- protocol/trust/state-machine versions; and
- the still-false GPU-preparation capability.

`tools/optical_closed_worker_supervisor.py` is also launched with `-I -S`. It
validates its own source digest before creating the worker, owns the worker
liveness sentinel, and records a no-restart process disposition. It imports no
benchmark, optical, Warp, Torch, or plugin module.

The source allowlist is enforced twice. Static source digests and the AST
policy validate what may be loaded; a live repository-module ledger then walks
the worker's actual `sys.modules` origins and rejects every repository source
outside that allowlist. The ledger paths and digest are part of HELLO evidence.
Prepared-input wrappers, their identity authority, and the Warp dtype mapping
now live in the CPU-safe allowlisted `optical_worker/native_contract.py`.
`native_setup.py` does not import `benchmarks.optical` directly or through its
package initializer.

The parent proxy uses an inherited Unix socketpair and a separate parent-
liveness pipe. The worker accepts only the eight operations frozen by design.
Canonical JSON has fixed 64 KiB request, 4 MiB response, depth-eight,
64-field, and 4096-byte-string limits. Floats, duplicate keys, noncanonical
encoding, extra fields, arbitrary operation names, Python objects, callables,
and pointer/handle fields are rejected.

The exact liveness values are versioned in
`benchmarks/results/q54_p12_4/closed_worker_runtime_contract.json`:

- `control_idle_timeout_ms = 120000`;
- `response_write_timeout_ms = 30000`; and
- `abort_watchdog_ms = 30000`.

They are transport liveness controls, not renderer budgets or backend
selection thresholds. `response_write_timeout_ms` applies only while the worker
writes a response; it is not a parent-side request/compile/warmup timeout. The
parent waits without that socket timeout. Any parent-side wait or transport
failure closes the liveness pipe and reaps the supervisor.

The supervisor owns a one-shot abort-start pipe and a separate one-shot
acknowledgement pipe. Before an emergency device synchronization, the lifecycle
sends abort-start. The supervisor must first commit the 30-second deadline and
only then return the exact acknowledgement byte. The worker records
`abort_watchdog_armed` and calls the synchronization only after receiving that
acknowledgement. An acknowledgement timeout, EOF, or malformed byte cannot
enter CUDA synchronization; it quarantines the whole graph and terminates the
worker. Thus a successful pipe write is not treated as proof that the watchdog
event loop ran. A synchronization that blocks after acknowledgement is killed
by the supervisor at process scope, even while the parent remains alive.

Each worker accepts one HELLO, one OPEN, at most one future frozen-row prepare,
and one terminal close. The close acknowledgement must be written before the
idle graph is released. In this slice, `PREPARE_FROZEN_ROW` returns the typed
`GPU_PREPARATION_DISABLED_PENDING_C4C0_REVIEW` terminal error with zero
measured-sample deltas and exits the one-shot worker.

## Device-scope and guardian authority

`DeviceScopeDescriptor` is stdlib-only immutable metadata. Constructing or
validating it does not import NumPy/Torch/Warp, query a current device, enumerate
devices, or initialize CUDA. It carries an explicit indexed ordinal, reviewed
runtime identity, operation kind, source, and unreturned-resource risk.

`NativeWorkerSetupLifecycle` records the descriptor before the leaf call. Its
possible-dirty ordinal set is monotonic inside the setup epoch; the unknown bit
is irreversible. A returned device owner is adopted into the guardian before
its metadata ordinal is compared with the target. This ordering retains a
foreign owner even when validation raises.

`NativeWorkerResourceGuardian` strongly owns partial host staging, device
allocations, streams, events, prepared inputs, and exception graphs. Additions
are legal only in `CONSTRUCTING`. Setup success seals the graph before the final
target barrier. Failure freezes the partial graph before selecting a terminal
disposition.

The terminal rule is exact:

- current setup quiescence plus no measured submission: cancel and release,
  with no emergency sync;
- dirty exact `{HELLO target}`, unknown false: exactly one target emergency
  sync, but only after the supervisor has acknowledged that its watchdog
  deadline is armed; success authorizes whole-graph release, failure or missing
  acknowledgement quarantines;
- any foreign ordinal or unknown scope: no emergency sync of any device,
  append-only whole-graph quarantine, and process exit.

The setup trace, guardian resource records and per-device counts are included
in `TransportTerminationEvidence`. Its parser replays operation counts,
dirty-set expansion, owner adoption, guardian transitions, sample latch,
emergency-sync outcome, release authority, cleanup state, and final
disposition. Recomputing the public SHA-256 cannot make a contradictory summary
valid.

## Register-before-copy a2 path

`optical_worker/native_setup.py` adds dedicated closed-worker a2 entry points;
the older in-process convenience functions are unchanged.

Both new paths use:

```text
copy canonical host value
-> guardian adopts host staging
-> create CPU runtime view and adopt it
-> allocate exact-target destination
-> guardian adopts returned destination
-> verify source/destination/stream ownership
-> enqueue H2D copy on the frozen target stream
```

Neither path uses `wp.array(host, device=cuda)` or
`torch.tensor(host, device=cuda)`. Ready events and streams are separately
created/adopted before record or wait. The final setup target synchronization
is distinct from the a2 ready-event wait and alone mints
`SetupQuiescenceEvidence` after the complete graph is sealed.

`NativeWorkerPreparationAborted` is consumed by the worker loop itself. Its
termination evidence must bind the exact session/lane/device. Synced aborts may
return normally because release was authorized; either quarantine disposition
forces immediate `os._exit(74)`. Callers cannot catch the exception and resume
the worker or run ordinary per-owner cleanup.

## Verification

The focused CPU contract suite covers canonical codec/schema rejection,
CUDA-free imports/descriptors, exact-target abort, foreign/unknown quarantine,
setup quiescence, clean cancellation, serialized termination replay, and the
closed product activation gate.

The real GPU matrix runs both Warp and CUDA and covers:

- successful register-before-copy a2 plus final target barrier;
- failure immediately after the first real H2D enqueue;
- ready-event host-wait failure;
- failure after a real warmup GPU operation;
- retention of all partial owners until target emergency sync; and
- target `cuda:1`, actual foreign allocation on `cuda:0`, for each backend in a
  sacrificial process. Both foreign cases prove zero target-sync calls, retained
  owner identity, no release authority, and the multi-device quarantine
  disposition.

At implementation time:

```text
ruff focused                                      passed
all benchmark unit tests                          117 passed
real GPU setup/native arena/command matrix         44 passed
closed-worker launcher/supervisor/terminal tests    7 passed
```

The clean-tree worker cases directly verify the live import ledger, a parent-
alive permanently blocked emergency sync, and sync-failure quarantine without
ordinary resource-owner finalization. The 44 GPU cases include the 10-case
Warp/CUDA setup fault matrix and 34 retained native arena/command regressions.

## Gates intentionally unchanged

- `native_result_activation.result_capable` remains `false`;
- `native_result_activation.activation_commit` remains `null`;
- `a3b_result_capable_before_C4_B3_review` remains `false`;
- host-payload arena sealing remains blocked until B3/joint review;
- product closed-worker GPU preparation remains `false` in the reviewed runtime
  contract; and
- no benchmark result or baseline artifact was generated.

The next authorized action after approval is C4c1. Enabling product
`PREPARE_FROZEN_ROW` needs a separate reviewed manifest/capability change; this
commit itself cannot do it.
