# C4c1b2 camera, evidence and recorded binding checkpoint

Initiative: q54-optical-lab-p12-4c4c1b2-recording
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-09-28
Status: in_review
Related Files: C4c1 design v1/v2/v3; optics/warp_native_render.py
Owner Summary: Immutable recording/preflight only; no recorded render execution or gate activation.

## Boundary

The parent implementation is independently approved fixed-ABI commit 845dbb3.
This branch's immediate base 4cda6ed records the unchanged independent report
and author test evidence (same evidence content as the checkpoint branch's
39db95d). Source ancestry and primary-host reconciliation limitations remain
documented; no original 183 commit identity is claimed by the recovery branch.

This checkpoint adds normalized query/camera parameters, strict local evidence,
an immutable factory, exact argument descriptors and recorded command ownership.
It does not launch a recorded command. There is no enqueue/execute/finish API,
terminal event, result view, arena or lease authority in the new bundle.
Neither public compatibility nor the closed product worker calls the bundle.

Execution warmup, the setup guardian, supervised partial-recording/launch failure
paths, numerical inspection and reuse remain C4c1c work. Compilation and recording
are untimed setup operations. A registration claiming zero warmup frames is not
upgraded into execution evidence by this code. Full C4c1 approval still requires
the reviewed sacrificial driver and its entire fault/oracle matrix.

## Implemented contracts

- `WarpRenderQueryParameters` holds exact scalar query identity and limits ray
  count to the int32 kernel range. Camera scalars are normalized to finite f32;
  width/height are positive, ray count is exact, focal lengths and maximum
  distance are positive. Host matrices are copied into immutable scalar tuples.
- The factory calls the production exact Warp prepared/readiness authority,
  verifies every local array is the exact original a2 object, verifies actual
  physical storage, counts and role mapping, and retains the native roots.
  Same-content reuploads and copied readiness objects fail before recording.
- Factory device and strongly owned nondefault stream are explicit. Kernel
  object/function/module identities, ordered argument ABI and source bytes are
  frozen. Preparation loads the production module on the declared device;
  recording uses a scoped stream and restores ambient context.
- R0 world/BVH render arrays must be the original prepared fixed allocations.
  Replay bindings require a valid independent replay-frame index. Dynamic world
  and acceleration arrays are retained/validated; their producer ordering is
  not claimed by this local render proof and still requires C4c2/C3 composition.
- Three profiles use the same physical render plan. Camera records raygen,
  BVH reset, initialized first-hit, shadow reset and one fixed shade entry.
  Resident-ray bindings omit raygen; zero rays retain exactly the two resets.
  Resident-ray evidence cannot masquerade as camera evidence.
- `validate_bound()` reconstructs plan, input/output storage and local evidence.
  It also checks the actual raw Launch kernel/device, dimensions, parameter
  owners and addresses, each array_t pointer/shape/stride/gradient field, scalar
  ctypes/value, hooks/module and launch options. A correct Python descriptor is
  insufficient if the packed native call drifted.
- Dataclass seals reject ordinary assignment/deletion and baseline recreation.
  Independent retention tuples and baseline comparisons detect single-point
  `object.__setattr__` drift. This is a supported-path integrity contract, not
  a sandbox against arbitrary same-process code rewriting all baselines.
- Local evidence rejects unknown fields, malformed hashes/types, incompatible
  fixture/replay/ray-source branches and wrong order/count. Hashes do not grant
  authority; the bundle rebuilds evidence from registered input and real owners.

No Warp/Torch bridge or new runtime allocation is introduced into a measured
submission; there is no measured submission API in this checkpoint. Full graph
checks, hashes, descriptor construction, module loading and recording are setup.
No authority registry or schema is duplicated from the benchmark package.

## Author verification

On 101 scratch `/data3/ang.gao/robot_sim_c4_binding_work`, conda robot_sim:

```text
python -m pytest tests/gpu/test_warp_render_recording.py tests/unit/optics/test_warp_render_command_contract.py tests/gpu/test_warp_render_buffers.py tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py tests/gpu/test_optical_gpu_runtime.py -q
200 passed in 6.81s
```

The 28 recording GPU tests forbid `wp.Launch.launch` for render bundles; actual
`record_cmd=True` runs and its raw packed parameters are inspected, but no bundle
work executes. The dynamic fixture uses previously approved C3 setup commands
before installing the render no-execution guard. Two real GPUs are present and
the target-one/ambient-zero recording test ran, not skipped. The 60 CPU contract
tests cover scalar normalization, source detachment, strict serialization,
branch exclusion, command counts/order and integer boundaries without huge GPU
allocations. Existing kernel/storage/compatibility regressions supply the rest.

The focused source files pass Ruff after mechanical formatting/import sorting.
Fresh-session independent review and clean-candidate full regression follow.
No result/baseline or performance measurement has been produced.

## Remaining stages and cleanup

The C4c1c driver must adopt setup/partial recording owners before another GPU
leaf, latch possible submission before every launch, obtain real supervisor ACK
before S/V/E waits, and quarantine unsafe failures without ordinary unwinding.
This binder alone does not meet those resource-release obligations. Product GPU
preparation, C4c2 arena execution, host payload/B3, shipping and a3b/results/
baselines remain closed. Retain the prior review's exact-counter microfixtures
in C4c1c as the recorded correctness matrix is implemented.

Final selected-backend cleanup follows the existing ledger: keep this one Warp
implementation only if selected; otherwise remove it with the losing optical
lane. No permanent comparator or fallback is added.
