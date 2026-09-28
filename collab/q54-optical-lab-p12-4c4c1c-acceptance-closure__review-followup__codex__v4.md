Initiative: q54-optical-lab-p12-4c4c1c-acceptance-closure
Stage: review-followup
Author: codex
Version: v4
Date: 2026-09-29
Status: in_review
Related Files:
- q54-optical-lab-p12-4c4c1c-acceptance-closure__design-request__codex__v1.md
- q54-optical-lab-p12-4c4c1c-gpu-validation__review__codex__v7.md
- tests/support/sacrificial_leaf_audit.py
- tests/support/sacrificial_warp_child.py
- tests/unit/optics/test_sacrificial_leaf_descriptors.py
- tests/gpu/test_warp_sacrificial_render.py
- tools/optical_sacrificial_warp_supervisor.py
Owner Summary: repair the remaining F4 generic Torch event/stream observation
gap; this is not C4c1c approval or product activation.

## Prior finding

Independent session `/root/c4_sacrificial_gpu_review_v7` reviewed `1c8b5f4`
and found that the distinct, immutable `torch.Event` and `torch.Stream`
native types escaped the Python CUDA-wrapper hooks. Already-created generic
objects could query/wait inside a recorded leaf without contamination evidence.
The report is preserved verbatim in `ea6845d`; its SHA-256 is
`15ea2b8a299f3439224e020fbfadf2af62f463234d146589d3dfc76d4b17b0ed`.

## Bounded repair

The test-only audit now installs CPython 3.12+ `sys.monitoring` CALL observation
around each recorded leaf. It rejects dispatch to the real generic native
constructors and public native event/stream descriptors before the C method
body. Existing instances, cached constructors/bound methods, unbound methods
and class methods are included; no mutable-class monkeypatch or replacement
constructor proxy substitutes for native receiver observation.

The three exact CUDA Python wrapper constructors retain their existing
`__new__` hooks instead of being mislabeled as their generic native bases.
Setup verifies that each such exception has its exact registered Python hook.
Arbitrary subclasses or same-named objects do not receive this exception.

The CALL monitor reserves an unused tool identity, preserves the previous
callback, restores its event mask/callback/ownership on success and failure,
and fails closed if no slot exists. It does not replace `sys.setprofile` or
another tool. CPU tests cover real immutable native methods, pre-body rejection,
cached/unbound/class dispatch, coexistence, exhaustion and partial installation
failure. This remains fixed-source test instrumentation, not a hostile-Python
sandbox or interception of arbitrary calls originating inside native code.

Sixteen fixed GPU negative controls cover the generic family, including cached
constructor and cached query references. They create and adopt target-device
stream/event owners during guarded setup, before seal, then exercise the real
entry inside the leaf. The second-device wait test covers both CUDA-wrapper
and generic families. Evidence separates attempted generic CALL dispatch from
native Warp submission, event record and query counts.

Installed Torch 2.9.1's generic `Event.ipc_handle()` is an unimplemented stub.
The import-rejection control therefore uses a genuine CUDA IPC handle from an
already-owned target-device Warp event, retained before seal; rejection must
occur before the generic import body. This does not claim working generic
Torch IPC export/import. The export control similarly rejects before its stub.

## Author verification before candidate freeze

Host 183 SSH transport remained unreachable; owner-authorized fallback 101 is
used. Environment: `robot_sim`, Python 3.12.13, Torch 2.9.1+cu128, Warp 1.12.0,
CUDA_HOME pointing to that Conda environment's CUDA 12.8 toolkit.

- The three unchanged v7 original generic query/wait/stream-query probes now
  raise `RecordedLeafContamination` at their corresponding public entry.
  Independent body-entry/restoration verification is still required.
- First draft GPU selection: 58 passed, 4 failed. Three failures exposed generic
  base-constructor labeling of CUDA Python wrappers; the fourth exposed the
  unimplemented generic IPC export during setup. These were repaired, not
  waived or converted to skips.
- Repaired draft: 24 passed in 59.45 s, including those four controls, cached
  references, real second-device waits/success, and 15 descriptor/monitor tests.
  Log: `/data3/ang.gao/c4-generic-event-draft-fixes.log`.
- A subsequent setup-only check explicitly couples the three delegated
  constructors to exact Python hook registration. Local final descriptor tests:
  15 passed; Ruff and `git diff --check` pass. The 24-test run predates this
  last check; full clean-checkout verification remains required after freeze.

The draft remote path is not clean-checkout evidence. The next review packet
must identify the new candidate commit, its clean detached checkout, full
regression results and the unchanged original v5/v6/v7 counterexamples.

## Scope and closed gates

Only the accepted C4c1c F4 validation repair is changed. Product code, kernels,
contracts, gate files, C4c0 A/K protocol and shared collaboration rules are
unchanged. No performance result or baseline was generated. C4c1c remains
unapproved until fresh independent review; C4c2-c4, C4d/B3, product GPU
preparation, a3b, result/baseline, host/shipping and backend selection remain
closed. This note makes no C4 completion claim.
