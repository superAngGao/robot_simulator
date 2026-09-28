# C4c1c public event/stream operation coverage

Initiative: q54-optical-lab-p12-4c4c1c-acceptance-closure
Stage: review-followup
Author: codex
Version: v3
Date: 2026-09-28
Status: in_review
Related Files: GPU-validation independent review v6; accepted acceptance-closure design/review v1
Owner Summary: Repair the verified Torch Event.wait false negative and audit adjacent
fixed-runtime event/stream operations. This is test-only supported-path observation,
not a product authority, arbitrary native-call sandbox, or stage approval.

Base: `ac4c6ad17c7cd542ab4ab1840d976d6837023801`; verbatim independent v6
report archived separately at `2e882ef`, SHA-256
`b3b6bd25d8ed7aa7177fbbd9fd31deb41c2cc58a9c9a0fceaf38c6d66e54ca13`.
The independent 364 passes did not close its confirmed F4 residual.

## Bounded repair

- Add Torch Event.wait, event/stream constructors, event IPC import, and Warp
  event IPC import. Installed Warp 1.12 imports an IPC event before constructing
  Event; intercepting the constructor alone would be too late.
- Preserve staticmethod/classmethod/property descriptors, not transient bound
  methods. Freeze/check/restore using `inspect.getattr_static`; keep the same
  exceptional ExitStack restoration. CPU tests exercise descriptor binding,
  original-body non-entry, constructor non-entry and exact restoration.
- In setup, compare the installed Warp Event/Stream and Torch CUDA
  Event/Stream/ExternalStream wrapper classes' public operation descriptors
  (plus constructors) against actual registered targets. A newly exposed
  operation cannot silently escape this coverage inventory. This inventory is
  not runtime observation and is not presented as such. Private C++ superclass
  entries, arbitrary cached native callables, dunder reflection and arbitrary
  Python heap activity are not a claimed security boundary.
- Thirteen additional fixed GPU negative controls cover the complete Torch
  CUDA Event public operation family, both Torch stream constructors and Warp
  event/stream construction plus IPC import. Existing Warp query/elapsed and
  allocation/copy/interop/sync controls remain. No transported callback, object,
  module name or native handle is added to the fixed test request.
- Control event and external stream are created in setup on the explicit target;
  the owning Warp stream, Torch view, event and any IPC bytes are all adopted
  before seal. The event is recorded on that exact stream before S proves setup
  quiescence. The wait negative is additionally exercised on real target 1.
  Physical stream/device mismatch goes straight to quarantine, not target-only
  release. Tests require no native render/event/query submission, exact rejected
  target, restored hooks/context and the existing post-latch disposition.

No renderer/kernel, guardian schedule, production A/K protocol, allowlist,
product preparation, arena authority or activation gate changes. No new release
or recovery path. Normal and exceptional recorded execution remain inside the
supervised one-shot validation program.

## Author evidence before freeze

Fallback 101 (`aig-a100`) remains the runtime host; primary 183 transport was
unreachable. Environment is robot_sim, Python 3.12.13, Warp 1.12.0,
Torch 2.9.1+cu128 with CUDA_HOME pointing to conda's CUDA 12.8 toolchain.
Old dirty worktrees remain untouched. No installation or benchmark execution.

- Local final descriptor/lifecycle/source/native-observation selection:
  `64 passed in 6.27s`; Ruff and diff checks pass.
- Author scratch checkout `/data3/ang.gao/robot_sim_c4_acceptance_draft_95e1c9f`:
  initial 25 real forbidden controls + eight repeated-sentinel cases + five
  descriptor tests: `38 passed in 206.90s`. This preceded the final public-surface
  inventory, explicit control-device check and sixth descriptor test; it is not
  mislabeled as the final complete candidate run.
- Updated scratch selection (explicit target-1 Torch wait rejection, target-1
  normal Warp render/context, six descriptor tests): `8 passed in 16.23s`.
  Logs: `/data3/ang.gao/c4-event-family-draft.log` and
  `/data3/ang.gao/c4-event-family-final-draft.log`.
- The unchanged v6 `torch_wait` probe now exits at
  `RecordedLeafContamination: torch.cuda.Event.wait`, before its old clean-audit
  assertions. The next reviewer must independently verify original/native-body
  non-entry and restoration, not infer all of that solely from this exception.
  Probe digest remains
  `f9cd070c1bae24a22a1f430903e505775a6c642dfbd76f94572c047f18c3449d`.
  Descriptor inspection in diagnostic wrappers must now use `getattr_static`,
  since classmethod binding generates new objects; the original operation repro
  itself remains unchanged.

Freeze a clean candidate and rerun full supervised validation plus affected
regressions. A fresh independent session decides closure. C4c1c is not approved
by this note. C4c2-c4, C4d/B3, product GPU preparation, a3b, result/baseline,
host-payload/shipping and backend selection all remain closed.
