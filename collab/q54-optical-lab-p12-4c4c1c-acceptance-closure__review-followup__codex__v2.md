# C4c1c event-query audit repair

Initiative: q54-optical-lab-p12-4c4c1c-acceptance-closure
Stage: review-followup
Author: codex
Version: v2
Date: 2026-09-28
Status: in_review
Related Files: GPU-validation independent review v5; acceptance-closure design v1
Owner Summary: Repair the demonstrated event-query false negative; no scope or gate change.

Repair base: `5f6f511`, the unchanged v5 report archive over `0c9e72b`.
The candidate is the commit containing this note. Independent re-review is
required; no passing-test total overrides the demonstrated contract violation.

## F4 residual

The monitor now supports readable property descriptors, freezes their exact
class identity, rejects inside their getter before the original body, and
restores the original descriptor in finally. `Warp Event.is_complete` and
`Stream.is_complete` are monitored along with `get_event_elapsed_time`.
Native Warp event/stream query and elapsed-time entries are separately wrapped
and restored. Their attempted entry count is recorded as `native_queries`;
positive render and public-query negative controls both require zero.

The neighboring event-family inventory also covers Warp event IPC-handle
creation, convenience record/wait functions and Torch event/stream query,
record/wait, elapsed-time and IPC-handle methods. This remains a fixed
supported-path observation tool, not a universal native security sandbox.

Three new fixed controls invoke the real retained event/stream completion
properties and elapsed-time API from inside the actual recorded-launch
wrapper. Tests require the exact rejected API, zero native render/event/query
entries, one successful hook restoration, retained owners and the existing
post-latch emergency disposition. No callback or native handle enters IPC.

## Author evidence before freeze

- Original v5 event probe, unchanged, against the draft: expected exit 1 with
  `RecordedLeafContamination: warp.Event.is_complete` at the property read.
  It no longer reaches its old clean-audit assertion. The scratch probe remains
  `/data3/ang.gao/c4-sacrificial-gpu-review-v5-event-probe.py`; SHA is recorded
  in the preserved v5 report.
- Real supervised forbidden-runtime selection: 12 passed, 92 deselected,
  77.23 s. Includes all three new query/elapsed controls.
- Local lifecycle/source regression: 50 passed in 5.30 s; ruff and diff checks
  passed. Final clean-candidate matrix and independent results follow separately.

The preceding `0c9e72b` clean-candidate author matrix passed 190 tests in
658.45 s and its additional contracts passed 104 in 5.09 s. Independent v5
passed 423 without skips while reproducing F4. Those are predecessor facts,
not proof for this repaired revision.

Only test audit, fixed fault enumeration and assertions change. No kernel,
guardian state machine, C0 A/K protocol, product source policy, gate, baseline
or result path changes. F1/F2/F3/F5 remain regression obligations. C4c1c is
still pending approval; all later and product paths remain closed.
