# C4c1c recorded acceptance closure

Initiative: q54-optical-lab-p12-4c4c1c-acceptance-closure
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-09-28
Status: in_review
Related Files: acceptance-closure design/review v1; GPU-validation review v4
Owner Summary: Implements F3/F4/F5 without opening product execution or results.

Base: `95e1c9f` (approved bounded design and verbatim review archive).
The code candidate is the commit containing this note. It still needs a fresh
independent code review; design approval is not implementation approval.

## Changes

- One sealed guardian supports a fixed one/two-pass schedule. Render and
  auxiliary attempts commit before each call; successful returns advance an
  ordered schedule. Neither the possible-submission latch nor seal is reset.
- A second pass uses the exact same bundle and recorded commands. Every physical
  output is copied to an independently preallocated GPU snapshot before the
  original output is filled with a different sentinel. Camera rays are included;
  zero-ray fixtures cover scalar resets. One supervised V drain copies both
  retained snapshots into preowned host storage. Both are checked against the
  independent CPU executor and against each other.
- Test-only runtime instrumentation wraps actual recorded launches and checks the
  exact native launch arguments/count. Fixed negative controls enter actual
  allocation/copy/materialization/NumPy/bridge/DLPack/module/event/sync APIs.
  Those hooks reject before their original bodies. Hook identities are checked
  on installation and after restoration, including exceptional exits.
- Setup establishes a distinct nondefault target current stream. CPU-only reads
  capture exact ambient/target stream owners and handles; an untouched foreign
  stream remains uninitialized. Every recorded call checks restoration before
  and in finally. Deliberate drift quarantines without a target-only sync.
- Progress text corrects stale recording/C0 review status, without changing any
  operative scope, gate, shared review rule or product protocol.

The instrumentation is a supported-path test observer, not a security boundary
against arbitrary native/Python code, and not a claim about Python heap allocation.
No timing or performance result is produced. Snapshot/fill work is validation-only.

## Author checks before candidate freeze

Local Windows Python 3.14: ruff format/check and git diff --check pass.
Six relevant CPU/source/contract files: 76 passed, 21 POSIX-only skipped. These
skips are not runtime evidence and must be rerun under Linux.

Primary SSH 183 timed out; authorized fallback 101 is reachable. Preserved all
old checkouts. Draft runtime tests used a new detached `95e1c9f` worktree with
only the candidate test-support changes overlaid:
`/data3/ang.gao/robot_sim_c4_acceptance_draft_95e1c9f`.
Environment: robot_sim, Warp 1.12, Torch 2.9.1+cu128, conda CUDA 12.8.
Actual Warp target 0 is H200; target 1 is A100. Normal caches were retained.

Executed selections in `tests/gpu/test_warp_sacrificial_render.py`:

- `-k two_complete_sentinels`: 8 passed, 93 deselected, 56.53 s.
- `-k forbidden_runtime`: 9 passed, 92 deselected, 58.20 s.
- second-real-GPU, exception restoration, deliberate stream drift and reuse
  partial-failure tests: 7 passed, 45.26 s.

These 24 draft checks precede the final strict rejection of reuse-only fault
requests with repetition=1. Full clean-candidate regression and independent
verification remain required; their outcomes will be recorded separately.

## Remaining scope

C4c1c remains unapproved pending review. C4c2-c4, C4d/B3, product GPU
preparation, host/shipping payloads, a3b, results/baselines and backend selection
remain closed. No public compatibility executor adopts this recorded driver.
All helpers follow the existing temporary validation/losing-lane cleanup rules.
