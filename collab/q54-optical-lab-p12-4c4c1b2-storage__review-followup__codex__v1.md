# C4c1b2 storage — first review repair

Initiative: q54-optical-lab-p12-4c4c1b2-storage
Stage: review-followup
Author: codex
Version: v1
Date: 2026-09-28
Status: in_review
Related Files: optics/warp_native_render.py, storage review v1
Owner Summary: Close two independently reproduced layout/launch-descriptor gaps.

Reviewed candidate: `fe292cf40f0eefc2a6ae999185d360e9f8a6d8c0`.
The first independent report is retained verbatim in the adjacent review v1.

## Repairs

The shared untimed layout checker is now used for outputs AND additional
read-only inputs. It independently verifies strides, rank, size and capacity
before deriving a contiguous interval; the cached is_contiguous flag is not
sufficient. The reviewer's stride-40 alias is a negative GPU regression.

Warp 1.12 array.__ctype__ caches its array_t, and Warp argument packing uses
that descriptor. When present, the exact descriptor type, data/grad pointer,
rank, all shape entries and all strides must agree with current array metadata.
Inconsistency is rejected, not repaired. Validation does not create a descriptor
when absent; legitimate later lazy materialization is allowed and tested.
Per-instance __ctype__ overrides are rejected as well. The descriptor type
comes from the pinned Warp runtime, not a mirrored ctypes schema.

The regression matrix modifies data, grad, shape, strides and rank in both
input and output descriptors. It checks rejection, unchanged bad descriptor,
restoration, and successful subsequent validation. Allocation/copy/launch/
synchronization/init entrypoints are forbidden during the checks. No render
command is recorded or submitted.

## Fresh evidence and its revision limits

- Revised code focused unit/GPU/source tests: **55 passed in 3.10s** on 101.
- Revised code Ruff check/format: passed.
- Earlier clean candidate `fe292cf` full non-slow suite: **1865 passed,
  39 skipped, 117 deselected, 1 warning in 374.32s**. This is explicitly NOT
  proof that the two new repairs passed the full suite; repeat after freezing
  the repaired revision. Log: `/data3/ang.gao/c4c1b2-storage-full-suite.log`.
- The warning is the pre-existing class-scoped instance-fixture pytest 10
  deprecation in `tests/unit/solvers/test_pgs_split_impulse.py`.

No design, output-profile, mathematical kernel, product worker or activation
gate changed. Request a NEW independent session for the repaired revision.
Approval remains limited to storage; the rest of b2 and c1c are not complete.
