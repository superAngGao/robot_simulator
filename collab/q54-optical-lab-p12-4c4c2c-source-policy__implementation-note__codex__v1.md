# C2c5 source-policy primitive (not runtime admission)

Initiative: q54-optical-lab-p12-4c4c2c-source-policy
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: runtime-integration design codex v3 section4;
optical_worker/native_source_policy.py; tests/unit/optics/test_native_source_policy.py;
benchmarks/results/q54_p12_4/cleanup_ledger.json
Owner Summary: Implement the CPU byte/AST checking primitive needed by the
accepted integrated-worker design. No loader, dispatcher or activation change.

Base: `7f32e6869e47fe31ea65a67ea317e3554144b41a`.
The candidate is the commit containing this note, reported separately in review.

## Delivered boundary

The new CPU-only verifier checks exact bytes against a supplied expected digest,
canonical production-relative Python paths, forbidden imports/dynamic entry
names and direct/literal-reflected runtime attributes. It permits the existing
read-only ctypes inspection only in the exact reviewed Warp renderer source
and its single `_render_launch_snapshot` function. Both full byte hash and
location-independent function AST hash are pinned; plain top-level module
import, exact allowed attributes and no alias escape are separately checked.

The source pin is the **LF Git/POSIX source-release bytes** of the already
approved C2c4 renderer, not text normalized by the verifier. Windows working
copies may contain CRLF. Tests obtain the actual Git blob without changing the
working source; file validation on POSIX must use the actual checkout bytes.
Even comment-only or newline-only changes reject under the exception and need
a reviewed source-pin update. No runtime renderer/kernel is imported or edited.

This is not a security boundary against arbitrary Python inside the trusted
worker. A caller-recomputed hash is only a consistency check, not authority.
Production roots/two launcher names restrict the verifier's input namespace;
they are **not** a wildcard import allowlist. The primitive returns no execution
capability and does not execute the checked bytes. The file helper rejects
symlink components/escaped paths, but does not claim to close later-loader
TOCTOU or bytecode-cache issues.

## Explicitly incomplete

Remaining integrated source-admission work must freeze literal per-lane complete
transitive source catalogs and exact digests, then enforce actual repository
import ledgers at the accepted startup/HELLO/request boundaries in the new
versioned bootstrap. This primitive is not that bootstrap, full import closure
or a replacement for lifecycle/guardian/supervision integration. Representative
source checks in its test are not evidence of a complete runtime import closure.

Old C0 AST policy, runtime manifest/allowlist, A/K protocol and preparation
rejection remain unchanged. C2d/e/f, C4 completion, product GPU preparation,
host payload, shipping, a3b and result/baseline activation are not approved by
this slice. No numerical performance/memory threshold is introduced.

## Validation before review

Local Windows `C:/Python314/python.exe`:

`python -m pytest -q tests/unit/optics/test_native_source_policy.py`

Observed **79 passed**. Checks include actual production source syntax, rehashed
renderer mutations, alias/CDLL/cast/write attempts, source pin/newline drift,
invalid digest/path/encoding, file and directory symlinks, cold `-I -S` CPU import,
non-execution of checked code and unchanged C0 gate. One lint line-length issue
was corrected afterward; final lint/regressions and native POSIX byte checks are
recorded with their actual commands/results in the subsequent evidence report.

The broader local command additionally including `test_closed_worker_protocol.py`
and `test_optical_benchmark_core.py` observed **98 passed, 9 failed**. All nine
failures resolve the existing `torch.empty` runtime instrumentation entry:
local Python3.14 has no Torch installation. They are not counted as passes;
the same collection must run in the existing remote robot_sim environment.
No package installation or gate/test weakening is used to bypass this gap.

No GPU tests or benchmark result records are appropriate for this read-only
CPU primitive. Earlier GPU results are not presented as validation of this delta.
Primary183 SSH timed out; authorized fallback101 is connected, exact clean
base7f32e68, H200 visible and no compute process reported before this round.
The historical hang cause is still unresolved; clean checks only cover the
observed round. No unrelated process is terminated.

## Cleanup

The new explicit ledger entry collapses the policy to its eventual selected
worker consumer at P12.4e, removes the unused Warp exception if CUDA wins, or
removes the module if no selected worker adopts it. Historical review records
are retained. This does not claim that backend selection/cleanup has happened.
