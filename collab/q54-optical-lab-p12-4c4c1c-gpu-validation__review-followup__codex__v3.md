# C4c1c source execution and readback ownership repairs

Initiative: q54-optical-lab-p12-4c4c1c-gpu-validation
Stage: review-followup
Author: codex
Version: v3
Date: 2026-09-28
Status: in_review
Related Files: GPU-validation review v2; implementation note v1
Owner Summary: Repair both independent code findings without waiving the
missing Linux/GPU proof. This is a new candidate, not C4c1c acceptance.

Prior reviewed candidate: `f2829d26233892bf9a17ead4484f48457477af37`.
The full independent review v2 is archived verbatim; its file SHA-256 is
`f9fe8b28b0f207a3a74d8a9ff55f9ceec54d151b9d62c54cad4ab11a469a76a6`.
The incomplete v1 review session is recorded separately as a tooling failure,
not approval evidence. The candidate containing this note is the re-review HEAD.

## F1: execute the bytes that were verified

The fixed supervisor and child create separate fresh empty cache namespaces
with `exist_ok=False`, set `sys.pycache_prefix`, and disable cache writes before
their first repository helper import. This covers bootstrap, including the
source ledger module itself; `-B` alone was not sufficient.

After ledger installation, `VerifiedRepositorySourceLoader.get_code` reads
source bytes, compares their digest with the manifest, and compiles that exact
byte buffer. It never calls the timestamp/hash-cache code path. An intervening
source change or a different module identity fails before compilation.

New CPU tests cover ordinary modules and package initializers with both
timestamp-colliding and unchecked-hash stale caches; both execute the current
fixed constant. Another test changes source after loader construction and
expects rejection. AST checks cover cache namespace creation before the first
repository import in both actual entry scripts. Actual Linux bootstrap remains
part of the required pending supervisor/GPU matrix, not inferred from AST.

The independent reviewer’s unchanged external reproduction was rerun. It now
prints `SOURCE: VALUE = 'new'`, `EXECUTED_VALUE: new`, and a valid source ledger.
Its old `assert VALUE == 'old'` fails, as expected for the repaired behavior.
The original reproduction file was not edited.

## F2: explicit readback owners, one formal comparison implementation

The fixed setup no longer calls array `.numpy()` for input parity. For every
array it creates a host destination and a Warp CPU view, adopts each into the
guardian, verifies exact custody, then enqueues the explicit copy on the frozen
stream. The host observations are inspected only after an S-ACK-supervised
stream wait. The destination arrays and views remain held across subsequent
setup leaves, rendering, and the required release/quarantine disposition.

The existing native validator's comparison loop/report construction is factored
into `_validate_native_input_host_observations`. The ordinary API still supplies
its existing runtime materializer lazily; the fixed driver supplies its ready
owned observations. There is one dtype/shape/order/content/hash checker, not a
test-only copy. This CPU helper neither certifies readback ownership nor creates
readiness authority. Existing readiness registration and exact physical-storage
checks are unchanged. No observation/callback input is added to worker IPC.

Production code delta is limited to that shared comparison extraction and one
accurate setup-leaf classification, `device_to_host_copy`, in the existing
CPU-safe descriptor. The product A/K protocol, request surface, source allowlist
and activation gates are unchanged. This classification does not enable host
payload delivery or GPU preparation.

Eight CPU tests compare ordinary-versus-explicit validation and reject missing,
extra, wrong-name, wrong-content, dtype, shape and stride observations. Five
new real-GPU cases require owner custody across host-storage/view adoption,
pre-copy failure, copy-submit-then-raise, and failure before readback return.
They are not yet executed here. A missing local Warp/Torch installation is not
replaced by fake GPU proof.

## Fresh local evidence and outstanding work

Windows Python 3.14:

```
python -m pytest tests/unit/benchmarks/test_native_input_observations.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py -q
```

Result: **130 passed, 21 skipped in 8.09s**. The 21 skips are POSIX subprocess
cases. Changed-code Ruff and diff checks pass. GPU tests are collection only.

An earlier broader local selection reported seven failures in existing
benchmark-core runner tests because the instrumentation plan cannot import
`torch.empty` in this Python environment. No package was installed and those
tests were not relabeled as passing. An initial new-test fixture used incorrect
constructor keyword names; it was corrected to the actual API before the
passing run above.

183:40031 and 101:22 still time out; local VPN adapters are disconnected.
The final repaired candidate needs clean Linux/POSIX/GPU execution and a fresh
independent review before acceptance. All GPU/runtime obligations in review v2
remain required, including the new readback cases and two-device test.

C4c2-c4, C4d/B3, product GPU preparation, a3b, host/shipping paths, result and
baseline outlets remain closed. No baseline was generated, no dependency was
installed, no default/backend choice was changed, and no remote push occurred.
